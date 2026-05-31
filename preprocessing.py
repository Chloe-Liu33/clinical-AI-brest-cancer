"""METABRIC preprocessing: clinical + gene expression -> training pickles.

Input  (raw/):
  - data_clinical_patient.txt   patient-level clinical data
  - data_clinical_sample.txt    sample-level clinical data (one row per sample)
  - data_mrna_illumina_microarray.txt   gene x sample expression matrix (log2)

Output (processed_metabric/):
  - X_struct.npy        [N, struct_dim] standardized clinical features
  - X_gene.npy          [N, gene_dim]   gene expression (log2)
  - y.npy               [N]             5-year overall survival label (0/1)
  - split.pkl           {'train': idx, 'val': idx, 'test': idx}
  - struct_cols.pkl     list of feature column names
  - gene_names.pkl      list of Hugo gene symbols (same order as X_gene cols)
  - scaler_struct.pkl   StandardScaler fit on train numeric features

Label:
  5-year OS = 1 if OS_STATUS == '1:DECEASED' AND OS_MONTHS <= 60
            = 0 if OS_MONTHS >= 60   (alive at 5 years)
            = drop otherwise (right-censored before 60 months -> uncertain)
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Features (no target leakage — molecular subtypes derived from mRNA are excluded).
NUMERIC = [
    "AGE_AT_DIAGNOSIS", "NPI", "LYMPH_NODES_EXAMINED_POSITIVE",
    "TUMOR_SIZE", "TMB_NONSYNONYMOUS", "TUMOR_STAGE", "GRADE",
]
CATEGORICAL = [
    "CELLULARITY", "CHEMOTHERAPY", "ER_IHC", "HER2_SNP6",
    "HORMONE_THERAPY", "INFERRED_MENOPAUSAL_STATE",
    "INTCLUST", "LATERALITY", "RADIO_THERAPY", "BREAST_SURGERY",
    "HISTOLOGICAL_SUBTYPE", "ER_STATUS", "PR_STATUS", "HER2_STATUS",
]


def read_cbioportal_clinical(path: str) -> pd.DataFrame:
    """cBioPortal clinical files have 4 metadata header lines starting with #."""
    return pd.read_csv(path, sep="\t", skiprows=4)


def build_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    deceased = df["OS_STATUS"].astype(str).str.startswith("1")

    label = pd.Series(np.nan, index=df.index, dtype=float)
    label[(deceased) & (df["OS_MONTHS"] <= 60)] = 1.0
    label[df["OS_MONTHS"] >= 60] = 0.0
    df["label_5y"] = label
    return df.dropna(subset=["label_5y"]).reset_index(drop=True)


def build_struct_features(df: pd.DataFrame, scaler: StandardScaler | None = None
                          ) -> tuple[np.ndarray, list[str], StandardScaler]:
    """Numeric: median-impute. Categorical: 'Unknown' fill, one-hot. Scale numeric only.

    Pass `scaler=None` on training to fit; reuse the returned scaler for val/test.
    """
    feats = df.copy()
    for c in NUMERIC:
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
        feats[c] = feats[c].fillna(feats[c].median())
    for c in CATEGORICAL:
        feats[c] = feats[c].fillna("Unknown").astype(str)

    X_num = feats[NUMERIC].astype(np.float32).values
    if scaler is None:
        scaler = StandardScaler().fit(X_num)
    X_num = scaler.transform(X_num).astype(np.float32)

    X_cat = pd.get_dummies(feats[CATEGORICAL], columns=CATEGORICAL, drop_first=False)
    X_cat_arr = X_cat.values.astype(np.float32)

    X = np.concatenate([X_num, X_cat_arr], axis=1)
    cols = NUMERIC + X_cat.columns.tolist()
    return X, cols, scaler


def load_gene_expression(path: str) -> pd.DataFrame:
    """Returns DataFrame indexed by SAMPLE_ID, columns = Hugo gene symbols."""
    print(f"  loading {os.path.basename(path)} (~660MB, ~30s) ...")
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["Hugo_Symbol"])
    if "Entrez_Gene_Id" in df.columns:
        df = df.drop(columns=["Entrez_Gene_Id"])
    # Multiple probes can map to the same Hugo symbol -> mean.
    df = df.groupby("Hugo_Symbol").mean(numeric_only=True)
    return df.T  # transpose: rows=samples, cols=genes


def main(raw_dir: str, out_dir: str, seed: int = 42) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("[1/5] Reading clinical files...")
    patient = read_cbioportal_clinical(os.path.join(raw_dir, "data_clinical_patient.txt"))
    sample  = read_cbioportal_clinical(os.path.join(raw_dir, "data_clinical_sample.txt"))
    clin    = sample.merge(patient, on="PATIENT_ID", how="inner")
    print(f"  merged: {clin.shape}")

    print("[2/5] Building 5-year survival label...")
    clin = build_label(clin)
    n_pos = int(clin["label_5y"].sum())
    n_neg = len(clin) - n_pos
    print(f"  kept {len(clin)} (pos={n_pos}, neg={n_neg}, "
          f"pos_rate={n_pos/len(clin):.3f})")

    print("[3/5] Loading gene expression...")
    gene_df = load_gene_expression(
        os.path.join(raw_dir, "data_mrna_illumina_microarray.txt"))
    print(f"  expression matrix: {gene_df.shape}")

    print("[4/5] Aligning samples between clinical and expression...")
    common = sorted(set(clin["SAMPLE_ID"]) & set(gene_df.index))
    clin   = clin[clin["SAMPLE_ID"].isin(common)].reset_index(drop=True)
    clin   = clin.sort_values("SAMPLE_ID").reset_index(drop=True)
    gene_df = gene_df.loc[clin["SAMPLE_ID"].tolist()]

    # Drop genes that are all-NaN after alignment, median-fill the rest.
    gene_df = gene_df.dropna(axis=1, how="all")
    gene_df = gene_df.fillna(gene_df.median())
    X_gene  = gene_df.values.astype(np.float32)
    gene_names = gene_df.columns.tolist()
    print(f"  aligned: {len(clin)} samples, {len(gene_names)} genes")

    print("[5/5] Splits, struct features, save...")
    y   = clin["label_5y"].astype(np.int64).values
    idx = np.arange(len(y))
    idx_tv, idx_test = train_test_split(idx, test_size=0.15, stratify=y, random_state=seed)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.15 / 0.85, stratify=y[idx_tv], random_state=seed,
    )
    print(f"  train/val/test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

    # Fit scalers on train only (no leakage).
    X_struct_train, struct_cols, scaler = build_struct_features(
        clin.iloc[idx_train], scaler=None)
    X_struct_all, _, _ = build_struct_features(clin, scaler=scaler)

    # Variance ranking on RAW gene expression (descending), train only.
    # Saved BEFORE z-score so the ranking reflects original biological variability.
    gene_var_rank = np.argsort(-X_gene[idx_train].var(axis=0)).astype(np.int64)

    # Z-score genes using train mean/std.
    gene_scaler = StandardScaler().fit(X_gene[idx_train])
    X_gene = gene_scaler.transform(X_gene).astype(np.float32)

    np.save(os.path.join(out_dir, "X_struct.npy"),     X_struct_all)
    np.save(os.path.join(out_dir, "X_gene.npy"),       X_gene)
    np.save(os.path.join(out_dir, "y.npy"),            y)
    np.save(os.path.join(out_dir, "gene_var_rank.npy"), gene_var_rank)
    with open(os.path.join(out_dir, "split.pkl"), "wb") as f:
        pickle.dump({"train": idx_train, "val": idx_val, "test": idx_test}, f)
    with open(os.path.join(out_dir, "struct_cols.pkl"), "wb") as f:
        pickle.dump(struct_cols, f)
    with open(os.path.join(out_dir, "gene_names.pkl"), "wb") as f:
        pickle.dump(gene_names, f)
    with open(os.path.join(out_dir, "scaler_struct.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "scaler_gene.pkl"), "wb") as f:
        pickle.dump(gene_scaler, f)

    print("\nSaved:")
    print(f"  X_struct {X_struct_all.shape}  X_gene {X_gene.shape}  y {y.shape}")
    print(f"  out_dir: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="raw")
    p.add_argument("--out_dir", default="processed_metabric")
    p.add_argument("--seed", type=int, default=42)
    main(**vars(p.parse_args()))
