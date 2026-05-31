"""METABRIC -- continuous OS prediction via Cox proportional hazards (DRAFT).

Cross-modal Transformer (Clinical EHR + Gene Expression) with HSIC top-K,
trained with Cox partial likelihood and evaluated by Concordance Index (C-index).
This makes results directly comparable to DeepSurv / DeepProg / MOFA+.

Key differences from main_AUROC.py (binary 5-year OS):
  * Labels = (OS_MONTHS, OS_STATUS) tuple, kept as continuous (time, event)
  * Early-censored patients are KEPT (vs dropped in main_AUROC.py preprocessing) ->
    ~25% more training samples
  * Loss = Cox partial likelihood (within-batch approximation, DeepSurv-style)
  * Eval = C-index (lifelines if installed; manual O(N^2) fallback otherwise)
  * R-Drop disabled (KL on sigmoid doesn't transfer to risk scores; can re-add
    as MSE on risk pairs if needed)

Reuses model architectures from main_AUROC.py via import (does not modify it).
Re-derives features from raw cBioPortal files because main_AUROC.py preprocessing
drops samples censored before 60 months, which we want to keep for survival.

Usage:
  python main_C-index.py --raw_dir ./raw
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup


# =============================================================
# 0. Reuse architectures from main_AUROC.py (no edits to that file)
# =============================================================

def _import_main_auroc():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "metabric_main_auroc", os.path.join(here, "main_AUROC.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_main = _import_main_auroc()
StructOnlyMLP        = _main.StructOnlyMLP
GeneOnlyMLP          = _main.GeneOnlyMLP
GeneVarTopKMLP       = _main.GeneVarTopKMLP
LateFusionConcat     = _main.LateFusionConcat
CrossModalGenomic    = _main.CrossModalGenomic
CnaOnlyMLP           = _main.CnaOnlyMLP
LateFusionConcat3    = _main.LateFusionConcat3
CrossModalGenomicTri = _main.CrossModalGenomicTri


# =============================================================
# 1. Configuration
# =============================================================

D_MODEL      = 128
N_HEADS      = 4
DROPOUT      = 0.3
BATCH_SIZE   = 128       # Cox needs more events per batch for risk-set stability
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 0.01
PATIENCE     = 15
WARMUP_RATIO = 0.1

HSIC_TOP_K_SWEEP = [256, 512, 1024, 2048]
HSIC_TOP_K_TRI   = [256, 512]   # Tri-modal: only sweep the 2-modal sweet spot

SEEDS = [42, 7, 123, 2024, 31415]

BOOTSTRAP_N  = 1000
BOOTSTRAP_CI = 0.95

DEVICE: torch.device = torch.device("cpu")
WORLD_SIZE = 1
RANK = 0
LOCAL_RANK = 0


# =============================================================
# 2. DDP / seed helpers
# =============================================================

def setup_ddp() -> None:
    global DEVICE, WORLD_SIZE, RANK, LOCAL_RANK
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        RANK       = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(LOCAL_RANK)
        DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    return RANK == 0


def log(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


def set_all_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


# =============================================================
# 3. Survival dataset
# =============================================================

class SurvivalDataset(Dataset):
    def __init__(self, X_struct: np.ndarray, X_gene: np.ndarray,
                 times: np.ndarray, events: np.ndarray,
                 X_cna: np.ndarray | None = None):
        self.X_struct = torch.tensor(X_struct, dtype=torch.float32)
        self.X_gene   = torch.tensor(X_gene,   dtype=torch.float32)
        self.X_cna    = (torch.tensor(X_cna, dtype=torch.float32)
                          if X_cna is not None else None)
        self.times    = torch.tensor(times,    dtype=torch.float32)
        self.events   = torch.tensor(events,   dtype=torch.float32)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        item = {
            "struct": self.X_struct[idx],
            "gene":   self.X_gene[idx],
            "time":   self.times[idx],
            "event":  self.events[idx],
        }
        if self.X_cna is not None:
            item["cna"] = self.X_cna[idx]
        return item


def _forward_risk(model, batch, device):
    """Forward through 2-modal or 3-modal model (dispatched on `is_tri` flag)."""
    s = batch["struct"].to(device, non_blocking=True)
    g = batch["gene"].to(device, non_blocking=True)
    raw = model.module if hasattr(model, "module") else model
    if getattr(raw, "is_tri", False):
        c = batch["cna"].to(device, non_blocking=True)
        return model(s, g, c)
    return model(s, g)


# =============================================================
# 4. Cox partial likelihood loss + C-index
# =============================================================

def cox_ph_loss(risks: torch.Tensor, times: torch.Tensor,
                events: torch.Tensor) -> torch.Tensor:
    """Negative log Cox partial likelihood (DeepSurv, within-batch approx).

    Sort by time descending; for each sample i, the risk set R(t_i) becomes the
    prefix of the sorted list, so log Sum_{j in R(t_i)} exp(risk_j) = logcumsumexp.

    L = - Sum_{i: event_i=1} [risk_i - logcumsumexp(risks)_i]  /  n_events
    """
    if events.sum() < 1:                             # no events in this batch
        return risks.sum() * 0.0                     # zero w/ grad path

    order  = torch.argsort(times, descending=True)
    risks  = risks[order]
    events = events[order]

    log_cumsum = torch.logcumsumexp(risks, dim=0)
    pl = risks - log_cumsum                          # per-sample partial logl
    return -(pl * events).sum() / events.sum().clamp(min=1)


def c_index(times: np.ndarray, risks: np.ndarray, events: np.ndarray) -> float:
    """Concordance index. Higher risk -> earlier expected event."""
    try:
        from lifelines.utils import concordance_index
        return float(concordance_index(times, -risks, events))
    except ImportError:
        # O(N^2) fallback. Fine for ~300 test samples.
        n = len(times)
        pairs = 0
        concordant = 0.0
        for i in range(n):
            if events[i] != 1:
                continue
            for j in range(n):
                if i == j or times[j] <= times[i]:
                    continue
                pairs += 1
                if risks[i] > risks[j]:
                    concordant += 1.0
                elif risks[i] == risks[j]:
                    concordant += 0.5
        return concordant / pairs if pairs > 0 else 0.5


def bootstrap_cindex(times: np.ndarray, risks: np.ndarray, events: np.ndarray,
                     n: int = BOOTSTRAP_N, ci: float = BOOTSTRAP_CI,
                     seed: int = 42) -> tuple[float, float, float]:
    """Test-set bootstrap CI for C-index. Resamples N samples with replacement."""
    rng = np.random.default_rng(seed)
    N = len(times)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        if events[idx].sum() < 1:
            continue
        try:
            scores.append(c_index(times[idx], risks[idx], events[idx]))
        except Exception:
            continue
    point = c_index(times, risks, events)
    if not scores:
        return point, float("nan"), float("nan")
    alpha = 1 - ci
    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_t_pvalue(scores_a: list, scores_b: list) -> float:
    """Paired t-test p-value (two-sided). Used for per-seed C-index pairs."""
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if len(a) < 2 or np.allclose(a, b):
        return float("nan")
    try:
        from scipy.stats import ttest_rel
        return float(ttest_rel(a, b).pvalue)
    except ImportError:
        # Manual paired t-test fallback.
        d = a - b
        from math import sqrt
        n = len(d)
        mean_d = d.mean()
        sd = d.std(ddof=1)
        if sd == 0:
            return float("nan")
        t = mean_d / (sd / sqrt(n))
        # Approximate two-sided p via normal (n=5 is small but fallback OK)
        from math import erf
        return float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


# =============================================================
# 5. Data prep -- re-derive from raw to keep early-censored samples
# =============================================================

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


def _load_genomic_table(path: str) -> pd.DataFrame:
    """Read a cBioPortal gene-by-sample TSV; return [samples x Hugo_Symbol] DF."""
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["Hugo_Symbol"])
    if "Entrez_Gene_Id" in df.columns:
        df = df.drop(columns=["Entrez_Gene_Id"])
    return df.groupby("Hugo_Symbol").mean(numeric_only=True).T


def prepare_survival_data(raw_dir: str, seed: int = 42) -> dict:
    """Load raw cBioPortal files, build (time, event) labels and features.

    Differs from main_AUROC.py preprocessing.py by KEEPING early-censored samples.
    All scalers / variance ranks fit on train indices only (no leakage).

    Tri-modal extension: also loads CNA. Only samples present in all three
    sources (clinical, mRNA, CNA) are kept so every model trains on the same N.
    Also returns X_struct_noICL (INTCLUST one-hot columns dropped) for the
    leakage-control ablation -- INTCLUST is a published 10-class clustering
    derived from joint mRNA+CNA, so it leaks both genomic modalities into the
    "clinical" feature.
    """
    log("[1/5] Reading raw clinical files...")
    patient = pd.read_csv(os.path.join(raw_dir, "data_clinical_patient.txt"),
                           sep="\t", skiprows=4)
    sample  = pd.read_csv(os.path.join(raw_dir, "data_clinical_sample.txt"),
                           sep="\t", skiprows=4)
    clin = sample.merge(patient, on="PATIENT_ID", how="inner")

    clin["OS_MONTHS"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    clin["event"] = clin["OS_STATUS"].astype(str).str.startswith("1").astype(np.float32)
    clin = clin.dropna(subset=["OS_MONTHS"]).reset_index(drop=True)
    log(f"  total with valid (time, status): {len(clin)} "
        f"(events={int(clin['event'].sum())}, "
        f"censored={int((1 - clin['event']).sum())})")

    log("[2/5] Loading mRNA expression (~660MB, ~30s)...")
    gene_df = _load_genomic_table(
        os.path.join(raw_dir, "data_mrna_illumina_microarray.txt"))

    log("[3/5] Loading CNA (~80MB)...")
    cna_df = _load_genomic_table(os.path.join(raw_dir, "data_cna.txt"))

    log("[4/5] Aligning samples (clinical ∩ mRNA ∩ CNA)...")
    common = sorted(set(clin["SAMPLE_ID"])
                    & set(gene_df.index)
                    & set(cna_df.index))
    clin = clin[clin["SAMPLE_ID"].isin(common)].sort_values("SAMPLE_ID").reset_index(drop=True)
    gene_df = gene_df.loc[clin["SAMPLE_ID"].tolist()]
    cna_df  = cna_df.loc[clin["SAMPLE_ID"].tolist()]

    gene_df = gene_df.dropna(axis=1, how="all")
    # Vectorized median fill: pandas' fillna(Series) is O(N*M log M) on a
    # 2k x 20k frame (~30 min); np.where on .values is ~1s.
    if gene_df.isna().values.any():
        col_med = gene_df.median(axis=0).values
        arr = np.where(np.isnan(gene_df.values), col_med, gene_df.values)
        gene_df = pd.DataFrame(arr, index=gene_df.index, columns=gene_df.columns)
    # CNA: discrete {-2,-1,0,1,2}; missing -> 0 (no alteration).
    cna_df  = cna_df.dropna(axis=1, how="all").fillna(0)

    times  = clin["OS_MONTHS"].astype(np.float32).values
    events = clin["event"].values.astype(np.float32)
    log(f"  aligned: {len(clin)} samples (events={int(events.sum())}), "
        f"{gene_df.shape[1]} mRNA genes, {cna_df.shape[1]} CNA genes")

    log("[5/5] Splits + features (event-stratified, fit-on-train)...")
    n   = len(clin)
    idx = np.arange(n)
    idx_tv, idx_test = train_test_split(idx, test_size=0.15,
                                          stratify=events, random_state=seed)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.15 / 0.85,
                                            stratify=events[idx_tv], random_state=seed)
    log(f"  train/val/test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

    feats = clin.copy()
    for c in NUMERIC:
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
        train_med = feats.iloc[idx_train][c].median()
        feats[c] = feats[c].fillna(train_med)
    for c in CATEGORICAL:
        feats[c] = feats[c].fillna("Unknown").astype(str)

    X_num    = feats[NUMERIC].astype(np.float32).values
    scaler_s = StandardScaler().fit(X_num[idx_train])
    X_num    = scaler_s.transform(X_num).astype(np.float32)
    X_cat_df = pd.get_dummies(feats[CATEGORICAL], columns=CATEGORICAL,
                               drop_first=False)
    X_cat    = X_cat_df.values.astype(np.float32)
    struct_cols = NUMERIC + X_cat_df.columns.tolist()
    X_struct    = np.concatenate([X_num, X_cat], axis=1)

    # INTCLUST_* one-hot columns leak joint mRNA+CNA cluster info into clinical.
    keep_mask = np.array([not c.startswith("INTCLUST_") for c in struct_cols])
    X_struct_noICL = X_struct[:, keep_mask]
    n_intclust = (~keep_mask).sum()
    log(f"  X_struct {X_struct.shape}; "
        f"X_struct_noICL {X_struct_noICL.shape} (dropped {n_intclust} INTCLUST cols)")

    gene_names    = gene_df.columns.tolist()
    X_gene_raw    = gene_df.values.astype(np.float32)
    gene_var_rank = np.argsort(-X_gene_raw[idx_train].var(axis=0)).astype(np.int64)
    scaler_g      = StandardScaler().fit(X_gene_raw[idx_train])
    X_gene        = scaler_g.transform(X_gene_raw).astype(np.float32)

    X_cna_raw = cna_df.values.astype(np.float32)
    scaler_c  = StandardScaler().fit(X_cna_raw[idx_train])
    X_cna     = scaler_c.transform(X_cna_raw).astype(np.float32)

    # DeepSurv-9d compatible "lean" baseline (Katzman 2018, BMC Med Res Methodol):
    # 5 basic clinical (age, ER, hormone/radio/chemo) + 4 mRNA marker genes.
    # Used as a published reference baseline that excludes derived features
    # (INTCLUST, NPI, PAM50, etc.).
    LEAN_NUM     = ["AGE_AT_DIAGNOSIS"]
    LEAN_CAT     = ["ER_STATUS", "HORMONE_THERAPY", "RADIO_THERAPY", "CHEMOTHERAPY"]
    LEAN_MARKERS = ["MKI67", "EGFR", "PGR", "ERBB2"]

    X_lean_num_raw = feats[LEAN_NUM].astype(np.float32).values
    scaler_lean    = StandardScaler().fit(X_lean_num_raw[idx_train])
    X_lean_num     = scaler_lean.transform(X_lean_num_raw).astype(np.float32)
    X_lean_cat     = pd.get_dummies(feats[LEAN_CAT], drop_first=False).values.astype(np.float32)
    marker_idx     = [gene_names.index(g) for g in LEAN_MARKERS if g in gene_names]
    missing        = [g for g in LEAN_MARKERS if g not in gene_names]
    if missing:
        log(f"  WARN: lean markers missing from mRNA panel: {missing}")
    X_lean_genes   = X_gene[:, marker_idx]
    X_struct_lean  = np.concatenate([X_lean_num, X_lean_cat, X_lean_genes], axis=1)

    log(f"  X_gene {X_gene.shape}, X_cna {X_cna.shape}, "
        f"X_struct_lean {X_struct_lean.shape} (DeepSurv-9d)")
    return {
        "X_struct":       X_struct,
        "X_struct_noICL": X_struct_noICL,
        "X_struct_lean":  X_struct_lean,
        "X_gene":         X_gene,
        "X_cna":          X_cna,
        "times":          times,
        "events":         events,
        "split":          {"train": idx_train, "val": idx_val, "test": idx_test},
        "gene_var_rank":  gene_var_rank,
    }


# =============================================================
# 6. Train / eval (Cox)
# =============================================================

def train_one_epoch_cox(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        times  = batch["time"].to(device, non_blocking=True)
        events = batch["event"].to(device, non_blocking=True)

        risks, _ = _forward_risk(model, batch, device)
        loss = cox_ph_loss(risks, times, events)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_cox(model, loader, device):
    model.eval()
    risks, times, events = [], [], []
    for batch in loader:
        r, _ = _forward_risk(model, batch, device)
        risks.extend(r.cpu().numpy().tolist())
        times.extend(batch["time"].numpy().tolist())
        events.extend(batch["event"].numpy().tolist())
    return np.array(risks), np.array(times), np.array(events)


def train_model_cox(model, data, max_epochs=EPOCHS):
    split  = data["split"]
    is_tri = getattr(model, "is_tri", False)

    def _make_ds(idx):
        return SurvivalDataset(
            data["X_struct"][idx],
            data["X_gene"][idx],
            data["times"][idx],
            data["events"][idx],
            X_cna=data["X_cna"][idx] if is_tri else None,
        )

    train_ds = _make_ds(split["train"])
    val_ds   = _make_ds(split["val"])

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp() else None
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                            sampler=train_sampler, shuffle=(train_sampler is None),
                            num_workers=2, pin_memory=True)
    val_dl = (DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=True)
                if is_main() else None)

    if is_ddp():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    total_steps  = len(train_dl) * max_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    log(f"    Total steps: {total_steps}, warmup: {warmup_steps}, batch: {BATCH_SIZE}")

    # Full-train tensors for epoch-level HSIC update (Ours / Ours-Tri only).
    X_struct_full = torch.tensor(data["X_struct"][split["train"]], dtype=torch.float32)
    X_gene_full   = torch.tensor(data["X_gene"][split["train"]],   dtype=torch.float32)
    X_cna_full    = (torch.tensor(data["X_cna"][split["train"]], dtype=torch.float32)
                      if is_tri else None)

    best_cidx, best_state, pat = 0.0, None, 0
    for epoch in range(1, max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Recompute HSIC on full train set at start of each epoch (epoch 2+).
        # Epoch 1 trains with no mask so struct_proj has a warm-up.
        raw_m = _unwrap(model)
        if epoch > 1 and hasattr(raw_m, "update_hsic"):
            if is_tri:
                raw_m.update_hsic(X_struct_full, X_gene_full, X_cna_full)
            else:
                raw_m.update_hsic(X_struct_full, X_gene_full)

        t0 = time.time()
        loss = train_one_epoch_cox(model, train_dl, optimizer, scheduler, DEVICE)

        if is_main():
            r, t, e = evaluate_cox(_unwrap(model), val_dl, DEVICE)
            cidx = c_index(t, r, e)
        else:
            cidx = 0.0
        if is_ddp():
            tt = torch.tensor([cidx], device=DEVICE)
            dist.broadcast(tt, src=0); cidx = tt.item()

        if is_main() and (epoch % 5 == 0 or epoch == 1):
            print(f"    Epoch {epoch:3d}/{max_epochs} | loss={loss:.4f} | "
                  f"val_cidx={cidx:.4f} | {time.time()-t0:.1f}s")

        if cidx > best_cidx:
            best_cidx = cidx
            if is_main():
                raw = _unwrap(model)
                best_state = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                log(f"    Early stop @ epoch {epoch} (best_cidx={best_cidx:.4f})")
                break

    raw = _unwrap(model)
    if is_main() and best_state is not None:
        raw.load_state_dict(best_state)
    return raw, best_cidx


# =============================================================
# 7. Multi-seed ablation
# =============================================================

def _train_one_config(name: str, build, data: dict, results: dict) -> None:
    """Train one model factory across SEEDS, write aggregated metrics into results."""
    log(f"\n{'='*60}")
    log(f"  Training: {name}  (seeds: {SEEDS})")
    log(f"{'='*60}")

    per_seed = []
    n_params = 0
    for seed in SEEDS:
        log(f"\n  --- Seed {seed} ---")
        set_all_seeds(seed)
        model = build()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model, val_cidx = train_model_cox(model, data, max_epochs=EPOCHS)

        if is_main():
            is_tri = getattr(model, "is_tri", False)
            test_ds = SurvivalDataset(
                data["X_struct"][data["split"]["test"]],
                data["X_gene"][data["split"]["test"]],
                data["times"][data["split"]["test"]],
                data["events"][data["split"]["test"]],
                X_cna=data["X_cna"][data["split"]["test"]] if is_tri else None,
            )
            test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=2, pin_memory=True)
            r, t, e = evaluate_cox(model, test_dl, DEVICE)
            test_cidx = c_index(t, r, e)
            _, ci_lo, ci_hi = bootstrap_cindex(t, r, e, seed=seed)

            # Also evaluate on the train split so calibration plots can fit a
            # Breslow baseline H0(t) per seed (needed for S(60), S(120)).
            train_ds = SurvivalDataset(
                data["X_struct"][data["split"]["train"]],
                data["X_gene"][data["split"]["train"]],
                data["times"][data["split"]["train"]],
                data["events"][data["split"]["train"]],
                X_cna=data["X_cna"][data["split"]["train"]] if is_tri else None,
            )
            train_dl_eval = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=2, pin_memory=True)
            r_tr, t_tr, e_tr = evaluate_cox(model, train_dl_eval, DEVICE)

            log(f"    [seed {seed}] val_cidx={val_cidx:.4f}  "
                f"test_cidx={test_cidx:.4f}  "
                f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
            per_seed.append({"seed": seed,
                              "val_cidx":  float(val_cidx),
                              "test_cidx": float(test_cidx),
                              "ci_lo":     ci_lo,
                              "ci_hi":     ci_hi,
                              # Per-patient arrays for post-hoc calibration:
                              "train_risks":  r_tr.astype(np.float32),
                              "train_times":  t_tr.astype(np.float32),
                              "train_events": e_tr.astype(np.int8),
                              "test_risks":   r.astype(np.float32),
                              "test_times":   t.astype(np.float32),
                              "test_events":  e.astype(np.int8)})
        if is_ddp():
            dist.barrier()

    if is_main():
        arr = np.array([s["test_cidx"] for s in per_seed])
        results[name] = {
            "cidx_mean":  float(arr.mean()),
            "cidx_std":   float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "_per_seed":  per_seed,
            "_n_params":  n_params,
        }
        log(f"\n  [{name} -- {len(SEEDS)}-seed mean ± std]")
        log(f"    test C-index = {results[name]['cidx_mean']:.4f}"
            f"±{results[name]['cidx_std']:.4f}")


def run_ablation(data: dict) -> dict:
    struct_dim = data["X_struct"].shape[1]
    gene_dim   = data["X_gene"].shape[1]
    cna_dim    = data["X_cna"].shape[1]

    n_train_events = int(data["events"][data["split"]["train"]].sum())
    log(f"\n  Train events: {n_train_events} / {len(data['split']['train'])}")

    # Note: model factories return models that output 1-d real-valued risk
    # scores. The Cox loss treats them as log-hazard ratios; no sigmoid.
    model_configs: dict = {
        "Clinical-only (MLP)":  lambda: StructOnlyMLP(struct_dim, gene_dim).to(DEVICE),
        "Gene-only (MLP)":      lambda: GeneOnlyMLP(struct_dim, gene_dim).to(DEVICE),
        "CNA-only (MLP)":       lambda: CnaOnlyMLP(struct_dim, gene_dim, cna_dim).to(DEVICE),
        "Late Fusion (Concat)": lambda: LateFusionConcat(struct_dim, gene_dim).to(DEVICE),
        "Late Fusion 3 (Concat)":
            lambda: LateFusionConcat3(struct_dim, gene_dim, cna_dim).to(DEVICE),
    }
    # Variance baseline swept over the same K as HSIC.
    for k in HSIC_TOP_K_SWEEP:
        var_top_idx_k = data["gene_var_rank"][:k]
        model_configs[f"Gene Var-{k} (MLP)"] = (
            lambda kk=k, idx=var_top_idx_k:
                GeneVarTopKMLP(struct_dim, gene_dim, idx).to(DEVICE)
        )
    for k in HSIC_TOP_K_SWEEP:
        model_configs[f"Ours (HSIC K={k})"] = (
            lambda kk=k: CrossModalGenomic(struct_dim, gene_dim, top_k=kk).to(DEVICE)
        )
    for k in HSIC_TOP_K_TRI:
        model_configs[f"Ours-Tri (HSIC K={k})"] = (
            lambda kk=k: CrossModalGenomicTri(
                struct_dim, gene_dim, cna_dim, top_k=kk).to(DEVICE)
        )

    results: dict = {}
    for name, build in model_configs.items():
        _train_one_config(name, build, data, results)

    # ---- Clinical-lean baseline (DeepSurv-9d compatible) -------------------
    # Strict published reference: 5 basic clinical + 4 mRNA markers, no
    # derived features. Lets the paper claim "we beat both rich-clinical
    # (CURE-style) and lean-clinical (DeepSurv-style) baselines."
    log(f"\n{'#'*60}")
    log(f"  Clinical-lean baseline (DeepSurv-9d compatible)")
    log(f"{'#'*60}")
    data_lean        = {**data, "X_struct": data["X_struct_lean"]}
    struct_dim_lean  = data_lean["X_struct"].shape[1]
    _train_one_config(
        "Clinical-lean (DeepSurv-9d)",
        lambda: StructOnlyMLP(struct_dim_lean, gene_dim).to(DEVICE),
        data_lean, results,
    )

    # ---- INTCLUST-leakage ablation -----------------------------------------
    # INTCLUST is a published 10-class clustering computed from joint mRNA+CNA;
    # leaving it in "clinical" credits the structured branch with mRNA/CNA
    # signal. Drop it and re-run the headline contrasts to see the *true* gap.
    log(f"\n{'#'*60}")
    log(f"  INTCLUST-leakage ablation (X_struct without INTCLUST_* one-hots)")
    log(f"{'#'*60}")
    data_noICL = {**data, "X_struct": data["X_struct_noICL"]}
    struct_dim_noICL = data_noICL["X_struct"].shape[1]

    noicl_configs = {
        "Clinical-noICL (MLP)":
            lambda: StructOnlyMLP(struct_dim_noICL, gene_dim).to(DEVICE),
        "Late Fusion 3-noICL (Concat)":
            lambda: LateFusionConcat3(struct_dim_noICL, gene_dim, cna_dim).to(DEVICE),
        "Ours-Tri-noICL (HSIC K=512)":
            lambda: CrossModalGenomicTri(
                struct_dim_noICL, gene_dim, cna_dim, top_k=512).to(DEVICE),
    }
    for name, build in noicl_configs.items():
        _train_one_config(name, build, data_noICL, results)

    return results


# =============================================================
# 8. Main
# =============================================================

def main(raw_dir: str, out_dir: str, data_seed: int):
    setup_ddp()

    log("=" * 60)
    log("  METABRIC -- Continuous OS prediction (Cox / C-index)")
    log("  Cross-modal Transformer (Clinical + Gene Expression)")
    log("=" * 60)
    log(f"Device: {DEVICE} | World size: {WORLD_SIZE}")

    data = prepare_survival_data(raw_dir, seed=data_seed)

    log(f"\nHSIC config: TOP_K_SWEEP={HSIC_TOP_K_SWEEP}, "
        f"estimation=full-train per epoch (epoch 1 = no mask)")
    log(f"Variance baseline: K_SWEEP={HSIC_TOP_K_SWEEP} (matched to HSIC)")
    log(f"Cox loss + C-index | {len(SEEDS)} seeds | data_seed={data_seed}")
    log(f"Bootstrap CI: n={BOOTSTRAP_N}, ci={BOOTSTRAP_CI}")

    results = run_ablation(data)

    if is_main():
        print("\n" + "=" * 60)
        print(f"  ABLATION RESULTS  (Test C-index; mean ± std over {len(SEEDS)} seeds)")
        print("=" * 60)
        print(f"\n  {'Model':<32} {'C-index':>17} {'Params':>11}")
        print(f"  {'-'*62}")
        for name, m in results.items():
            mark = " <--" if "Ours" in name else ""
            cstr = f"{m['cidx_mean']:.4f}±{m['cidx_std']:.4f}"
            print(f"  {name:<32} {cstr:>17} {m['_n_params']:>11,}{mark}")

        print(f"\n  [Per-seed test C-index]")
        hdr = "  " + f"{'Model':<32}" + "".join(f"{f'seed={s}':>11}" for s in SEEDS)
        print(hdr)
        print("  " + "-" * (32 + 11 * len(SEEDS)))
        for name, m in results.items():
            mark = " <--" if "Ours" in name else ""
            row = "  " + f"{name:<32}"
            for s in m["_per_seed"]:
                row += f"{s['test_cidx']:>11.4f}"
            print(row + mark)

        # Per-seed bootstrap 95% CI on the test set (sample-level uncertainty,
        # complementary to seed-level std above).
        print(f"\n  [Per-seed test bootstrap 95% CI -- mean across seeds]")
        for name, m in results.items():
            los = np.array([s["ci_lo"] for s in m["_per_seed"]])
            his = np.array([s["ci_hi"] for s in m["_per_seed"]])
            mark = " <--" if "Ours" in name else ""
            print(f"  {name:<32}  mean CI = "
                  f"[{los.mean():.4f}, {his.mean():.4f}]{mark}")

        # Paired t-test of each "Ours" against every baseline (n=5 seeds).
        # Same train/val/test split across seeds, so test C-indices are paired.
        # noICL models are excluded here -- they belong to the leakage block below.
        print(f"\n  [Paired t-test p-values (Ours vs baseline, n={len(SEEDS)} seeds)]")
        baselines  = [n for n in results if "Ours" not in n and "noICL" not in n]
        ours_names = [n for n in results if "Ours" in n and "noICL" not in n]
        hdr = "  " + f"{'Ours model':<24}" + "".join(f"{b[:17]:>18}" for b in baselines)
        print(hdr)
        print("  " + "-" * (24 + 18 * len(baselines)))
        for o in ours_names:
            o_seeds = [s["test_cidx"] for s in results[o]["_per_seed"]]
            row = "  " + f"{o:<24}"
            for b in baselines:
                b_seeds = [s["test_cidx"] for s in results[b]["_per_seed"]]
                p = paired_t_pvalue(o_seeds, b_seeds)
                row += f"{p:>18.4f}"
            print(row)
        print(f"  (n=5 paired observations is small; treat p<0.05 as suggestive, not conclusive)")

        # INTCLUST-leakage contrasts: same model trained with vs without the
        # INTCLUST one-hots. Negative delta = INTCLUST was carrying signal that
        # CNA / mRNA cannot fully replace.
        leakage_pairs = [
            ("Clinical-only (MLP)",       "Clinical-noICL (MLP)"),
            ("Late Fusion 3 (Concat)",    "Late Fusion 3-noICL (Concat)"),
            ("Ours-Tri (HSIC K=512)",     "Ours-Tri-noICL (HSIC K=512)"),
        ]
        print(f"\n  [INTCLUST-leakage contrasts (with-ICL  vs  noICL, paired across seeds)]")
        for full, noicl in leakage_pairs:
            if full in results and noicl in results:
                a = np.array([s["test_cidx"] for s in results[full]["_per_seed"]])
                b = np.array([s["test_cidx"] for s in results[noicl]["_per_seed"]])
                delta = (b - a).mean()
                p = paired_t_pvalue(b.tolist(), a.tolist())
                print(f"  {full:<32}  ICL: {a.mean():.4f}  noICL: {b.mean():.4f}  "
                      f"delta={delta:+.4f}  p={p:.4f}")

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "ablation_results_cindex.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {out_path}")

    cleanup_ddp()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",   default="raw")
    p.add_argument("--out_dir",   default="processed_metabric")
    p.add_argument("--data_seed", type=int, default=42,
                    help="Seed for train/val/test split. Held fixed across SEEDS.")
    args = p.parse_args()
    main(args.raw_dir, args.out_dir, args.data_seed)
