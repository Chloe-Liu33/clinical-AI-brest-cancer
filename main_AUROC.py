"""METABRIC -- 5-year overall survival prediction.

Cross-modal Transformer (Clinical EHR + Gene Expression) with HSIC-based
hard top-K gene selection.

Adapted from the Diabetes 130-US codebase. Differences:
  - Text/BERT path -> gene expression path (raw [B, ~20000] vector)
  - HSIC top-K operates on gene dims, identifying genes most associated
    with the clinical (structured) embedding
  - No BERT pre-encoding, no tokenizer, no cached hidden states
  - Smaller dataset (~1900 samples) -> smaller batch size, longer epochs

Usage:
  Single-GPU:    python main.py --processed_dir ./processed_metabric
  Multi-GPU DDP: torchrun --standalone --nproc_per_node=2 main.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup


# =============================================================
# 0. Configuration
# =============================================================

D_MODEL      = 128
N_HEADS      = 4
DROPOUT      = 0.3
BATCH_SIZE   = 64       # METABRIC has ~1300 train samples -> small batch
ACCUM_STEPS  = 1
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 0.01
PATIENCE     = 15
WARMUP_RATIO = 0.1

# HSIC-based gene selection.
# HSIC is recomputed once per epoch over the full training set (see
# CrossModalGenomic.update_hsic). Epoch 1 trains with no mask so struct_proj
# has a reasonable warm-up before HSIC is queried.
HSIC_TOP_K_SWEEP = [256, 512, 1024, 2048]    # ablate K to find best

RDROP_ALPHA   = 0.5
RDROP_ENABLED = True

# Multi-seed averaging: each model trains once per seed; results report mean ± std.
SEEDS = [42, 7, 123, 2024, 31415]

BOOTSTRAP_N  = 1000
BOOTSTRAP_CI = 0.95

DEVICE: torch.device = torch.device("cpu")
WORLD_SIZE = 1
RANK = 0
LOCAL_RANK = 0


# =============================================================
# 0.5 DDP helpers
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


def set_all_seeds(seed: int) -> None:
    """Seed every RNG that affects model init / dropout / dataloader shuffle."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    return RANK == 0


def log(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


# =============================================================
# 1. Dataset
# =============================================================

class GenomicDataset(Dataset):
    def __init__(self, X_struct: np.ndarray, X_gene: np.ndarray, y: np.ndarray):
        self.X_struct = torch.tensor(X_struct, dtype=torch.float32)
        self.X_gene   = torch.tensor(X_gene,   dtype=torch.float32)
        self.y        = torch.tensor(y,        dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "struct": self.X_struct[idx],
            "gene":   self.X_gene[idx],
            "label":  self.y[idx],
        }


# =============================================================
# 2. Models
# =============================================================

class StructOnlyMLP(nn.Module):
    """Baseline: MLP on clinical features only."""
    def __init__(self, struct_dim: int, _gene_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(struct_dim, D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 1),
        )

    def forward(self, struct, gene):
        return self.net(struct).squeeze(-1), None


class GeneOnlyMLP(nn.Module):
    """Baseline: MLP on gene expression only (all genes, no selection)."""
    def __init__(self, _struct_dim: int, gene_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gene_dim, D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 1),
        )

    def forward(self, struct, gene):
        return self.net(gene).squeeze(-1), None


class GeneVarTopKMLP(nn.Module):
    """Baseline: MLP on top-K highest-variance genes (selected on train).

    Direct counterpart to the HSIC top-K selection — same K, same downstream
    classifier shape — so any gap reflects HSIC's choice of *which* K genes.
    """
    def __init__(self, _struct_dim: int, _gene_dim: int, top_idx: np.ndarray):
        super().__init__()
        self.register_buffer("top_idx",
                             torch.tensor(top_idx, dtype=torch.long))
        k = len(top_idx)
        self.net = nn.Sequential(
            nn.Linear(k, D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 1),
        )

    def forward(self, struct, gene):
        gene_sel = gene.index_select(1, self.top_idx)
        return self.net(gene_sel).squeeze(-1), None


class LateFusionConcat(nn.Module):
    """Baseline: project each modality, concat, classify.

    Projections mirror Ours' struct_proj (Linear+LN+ReLU+Dropout) on BOTH
    branches so any gap to Ours reflects the cross-attention + HSIC choice
    rather than a weaker projection head.
    """
    def __init__(self, struct_dim: int, gene_dim: int):
        super().__init__()
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_dim, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.classifier  = nn.Sequential(
            nn.Linear(2 * D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 1),
        )

    def forward(self, struct, gene):
        s = self.struct_proj(struct)
        g = self.gene_proj(gene)
        return self.classifier(torch.cat([s, g], dim=-1)).squeeze(-1), None


class CrossModalGenomic(nn.Module):
    """
    Cross-modal: clinical (struct) + gene expression with HSIC top-K.
      - HSIC(gene_dim_d, struct_emb) per d on raw [B, gene_dim] expression
      - running_hsic [gene_dim] via EMA -> deterministic top-K mask
      - gene_proj sees masked expression
      - Cross-attn: denoised gene queries, struct K/V (residual carries gene)
      - Final concat([ffn_out, struct_emb]) -> classifier

    `top_k` is per-instance so the ablation can sweep multiple K values.
    """
    def __init__(self, struct_dim: int, gene_dim: int, top_k: int):
        super().__init__()
        self.gene_dim   = gene_dim
        self.hsic_top_k = top_k

        self.gene_proj = nn.Linear(gene_dim, D_MODEL)

        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D_MODEL, num_heads=N_HEADS,
            dropout=DROPOUT, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(D_MODEL)

        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL * 2, D_MODEL),
        )
        self.ffn_norm = nn.LayerNorm(D_MODEL)

        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL * 2, D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 1),
        )

        self.register_buffer("running_hsic", torch.zeros(gene_dim))
        self.register_buffer("hsic_ready", torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _rbf_kernel(Y: torch.Tensor) -> torch.Tensor:
        pdist = torch.cdist(Y, Y, p=2)
        off = pdist[~torch.eye(pdist.shape[0], dtype=torch.bool, device=Y.device)]
        sigma = off.median().clamp(min=1e-6)
        return torch.exp(-(pdist ** 2) / (2.0 * sigma ** 2))

    @classmethod
    def hsic_per_dim(cls, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Per-dimension HSIC. X: [N, gene_dim], Y: [N, D]. Returns [gene_dim]."""
        N = X.shape[0]
        if N < 2:
            return torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)
        L = cls._rbf_kernel(Y)
        L_c = L - L.mean(0, keepdim=True) - L.mean(1, keepdim=True) + L.mean()
        X_c = X - X.mean(0, keepdim=True)
        # tr(K_X[d] @ L_c) with K_X[d] = X_c[:,d] X_c[:,d]^T
        return torch.einsum("bd,bm,md->d", X_c, L_c, X_c) / (N - 1) ** 2

    @torch.no_grad()
    def update_hsic(self, X_struct: torch.Tensor, X_gene: torch.Tensor) -> None:
        """Recompute HSIC over the full training set. Call between epochs.

        Replaces the old per-batch + EMA estimate: B << gene_dim made the
        per-batch HSIC very noisy, and EMA mixed early (random struct_emb)
        signal into the running statistic. Full-train every epoch is cheap
        (N ~ 1300) and gives a stable top-K.
        """
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        X_s = X_struct.to(device, non_blocking=True)
        X_g = X_gene.to(device, non_blocking=True)
        struct_emb = self.struct_proj(X_s)
        hsic = self.hsic_per_dim(X_g, struct_emb)
        self.running_hsic.copy_(hsic)
        self.hsic_ready.fill_(1)
        if was_training:
            self.train()

    def _topk_mask(self, gene_expr: torch.Tensor) -> torch.Tensor:
        if int(self.hsic_ready) == 0:
            return torch.ones(self.gene_dim,
                              device=gene_expr.device, dtype=gene_expr.dtype)
        topk_idx = self.running_hsic.topk(self.hsic_top_k).indices
        mask = torch.zeros(self.gene_dim,
                           device=gene_expr.device, dtype=gene_expr.dtype)
        mask[topk_idx] = 1.0
        return mask

    def forward(self, struct, gene):
        struct_emb = self.struct_proj(struct)                # [B, D]
        mask = self._topk_mask(gene)                         # [gene_dim], 0/1
        gene_masked = gene * mask                            # [B, gene_dim]
        gene_emb = self.gene_proj(gene_masked)               # [B, D]

        # Denoised gene drives the query (residual preserves it);
        # struct provides K/V context.
        gene_q    = gene_emb.unsqueeze(1)                    # [B, 1, D]
        struct_kv = struct_emb.unsqueeze(1)                  # [B, 1, D]
        cross_out, _ = self.cross_attn(gene_q, struct_kv, struct_kv)
        cross_out = self.cross_norm(cross_out + gene_q)

        ffn_out = self.ffn(cross_out)
        ffn_out = self.ffn_norm(ffn_out + cross_out).squeeze(1)

        combined = torch.cat([ffn_out, struct_emb], dim=-1)
        logits   = self.classifier(combined).squeeze(-1)
        return logits, mask


# =============================================================
# 2.5. Tri-modal models (clinical + mRNA + CNA)
# is_tri=True is a flag the training loop reads to pass the cna tensor.
# =============================================================

class CnaOnlyMLP(nn.Module):
    """Baseline: MLP on copy-number alteration only."""
    is_tri = True

    def __init__(self, _struct_dim: int, _gene_dim: int, cna_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cna_dim, D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 1),
        )

    def forward(self, struct, gene, cna):
        return self.net(cna).squeeze(-1), None


class LateFusionConcat3(nn.Module):
    """Baseline: project clinical + mRNA + CNA, concat, classify."""
    is_tri = True

    def __init__(self, struct_dim: int, gene_dim: int, cna_dim: int):
        super().__init__()
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_dim, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.cna_proj = nn.Sequential(
            nn.Linear(cna_dim, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 1),
        )

    def forward(self, struct, gene, cna):
        s = self.struct_proj(struct)
        g = self.gene_proj(gene)
        c = self.cna_proj(cna)
        return self.classifier(torch.cat([s, g, c], dim=-1)).squeeze(-1), None


class CrossModalGenomicTri(nn.Module):
    """Tri-modal cross-modal Transformer with parallel HSIC selection.

    Symmetric design:
      - HSIC selects top-K mRNA dims against struct_emb (independent buffer).
      - HSIC selects top-K CNA  dims against struct_emb (independent buffer).
      - Two parallel cross-attention paths: mRNA_q ← struct K/V, CNA_q ← struct K/V.
      - Classifier sees concat(ffn_mrna, ffn_cna, struct_emb) → 3*D_MODEL.

    Same `top_k` is used for both modalities to keep the comparison clean; the
    HSIC ranking is per-modality so each picks its own genes.
    """
    is_tri = True

    def __init__(self, struct_dim: int, gene_dim: int, cna_dim: int, top_k: int):
        super().__init__()
        self.gene_dim   = gene_dim
        self.cna_dim    = cna_dim
        self.hsic_top_k = top_k

        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.gene_proj = nn.Linear(gene_dim, D_MODEL)
        self.cna_proj  = nn.Linear(cna_dim,  D_MODEL)

        self.gene_attn = nn.MultiheadAttention(
            embed_dim=D_MODEL, num_heads=N_HEADS,
            dropout=DROPOUT, batch_first=True,
        )
        self.gene_norm = nn.LayerNorm(D_MODEL)
        self.gene_ffn  = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL * 2, D_MODEL),
        )
        self.gene_ffn_norm = nn.LayerNorm(D_MODEL)

        self.cna_attn = nn.MultiheadAttention(
            embed_dim=D_MODEL, num_heads=N_HEADS,
            dropout=DROPOUT, batch_first=True,
        )
        self.cna_norm = nn.LayerNorm(D_MODEL)
        self.cna_ffn  = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL * 2, D_MODEL),
        )
        self.cna_ffn_norm = nn.LayerNorm(D_MODEL)

        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL * 3, D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 1),
        )

        self.register_buffer("running_hsic_gene", torch.zeros(gene_dim))
        self.register_buffer("running_hsic_cna",  torch.zeros(cna_dim))
        self.register_buffer("hsic_ready",        torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update_hsic(self, X_struct: torch.Tensor, X_gene: torch.Tensor,
                    X_cna: torch.Tensor) -> None:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        X_s = X_struct.to(device, non_blocking=True)
        X_g = X_gene.to(device, non_blocking=True)
        X_c = X_cna.to(device, non_blocking=True)
        struct_emb = self.struct_proj(X_s)
        self.running_hsic_gene.copy_(CrossModalGenomic.hsic_per_dim(X_g, struct_emb))
        self.running_hsic_cna.copy_(CrossModalGenomic.hsic_per_dim(X_c, struct_emb))
        self.hsic_ready.fill_(1)
        if was_training:
            self.train()

    def _mask(self, dim: int, hsic: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if int(self.hsic_ready) == 0:
            return torch.ones(dim, device=x.device, dtype=x.dtype)
        k = min(self.hsic_top_k, dim)
        topk_idx = hsic.topk(k).indices
        mask = torch.zeros(dim, device=x.device, dtype=x.dtype)
        mask[topk_idx] = 1.0
        return mask

    def forward(self, struct, gene, cna):
        struct_emb = self.struct_proj(struct)                  # [B, D]

        gene_mask = self._mask(self.gene_dim, self.running_hsic_gene, gene)
        cna_mask  = self._mask(self.cna_dim,  self.running_hsic_cna,  cna)
        gene_emb  = self.gene_proj(gene * gene_mask)           # [B, D]
        cna_emb   = self.cna_proj(cna  * cna_mask)             # [B, D]

        struct_kv = struct_emb.unsqueeze(1)                    # [B, 1, D]

        gene_q = gene_emb.unsqueeze(1)
        gene_out, _ = self.gene_attn(gene_q, struct_kv, struct_kv)
        gene_out = self.gene_norm(gene_out + gene_q)
        gene_out = self.gene_ffn_norm(self.gene_ffn(gene_out) + gene_out).squeeze(1)

        cna_q = cna_emb.unsqueeze(1)
        cna_out, _ = self.cna_attn(cna_q, struct_kv, struct_kv)
        cna_out = self.cna_norm(cna_out + cna_q)
        cna_out = self.cna_ffn_norm(self.cna_ffn(cna_out) + cna_out).squeeze(1)

        combined = torch.cat([gene_out, cna_out, struct_emb], dim=-1)
        logits   = self.classifier(combined).squeeze(-1)
        return logits, (gene_mask, cna_mask)


# =============================================================
# 3. R-Drop KL loss
# =============================================================

def compute_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p_prob = torch.sigmoid(p)
    q_prob = torch.sigmoid(q)
    p_dist = torch.stack([1 - p_prob, p_prob], dim=-1).clamp(1e-7, 1.0)
    q_dist = torch.stack([1 - q_prob, q_prob], dim=-1).clamp(1e-7, 1.0)
    kl_pq = F.kl_div(q_dist.log(), p_dist, reduction="batchmean")
    kl_qp = F.kl_div(p_dist.log(), q_dist, reduction="batchmean")
    return (kl_pq + kl_qp) / 2


# =============================================================
# 4. Train / eval helpers
# =============================================================

def wrap_ddp(model):
    if is_ddp():
        return nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], find_unused_parameters=False,
        )
    return model


def unwrap_ddp(model):
    return model.module if hasattr(model, "module") else model


def _forward(model, batch, device):
    s = batch["struct"].to(device, non_blocking=True)
    g = batch["gene"].to(device, non_blocking=True)
    return model(s, g)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device,
                    use_rdrop=False):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        labels = batch["label"].to(device, non_blocking=True)

        if use_rdrop:
            logits1, _ = _forward(model, batch, device)
            logits2, _ = _forward(model, batch, device)
            ce = (criterion(logits1, labels) + criterion(logits2, labels)) / 2
            kl = compute_kl_loss(logits1, logits2)
            loss = (ce + RDROP_ALPHA * kl) / ACCUM_STEPS
        else:
            logits, _ = _forward(model, batch, device)
            loss = criterion(logits, labels) / ACCUM_STEPS

        loss.backward()
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * ACCUM_STEPS
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        logits, _ = _forward(model, batch, device)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def evaluate_binary(y_true, y_prob, threshold=0.5, verbose=True):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {}
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["auprc"] = float("nan")
    metrics["f1"]        = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"]    = recall_score(y_true, y_pred, zero_division=0)
    if verbose:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        print(f"    AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
              f"F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}  "
              f"Rec={metrics['recall']:.4f}")
        print(f"    Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    return metrics


# =============================================================
# 5. Bootstrap CI
# =============================================================

def bootstrap_ci(y_true, y_prob, metric_fn, n=BOOTSTRAP_N, ci=BOOTSTRAP_CI,
                 random_state=42):
    rng = np.random.default_rng(random_state)
    n_sample = len(y_true)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, n_sample, size=n_sample)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(metric_fn(yt, yp))
        except Exception:
            continue
    if not scores:
        return float("nan"), float("nan"), float("nan")
    alpha = 1 - ci
    return (metric_fn(y_true, y_prob),
            np.percentile(scores, 100 * alpha / 2),
            np.percentile(scores, 100 * (1 - alpha / 2)))


def bootstrap_all_metrics(y_true, y_prob):
    out = {}
    out["auroc"] = bootstrap_ci(y_true, y_prob, roc_auc_score)
    out["auprc"] = bootstrap_ci(y_true, y_prob, average_precision_score)
    out["f1"]    = bootstrap_ci(
        y_true, y_prob,
        lambda yt, yp: f1_score(yt, (yp >= 0.5).astype(int), zero_division=0),
    )
    return out


def format_ci(point, lower, upper):
    if np.isnan(point):
        return "N/A"
    return f"{point:.4f} (95% CI: {lower:.4f}-{upper:.4f})"


# =============================================================
# 6. Single model training
# =============================================================

def build_optimizer(model):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )


def train_model(model, X_train, X_train_gene, y_train,
                X_val, X_val_gene, y_val, pos_weight,
                use_rdrop=False, max_epochs=EPOCHS):
    train_ds = GenomicDataset(X_train, X_train_gene, y_train)
    val_ds   = GenomicDataset(X_val,   X_val_gene,   y_val)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp() else None
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True,
    )
    val_dl = (DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)
              if is_main() else None)

    model = wrap_ddp(model)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = build_optimizer(model)

    total_steps  = (len(train_dl) // ACCUM_STEPS) * max_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    log(f"    Total steps/rank: {total_steps}, warmup: {warmup_steps}")
    log(f"    R-Drop: {'ON' if use_rdrop else 'OFF'}, max_epochs: {max_epochs}, "
        f"batch/rank: {BATCH_SIZE}, world_size: {WORLD_SIZE}")

    # Full-train tensors for epoch-level HSIC update (only used by Ours).
    X_struct_full = torch.tensor(X_train, dtype=torch.float32)
    X_gene_full   = torch.tensor(X_train_gene, dtype=torch.float32)

    best_auroc, best_state, patience_cnt = 0.0, None, 0
    for epoch in range(1, max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Recompute HSIC on full train set at start of each epoch (epoch 2+).
        # Epoch 1 trains with no mask so struct_proj has a warm-up.
        raw = unwrap_ddp(model)
        if epoch > 1 and hasattr(raw, "update_hsic"):
            raw.update_hsic(X_struct_full, X_gene_full)

        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_dl, optimizer, scheduler, criterion, DEVICE, use_rdrop,
        )

        if is_main():
            probs, labels = evaluate(unwrap_ddp(model), val_dl, DEVICE)
            auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        else:
            auroc = 0.0
        if is_ddp():
            t = torch.tensor([auroc], device=DEVICE)
            dist.broadcast(t, src=0)
            auroc = t.item()

        if is_main() and (epoch % 5 == 0 or epoch == 1):
            print(f"    Epoch {epoch:3d}/{max_epochs} | loss={train_loss:.4f} | "
                  f"val_auroc={auroc:.4f} | {time.time()-t0:.1f}s")

        if auroc > best_auroc:
            best_auroc = auroc
            if is_main():
                raw = unwrap_ddp(model)
                best_state = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                log(f"    Early stop @ epoch {epoch} (best_auroc={best_auroc:.4f})")
                break

    raw_model = unwrap_ddp(model)
    if is_main() and best_state is not None:
        raw_model.load_state_dict(best_state)

    if is_main():
        probs, labels = evaluate(raw_model, val_dl, DEVICE)
        metrics = evaluate_binary(labels, probs, verbose=True)
    else:
        probs, labels, metrics = None, None, None
    return metrics, raw_model, probs, labels


# =============================================================
# 7. Ablation study
# =============================================================

def run_ablation(X_struct, X_gene, y, split, struct_dim, gene_dim, gene_var_rank):
    X_train, y_train       = X_struct[split["train"]], y[split["train"]]
    X_val,   y_val         = X_struct[split["val"]],   y[split["val"]]
    X_test,  y_test        = X_struct[split["test"]],  y[split["test"]]
    G_train, G_val, G_test = X_gene[split["train"]], X_gene[split["val"]], X_gene[split["test"]]

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    log(f"\n  Class balance: {int(n_neg)} neg / {int(n_pos)} pos "
        f"(pos_weight={pos_weight.item():.2f})")

    # R-Drop is applied uniformly to ALL models so the comparison isolates the
    # architectural difference (HSIC + cross-attn) rather than which model gets
    # extra regularization.
    use_rdrop = RDROP_ENABLED

    model_configs = {
        "Clinical-only (MLP)": {
            "build": lambda: StructOnlyMLP(struct_dim, gene_dim).to(DEVICE),
            "rdrop": use_rdrop, "epochs": EPOCHS,
        },
        "Gene-only (MLP)": {
            "build": lambda: GeneOnlyMLP(struct_dim, gene_dim).to(DEVICE),
            "rdrop": use_rdrop, "epochs": EPOCHS,
        },
        "Late Fusion (Concat)": {
            "build": lambda: LateFusionConcat(struct_dim, gene_dim).to(DEVICE),
            "rdrop": use_rdrop, "epochs": EPOCHS,
        },
    }
    # Variance baseline swept over the SAME K as HSIC so any gap reflects
    # the *choice* of K genes, not just the K value.
    for k in HSIC_TOP_K_SWEEP:
        var_top_idx_k = gene_var_rank[:k]
        model_configs[f"Gene Var-{k} (MLP)"] = {
            "build": (lambda kk=k, idx=var_top_idx_k:
                      GeneVarTopKMLP(struct_dim, gene_dim, idx).to(DEVICE)),
            "rdrop": use_rdrop, "epochs": EPOCHS,
        }
    # Ours: sweep K to ablate the HSIC top-K hyperparameter.
    for k in HSIC_TOP_K_SWEEP:
        model_configs[f"Ours (HSIC K={k})"] = {
            "build": (lambda kk=k:
                      CrossModalGenomic(struct_dim, gene_dim, top_k=kk).to(DEVICE)),
            "rdrop": use_rdrop, "epochs": EPOCHS,
        }

    metric_keys = ["auroc", "auprc", "f1", "precision", "recall"]

    results = {}
    for name, cfg in model_configs.items():
        log(f"\n{'='*60}")
        log(f"  Training: {name}  (seeds: {SEEDS})")
        log(f"{'='*60}")

        per_seed = []         # list of test-metric dicts, one per seed
        n_params = 0
        for seed in SEEDS:
            log(f"\n  --- Seed {seed} ---")
            set_all_seeds(seed)

            model = cfg["build"]()
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            val_metrics, model, _, _ = train_model(
                model, X_train, G_train, y_train,
                X_val, G_val, y_val, pos_weight,
                use_rdrop=cfg["rdrop"], max_epochs=cfg["epochs"],
            )

            if is_main():
                test_ds = GenomicDataset(X_test, G_test, y_test)
                test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=2, pin_memory=True)
                test_probs, test_labels = evaluate(model, test_dl, DEVICE)
                test_metrics = evaluate_binary(test_labels, test_probs, verbose=False)
                log(f"    [seed {seed}] test AUROC={test_metrics['auroc']:.4f} "
                    f"AUPRC={test_metrics['auprc']:.4f} F1={test_metrics['f1']:.4f}")
                per_seed.append({"seed": seed, "val": val_metrics, "test": test_metrics})

            if is_ddp():
                dist.barrier()

        if is_main():
            test_arr = {k: np.array([s["test"][k] for s in per_seed]) for k in metric_keys}
            agg = {f"{k}_mean": float(test_arr[k].mean()) for k in metric_keys}
            agg.update({f"{k}_std": float(test_arr[k].std(ddof=1)) for k in metric_keys})
            agg["_per_seed"] = per_seed
            agg["_n_params"] = n_params
            results[name] = agg
            log(f"\n  [{name} -- {len(SEEDS)}-seed mean ± std]")
            log(f"    AUROC={agg['auroc_mean']:.4f}±{agg['auroc_std']:.4f}  "
                f"AUPRC={agg['auprc_mean']:.4f}±{agg['auprc_std']:.4f}  "
                f"F1={agg['f1_mean']:.4f}±{agg['f1_std']:.4f}")

    return results


# =============================================================
# 8. Main
# =============================================================

def main(processed_dir: str):
    setup_ddp()

    log("=" * 60)
    log("  METABRIC -- 5-year OS prediction")
    log("  Cross-modal Transformer (Clinical + Gene Expression)")
    log("=" * 60)
    log(f"Device: {DEVICE} | World size: {WORLD_SIZE}")

    log("\nLoading processed data...")
    X_struct      = np.load(os.path.join(processed_dir, "X_struct.npy"))
    X_gene        = np.load(os.path.join(processed_dir, "X_gene.npy"))
    y             = np.load(os.path.join(processed_dir, "y.npy"))
    gene_var_rank = np.load(os.path.join(processed_dir, "gene_var_rank.npy"))
    with open(os.path.join(processed_dir, "split.pkl"), "rb") as f:
        split = pickle.load(f)

    struct_dim = X_struct.shape[1]
    gene_dim   = X_gene.shape[1]
    log(f"  Samples: {len(y)} (pos={int(y.sum())}, neg={int((1-y).sum())})")
    log(f"  Clinical features: {struct_dim}")
    log(f"  Gene features:     {gene_dim}")
    log(f"  train/val/test:    {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")

    log(f"\nHSIC config: TOP_K_SWEEP={HSIC_TOP_K_SWEEP}/{gene_dim}, "
        f"estimation=full-train per epoch (epoch 1 = no mask)")
    log(f"Variance baseline: K_SWEEP={HSIC_TOP_K_SWEEP} (matched to HSIC)")

    results = run_ablation(X_struct, X_gene, y, split, struct_dim, gene_dim,
                           gene_var_rank)

    if is_main():
        print("\n" + "=" * 60)
        print(f"  ABLATION RESULTS  (Test set; mean ± std over {len(SEEDS)} seeds)")
        print("=" * 60)
        print(f"\n  {'Model':<30} {'AUROC':>17} {'AUPRC':>17} {'F1':>17} {'Params':>11}")
        print(f"  {'-'*98}")
        for name, m in results.items():
            mark = " <--" if "Ours" in name else ""
            auroc = f"{m['auroc_mean']:.4f}±{m['auroc_std']:.4f}"
            auprc = f"{m['auprc_mean']:.4f}±{m['auprc_std']:.4f}"
            f1    = f"{m['f1_mean']:.4f}±{m['f1_std']:.4f}"
            print(f"  {name:<30} {auroc:>17} {auprc:>17} {f1:>17} "
                  f"{m['_n_params']:>11,}{mark}")

        # Per-seed breakdown for transparency.
        print(f"\n  [Per-seed test AUROC]")
        seed_hdr = "  " + f"{'Model':<30}" + "".join(f"{f'seed={s}':>11}" for s in SEEDS)
        print(seed_hdr)
        print(f"  {'-'*(30 + 11*len(SEEDS))}")
        for name, m in results.items():
            mark = " <--" if "Ours" in name else ""
            row = "  " + f"{name:<30}"
            for s in m["_per_seed"]:
                row += f"{s['test']['auroc']:>11.4f}"
            print(row + mark)

        out_path = os.path.join(processed_dir, "ablation_results_metabric.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {out_path}")

    cleanup_ddp()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", default="./processed_metabric")
    args = p.parse_args()
    main(args.processed_dir)
