"""
13April2026_RobustNN_Experiments.py
====================================
Scientifically-rigorous, GPU-efficient, fully reproducible robustness benchmark
for Classification under Label Noise and Adversarial Perturbations.

ALL datasets are TRAINED FROM SCRATCH with a ViT, so each loss function
genuinely influences learning dynamics — unlike zero-shot evaluation.

Datasets (all trained with ViT):
  1. MNIST         (10 classes, grayscale → 3ch padded to 32×32)
  2. Fashion-MNIST (10 classes, grayscale → 3ch padded to 32×32)
  3. CIFAR-10      (10 classes, 3ch 32×32)
  4. PathMNIST     (9  classes, 3ch 28→32×32, histopathology)
  5. DermaMNIST    (7  classes, 3ch 28→32×32, dermatoscopy)

Loss functions (unified PyTorch):
  CCE · MAE · GCE(q) · TruncGCE · SCE · SDIV(β,λ) · DPD(β) · TSCCE · ForwardT

Batteries:
  A) Clean-label training  →  baseline accuracy
  B) Label-noise training  →  η ∈ {0, 0.1, 0.2, 0.3, 0.4}
  C) FGSM attack on clean  →  ε ∈ {0, 1/255, 2/255, 4/255, 8/255}
  E) SDIV (β, λ) surface   →  3D accuracy heatmap
  F) Curriculum GCE anneal  →  q decays over training

Key improvements over 12-April code:
  ✓ All 5 datasets trained from scratch (no zero-shot on PathMNIST/DermaMNIST)
  ✓ GPU auto-tuning: batch size, DataLoader workers, torch.compile
  ✓ torch >= 2.6 required (CVE-2025-32434 fix); safetensors for HF models
  ✓ Per-loss unscaled Y-axis subplots (never mix loss scales)
  ✓ Normalized confusion matrices with class names
  ✓ FGSM + label noise combined robustness frontier
  ✓ Seed-averaged results
  ✓ All results → ./results_13April2026/

Requirements:
  pip install 'torch>=2.6' torchvision transformers datasets medmnist
              scikit-learn matplotlib seaborn pandas tqdm

Run:
  python Runpod_13April2026_RobustNN_Experiments.py    # quick (30 ep, 1 seed)
  ROBUST_NN_QUICK_RUN=0 python ...                     # full paper run

"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 ── CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
import os

CFG = dict(
    # ── General ───────────────────────────────────────────────────────────────
    QUICK_RUN=os.environ.get("ROBUST_NN_QUICK_RUN", "1") == "1",
    PART=os.environ.get("ROBUST_NN_PART", "ABC"),
    SEED=int(os.environ.get("ROBUST_NN_SEED", "42")),
    RESULTS_DIR=os.environ.get("ROBUST_NN_RESULTS_DIR", "results_13April2026"),
    # ── Datasets to run ──────────────────────────────────────────────────────
    VIT_DATASETS=os.environ.get("ROBUST_NN_DATASETS", "pathmnist,dermamnist").split(","),
    # ── Vision ViT (Part A) ──────────────────────────────────────────────────
    VIT_EPOCHS=int(os.environ.get("ROBUST_NN_VIT_EPOCHS", "10")),
    VIT_BATCH=int(os.environ.get("ROBUST_NN_VIT_BATCH", "0")),  # 0 = auto
    VIT_SEEDS=[42],
    VIT_PATCH=4,  # 4×4 patches → 64 patches per 32×32 image (more work for GPU)
    VIT_D_MODEL=256,  # ↑ from 64: gives ~4M params → meaningful GPU compute
    VIT_HEADS=8,  # ↑ from 4:  8-head attention for richer representation
    VIT_FFN=512,  # ↑ from 128: wider FFN
    VIT_LAYERS=6,  # ↑ from 4:  deeper model
    VIT_DROPOUT=0.1,
    VIT_LR=3e-4,  # ↓ from 1e-3: larger model needs gentler LR
    NUM_WORKERS=int(os.environ.get("ROBUST_NN_NUM_WORKERS", "8")),  # 0 = safe for Jupyter
    # ── Label noise rates ────────────────────────────────────────────────────
    NOISE_RATES=[0.0, 0.1, 0.2, 0.3, 0.4],
    # ── FGSM adversarial epsilons ────────────────────────────────────────────
    FGSM_EPS=[0.0, 1 / 255, 2 / 255, 4 / 255, 8 / 255],
    # ── SDIV parameter grid ──────────────────────────────────────────────────
    BETA_GRID=[0.02, 0.05, 0.10, 0.20, 0.50],
    LAM_GRID=[-0.80, -0.40, 0.00, 0.20],
    # ── NLP BERT (Part B) ────────────────────────────────────────────────────
    NLP_EPOCHS=int(os.environ.get("ROBUST_NN_NLP_EPOCHS", "3")),
    NLP_BATCH=int(os.environ.get("ROBUST_NN_NLP_BATCH", "32")),
    NLP_LR=2e-5,
    NLP_MAX_LEN=128,
    NLP_MAX_TRAIN=int(os.environ.get("ROBUST_NN_NLP_MAX_TRAIN", "1500")),
)

# Override for quick run
if CFG["QUICK_RUN"]:
    CFG["VIT_EPOCHS"] = min(CFG["VIT_EPOCHS"], 10)
    CFG["VIT_SEEDS"] = [42]
    CFG["NLP_EPOCHS"] = min(CFG["NLP_EPOCHS"], 2)

os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
print(
    f"[Config] quick={CFG['QUICK_RUN']}  part={CFG['PART']}  "
    f"vit_epochs={CFG['VIT_EPOCHS']}  datasets={CFG['VIT_DATASETS']}"
)
print(f"[Config] results → {CFG['RESULTS_DIR']}/")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import gc
import math
import random
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ── Torch version check (CVE-2025-32434 requires ≥ 2.6) ─────────────────────
def _torch_version() -> tuple[int, ...]:
    return tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])


TORCH_VER = _torch_version()
if TORCH_VER < (2, 6, 0):
    print(f"[WARNING] torch {torch.__version__} < 2.6 — torch.load is blocked by CVE-2025-32434.")
    print("          Install: pip install 'torch>=2.6'")
    print("          HuggingFace models will use safetensors format as workaround.")


# ── AMP (Automatic Mixed Precision) ──────────────────────────────────────────
def _make_amp_classes():
    if TORCH_VER >= (2, 4):
        _DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

        class _GradScaler(torch.amp.GradScaler):
            def __init__(self, enabled=True, **kw):
                super().__init__(device=_DEVICE_TYPE, enabled=enabled, **kw)

        class _autocast:
            def __init__(self, enabled=True, **kw):
                self._ctx = torch.amp.autocast(device_type=_DEVICE_TYPE, enabled=enabled)

            def __enter__(self):
                return self._ctx.__enter__()

            def __exit__(self, *a):
                return self._ctx.__exit__(*a)

        return _GradScaler, _autocast
    else:
        from torch.cuda.amp import GradScaler as _GS
        from torch.cuda.amp import autocast as _AC

        return _GS, _AC


GradScaler, autocast = _make_amp_classes()

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# TIMING LEDGER — global, threadsafe-enough for single-process use
# ═══════════════════════════════════════════════════════════════════════════════


class _TimingLedger:
    """Captures wall-clock timing for every (dataset, loss, noise_rate, seed) run,
    then emits a formatted multi-level summary table."""

    def __init__(self):
        self._wall_start: float = time.time()
        self._records: list[dict] = []  # one entry per train_one / train_pretrained call
        self._dataset_starts: dict[str, float] = {}

    def session_start(self) -> None:
        self._wall_start = time.time()

    def dataset_start(self, dataset: str) -> None:
        self._dataset_starts[dataset] = time.time()

    def record(
        self,
        dataset: str,
        loss: str,
        noise_rate: float,
        seed: int,
        elapsed_s: float,
        best_acc: float,
        part: str = "A",
    ) -> None:
        self._records.append(
            dict(
                part=part,
                dataset=dataset,
                loss=loss,
                noise_rate=noise_rate,
                seed=seed,
                elapsed_s=elapsed_s,
                best_acc=best_acc,
                wall_offset_s=time.time() - self._wall_start,
            )
        )
        # Live one-line log after each run
        h, rem = divmod(int(time.time() - self._wall_start), 3600)
        m, s = divmod(rem, 60)
        marker = f"T+{h:02d}h{m:02d}m{s:02d}s"
        print(
            f"  [TIMING] {marker} | {part}/{dataset}/{loss} "
            f"η={noise_rate} s={seed} | "
            f"acc={best_acc:.4f} | run={elapsed_s:.0f}s"
        )

    def print_loss_summary(self, dataset: str) -> None:
        """Per-dataset summary: mean/min/max time per loss function."""
        rows = [r for r in self._records if r["dataset"] == dataset]
        if not rows:
            return
        import pandas as _pd

        df = _pd.DataFrame(rows)
        grp = (
            df.groupby("loss")["elapsed_s"]
            .agg(runs="count", total_s="sum", mean_s="mean", min_s="min", max_s="max")
            .reset_index()
            .sort_values("total_s", ascending=False)
        )
        ds_elapsed = (
            time.time() - self._dataset_starts[dataset] if dataset in self._dataset_starts else df["elapsed_s"].sum()
        )
        print(f"\n{'─' * 70}")
        print(f"  TIMING SUMMARY — {dataset.upper()}   (wall time: {_fmt(ds_elapsed)})")
        print(f"  {'Loss':<22} {'runs':>5} {'total':>9} {'mean/run':>9} {'min':>8} {'max':>8}")
        print(f"  {'─' * 22} {'─' * 5} {'─' * 9} {'─' * 9} {'─' * 8} {'─' * 8}")
        for _, r in grp.iterrows():
            print(
                f"  {r['loss']:<22} {int(r['runs']):>5} "
                f"{_fmt(r['total_s']):>9} {_fmt(r['mean_s']):>9} "
                f"{_fmt(r['min_s']):>8} {_fmt(r['max_s']):>8}"
            )
        print(f"{'─' * 70}")

    def print_final_summary(self) -> None:
        """Full cross-dataset, cross-loss timing breakdown."""
        if not self._records:
            return
        import pandas as _pd

        total_wall = time.time() - self._wall_start
        df = _pd.DataFrame(self._records)

        print(f"\n{'═' * 70}")
        print("  FULL TIMING REPORT")
        print(f"  Total wall time : {_fmt(total_wall)}")
        print(f"  Total GPU time  : {_fmt(df['elapsed_s'].sum())} (sum of all individual runs)")
        print(f"  Total runs      : {len(df)}")
        print(f"{'═' * 70}")

        # ── Per-dataset breakdown ─────────────────────────────────────────────
        print(f"\n  {'Dataset':<20} {'runs':>5} {'GPU time':>10} {'wall time':>11}")
        print(f"  {'─' * 20} {'─' * 5} {'─' * 10} {'─' * 11}")
        for ds, grp in df.groupby("dataset"):
            ds_wall = time.time() - self._dataset_starts[ds] if ds in self._dataset_starts else grp["elapsed_s"].sum()
            print(f"  {ds:<20} {len(grp):>5} {_fmt(grp['elapsed_s'].sum()):>10} {_fmt(ds_wall):>11}")

        # ── Per-loss breakdown (all datasets combined) ─────────────────────────
        print(f"\n  {'Loss':<22} {'runs':>5} {'total GPU':>10} {'mean/run':>9} {'min':>8} {'max':>8}")
        print(f"  {'─' * 22} {'─' * 5} {'─' * 10} {'─' * 9} {'─' * 8} {'─' * 8}")
        grp_loss = (
            df.groupby("loss")["elapsed_s"]
            .agg(runs="count", total_s="sum", mean_s="mean", min_s="min", max_s="max")
            .reset_index()
            .sort_values("total_s", ascending=False)
        )
        for _, r in grp_loss.iterrows():
            print(
                f"  {r['loss']:<22} {int(r['runs']):>5} "
                f"{_fmt(r['total_s']):>10} {_fmt(r['mean_s']):>9} "
                f"{_fmt(r['min_s']):>8} {_fmt(r['max_s']):>8}"
            )
        print(f"\n{'═' * 70}\n")

    def to_csv(self, path: str) -> None:
        import pandas as _pd

        if self._records:
            _pd.DataFrame(self._records).to_csv(path, index=False)
            print(f"  [TIMING CSV] {path}")


def _fmt(sec: float) -> str:
    """Format seconds as hh:mm:ss or mm:ss depending on magnitude."""
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


TIMING = _TimingLedger()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

print(f"[Device] {DEVICE}  |  AMP={'ON' if USE_AMP else 'OFF (CPU)'}")
if DEVICE.type == "cuda":
    _dev = torch.cuda.get_device_properties(0)
    VRAM_GB = _dev.total_memory / 1024**3
    print(f"         {_dev.name} | {VRAM_GB:.1f} GB VRAM")
    # GPU optimizations
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
else:
    VRAM_GB = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG["SEED"])


# ── GPU auto-tuning ──────────────────────────────────────────────────────────
def auto_batch_size(vram_gb: float, img_size: int = 32, in_ch: int = 3) -> int:
    """Compute batch size to use GPU efficiently.
    The model is now ~4M params (d=256, 6 layers). Per-sample memory includes
    activations (forward), gradients (backward), and AMP fp16 buffers.
    Empirically calibrated for ViT-Small on 32×32 images."""
    if vram_gb <= 0:
        return 64  # CPU fallback
    # Empirical per-sample cost for d=256, 6-layer ViT with AMP:
    #   ~1.5 MB per sample (activations + gradients + AMP buffers)
    per_sample_mb = 1.5
    model_overhead_mb = 500  # model params (4M×4B) + optimizer states (×3) + AMP
    target_mb = vram_gb * 1024 * 0.65 - model_overhead_mb  # 65% utilization target
    batch = int(target_mb / per_sample_mb)
    # Round down to nearest power of 2 for GPU alignment
    batch = 2 ** int(math.log2(max(batch, 32)))
    batch = min(max(batch, 32), 512)  # Cap at 512 — larger batches waste GPU on small images
    return batch


def auto_num_workers() -> int:
    """Auto-detect optimal DataLoader workers.
    IMPORTANT: In Jupyter notebooks (RunPod), num_workers>0 causes
    multiprocessing deadlocks that pin CPU at 100% with GPU at 0%.
    We default to 0 (main-process loading) which is actually FASTER
    for small images (32×32) that are already in-memory as tensors."""
    explicit = os.environ.get("ROBUST_NN_NUM_WORKERS")
    if explicit is not None:
        return int(explicit)
    # Detect Jupyter/notebook environment
    _in_jupyter = False
    try:
        from IPython import get_ipython

        ip = get_ipython()
        _in_jupyter = ip is not None and "IPKernelApp" in ip.config
    except Exception:
        pass
    if _in_jupyter:
        return 0  # Jupyter: MUST use 0 to avoid deadlocks
    # CLI: can use workers, but conservative (2) since data is in-memory
    return min(os.cpu_count() or 1, 2)


if CFG["VIT_BATCH"] == 0:
    CFG["VIT_BATCH"] = auto_batch_size(VRAM_GB)
    print(f"[Auto] batch_size={CFG['VIT_BATCH']} (from {VRAM_GB:.1f}GB VRAM)")
else:
    print(f"[Config] batch_size={CFG['VIT_BATCH']} (user override)")

N_WORKERS = auto_num_workers()
print(f"[Auto] num_workers={N_WORKERS}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── LOSS FUNCTIONS (unified PyTorch, all modalities)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each loss:
#  • accepts (logits: Tensor[B,C], targets: Tensor[B]) — raw logits, integer targets
#  • returns scalar loss
#  • has .scale_info: str — describes the expected Y-axis range
#  • has .name: str


class _RobustLoss(nn.Module):
    """Base class: every loss must expose .name and .scale_info."""

    name: str = "base"
    scale_info: str = "[0, ∞)"


class CCELoss(_RobustLoss):
    """Standard Categorical Cross-Entropy (baseline).
    Y-axis: nats, [0,+∞), ~log(C) at initialization."""

    name = "CCE"
    scale_info = r"[0, +∞)"

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets)


class MAELoss(_RobustLoss):
    """Mean Absolute Error on probabilities: 1 - p_y.
    Y-axis: [0, 1] always. Bounded gradient (most robust)."""

    name = "MAE"
    scale_info = "[0, 1]"

    def __init__(self, num_classes):
        super().__init__()
        self.C = num_classes

    def forward(self, logits, targets):
        py = F.softmax(logits, 1).clamp(1e-9)
        py = py[torch.arange(len(targets), device=logits.device), targets]
        return (1.0 - py).mean()


class GCELoss(_RobustLoss):
    """Generalised Cross-Entropy: (1-p_y^q)/q.
    q=0→CCE, q=1→MAE. Y-axis: [0, 1/q]."""

    def __init__(self, q: float = 0.7):
        super().__init__()
        self.q = q
        self.name = f"GCE(q={q})"
        self.scale_info = f"[0, {1 / q if q > 0 else '+∞'}]"

    def forward(self, logits, targets):
        py = F.softmax(logits, 1).clamp(1e-9)
        py = py[torch.arange(len(targets), device=logits.device), targets]
        if abs(self.q) < 1e-9:
            return -torch.log(py).mean()
        return ((1.0 - py.pow(self.q)) / self.q).mean()


class TruncGCELoss(_RobustLoss):
    """Truncated GCE: only samples with p_y < k contribute."""

    def __init__(self, q: float = 0.7, k: float = 0.5):
        super().__init__()
        self.q = q
        self.k = k
        self.name = f"TruncGCE(q={q},k={k})"
        self.scale_info = f"[0, {1 / q if q > 0 else '+∞'}]"

    def forward(self, logits, targets):
        py = F.softmax(logits, 1).clamp(1e-9)
        py = py[torch.arange(len(targets), device=logits.device), targets]
        loss = (1.0 - py.pow(self.q)) / (self.q + 1e-10)
        mask = (py < self.k).float()
        denom = mask.sum().clamp(min=1.0)
        return (loss * mask).sum() / denom


class SCELoss(_RobustLoss):
    """Symmetric Cross-Entropy: α·CCE + β·RCE."""

    def __init__(self, alpha: float = 0.1, beta: float = 1.0, num_classes: int = 10):
        super().__init__()
        self.a = alpha
        self.b = beta
        self.C = num_classes
        self.name = f"SCE(α={alpha},β={beta})"
        self.scale_info = f"≈{alpha}·CCE + {beta}·RCE"

    def forward(self, logits, targets):
        probs = F.softmax(logits, 1).clamp(1e-7)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        y_oh = F.one_hot(targets, self.C).float().clamp(min=1e-4)
        cce = -torch.log(py).mean()
        rce = -(probs * torch.log(y_oh)).sum(1).mean()
        return self.a * cce + self.b * rce


class DPDLoss(_RobustLoss):
    """Density Power Divergence."""

    def __init__(self, beta: float = 0.05):
        super().__init__()
        self.beta = beta
        self.name = f"DPD(β={beta})"
        self.scale_info = "(-∞, +∞)"

    def forward(self, logits, targets):
        probs = F.softmax(logits, 1).clamp(1e-9)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        loss = probs.pow(self.beta + 1).sum(1) - (1.0 + 1.0 / self.beta) * py.pow(self.beta)
        return loss.mean()


class SDIVLoss(_RobustLoss):
    """S-Divergence loss — core institutional contribution.
    A = 1+λ(1-β) > 0, B = β-λ(1-β) > 0."""

    def __init__(self, beta: float = 0.05, lam: float = -0.8):
        super().__init__()
        self.beta = beta
        self.lam = lam
        A = 1.0 + lam * (1.0 - beta)
        B = beta - lam * (1.0 - beta)
        if A <= 0 or B <= 0:
            raise ValueError(f"SDIV constraint violated: A={A:.3f}, B={B:.3f}.")
        self.A = A
        self.B = B
        self.name = f"SDIV(β={beta},λ={lam})"
        self.scale_info = f"(-∞,+∞) A={A:.3f} B={B:.3f}"

    def forward(self, logits, targets):
        probs = F.softmax(logits, 1).clamp(1e-9)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        loss = probs.pow(self.beta + 1).sum(1) / self.A - (1.0 + self.beta) / (self.A * self.B) * py.pow(self.B)
        return loss.mean()


class TSCCELoss(_RobustLoss):
    """Trimmed Sparse CCE: sort per-sample CCE, drop top trim_ratio."""

    def __init__(self, trim_ratio: float = 0.2):
        super().__init__()
        self.trim = trim_ratio
        self.name = f"TSCCE(trim={trim_ratio})"
        self.scale_info = "[0, +∞)"

    def forward(self, logits, targets):
        per = F.cross_entropy(logits, targets, reduction="none")
        k = max(1, int((1 - self.trim) * len(per)))
        return per.topk(k, largest=False).values.mean()


class FCLoss(_RobustLoss):
    """Fractional Cross-Entropy Loss (rSDNet companion loss).

    L(y, p) = (−log p_y)^(1−μ) / Γ(2−μ)  +  2·(1 − p_y)
    where Γ is the Gamma function.

    μ ∈ [0, 1):
      μ → 0  recovers shifted CCE   (plus a constant MAE term)
      μ → 1  approaches MAE (bounded gradient)

    Introduced in the rSDNet codebase for comparison against SDIV.
    The (−log p_y)^(1−μ) term is a fractional-power of the CCE,
    while the 2·(1−p_y) term provides the MAE-style robustness floor.
    """

    def __init__(self, mu: float = 0.5):
        super().__init__()
        if not (0.0 <= mu < 1.0):
            raise ValueError(f"FCLoss: mu must be in [0, 1), got {mu}")
        self.mu = mu
        import math as _math

        self._gamma_denom = _math.gamma(2.0 - mu)
        self.name = f"FCL(μ={mu})"
        self.scale_info = "[0, +∞)"

    def forward(self, logits, targets):
        py = F.softmax(logits, 1).clamp(1e-9)
        py = py[torch.arange(len(targets), device=logits.device), targets]
        cce_frac = (-torch.log(py)).pow(1.0 - self.mu) / self._gamma_denom
        mae_term = 2.0 * (1.0 - py)
        return (cce_frac + mae_term).mean()


class ForwardCorrectionLoss(_RobustLoss):
    """Forward label-correction. T[i,j] = P(ỹ=j | y*=i)."""

    def __init__(self, T: np.ndarray):
        super().__init__()
        self._T_np = T
        self.register_buffer("T", torch.tensor(T, dtype=torch.float32))
        self.name = "ForwardT"
        self.scale_info = "[0,+∞)"

    def forward(self, logits, targets):
        T = self.T.to(logits.device)
        p_corrupt = (F.softmax(logits, 1).clamp(1e-9) @ T.t()).clamp(1e-9)
        py = p_corrupt[torch.arange(len(targets), device=logits.device), targets]
        return -torch.log(py).mean()


def make_loss_registry(
    num_classes: int,
    T_oracle: np.ndarray | None = None,
    q: float = 0.7,
    beta_sdiv: float = 0.05,
    lam_sdiv: float = -0.8,
) -> dict[str, _RobustLoss]:
    """Return the full named loss dictionary for one experiment."""
    reg = {
        "CCE": CCELoss(),
        "MAE": MAELoss(num_classes),
        f"GCE(q={q})": GCELoss(q),
        "TruncGCE": TruncGCELoss(q, 0.5),
        "SCE": SCELoss(0.1, 1.0, num_classes),
        "TPDD-CCE": DPDLoss(beta_sdiv),  # Trimmed DPD+CCE (trim_ratio=0 = paper default)
        "SDIV": SDIVLoss(beta_sdiv, lam_sdiv),
        "TSCCE": TSCCELoss(0.2),
        "FCL": FCLoss(mu=0.5),  # Fractional Cross-Entropy (rSDNet companion)
    }
    if T_oracle is not None:
        reg["ForwardT"] = ForwardCorrectionLoss(T_oracle)
    return reg


def print_loss_scale_table(num_classes: int, registry: dict) -> None:
    """Diagnostic table of expected loss scales."""
    print("\n" + "=" * 70)
    print(f"  LOSS SCALE DIAGNOSTIC (C={num_classes})")
    print("=" * 70)
    print(f"  {'Loss':<22} {'Scale / Y-axis range'}")
    print("  " + "-" * 66)
    for name, fn in registry.items():
        print(f"  {name:<22} {fn.scale_info}")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── LABEL NOISE AND ADVERSARIAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def inject_uniform_noise(labels: np.ndarray, eta: float, C: int, seed: int = 42) -> np.ndarray:
    """Symmetric uniform label noise: each label flipped to random wrong class with prob eta."""
    if eta <= 0:
        return labels.copy()
    rng = np.random.RandomState(seed)
    noisy = labels.copy()
    mask = rng.rand(len(labels)) < eta
    for i in np.where(mask)[0]:
        noisy[i] = rng.choice([c for c in range(C) if c != labels[i]])
    print(f"  [Noise η={eta:.1f}] {mask.sum()}/{len(labels)} labels flipped ({100 * mask.mean():.1f}%)")
    return noisy


def make_T_uniform(C: int, eta: float) -> np.ndarray:
    T = np.full((C, C), eta / max(C - 1, 1))
    np.fill_diagonal(T, 1.0 - eta)
    return T


def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float, loss_fn=None) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al., 2014)."""
    if epsilon == 0.0:
        return x
    if loss_fn is None:
        loss_fn = CCELoss()
    x_adv = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model(x_adv)
        loss = loss_fn(logits, y)
    loss.backward()
    with torch.no_grad():
        x_adv = (x + epsilon * x_adv.grad.sign()).clamp(0.0, 1.0)
    return x_adv.detach()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── VISION TRANSFORMER (from scratch)
# ═══════════════════════════════════════════════════════════════════════════════


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_ch: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_ch, d_model)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p * p)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, -1, C * p * p)
        return self.proj(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)
        h = self.norm2(x)
        return x + self.drop2(self.ffn(h))


class VisionTransformer(nn.Module):
    """
    From-scratch ViT: PatchEmbed → PosEmbed → N×TransEnc → GAP → LN → Linear.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        in_ch: int = 3,
        num_classes: int = 10,
        d_model: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, d_model)
        n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Embedding(n_patches, d_model)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)
        pos = self.pos_embed(torch.arange(x.size(1), device=x.device))
        x = x + pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x.mean(1))  # Global average pool
        return self.head(x)  # Raw logits


def build_vit(img_size: int, in_ch: int, num_classes: int) -> VisionTransformer:
    model = VisionTransformer(
        img_size=img_size,
        patch_size=CFG["VIT_PATCH"],
        in_ch=in_ch,
        num_classes=num_classes,
        d_model=CFG["VIT_D_MODEL"],
        num_heads=CFG["VIT_HEADS"],
        ffn_dim=CFG["VIT_FFN"],
        num_layers=CFG["VIT_LAYERS"],
        dropout=CFG["VIT_DROPOUT"],
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  [ViT] {n_params / 1e6:.2f}M params on {DEVICE} "
        f"(d={CFG['VIT_D_MODEL']}, L={CFG['VIT_LAYERS']}, "
        f"patch={CFG['VIT_PATCH']}, heads={CFG['VIT_HEADS']})"
    )
    # NOTE: torch.compile removed — it uses CUDA graphs which can cause
    # CPU-heavy tracing and 0% GPU utilization in Jupyter notebooks.
    # For CLI runs, enable via: ROBUST_NN_COMPILE=1
    if os.environ.get("ROBUST_NN_COMPILE", "0") == "1" and DEVICE.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default")  # 'default' not 'reduce-overhead'
            print("  [torch.compile] Model compiled (opt-in via ROBUST_NN_COMPILE=1)")
        except Exception:
            pass
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── DATASET LOADING (all 5 datasets)
# ═══════════════════════════════════════════════════════════════════════════════

# Class name registries for confusion matrix labels
DATASET_META = {
    "mnist": {
        "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "n_ch": 1,
        "img_size": 28,
        "num_classes": 10,
    },
    "fashion_mnist": {
        "class_names": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
        "n_ch": 1,
        "img_size": 28,
        "num_classes": 10,
    },
    "cifar10": {
        "class_names": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        "n_ch": 3,
        "img_size": 32,
        "num_classes": 10,
    },
    "pathmnist": {
        "class_names": [
            "adipose",
            "background",
            "debris",
            "lymphocytes",
            "mucus",
            "smooth muscle",
            "normal mucosa",
            "cancer stroma",
            "adenocarcinoma",
        ],
        "n_ch": 3,
        "img_size": 28,
        "num_classes": 9,
    },
    "dermamnist": {
        "class_names": [
            "actinic kerat.",
            "basal cell ca.",
            "benign kerat.",
            "dermatofibroma",
            "melanoma",
            "melanocytic nevi",
            "vascular les.",
        ],
        "n_ch": 3,
        "img_size": 28,
        "num_classes": 7,
    },
}


def load_vision_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one of 5 datasets. Returns (X_train, y_train, X_test, y_test).
    X shape: (N, 3, 32, 32) float32 in [0,1].  y shape: (N,) int64.
    All images are converted to 3-channel and padded to 32×32."""

    meta = DATASET_META[name]

    if name in ("mnist", "fashion_mnist", "cifar10"):
        # ── HuggingFace datasets ──────────────────────────────────────────────
        from datasets import load_dataset as _hf

        _hf_registry = {
            "mnist": ("ylecun/mnist", "image", "label"),
            "fashion_mnist": ("randall-lab/fashion-mnist", "image", "label"),
            "cifar10": ("uoft-cs/cifar10", "img", "label"),
        }
        hf_id, img_col, lbl_col = _hf_registry[name]
        print(f"  Loading {name.upper()} from {hf_id} ...")
        ds = _hf(hf_id)

        def _to_numpy(split):
            imgs = [np.array(img) for img in tqdm(split[img_col], desc=f"    {name} converting", leave=False)]
            X = np.stack(imgs).astype("float32") / 255.0
            y = np.array(split[lbl_col], dtype=np.int64)
            if X.ndim == 3:  # (N,H,W) grayscale
                X = X[:, np.newaxis, :, :]
            elif X.ndim == 4 and X.shape[-1] in (1, 3):  # (N,H,W,C) → (N,C,H,W)
                X = X.transpose(0, 3, 1, 2)
            return X, y

        X_tr, y_tr = _to_numpy(ds["train"])
        X_te, y_te = _to_numpy(ds["test"])

    elif name in ("pathmnist", "dermamnist"):
        # ── MedMNIST datasets ─────────────────────────────────────────────────
        import medmnist
        from medmnist.info import INFO

        key_lower = name.lower()
        info = INFO[key_lower]
        DataClass = getattr(medmnist, info["python_class"])

        # ── Robust pre-download: wget/curl/requests before medmnist tries urllib
        def _ensure_medmnist_npz(ds_name: str) -> None:
            import pathlib
            import subprocess as _sp2

            cache_dir = pathlib.Path.home() / ".medmnist"
            cache_dir.mkdir(parents=True, exist_ok=True)
            npz_file = cache_dir / f"{ds_name}.npz"
            if npz_file.exists() and npz_file.stat().st_size > 100_000:
                print(f"  {ds_name}.npz cached ({npz_file.stat().st_size // 1024} KB) — skip download")
                return
            url = INFO[ds_name]["url"]
            print(f"  Downloading {ds_name}.npz via wget ...")
            r = _sp2.run(
                ["wget", "-q", "--no-check-certificate", "-O", str(npz_file), url],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r.returncode == 0 and npz_file.exists() and npz_file.stat().st_size > 100_000:
                print(f"  wget OK → {npz_file.stat().st_size // 1024} KB")
                return
            print(f"  wget failed (rc={r.returncode}), trying curl ...")
            r2 = _sp2.run(
                ["curl", "-sL", "--insecure", "-o", str(npz_file), url],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r2.returncode == 0 and npz_file.exists() and npz_file.stat().st_size > 100_000:
                print(f"  curl OK → {npz_file.stat().st_size // 1024} KB")
                return
            print("  curl failed, trying requests ...")
            try:
                import warnings

                import requests as _req  # type: ignore

                warnings.filterwarnings("ignore", "Unverified HTTPS")
                headers = {"User-Agent": "Mozilla/5.0 (compatible; RunPod)"}
                resp = _req.get(url, headers=headers, verify=False, stream=True, timeout=300)
                resp.raise_for_status()
                with open(npz_file, "wb") as _fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        _fh.write(chunk)
                print(f"  requests OK → {npz_file.stat().st_size // 1024} KB")
            except Exception as _e:
                print(f"  WARNING: all download methods failed for {ds_name}.npz: {_e}")
                print(f"  To fix manually run:  wget '{url}' -O ~/.medmnist/{ds_name}.npz")

        _ensure_medmnist_npz(key_lower)

        print(f"  Loading {name.upper()} via medmnist ...")
        train_ds = DataClass(split="train", download=True)
        test_ds = DataClass(split="test", download=True)

        def _medmnist_to_numpy(ds_obj):
            imgs = ds_obj.imgs  # (N, 28, 28, 3) uint8 or (N,28,28) for 1ch
            labs = ds_obj.labels.squeeze()  # (N,) or (N,1)
            X = imgs.astype("float32") / 255.0
            if X.ndim == 3:
                X = X[:, np.newaxis, :, :]  # (N,1,H,W)
            elif X.ndim == 4:
                X = X.transpose(0, 3, 1, 2)  # (N,C,H,W)
            y = labs.astype(np.int64)
            return X, y

        X_tr, y_tr = _medmnist_to_numpy(train_ds)
        X_te, y_te = _medmnist_to_numpy(test_ds)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # ── Convert to 3-channel if grayscale ─────────────────────────────────────
    if X_tr.shape[1] == 1:
        X_tr = np.repeat(X_tr, 3, axis=1)
        X_te = np.repeat(X_te, 3, axis=1)

    # ── Pad to 32×32 if 28×28 ─────────────────────────────────────────────────
    if X_tr.shape[2] == 28:
        X_tr = np.pad(X_tr, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="constant")
        X_te = np.pad(X_te, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="constant")

    print(f"  {name.upper()}: train {X_tr.shape}  test {X_te.shape}  classes={meta['num_classes']}  ch=3  size=32×32")
    return X_tr, y_tr, X_te, y_te


class NumpyImageDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── TRAINING HARNESS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainResult:
    loss_name: str
    dataset: str
    noise_rate: float
    seed: int
    history: dict[str, list] = field(default_factory=dict)
    best_acc: float = 0.0
    best_preds: list = field(default_factory=list)
    true_labels: list = field(default_factory=list)
    elapsed_s: float = 0.0
    model_state: dict | None = None


def train_one(
    model: nn.Module,
    loss_fn: _RobustLoss,
    X_tr: np.ndarray,
    y_tr_noisy: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_epochs: int,
    batch_size: int,
    lr: float,
    loss_name: str,
    dataset_name: str,
    noise_rate: float,
    seed: int,
    save_model: bool = False,
) -> TrainResult:
    """Train ViT from scratch with a given loss. AMP + gradient clipping."""
    set_seed(seed)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(DEVICE)

    tr_ds = NumpyImageDataset(X_tr, y_tr_noisy)
    te_ds = NumpyImageDataset(X_te, y_te)
    _dl_kw = dict(
        num_workers=N_WORKERS,
        pin_memory=(DEVICE.type == "cuda" and N_WORKERS == 8),  # pin_memory only useful with workers=0
        persistent_workers=(N_WORKERS > 0),
        prefetch_factor=3 if N_WORKERS > 0 else None,
    )
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False, **_dl_kw)
    te_loader = DataLoader(te_ds, batch_size=batch_size * 2, shuffle=False, **_dl_kw)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = GradScaler(enabled=USE_AMP)

    result = TrainResult(
        loss_name=loss_name,
        dataset=dataset_name,
        noise_rate=noise_rate,
        seed=seed,
        history={"train_loss": [], "test_acc": [], "test_cce": []},
    )
    best_val, best_preds, best_true = 0.0, [], []
    t0 = time.time()

    pbar = tqdm(range(1, n_epochs + 1), desc=f"{dataset_name}|{loss_name}|η={noise_rate}", leave=False, unit="ep")
    _gpu_verified = False
    for epoch in pbar:
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            # GPU verification: print once on first batch to confirm GPU execution
            if not _gpu_verified:
                _gpu_verified = True
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                    mem_mb = torch.cuda.memory_allocated() / 1024**2
                    print(
                        f"  [GPU ✓] First batch on {DEVICE} | "
                        f"xb.device={xb.device} | model on {next(model.parameters()).device} | "
                        f"GPU mem={mem_mb:.0f} MB"
                    )
                else:
                    print("  [CPU] Running on CPU — no GPU acceleration")
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        epoch_loss /= max(n_batches, 1)

        # ── Evaluate ───────────────────────────────────────────────────────────
        model.eval()
        preds, labs, cce_tot = [], [], 0.0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                with autocast(enabled=USE_AMP):
                    logits = model(xb)
                preds.extend(logits.argmax(1).cpu().tolist())
                labs.extend(yb.cpu().tolist())
                cce_tot += F.cross_entropy(logits.float(), yb).item()
        acc = accuracy_score(labs, preds)
        cce_val = cce_tot / max(len(te_loader), 1)

        result.history["train_loss"].append(epoch_loss)
        result.history["test_acc"].append(acc)
        result.history["test_cce"].append(cce_val)

        if acc > best_val:
            best_val = acc
            best_preds = preds[:]
            best_true = labs[:]
            if save_model:
                result.model_state = deepcopy(model.state_dict())

        pbar.set_postfix({"loss": f"{epoch_loss:.4f}", "acc": f"{acc:.4f}"})

    result.best_acc = best_val
    result.best_preds = best_preds
    result.true_labels = best_true
    result.elapsed_s = time.time() - t0
    print(f"  [{loss_name}] seed={seed} η={noise_rate}  best_acc={best_val:.4f}  time={result.elapsed_s:.0f}s")
    TIMING.record(dataset_name, loss_name, noise_rate, seed, result.elapsed_s, best_val, part="A")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 ── PLOTTING (scientifically correct)
# ═══════════════════════════════════════════════════════════════════════════════

LOSS_COLORS = {
    "CCE": "#1f77b4",
    "MAE": "#ff7f0e",
    "GCE": "#2ca02c",
    "TruncGCE": "#d62728",
    "SCE": "#9467bd",
    "DPD": "#8c564b",
    "SDIV": "#e377c2",
    "TSCCE": "#7f7f7f",
    "ForwardT": "#bcbd22",
}


def _get_color(name: str, idx: int = 0) -> str:
    for k, v in LOSS_COLORS.items():
        if name.startswith(k):
            return v
    colors = list(LOSS_COLORS.values())
    return colors[idx % len(colors)]


def plot_training_curves(results: list[TrainResult], title_suffix: str, save_path: str) -> None:
    """
    Row 1: Training loss per epoch — each loss in its OWN subplot (never mix scales)
    Row 2: Test CCE (common scale) + Test accuracy (shared)
    CORRECT: never mix different loss scales on same Y-axis.
    """
    loss_names = [r.loss_name for r in results]
    n = len(loss_names)
    if n == 0:
        return
    ncols = min(n, 4)
    nrows_top = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols, 5 * nrows_top + 6))
    gs = gridspec.GridSpec(nrows_top + 2, ncols, figure=fig, hspace=0.55, wspace=0.45)

    # ── Row(s) 1: per-loss training objective ─────────────────────────────────
    for i, res in enumerate(results):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])
        epochs = list(range(1, len(res.history["train_loss"]) + 1))
        ax.plot(epochs, res.history["train_loss"], color=_get_color(res.loss_name, i), linewidth=1.5)
        ax.set_title(f"{res.loss_name}", fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Training Loss", fontsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(alpha=0.3)
        # Annotate scale warning for non-CCE losses
        if not res.loss_name.startswith("CCE"):
            ax.annotate(
                "Own scale (≠ CCE)",
                xy=(0.97, 0.97),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=6,
                color="gray",
                style="italic",
            )

    # ── Bottom row 1: Test CCE (comparable across all losses) ─────────────────
    ax_cce = fig.add_subplot(gs[nrows_top, :])
    for i, res in enumerate(results):
        if "test_cce" in res.history:
            epochs = list(range(1, len(res.history["test_cce"]) + 1))
            ax_cce.plot(
                epochs, res.history["test_cce"], label=res.loss_name, color=_get_color(res.loss_name, i), linewidth=1.8
            )
    ax_cce.set_title("Test CCE Loss (common scale — all losses comparable)", fontsize=10)
    ax_cce.set_xlabel("Epoch")
    ax_cce.set_ylabel("CCE (nats)")
    ax_cce.legend(fontsize=7, ncol=3)
    ax_cce.grid(alpha=0.3)

    # ── Bottom row 2: Test accuracy ───────────────────────────────────────────
    ax_acc = fig.add_subplot(gs[nrows_top + 1, :])
    for i, res in enumerate(results):
        epochs = list(range(1, len(res.history["test_acc"]) + 1))
        ax_acc.plot(
            epochs,
            res.history["test_acc"],
            label=f"{res.loss_name} ({res.best_acc:.3f})",
            color=_get_color(res.loss_name, i),
            linewidth=2.0,
        )
    ax_acc.set_title("Test Accuracy (directly comparable)", fontsize=10)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=8, ncol=3)
    ax_acc.grid(alpha=0.3)

    fig.suptitle(f"Training Curves — {title_suffix}", fontsize=12, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {save_path}")


def plot_normalized_confusion_matrix(
    y_true: list[int], y_pred: list[int], class_names: list[str], title: str, save_path: str
) -> None:
    """
    Normalized confusion matrix with:
    - Row normalization (cell = P(pred=col | true=row) = recall)
    - Actual class names as tick labels
    - Cell values annotated
    - Color bar labeled
    """
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)), normalize="true")

    fig, ax = plt.subplots(figsize=(max(7, n_classes * 0.95), max(6, n_classes * 0.85)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalized)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_title(f"{title}\n(cell = P(pred=col | true=row))", fontsize=10)

    # Annotate cells with values
    thresh = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > thresh else "black", fontsize=7
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ConfMat] {save_path}")


def plot_robustness_curves(summary_df: pd.DataFrame, dataset: str, save_path: str) -> None:
    """Accuracy vs. noise rate η for each loss — PRIMARY paper figure."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (loss_name, grp) in enumerate(summary_df.groupby("loss")):
        grp = grp.sort_values("noise_rate")
        ax.plot(
            grp["noise_rate"],
            grp["accuracy"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=loss_name,
            color=_get_color(loss_name, i),
        )
    ax.set_xlabel("Label Noise Rate η", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Robustness to Label Noise — {dataset.upper()}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Robustness] {save_path}")


def plot_adversarial_curves(fgsm_df: pd.DataFrame, dataset: str, save_path: str) -> None:
    """Dual-panel: absolute accuracy + accuracy retention (acc/acc_clean).
    Retention is the scientifically correct metric for adversarial robustness:
    it asks 'what fraction of clean accuracy does each loss preserve under attack?'
    A loss that retains 90% of its clean accuracy is more adversarially robust
    than one that retains 50%, regardless of baseline accuracy differences.
    """
    # Compute accuracy retention = acc(eps) / acc(eps=0)
    baseline = fgsm_df[fgsm_df["epsilon"] == 0.0][["loss", "accuracy"]].rename(columns={"accuracy": "acc0"})
    df_norm = fgsm_df.merge(baseline, on="loss", how="left")
    df_norm["retention"] = df_norm["accuracy"] / df_norm["acc0"].clip(lower=1e-6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    for i, (loss_name, grp) in enumerate(fgsm_df.groupby("loss")):
        grp = grp.sort_values("epsilon")
        eps_pct = grp["epsilon"] * 255
        color = _get_color(loss_name, i)
        ax1.plot(eps_pct, grp["accuracy"], marker="s", linewidth=2.2, markersize=6, label=loss_name, color=color)
        norm_grp = df_norm[df_norm["loss"] == loss_name].sort_values("epsilon")
        ax2.plot(
            norm_grp["epsilon"] * 255,
            norm_grp["retention"],
            marker="s",
            linewidth=2.2,
            markersize=6,
            label=loss_name,
            color=color,
        )

    # Panel 1: absolute accuracy
    ax1.set_xlabel("FGSM Perturbation ε (×255)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (absolute)", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"FGSM — Absolute Accuracy\n{dataset.upper()}", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(alpha=0.3)

    # Panel 2: retention — the research contribution metric
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.4)
    ax2.set_xlabel("FGSM Perturbation ε (×255)", fontsize=12)
    ax2.set_ylabel("Accuracy Retention = acc(ε) / acc(0)", fontsize=12)
    ax2.set_ylim(0, 1.10)
    ax2.set_title(
        f"FGSM — Relative Adversarial Robustness\n{dataset.upper()}\n"
        "(higher = better: loss preserves more of its clean accuracy)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FGSM] {save_path}")


def plot_sdiv_surface(surface_df: pd.DataFrame, dataset: str, save_path: str) -> None:
    """3D accuracy surface over (β, λ) grid."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    pivot = surface_df.pivot(index="beta", columns="lam", values="accuracy")
    B = pivot.index.values.astype(float)
    L = pivot.columns.values.astype(float)
    Z = pivot.values
    Bgrid, Lgrid = np.meshgrid(L, B)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Bgrid, Lgrid, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("λ", fontsize=10)
    ax.set_ylabel("β", fontsize=10)
    ax.set_zlabel("Test Accuracy", fontsize=10)
    ax.set_title(f"SDIV (β,λ) Accuracy Surface — {dataset.upper()}", fontsize=11, fontweight="bold")
    fig.colorbar(surf, ax=ax, pad=0.1, label="Accuracy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3D Surface] {save_path}")


def plot_dual_robustness_frontier(
    summary_df: pd.DataFrame, fgsm_df: pd.DataFrame, dataset: str, noise_rate: float, epsilon: float, save_path: str
) -> None:
    """Dual robustness scatter: X=noise acc, Y=adversarial acc."""
    noise_acc = summary_df[summary_df["noise_rate"] == noise_rate][["loss", "accuracy"]].rename(
        columns={"accuracy": "noise_acc"}
    )
    fgsm_acc = fgsm_df[fgsm_df["epsilon"] == epsilon][["loss", "accuracy"]].rename(columns={"accuracy": "adv_acc"})
    merged = noise_acc.merge(fgsm_acc, on="loss", how="inner")

    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, row in merged.iterrows():
        c = _get_color(row["loss"], i)
        ax.scatter(row["noise_acc"], row["adv_acc"], s=140, color=c, zorder=3, edgecolors="black", linewidths=0.5)
        ax.annotate(
            row["loss"],
            (row["noise_acc"], row["adv_acc"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            color=c,
            fontweight="bold",
        )
    ax.set_xlabel(f"Accuracy under label noise (η={noise_rate})", fontsize=12)
    ax.set_ylabel(f"Accuracy under FGSM (ε={epsilon * 255:.0f}/255)", fontsize=12)
    ax.set_title(f"Dual Robustness Frontier — {dataset.upper()}", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.annotate(
        "← Ideal loss", xy=(0.85, 0.95), xycoords="axes fraction", fontsize=10, color="green", fontweight="bold"
    )
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Dual frontier] {save_path}")


def plot_summary_bar(summary_df: pd.DataFrame, dataset: str, noise_rates: list[float], save_path: str) -> None:
    """Bar chart: accuracy by loss at selected noise rates."""
    sub = summary_df[summary_df["noise_rate"].isin(noise_rates)].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, len(noise_rates), figsize=(5 * len(noise_rates), 5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, eta in zip(axes, noise_rates):
        data = sub[sub["noise_rate"] == eta].sort_values("accuracy", ascending=True)
        colors = [_get_color(n, i) for i, n in enumerate(data["loss"])]
        ax.barh(data["loss"], data["accuracy"], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Test Accuracy", fontsize=10)
        ax.set_title(f"η={eta}", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.grid(alpha=0.3, axis="x")
        # Annotate values on bars
        for j, (_, row) in enumerate(data.iterrows()):
            ax.text(row["accuracy"] + 0.01, j, f"{row['accuracy']:.3f}", va="center", fontsize=8)

    fig.suptitle(f"Loss Comparison — {dataset.upper()}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Bar chart] {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 ── PART A: VISION ViT EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def run_vision_battery(dataset_name: str) -> None:
    """Full battery for one dataset: clean + noise + FGSM + SDIV surface."""
    print(f"\n{'=' * 70}")
    print(f"  PART A: Vision ViT — {dataset_name.upper()}")
    print(f"{'=' * 70}")
    TIMING.dataset_start(dataset_name)

    X_tr, y_tr, X_te, y_te = load_vision_dataset(dataset_name)
    meta = DATASET_META[dataset_name]
    num_classes = meta["num_classes"]
    class_names = meta["class_names"]
    img_size = X_tr.shape[2]  # 32 after padding
    in_ch = X_tr.shape[1]  # 3 after conversion

    # Build loss registry and print scale table
    registry = make_loss_registry(num_classes, q=0.7, beta_sdiv=0.05, lam_sdiv=-0.8)
    print_loss_scale_table(num_classes, registry)

    results_dir = Path(CFG["RESULTS_DIR"])
    all_noise_rows, all_fgsm_rows, all_sdiv_rows = [], [], []
    _clean_models: dict = {}  # (loss_name, seed) → state_dict — reused by Battery C

    # ══════════════════════════════════════════════════════════════════════════
    # Battery A + B: Clean and noisy label training
    # ══════════════════════════════════════════════════════════════════════════
    for seed in CFG["VIT_SEEDS"]:
        print(
            f"\n  [Seed {seed}] Starting noise battery ({len(CFG['NOISE_RATES'])} rates × {len(registry)} losses) ..."
        )
        for eta in CFG["NOISE_RATES"]:
            y_noisy = inject_uniform_noise(y_tr, eta, num_classes, seed)
            T_oracle = make_T_uniform(num_classes, eta) if eta > 0 else None
            run_registry = make_loss_registry(num_classes, T_oracle, 0.7, 0.05, -0.8)

            run_results = []
            for loss_name, loss_fn in run_registry.items():
                # FRESH model for each loss × noise combo
                model = build_vit(img_size, in_ch, num_classes)
                res = train_one(
                    model,
                    loss_fn,
                    X_tr,
                    y_noisy,
                    X_te,
                    y_te,
                    n_epochs=CFG["VIT_EPOCHS"],
                    batch_size=CFG["VIT_BATCH"],
                    lr=CFG["VIT_LR"],
                    loss_name=loss_name,
                    dataset_name=dataset_name,
                    noise_rate=eta,
                    seed=seed,
                    save_model=(eta == 0.0),
                )
                run_results.append(res)
                all_noise_rows.append(
                    dict(
                        dataset=dataset_name,
                        loss=loss_name,
                        noise_rate=eta,
                        seed=seed,
                        accuracy=res.best_acc,
                    )
                )
                # Cache clean-trained model for Battery C (avoids re-training)
                if eta == 0.0 and res.model_state:
                    _clean_models[(loss_name, seed)] = res.model_state

                # Free GPU memory between runs
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            # ── Per-noise plots ────────────────────────────────────────────────
            plot_training_curves(
                run_results,
                title_suffix=f"{dataset_name.upper()} | η={eta} | seed={seed}",
                save_path=str(results_dir / f"{dataset_name}_curves_eta{eta}_s{seed}.png"),
            )
            # Confusion matrices for clean and high-noise
            if eta in (0.0, 0.3):
                for res in run_results:
                    if res.best_preds and res.true_labels:
                        safe_name = (
                            res.loss_name.replace("/", "_")
                            .replace("(", "")
                            .replace(")", "")
                            .replace(",", "_")
                            .replace("=", "")
                        )
                        plot_normalized_confusion_matrix(
                            res.true_labels,
                            res.best_preds,
                            class_names,
                            title=f"{res.loss_name} — {dataset_name.upper()} η={eta}",
                            save_path=str(results_dir / f"{dataset_name}_confmat_{safe_name}_eta{eta}_s{seed}.png"),
                        )

    # ══════════════════════════════════════════════════════════════════════════
    # Battery C: FGSM adversarial — reuse clean-trained models from Battery A
    # ══════════════════════════════════════════════════════════════════════════
    print("\n  FGSM battery on clean-trained models (reusing Battery A models) ...")
    for seed in CFG["VIT_SEEDS"]:
        for loss_name, loss_fn in make_loss_registry(num_classes, None, 0.7, 0.05, -0.8).items():
            model = build_vit(img_size, in_ch, num_classes)
            cached_key = (loss_name, seed)
            if cached_key in _clean_models:
                # Reuse exact clean model from Battery A — consistent with noise results
                model.load_state_dict(_clean_models[cached_key])
                print(f"  [Battery C] Reusing Battery A model: {loss_name}, seed={seed}")
            else:
                # Fallback: train fresh clean model (should not happen normally)
                print(f"  [Battery C] No cached model for {loss_name} seed={seed}, training fresh ...")
                res_clean = train_one(
                    model,
                    loss_fn,
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    n_epochs=CFG["VIT_EPOCHS"],
                    batch_size=CFG["VIT_BATCH"],
                    lr=CFG["VIT_LR"],
                    loss_name=loss_name,
                    dataset_name=dataset_name,
                    noise_rate=0.0,
                    seed=seed,
                    save_model=True,
                )
                if res_clean.model_state:
                    model.load_state_dict(res_clean.model_state)
            model.eval()

            te_ds = NumpyImageDataset(X_te, y_te)
            te_loader = DataLoader(
                te_ds,
                batch_size=CFG["VIT_BATCH"],
                shuffle=False,
                num_workers=N_WORKERS,
                pin_memory=(DEVICE.type == "cuda" and N_WORKERS == 8),
            )
            cce_fn = CCELoss()

            for eps in CFG["FGSM_EPS"]:
                preds, labs = [], []
                for xb, yb in te_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    x_adv = fgsm_attack(model, xb, yb, eps, cce_fn)
                    with torch.no_grad(), autocast(enabled=USE_AMP):
                        out = model(x_adv)
                    preds.extend(out.argmax(1).cpu().tolist())
                    labs.extend(yb.cpu().tolist())
                acc = accuracy_score(labs, preds)
                all_fgsm_rows.append(
                    dict(
                        dataset=dataset_name,
                        loss=loss_name,
                        epsilon=eps,
                        seed=seed,
                        accuracy=acc,
                    )
                )
                print(f"    FGSM ε={eps * 255:.1f}/255 | {loss_name:25s} | acc={acc:.4f}")

            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # ══════════════════════════════════════════════════════════════════════════
    # Save CSVs and produce summary plots
    # ══════════════════════════════════════════════════════════════════════════
    noise_df = pd.DataFrame(all_noise_rows)
    fgsm_df = pd.DataFrame(all_fgsm_rows)

    noise_csv = str(results_dir / f"{dataset_name}_noise_results.csv")
    fgsm_csv = str(results_dir / f"{dataset_name}_fgsm_results.csv")
    noise_df.to_csv(noise_csv, index=False)
    fgsm_df.to_csv(fgsm_csv, index=False)
    print(f"  [CSV] {noise_csv}")
    print(f"  [CSV] {fgsm_csv}")

    # Aggregate over seeds
    noise_agg = noise_df.groupby(["dataset", "loss", "noise_rate"], as_index=False).agg(
        accuracy=("accuracy", "mean"), std=("accuracy", "std")
    )
    fgsm_agg = fgsm_df.groupby(["dataset", "loss", "epsilon"], as_index=False).agg(
        accuracy=("accuracy", "mean"), std=("accuracy", "std")
    )

    plot_robustness_curves(
        noise_agg[noise_agg["dataset"] == dataset_name],
        dataset_name,
        str(results_dir / f"{dataset_name}_robustness_noise.png"),
    )
    plot_adversarial_curves(
        fgsm_agg[fgsm_agg["dataset"] == dataset_name],
        dataset_name,
        str(results_dir / f"{dataset_name}_robustness_fgsm.png"),
    )
    plot_dual_robustness_frontier(
        noise_agg[noise_agg["dataset"] == dataset_name],
        fgsm_agg[fgsm_agg["dataset"] == dataset_name],
        dataset_name,
        noise_rate=0.2,
        epsilon=4 / 255,
        save_path=str(results_dir / f"{dataset_name}_dual_frontier.png"),
    )
    plot_summary_bar(
        noise_agg[noise_agg["dataset"] == dataset_name],
        dataset_name,
        noise_rates=[0.0, 0.2, 0.4],
        save_path=str(results_dir / f"{dataset_name}_summary_bar.png"),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Battery E: SDIV (β, λ) surface
    # ══════════════════════════════════════════════════════════════════════════
    print("\n  Battery E: SDIV (β,λ) accuracy surface ...")
    for seed in CFG["VIT_SEEDS"]:
        for beta in CFG["BETA_GRID"]:
            for lam in CFG["LAM_GRID"]:
                A = 1.0 + lam * (1 - beta)
                B_val = beta - lam * (1 - beta)
                if A <= 0 or B_val <= 0:
                    continue
                model = build_vit(img_size, in_ch, num_classes)
                try:
                    loss_fn = SDIVLoss(beta, lam)
                except ValueError:
                    continue
                res = train_one(
                    model,
                    loss_fn,
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    n_epochs=CFG["VIT_EPOCHS"],
                    batch_size=CFG["VIT_BATCH"],
                    lr=CFG["VIT_LR"],
                    loss_name=f"SDIV(β={beta},λ={lam})",
                    dataset_name=dataset_name,
                    noise_rate=0.0,
                    seed=seed,
                )
                all_sdiv_rows.append(
                    dict(
                        dataset=dataset_name,
                        beta=beta,
                        lam=lam,
                        seed=seed,
                        accuracy=res.best_acc,
                    )
                )
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    if all_sdiv_rows:
        sdiv_df = pd.DataFrame(all_sdiv_rows)
        sdiv_agg = sdiv_df.groupby(["dataset", "beta", "lam"], as_index=False)["accuracy"].mean()
        sdiv_df.to_csv(str(results_dir / f"{dataset_name}_sdiv_surface.csv"), index=False)
        plot_sdiv_surface(
            sdiv_agg[sdiv_agg["dataset"] == dataset_name],
            dataset_name,
            save_path=str(results_dir / f"{dataset_name}_sdiv_surface_3d.png"),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Curriculum GCE annealing
    # ══════════════════════════════════════════════════════════════════════════
    run_curriculum_annealing(dataset_name, X_tr, y_tr, X_te, y_te, num_classes, img_size, in_ch, eta=0.3, seed=42)

    TIMING.print_loss_summary(dataset_name)
    print(f"\n  ✓ Part A done for {dataset_name}.\n")


def run_curriculum_annealing(
    dataset_name, X_tr, y_tr, X_te, y_te, num_classes, img_size, in_ch, eta: float = 0.3, seed: int = 42
) -> None:
    """Curriculum GCE annealing: q(t) = q_max · (1 - t/T)"""
    print(f"\n  [Curriculum] GCE annealing | {dataset_name} | η={eta}")
    y_noisy = inject_uniform_noise(y_tr, eta, num_classes, seed)

    model = build_vit(img_size, in_ch, num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["VIT_LR"], weight_decay=1e-4)
    scaler = GradScaler(enabled=USE_AMP)
    _dl_kw = dict(num_workers=N_WORKERS, pin_memory=(DEVICE.type == "cuda" and N_WORKERS == 8))
    tr_loader = DataLoader(NumpyImageDataset(X_tr, y_noisy), CFG["VIT_BATCH"], shuffle=True, **_dl_kw)
    te_loader = DataLoader(NumpyImageDataset(X_te, y_te), CFG["VIT_BATCH"] * 2, shuffle=False, **_dl_kw)

    T = CFG["VIT_EPOCHS"]
    q_max = 0.9
    history = {"epoch": [], "q": [], "acc": []}

    for epoch in tqdm(range(1, T + 1), desc="Curriculum GCE", leave=True):
        q_t = q_max * (1.0 - (epoch - 1) / T)
        loss_fn = GCELoss(q=max(q_t, 1e-3))

        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for xb, yb in te_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                with autocast(enabled=USE_AMP):
                    out = model(xb)
                preds.extend(out.argmax(1).cpu().tolist())
                labs.extend(yb.tolist())
        acc = accuracy_score(labs, preds)
        history["epoch"].append(epoch)
        history["q"].append(q_t)
        history["acc"].append(acc)

    results_dir = Path(CFG["RESULTS_DIR"])
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(history["epoch"], history["acc"], "b-o", ms=3, label="Test Accuracy")
    ax2.plot(history["epoch"], history["q"], "r--", alpha=0.7, label="q (GCE param)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy", color="b")
    ax2.set_ylabel("GCE q value", color="r")
    ax1.set_title(
        f"Curriculum GCE Annealing — {dataset_name.upper()} η={eta}\n(q decays from {q_max} → 0 over training)"
    )
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower left")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    save_path = str(results_dir / f"{dataset_name}_curriculum_gce_eta{eta}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Curriculum] {save_path}")
    pd.DataFrame(history).to_csv(str(results_dir / f"{dataset_name}_curriculum_history.csv"), index=False)

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8B ── PART C: PRETRAINED FOUNDATION MODEL FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════════════
#
# Fine-tunes 3 pretrained vision models on PathMNIST and DermaMNIST:
#   1. google/vit-base-patch16-224      (ImageNet ViT, 86M params, 224×224)
#   2. openai/clip-vit-base-patch32     (CLIP vision encoder, 86M, 224×224)
#   3. google/medsiglip-448             (Medical SigLIP, 400M, 448×448)
#
# Strategy: FULL FINE-TUNING (freeze_backbone=False)
#   All backbone weights are updated during training using the robust loss.
#   Differential learning rates: head=1e-3, backbone=1e-5 (100× slower).
#   This is the ONLY mode that properly tests rSDNet's robustness guarantees,
#   because SDIV/TSCCE/FCL/GCE gradient-shaping must influence the full
#   feature representation, not just a single linear layer.
#
# Note: linear probe (frozen backbone) tests a DIFFERENT question
#   ("which loss is best for noisy-label head-only classification") and
#   CANNOT demonstrate the loss function robustness claimed in rSDNet.
#
# This tests: "Do robust loss functions protect pretrained model adaptation
#              under label noise — the transfer learning robustness claim."
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModel,
        AutoProcessor,
        CLIPProcessor,
        CLIPVisionModel,
        SiglipVisionModel,
        ViTImageProcessor,
        ViTModel,
    )

    _HAS_PRETRAINED = True
except ImportError:
    _HAS_PRETRAINED = False
    print("[Part C] transformers not found — pretrained fine-tuning disabled.")


# ── Model registry ────────────────────────────────────────────────────────────
PRETRAINED_MODELS = {
    "ViT-B/16": {
        "hf_id": "google/vit-base-patch16-224",
        "model_class": "ViTModel",
        "proc_class": "ViTImageProcessor",
        "hidden_dim": 768,
        "input_size": 224,
        "gated": False,  # no terms-of-use gate
    },
    "CLIP-ViT": {
        "hf_id": "openai/clip-vit-base-patch32",
        "model_class": "CLIPVisionModel",
        "proc_class": "CLIPProcessor",
        "hidden_dim": 768,
        "input_size": 224,
        "gated": False,
    },
    "MedSigLIP": {
        "hf_id": "google/medsiglip-448",
        "model_class": "SiglipVisionModel",
        "proc_class": "AutoImageProcessor",
        "hidden_dim": 1152,
        "input_size": 448,
        "gated": True,  # requires HF terms acceptance
    },
}


class PretrainedImageDataset(Dataset):
    """Dataset that resizes raw numpy images to the pretrained model's resolution
    and applies model-specific normalization via PIL + manual normalization."""

    def __init__(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        target_size: int = 224,
        mean: tuple[float, ...] = (0.5, 0.5, 0.5),
        std: tuple[float, ...] = (0.5, 0.5, 0.5),
    ):
        """
        Args:
            X_np: (N, C, H, W) float32 in [0,1]
            y_np: (N,) int64
            target_size: resize to (target_size, target_size)
            mean/std: channel-wise normalization
        """
        from PIL import Image

        self.y = torch.from_numpy(y_np).long()
        self.mean = torch.tensor(mean).float().view(3, 1, 1)
        self.std = torch.tensor(std).float().view(3, 1, 1)

        # Resize all images at init (they're small — 28×28 → 224 or 448)
        print(f"    Resizing {len(y_np)} images: {X_np.shape[2]}×{X_np.shape[3]} → {target_size}×{target_size} ...")
        resized = []
        for i in range(len(X_np)):
            # Convert (C,H,W) → PIL → resize → back to tensor
            img = X_np[i].transpose(1, 2, 0)  # CHW → HWC
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode="RGB")
            pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
            arr = np.array(pil_img, dtype=np.float32) / 255.0  # back to [0,1]
            resized.append(arr.transpose(2, 0, 1))  # HWC → CHW

        self.X = torch.from_numpy(np.stack(resized)).float()
        # Apply normalization: (x - mean) / std
        self.X = (self.X - self.mean.unsqueeze(0)) / self.std.unsqueeze(0)
        print(f"    Done. Dataset shape: {self.X.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class PretrainedClassifier(nn.Module):
    """Wraps a frozen pretrained backbone + trainable classification head."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        num_classes: int,
        model_type: str = "vit",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_dim, num_classes)
        self.model_type = model_type  # 'vit', 'clip', 'siglip'
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            n_frozen = sum(p.numel() for p in self.backbone.parameters())
            n_head = sum(p.numel() for p in self.head.parameters())
            print(f"    [Freeze] Backbone: {n_frozen / 1e6:.1f}M frozen | Head: {n_head / 1e3:.1f}K trainable")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            if self.model_type == "clip" or self.model_type == "siglip":
                out = self.backbone(pixel_values=pixel_values)
                features = out.pooler_output  # (B, hidden_dim)
            else:  # 'vit'
                out = self.backbone(pixel_values=pixel_values)
                features = out.last_hidden_state[:, 0, :]  # CLS token
        return self.head(features)


def build_pretrained_classifier(
    model_key: str, num_classes: int, freeze_backbone: bool = False
) -> tuple[PretrainedClassifier, dict]:
    """Load a pretrained model from HuggingFace and wrap with classifier head.

    Returns (model, info_dict) where info_dict has 'input_size', 'mean', 'std'.
    """
    info = PRETRAINED_MODELS[model_key]
    hf_id = info["hf_id"]
    hidden_dim = info["hidden_dim"]
    input_size = info["input_size"]

    print(f"  [Loading] {model_key} ({hf_id}) ...")

    # Common kwargs for safe loading
    load_kw = dict(use_safetensors=True)

    # Add HF token for gated models (MedSigLIP)
    hf_token = os.environ.get("HF_TOKEN", None)
    if info["gated"] and hf_token:
        load_kw["token"] = hf_token
    elif info["gated"] and not hf_token:
        print(f"    ⚠ {model_key} is gated — set HF_TOKEN env variable.")
        print(f"    → Accept terms at: https://huggingface.co/{hf_id}")
        raise OSError(f"HF_TOKEN required for {hf_id}")

    # Load backbone
    if info["model_class"] == "ViTModel":
        backbone = ViTModel.from_pretrained(hf_id, **load_kw)
        model_type = "vit"
        # ViT normalization: ImageNet mean/std
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif info["model_class"] == "CLIPVisionModel":
        backbone = CLIPVisionModel.from_pretrained(hf_id, **load_kw)
        model_type = "clip"
        # CLIP normalization
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    elif info["model_class"] == "SiglipVisionModel":
        backbone = SiglipVisionModel.from_pretrained(hf_id, **load_kw)
        model_type = "siglip"
        # SigLIP normalization: (-1, 1) range → mean=0.5, std=0.5
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        raise ValueError(f"Unknown model class: {info['model_class']}")

    backbone = backbone.to(DEVICE)
    model = PretrainedClassifier(
        backbone, hidden_dim, num_classes, model_type=model_type, freeze_backbone=freeze_backbone
    ).to(DEVICE)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  [{model_key}] {n_total / 1e6:.1f}M total | "
        f"{n_trainable / 1e3:.1f}K trainable | {input_size}×{input_size} | on {DEVICE}"
    )

    return model, {
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "model_key": model_key,
    }


def train_pretrained(
    model: PretrainedClassifier,
    loss_fn: _RobustLoss,
    X_tr: np.ndarray,
    y_tr_noisy: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_epochs: int,
    batch_size: int,
    lr_head: float,
    model_info: dict,
    loss_name: str,
    dataset_name: str,
    model_key: str,
    noise_rate: float,
    seed: int,
) -> TrainResult:
    """Train a pretrained classifier with a given loss. Linear probe or full FT."""
    set_seed(seed)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(DEVICE)

    input_size = model_info["input_size"]
    mean, std = model_info["mean"], model_info["std"]

    tr_ds = PretrainedImageDataset(X_tr, y_tr_noisy, input_size, mean, std)
    te_ds = PretrainedImageDataset(X_te, y_te, input_size, mean, std)

    # Smaller batch for pretrained (larger images)
    effective_batch = min(batch_size, 64 if input_size >= 448 else 128)
    _dl_kw = dict(num_workers=N_WORKERS, pin_memory=(DEVICE.type == "cuda" and N_WORKERS == 8))
    tr_loader = DataLoader(tr_ds, batch_size=effective_batch, shuffle=True, drop_last=False, **_dl_kw)
    te_loader = DataLoader(te_ds, batch_size=effective_batch * 2, shuffle=False, **_dl_kw)

    # Separate param groups: backbone (lower LR) vs head (higher LR)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    param_groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_head * 0.01})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = GradScaler(enabled=USE_AMP)

    result = TrainResult(
        loss_name=f"{model_key}/{loss_name}",
        dataset=dataset_name,
        noise_rate=noise_rate,
        seed=seed,
        history={"train_loss": [], "test_acc": [], "test_cce": []},
    )
    best_val, best_preds, best_true = 0.0, [], []
    t0 = time.time()
    _gpu_verified = False

    pbar = tqdm(
        range(1, n_epochs + 1), desc=f"{model_key}|{dataset_name}|{loss_name}|η={noise_rate}", leave=False, unit="ep"
    )
    for epoch in pbar:
        model.train()
        # Full fine-tune: backbone in train mode so BN/dropout adapt.
        # If backbone were frozen, we'd call model.backbone.eval() here — but
        # that would prevent the robust loss from shaping backbone features.

        epoch_loss, n_b = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            if not _gpu_verified:
                _gpu_verified = True
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                    mem_mb = torch.cuda.memory_allocated() / 1024**2
                    print(f"  [GPU ✓] {model_key} | xb={xb.shape} on {xb.device} | GPU mem={mem_mb:.0f} MB")
                else:
                    print(f"  [CPU] {model_key} running on CPU")

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_b += 1
        scheduler.step()
        epoch_loss /= max(n_b, 1)

        # Evaluate
        model.eval()
        preds, labs, cce_tot = [], [], 0.0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                with autocast(enabled=USE_AMP):
                    logits = model(xb)
                preds.extend(logits.argmax(1).cpu().tolist())
                labs.extend(yb.cpu().tolist())
                cce_tot += F.cross_entropy(logits.float(), yb).item()
        acc = accuracy_score(labs, preds)
        cce_val = cce_tot / max(len(te_loader), 1)

        result.history["train_loss"].append(epoch_loss)
        result.history["test_acc"].append(acc)
        result.history["test_cce"].append(cce_val)

        if acc > best_val:
            best_val = acc
            best_preds, best_true = preds[:], labs[:]

        pbar.set_postfix({"loss": f"{epoch_loss:.4f}", "acc": f"{acc:.4f}"})

    result.best_acc = best_val
    result.best_preds = best_preds
    result.true_labels = best_true
    result.elapsed_s = time.time() - t0
    print(f"  [{model_key}/{loss_name}] η={noise_rate} best_acc={best_val:.4f} time={result.elapsed_s:.0f}s")
    TIMING.record(dataset_name, f"{model_key}/{loss_name}", noise_rate, seed, result.elapsed_s, best_val, part="C")
    return result


def plot_backbone_comparison(all_results: dict[str, pd.DataFrame], dataset_name: str, save_path: str) -> None:
    """Compare from-scratch ViT vs pretrained backbones on the same dataset."""
    fig, ax = plt.subplots(figsize=(12, 6))

    backbone_colors = {
        "ViT-Scratch": "#1f77b4",
        "ViT-B/16": "#ff7f0e",
        "CLIP-ViT": "#2ca02c",
        "MedSigLIP": "#d62728",
    }

    for backbone_name, df in all_results.items():
        if df.empty:
            continue
        agg = df.groupby("noise_rate")["accuracy"].mean().reset_index()
        color = backbone_colors.get(backbone_name, "#333333")
        marker = "o" if backbone_name == "ViT-Scratch" else "s"
        linestyle = "--" if backbone_name == "ViT-Scratch" else "-"
        ax.plot(
            agg["noise_rate"],
            agg["accuracy"],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=8,
            color=color,
            label=backbone_name,
        )

    ax.set_xlabel("Label Noise Rate η", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"{dataset_name.upper()} — Backbone Comparison\n(Best across all losses at each η)", fontsize=13)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {save_path}")


def run_pretrained_battery(dataset_name: str) -> None:
    """Run pretrained model fine-tuning battery on one medical dataset."""
    if not _HAS_PRETRAINED:
        print("  [Skip] Part C requires 'transformers'. Install it.")
        return

    results_dir = Path(CFG["RESULTS_DIR"])
    meta = DATASET_META[dataset_name]
    num_classes = meta["num_classes"]
    class_names = meta["class_names"]

    print(f"\n{'=' * 70}")
    print(f"  PART C: Pretrained Fine-Tuning — {dataset_name.upper()}")
    print(f"  Models: {list(PRETRAINED_MODELS.keys())}")
    print(f"  Loss functions: 9 | Noise rates: {CFG['NOISE_RATES']}")
    print(f"{'=' * 70}")

    # Load dataset once
    X_tr, y_tr, X_te, y_te = load_vision_dataset(dataset_name)

    # Full fine-tune needs more epochs than a linear probe, but fewer than from-scratch ViT.
    # backbone LR = lr_head * 0.01 (differential LR, see train_pretrained param_groups)
    n_epochs = min(CFG["VIT_EPOCHS"], 30) if CFG["QUICK_RUN"] else min(CFG["VIT_EPOCHS"], 60)
    lr_head = 1e-3  # head LR; backbone receives lr_head * 0.01 = 1e-5

    all_backbone_results = {}  # backbone_name → DataFrame

    for model_key, model_info_cfg in PRETRAINED_MODELS.items():
        print(f"\n  ═══ {model_key} ═══")
        all_rows = []

        try:
            # Noise battery
            for seed in CFG["VIT_SEEDS"]:
                for eta in CFG["NOISE_RATES"]:
                    y_noisy = inject_uniform_noise(y_tr, eta, num_classes, seed)
                    T_oracle = make_T_uniform(num_classes, eta) if eta > 0 else None
                    registry = make_loss_registry(num_classes, T_oracle, 0.7, 0.05, -0.8)

                    for loss_name, loss_fn in registry.items():
                        try:
                            # freeze_backbone=False: robust loss updates ALL weights
                            # (backbone at lr*0.01, head at lr — differential LR)
                            model, m_info = build_pretrained_classifier(model_key, num_classes, freeze_backbone=False)
                            res = train_pretrained(
                                model,
                                loss_fn,
                                X_tr,
                                y_noisy,
                                X_te,
                                y_te,
                                n_epochs=n_epochs,
                                batch_size=CFG["VIT_BATCH"],
                                lr_head=lr_head,
                                model_info=m_info,
                                loss_name=loss_name,
                                dataset_name=dataset_name,
                                model_key=model_key,
                                noise_rate=eta,
                                seed=seed,
                            )
                            all_rows.append(
                                dict(
                                    dataset=dataset_name,
                                    backbone=model_key,
                                    loss=loss_name,
                                    noise_rate=eta,
                                    seed=seed,
                                    accuracy=res.best_acc,
                                )
                            )

                            # Confusion matrix at clean and high noise
                            if eta in (0.0, 0.3) and seed == CFG["VIT_SEEDS"][0]:
                                _safe = (
                                    f"{model_key}_{loss_name}".replace("/", "_")
                                    .replace("(", "")
                                    .replace(")", "")
                                    .replace(",", "_")
                                    .replace("=", "")
                                )
                                plot_normalized_confusion_matrix(
                                    y_true=res.true_labels,
                                    y_pred=res.best_preds,
                                    class_names=class_names,
                                    title=f"{model_key}/{loss_name} — {dataset_name.upper()} η={eta}",
                                    save_path=str(results_dir / f"{dataset_name}_{_safe}_confmat_eta{eta}.png"),
                                )
                        except Exception as e:
                            print(f"    ⚠ {model_key}/{loss_name} η={eta}: {e}")
                        finally:
                            if "model" in dir():
                                del model
                            if DEVICE.type == "cuda":
                                torch.cuda.empty_cache()
                            gc.collect()

            # Save results CSV
            if all_rows:
                df = pd.DataFrame(all_rows)
                csv_path = results_dir / f"{dataset_name}_{model_key}_pretrained_results.csv".replace("/", "_")
                df.to_csv(str(csv_path), index=False)
                print(f"  [CSV] {csv_path.name}")

                # Robustness curve per backbone
                agg = df.groupby(["loss", "noise_rate"], as_index=False).agg(accuracy=("accuracy", "mean"))
                plot_robustness_curves(
                    agg,
                    f"{dataset_name}/{model_key}",
                    str(results_dir / f"{dataset_name}_{model_key}_robustness_noise.png".replace("/", "_")),
                )

                all_backbone_results[model_key] = df

        except OSError as e:
            print(f"    ⚠ Skipping {model_key}: {e}")
            continue
        except Exception as e:
            import traceback

            print(f"    ✗ {model_key} failed: {e}")
            traceback.print_exc()
            continue

    # ── Load from-scratch ViT results for comparison ──────────────────────────
    scratch_csv = results_dir / f"{dataset_name}_noise_results.csv"
    if scratch_csv.exists():
        scratch_df = pd.read_csv(str(scratch_csv))
        # Get best accuracy per noise rate (across losses)
        all_backbone_results["ViT-Scratch"] = scratch_df

    # ── Backbone comparison plot ──────────────────────────────────────────────
    if len(all_backbone_results) > 1:
        # For comparison: best accuracy across all losses at each noise rate
        comparison_data = {}
        for bk_name, df in all_backbone_results.items():
            best_per_eta = df.groupby("noise_rate")["accuracy"].max().reset_index()
            best_per_eta["backbone"] = bk_name
            comparison_data[bk_name] = best_per_eta

        plot_backbone_comparison(
            comparison_data,
            dataset_name,
            str(results_dir / f"{dataset_name}_backbone_comparison.png"),
        )

    print(f"\n  ✓ Part C done for {dataset_name}.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 ── PART B: NLP BERT FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from datasets import load_dataset as _hf_load
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    _HF_OK = True
except ImportError:
    _HF_OK = False
    print("[Warning] transformers / datasets not installed — Part B (NLP) skipped.")

NLP_DATASETS = {
    "PubMedQA": {
        "hf_name": "qiaojin/PubMedQA",
        "hf_config": "pqa_labeled",
        "text_field": "question",
        "label_field": "final_decision",
        "label_map": {"yes": 0, "no": 1, "maybe": 2},
        "num_classes": 3,
        "class_names": ["yes", "no", "maybe"],
        "model": "allenai/scibert_scivocab_uncased",
        "splits": {"train": "train", "val": None, "test": None},
    },
    "Emotion": {
        "hf_name": "dair-ai/emotion",
        "hf_config": None,
        "text_field": "text",
        "label_field": "label",
        "label_map": None,
        "num_classes": 6,
        "class_names": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "model": "distilbert-base-uncased",
        "splits": {"train": "train", "val": "validation", "test": "test"},
    },
}


class _TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        enc = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        self.inp = enc
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.inp.items()}, self.labels[i]


def run_nlp_battery(ds_name: str) -> None:
    if not _HF_OK:
        return
    print(f"\n{'=' * 70}\n  PART B: NLP BERT — {ds_name}\n{'=' * 70}")
    TIMING.dataset_start(f"nlp:{ds_name}")
    dcfg = NLP_DATASETS[ds_name]
    C = dcfg["num_classes"]
    results_dir = Path(CFG["RESULTS_DIR"])

    # Load data
    raw = _hf_load(dcfg["hf_name"], dcfg.get("hf_config"))

    def _extract(skey):
        if not skey or skey not in raw:
            return [], []
        split = raw[skey]
        texts = list(split[dcfg["text_field"]])
        labels = list(split[dcfg["label_field"]])
        lmap = dcfg["label_map"]
        if lmap:
            pairs = [(t, lmap[l]) for t, l in zip(texts, labels) if l in lmap]
            if not pairs:
                return [], []
            texts, labels = zip(*pairs)
            return list(texts), list(labels)
        return texts, [int(l) for l in labels]

    tr_t, tr_l = _extract(dcfg["splits"]["train"])
    val_t, val_l = _extract(dcfg["splits"].get("val"))
    te_t, te_l = _extract(dcfg["splits"].get("test"))

    if not te_t and tr_t:
        tr_t, te_t, tr_l, te_l = train_test_split(tr_t, tr_l, test_size=0.15, random_state=42, stratify=tr_l)
    if not val_t and tr_t:
        tr_t, val_t, tr_l, val_l = train_test_split(tr_t, tr_l, test_size=0.15, random_state=42, stratify=tr_l)

    max_tr = CFG["NLP_MAX_TRAIN"]
    if max_tr and len(tr_t) > max_tr:
        idx = np.random.RandomState(42).choice(len(tr_t), max_tr, replace=False)
        tr_t = [tr_t[i] for i in idx]
        tr_l = [tr_l[i] for i in idx]

    print(f"  Data: train={len(tr_t)} val={len(val_t)} test={len(te_t)}")

    tok = AutoTokenizer.from_pretrained(dcfg["model"])
    mk_loader = lambda t, l, sh: DataLoader(
        _TextDataset(t, l, tok, CFG["NLP_MAX_LEN"]), CFG["NLP_BATCH"], shuffle=sh, num_workers=0
    )
    tr_ldr = mk_loader(tr_t, tr_l, True)
    val_ldr = mk_loader(val_t, val_l, False)
    te_ldr = mk_loader(te_t, te_l, False)

    nlp_registry = make_loss_registry(C, None, 0.7, 0.05, -0.8)
    all_rows = []

    for loss_name, loss_fn in nlp_registry.items():
        set_seed(42)
        _nlp_t0 = time.time()
        # Use safetensors to avoid torch.load CVE
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                dcfg["model"], num_labels=C, ignore_mismatched_sizes=True, use_safetensors=True
            ).to(DEVICE)
        except Exception:
            # Fallback: some models may not have safetensors
            model = AutoModelForSequenceClassification.from_pretrained(
                dcfg["model"], num_labels=C, ignore_mismatched_sizes=True
            ).to(DEVICE)

        if hasattr(loss_fn, "to"):
            loss_fn = loss_fn.to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["NLP_LR"], weight_decay=0.01)
        steps = len(tr_ldr) * CFG["NLP_EPOCHS"]
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = GradScaler(enabled=USE_AMP)

        best_acc = 0.0
        hist = {"epoch": [], "train_loss": [], "val_acc": [], "test_acc": []}
        for ep in tqdm(range(1, CFG["NLP_EPOCHS"] + 1), desc=f"{ds_name}|{loss_name}", leave=False):
            model.train()
            tr_loss = 0.0
            for batch, labs in tr_ldr:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                labs = labs.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=USE_AMP):
                    logits = model(**batch).logits
                    loss = loss_fn(logits, labs)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                tr_loss += loss.item()
            tr_loss /= max(len(tr_ldr), 1)

            model.eval()
            for split_ldr, key in [(val_ldr, "val_acc"), (te_ldr, "test_acc")]:
                ps, ls = [], []
                with torch.no_grad():
                    for batch, labs in split_ldr:
                        batch = {k: v.to(DEVICE) for k, v in batch.items()}
                        with autocast(enabled=USE_AMP):
                            out = model(**batch).logits
                        ps.extend(out.argmax(1).cpu().tolist())
                        ls.extend(labs.tolist())
                hist[key].append(accuracy_score(ls, ps))
            hist["train_loss"].append(tr_loss)
            hist["epoch"].append(ep)
            if hist["val_acc"][-1] > best_acc:
                best_acc = hist["test_acc"][-1]

        _nlp_elapsed = time.time() - _nlp_t0
        all_rows.append(dict(dataset=ds_name, loss=loss_name, best_acc=best_acc))
        print(f"  {ds_name}|{loss_name}: best_test_acc={best_acc:.4f}  time={_nlp_elapsed:.0f}s")
        TIMING.record(f"nlp:{ds_name}", loss_name, 0.0, 42, _nlp_elapsed, best_acc, part="B")

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    pd.DataFrame(all_rows).to_csv(str(results_dir / f"nlp_{ds_name}_results.csv"), index=False)
    print(f"  [CSV] nlp_{ds_name}_results.csv")
    TIMING.print_loss_summary(f"nlp:{ds_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 ── MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def _auto_archive_and_download(results_dir: Path, script_path: Path) -> str:
    """Create a ZIP of ALL outputs + the experiment script, print download hint.

    Returns the archive path string.
    """
    import datetime
    import shutil

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"robust_nn_results_{ts}"
    archive_base = Path("/workspace") / name

    # Collect everything: results dir + the .py script + this notebook (if present)
    staging = Path("/workspace") / f"_stage_{ts}"
    staging.mkdir(parents=True, exist_ok=True)

    # 1. Copy results directory
    if results_dir.exists():
        shutil.copytree(str(results_dir), str(staging / results_dir.name))

    # 2. Copy the experiment script
    if script_path.exists():
        shutil.copy2(str(script_path), str(staging / script_path.name))

    # 3. Copy any .ipynb in same folder
    for nb in script_path.parent.glob("*.ipynb"):
        shutil.copy2(str(nb), str(staging / nb.name))

    # 4. Copy timing CSV if saved separately
    timing_csv = results_dir / "timing_ledger.csv"
    if timing_csv.exists():
        shutil.copy2(str(timing_csv), str(staging / "timing_ledger.csv"))

    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(staging.parent), base_dir=staging.name)
    shutil.rmtree(str(staging), ignore_errors=True)

    zip_mb = Path(archive_path).stat().st_size / 1024**2
    n_files = sum(1 for _ in Path("/workspace").glob(f"{name}*"))

    print(f"\n{'█' * 70}")
    print("  AUTO-ARCHIVE COMPLETE")
    print(f"  File   : {archive_path}")
    print(f"  Size   : {zip_mb:.1f} MB")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  DOWNLOAD BEFORE STOPPING POD:")
    print(f"    RunPod Dashboard → Pod → Files → /workspace/{Path(archive_path).name}")
    print("  Or from Jupyter terminal:")
    print(f"    scp <pod-ip>:/workspace/{Path(archive_path).name} ./")
    print(f"{'█' * 70}\n")

    # Try to trigger a Jupyter download link (works when inside Jupyter)
    try:
        from IPython.display import HTML, display

        display(
            HTML(
                f"<div style='background:#1a472a;color:#fff;padding:14px;"
                f"border-radius:6px;font-family:monospace;'>"
                f"<b style='font-size:1.1em'>📦 Archive ready — DOWNLOAD NOW</b><br><br>"
                f"<b>File:</b> <code style='color:#7fff00'>{archive_path}</code><br>"
                f"<b>Size:</b> {zip_mb:.1f} MB<br><br>"
                f"<b>Steps:</b><br>"
                f"1. RunPod Dashboard → your pod → <b>Files</b> tab<br>"
                f"2. Navigate to <code>/workspace/</code><br>"
                f"3. Click <b>{Path(archive_path).name}</b> → Download<br>"
                f"</div>"
            )
        )
    except Exception:
        pass  # Not in Jupyter — terminal-only is fine

    return archive_path


if __name__ == "__main__":
    TIMING.session_start()
    t_start = time.time()
    print(f"\n{'=' * 70}")
    print("  Robust NN Experiments — 13 April 2026")
    print(f"  Quick={CFG['QUICK_RUN']}  Part={CFG['PART']}")
    print(f"  Datasets={CFG['VIT_DATASETS']}")
    print(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results → {CFG['RESULTS_DIR']}/")
    print(f"{'=' * 70}\n")

    if "A" in CFG["PART"]:
        for ds in CFG["VIT_DATASETS"]:
            ds = ds.strip()
            if ds in DATASET_META:
                run_vision_battery(ds)
            else:
                print(f"[Skip] Unknown dataset: {ds}")

    if "C" in CFG["PART"]:
        # Run pretrained fine-tuning only on medical datasets
        medical_ds = [d.strip() for d in CFG["VIT_DATASETS"] if d.strip() in ("pathmnist", "dermamnist")]
        if not medical_ds:
            medical_ds = ["pathmnist", "dermamnist"]
        print(f"\n  Part C datasets: {medical_ds}")
        for ds in medical_ds:
            run_pretrained_battery(ds)

    if "B" in CFG["PART"]:
        for ds in ["Emotion", "PubMedQA"]:
            run_nlp_battery(ds)

    # ── Full timing report ────────────────────────────────────────────────────
    TIMING.print_final_summary()
    TIMING.to_csv(str(Path(CFG["RESULTS_DIR"]) / "timing_ledger.csv"))

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  All experiments done. Total wall time: {_fmt(elapsed)}")
    print(f"  Finished : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results saved to: {CFG['RESULTS_DIR']}/")
    print(f"{'=' * 70}")

    # ── Auto-archive + download everything ───────────────────────────────────
    try:
        _script = Path(__file__)
    except NameError:
        _script = Path(os.getcwd())  # notebook: __file__ not defined
    _auto_archive_and_download(
        results_dir=Path(CFG["RESULTS_DIR"]),
        script_path=_script,
    )
