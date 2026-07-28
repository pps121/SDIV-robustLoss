"""
Part 3 — BERT-Style Robust Classification: Mental Health & Medical Text
========================================================================
Replaces the vision-based Part 2 grid with pre-trained language models
fine-tuned under noisy labels and using robust loss functions.

  Domains   :  Biomedical / Clinical NLP  (multimodal-adjacent: medical QA, radiology reasoning)
  Datasets  :  medmcqa                  (4-cls, PG medical entrance QA, ~234k) ← medium
               GBaker/MedQA-USMLE-4-options (4-cls, USMLE Step exam QA, ~10k)  ← hard
               qiaojin/PubMedQA         (3-cls, Biomedical literature QA, ~1k) ← hard
  Models    :  microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext  (MedMCQA)
               dmis-lab/biobert-v1.1                                           (MedQA-USMLE)
               allenai/scibert_scivocab_uncased                                (PubMedQA)
  Losses    :  CCE · MAE · GCE(q) · TruncGCE · SCE · SDIV · ForwardT · ForwardThat
  Batteries :  A = clean baseline
               B = uniform label-noise   η ∈ {0, 0.2, 0.4, 0.6, 0.8}
               C = class-dependent noise η ∈ {0.1, 0.2, 0.3, 0.4}
               D = q-sensitivity sweep  → reproduces Figure 2 style plots
  Outputs   :  $ROBUST_NN_WORKSPACE/results_bert/ (or ./results_bert/)  *.csv + *.png

Install   :
  pip install torch transformers datasets scikit-learn
             matplotlib seaborn pandas tqdm

Design notes
  • All sections are self-contained → plug-and-play for future extensions.
  • Edit the CFG block below; nothing else needs changing for most experiments.
  • QUICK_RUN = True  → 5 epochs, 1 seed, 1500 training samples (fast debug).
    QUICK_RUN = False → 15 epochs, 5 seeds, full data (paper-quality results).
"""

# ══════════════════════════════════════════════════════════════════════════════
# TORCH VERSION GUARD — torch ≥ 2.6 required (CVE fix for torch.load)
# ══════════════════════════════════════════════════════════════════════════════
import torch as _torch_check


def _torch_version_tuple():
    return tuple(int(x) for x in _torch_check.__version__.split("+")[0].split(".")[:2])


if _torch_version_tuple() < (2, 6):
    import warnings as _w

    _w.warn(
        f"\n{'=' * 70}\n"
        f"  WARNING: torch {_torch_check.__version__} detected — torch >= 2.6 is REQUIRED.\n"
        f"  A critical vulnerability in torch.load was fixed in 2.6.\n"
        f"  transformers / HuggingFace Hub model loading will FAIL.\n\n"
        f"  Fix:  pip install --upgrade 'torch>=2.6.0'\n"
        f"{'=' * 70}\n",
        stacklevel=1,
    )
del _torch_check, _torch_version_tuple

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONFIG  ← edit here; nothing else needs touching for most runs
# ══════════════════════════════════════════════════════════════════════════════

QUICK_RUN = os.environ.get("ROBUST_NN_QUICK_RUN", "1") == "1"  # False → full paper runs (slower)
SEEDS = [42] if QUICK_RUN else [42, 0, 1, 2, 3]
N_EPOCHS = int(os.environ.get("ROBUST_NN_EPOCHS", "5" if QUICK_RUN else "15"))
MAX_TRAIN = int(os.environ.get("ROBUST_NN_MAX_TRAIN", "1500" if QUICK_RUN else "0")) or None
NOISE_RATES = [0.0, 0.2, 0.4, 0.6] if QUICK_RUN else [0.0, 0.2, 0.4, 0.6, 0.8]
Q_SWEEP = [0.0, 0.4, 0.8, 1.0]  # Figure 2 q-values
BATCH_SIZE = int(os.environ.get("ROBUST_NN_BATCH_SIZE", "32"))
MAX_LEN = 128
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

# ── Dataset registry  (add new datasets here without touching any other code) ─
DATASETS = {
    "MedMCQA": {
        # ~234 k Indian PG medical entrance questions (4-option MC) —
        # proxy for diagnostic image-description reasoning in clinical NLP.
        # Wang et al. https://arxiv.org/abs/2203.14371
        "hf_name": "medmcqa",
        "hf_config": None,
        "text_field": "question",
        "label_field": "cop",  # correct option index 0-3 (already int)
        "label_map": None,
        "num_classes": 4,
        "class_names": ["option_A", "option_B", "option_C", "option_D"],
        "model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "domain": "Clinical NLP / Medical QA",
        "splits": {"train": "train", "val": "validation", "test": "test"},
    },
    "MedQA-USMLE": {
        # USMLE Step 1/2/3 4-option exam questions
        # Jin et al. https://arxiv.org/abs/2009.13081
        "hf_name": "GBaker/MedQA-USMLE-4-options",
        "hf_config": None,
        "text_field": "question",
        "label_field": "answer_idx",  # string 'A'/'B'/'C'/'D'
        "label_map": {"A": 0, "B": 1, "C": 2, "D": 3},
        "num_classes": 4,
        "class_names": ["A", "B", "C", "D"],
        "model": "dmis-lab/biobert-v1.1",
        "domain": "Clinical NLP / USMLE",
        "splits": {"train": "train", "val": None, "test": "test"},
    },
    "PubMedQA": {
        # Hard; analogous to CIFAR-100 in Table 1; self-split from train
        "hf_name": "qiaojin/PubMedQA",
        "hf_config": "pqa_labeled",
        "text_field": "question",
        "label_field": "final_decision",
        "label_map": {"yes": 0, "no": 1, "maybe": 2},
        "num_classes": 3,
        "class_names": ["yes", "no", "maybe"],
        "model": "allenai/scibert_scivocab_uncased",
        "domain": "Medical",
        "splits": {"train": "train", "val": None, "test": None},
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import os
import random
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

if "COLAB_RELEASE_TAG" not in os.environ:
    matplotlib.use("Agg")  # headless backend for scripts / RunPod
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset as _hf_load
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


def _results_dir(default_subdir: str) -> Path:
    """All CSV/PNG outputs go under ROBUST_NN_WORKSPACE / ROBUST_NN_RESULTS_SUBDIR (or default_subdir)."""
    root = Path(os.environ.get("ROBUST_NN_WORKSPACE", os.getcwd())).resolve()
    sub = os.environ.get("ROBUST_NN_RESULTS_SUBDIR", default_subdir)
    p = root / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


RESULTS_DIR = _results_dir("results_bert")


def maybe_hf_login() -> None:
    """Uses HF_TOKEN or HUGGING_FACE_HUB_TOKEN from the environment only — never hardcode tokens."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        print("[HF] Authenticated via environment token (HF_TOKEN).")
    except Exception as exc:
        print(f"[HF] Optional Hugging Face login skipped: {exc}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[CONFIG] device={DEVICE}  quick={QUICK_RUN}  epochs={N_EPOCHS}  batch={BATCH_SIZE}  seeds={SEEDS}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _now() -> str:
    return time.strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ROBUST LOSS FUNCTIONS  (PyTorch; add new losses here)
# ══════════════════════════════════════════════════════════════════════════════


class CCELoss(nn.Module):
    """Standard Categorical Cross-Entropy."""

    def __init__(self, **_):
        super().__init__()
        self._ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._ce(logits, targets)


class MAELoss(nn.Module):
    """Mean Absolute Error on one-hot softmax outputs."""

    def __init__(self, num_classes: int, **_):
        super().__init__()
        self.C = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        y_oh = F.one_hot(targets, self.C).float()
        return (1.0 - (y_oh * probs).sum(dim=1)).mean()


class GCELoss(nn.Module):
    """Generalised Cross-Entropy  L_q(y, p) = (1 − p_y^q) / q.
    q=0.0 → CCE;  q=1.0 → MAE."""

    def __init__(self, q: float = 0.7, **_):
        super().__init__()
        self.q = q

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        p_y = probs[torch.arange(len(targets)), targets]
        if abs(self.q) < 1e-8:
            return -torch.log(p_y).mean()  # q→0 limit = CCE
        return ((1.0 - p_y**self.q) / self.q).mean()


class TruncGCELoss(nn.Module):
    """Truncated GCE: use the GCE loss only on samples where p_y < k.
    Ignores confidently-correct predictions — reduces impact of noisy labels."""

    def __init__(self, q: float = 0.7, k: float = 0.5, **_):
        super().__init__()
        self.q = q
        self.k = k

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        p_y = probs[torch.arange(len(targets)), targets]
        loss = (1.0 - p_y**self.q) / self.q
        mask = (p_y < self.k).float()
        denom = mask.sum().clamp(min=1.0)
        return (loss * mask).sum() / denom


class SCELoss(nn.Module):
    """Symmetric Cross-Entropy  α·CE + β·RCE  (Wang et al., 2019)."""

    def __init__(self, alpha: float = 0.1, beta: float = 1.0, num_classes: int = 2, **_):
        super().__init__()
        self.a = alpha
        self.b = beta
        self.C = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-7)
        p_y = probs[torch.arange(len(targets)), targets]
        ce = -torch.log(p_y).mean()
        y_oh = F.one_hot(targets, self.C).float().clamp(min=1e-4)
        rce = -(probs * torch.log(y_oh)).sum(dim=1).mean()
        return self.a * ce + self.b * rce


class SDIVLoss(nn.Module):
    """S-Divergence loss — directly ported from part2_rSDNet_Transformer_Experiments.py.

    Parameters (A = 1 + λ(1−β),  B = β − λ(1−β)):
      L = Σ_k p_k^(β+1) / A  −  (1+β)/(A·B) · p_y^B

    β=0.05, λ=-0.8 → paper default for rSDNet.  λ=0 → standard DPD.

    Constraint: A > 0 AND B > 0  (required for theoretical robustness guarantees).
    """

    def __init__(self, beta: float = 0.05, lam: float = -0.8, trim_ratio: float = 0.0, **_):
        super().__init__()
        A = 1.0 + lam * (1.0 - beta)
        B = beta - lam * (1.0 - beta)
        if A <= 0 or B <= 0:
            raise ValueError(
                f"SDIVLoss constraint violated: A={A:.4f}, B={B:.4f} must both be > 0. "
                f"(β={beta}, λ={lam})  "
                f"Theoretical robustness guarantees require A>0 and B>0."
            )
        self.beta = beta
        self.lam = lam
        self.trim = trim_ratio

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        A = 1.0 + self.lam * (1.0 - self.beta)
        B = self.beta - self.lam * (1.0 - self.beta)
        p_y = probs[torch.arange(len(targets)), targets]
        loss = probs.pow(self.beta + 1.0).sum(dim=1) / A - (1.0 + self.beta) / (A * B) * p_y.pow(B)
        if self.trim > 0.0:
            k = max(1, int((1.0 - self.trim) * len(loss)))
            loss = loss.sort().values[:k]
        return loss.mean()


class FCLoss(nn.Module):
    """Fractional Cross-Entropy Loss (rSDNet companion loss).

    L(y, p) = (−log p_y)^(1−μ) / Γ(2−μ)  +  2·(1 − p_y)
    where Γ is the Gamma function.

    μ ∈ [0, 1):
      μ → 0  recovers shifted CCE  (CCE term dominates)
      μ → 1  approaches MAE  (bounded gradient, robust floor)

    Introduced in the rSDNet codebase alongside SDIV.  The fractional power
    of CCE softens sensitivity to large individual losses, while the
    2·(1−p_y) term provides the MAE-style robustness floor.

    Reference: rSDNet.ipynb, FCL class (TensorFlow → ported to PyTorch).
    """

    def __init__(self, mu: float = 0.5, **_):
        super().__init__()
        if not (0.0 <= mu < 1.0):
            raise ValueError(f"FCLoss: mu must be in [0, 1), got {mu}")
        self.mu = mu
        import math as _math

        self._gamma_denom = _math.gamma(2.0 - mu)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        p_y = probs[torch.arange(len(targets)), targets]
        cce_frac = (-torch.log(p_y)).pow(1.0 - self.mu) / self._gamma_denom
        mae_term = 2.0 * (1.0 - p_y)
        return (cce_frac + mae_term).mean()


class DPDLoss(nn.Module):
    """Density Power Divergence loss (TPDD-CCE in rSDNet notation).

    L(y, p) = Σ_k p_k^(β+1)  −  (1 + 1/β) · p_y^β

    Equivalent to SDIV with λ=0.  Untrimmed version is the paper default
    (rSDNet uses trim_ratio=0).
    Reference: Fujisawa & Eguchi (2008); rSDNet TDPDSCCE class.
    """

    def __init__(self, beta: float = 0.05, **_):
        super().__init__()
        if beta <= 0:
            raise ValueError(f"DPDLoss: beta must be > 0, got {beta}")
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        p_y = probs[torch.arange(len(targets)), targets]
        loss = probs.pow(self.beta + 1.0).sum(dim=1) - (1.0 + 1.0 / self.beta) * p_y.pow(self.beta)
        return loss.mean()


class TSCCELoss(nn.Module):
    """Trimmed Sparse CCE: sort per-sample CCE losses, drop the top `trim_ratio`
    highest-loss samples (likely noisy), average the rest.

    Reference: rSDNet TSCCE class.
    """

    def __init__(self, trim_ratio: float = 0.2, **_):
        super().__init__()
        self.trim = trim_ratio

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per = F.cross_entropy(logits, targets, reduction="none")
        k = max(1, int((1.0 - self.trim) * len(per)))
        return per.topk(k, largest=False).values.mean()


class ForwardCorrectionLoss(nn.Module):
    """Label-correction loss via the noise transition matrix T.
    T[i, j] = P(observed label = j | true label = i).
    (Patrini et al., 2017 — "Making Deep Neural Networks Robust to Label Noise")"""

    def __init__(self, T: np.ndarray, **_):
        super().__init__()
        self._T = torch.tensor(T, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        T = self._T.to(logits.device)
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        corrected = torch.mm(probs, T.t()).clamp(min=1e-9)
        p_y = corrected[torch.arange(len(targets)), targets]
        return -torch.log(p_y).mean()


# ── Transition matrix factories ──────────────────────────────────────────────


def make_T_uniform(num_classes: int, noise_rate: float) -> np.ndarray:
    """Oracle T for symmetric uniform noise."""
    T = np.full((num_classes, num_classes), noise_rate / max(num_classes - 1, 1))
    np.fill_diagonal(T, 1.0 - noise_rate)
    return T


def make_T_classdep(num_classes: int, noise_rate: float) -> np.ndarray:
    """Oracle T for class-dependent (cyclic) noise: c → (c+1) % C."""
    T = np.eye(num_classes) * (1.0 - noise_rate)
    for i in range(num_classes):
        T[i, (i + 1) % num_classes] += noise_rate
    return T


def estimate_T_hat(model: nn.Module, loader: DataLoader, num_classes: int) -> np.ndarray:
    """Estimate T̂ via Patrini et al. (2017):
    For each class c, take the sample with highest P(c|x) — its full
    softmax row becomes row c of T̂."""
    model.eval()
    best_prob = np.zeros(num_classes)
    T_hat = np.eye(num_classes)
    with torch.no_grad():
        for batch, _ in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            probs = F.softmax(model(**batch).logits, dim=1).cpu().numpy()
            for c in range(num_classes):
                idx = int(probs[:, c].argmax())
                if probs[idx, c] > best_prob[c]:
                    best_prob[c] = probs[idx, c]
                    T_hat[c] = probs[idx]
    row_sums = T_hat.sum(axis=1, keepdims=True).clip(min=1e-9)
    return T_hat / row_sums


# ── Loss registry ─────────────────────────────────────────────────────────────


def make_loss_registry(
    num_classes: int, T_oracle: np.ndarray | None = None, q: float = 0.7, beta: float = 0.05, lam: float = -0.8
) -> dict[str, nn.Module]:
    """Returns {loss_name: loss_module}.  Add new losses here."""
    reg = {
        "CCE": CCELoss(),
        "MAE": MAELoss(num_classes=num_classes),
        f"GCE(q={q})": GCELoss(q=q),
        "TruncGCE": TruncGCELoss(q=q, k=0.5),
        "SCE": SCELoss(alpha=0.1, beta=1.0, num_classes=num_classes),
        "TPDD-CCE": DPDLoss(beta=beta),  # DPD, untrimmed (paper default trim=0)
        "SDIV": SDIVLoss(beta=beta, lam=lam, trim_ratio=0.0),  # rSDNet default (β=0.05, λ=-0.8)
        "TSCCE": TSCCELoss(trim_ratio=0.2),  # Trimmed Sparse CCE (rSDNet)
        "FCL": FCLoss(mu=0.5),  # Fractional Cross-Entropy (rSDNet)
    }
    if T_oracle is not None:
        reg["ForwardT"] = ForwardCorrectionLoss(T=T_oracle)
    return reg


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADING  (HuggingFace → tokenised PyTorch Dataset)
# ══════════════════════════════════════════════════════════════════════════════

_DATA_CACHE: dict[str, tuple] = {}


def load_nlp_dataset(dataset_name: str) -> tuple[list, list, list, list, list, list]:
    """Returns (tr_texts, tr_labels, val_texts, val_labels, te_texts, te_labels).
    Pulls from HuggingFace Hub; caches in memory (and ~/.cache/huggingface/)."""
    if dataset_name in _DATA_CACHE:
        return _DATA_CACHE[dataset_name]

    dcfg = DATASETS[dataset_name]
    t0 = time.time()
    print(f"\n  Fetching '{dataset_name}' from HuggingFace ({dcfg['hf_name']}) …  [{_now()}]")

    _load_args = [dcfg["hf_name"]]
    if dcfg["hf_config"]:
        _load_args.append(dcfg["hf_config"])
    raw = _hf_load(*_load_args)

    def _extract(split_key: str | None) -> tuple[list, list]:
        if split_key is None or split_key not in raw:
            return [], []
        split = raw[split_key]
        texts = list(split[dcfg["text_field"]])
        labels_raw = list(split[dcfg["label_field"]])
        lmap = dcfg["label_map"]
        if lmap is not None:
            pairs = [(t, lmap[l]) for t, l in zip(texts, labels_raw) if l in lmap]
            if not pairs:
                return [], []
            texts, labels = zip(*pairs)
            return list(texts), list(labels)
        return texts, [int(l) for l in labels_raw]

    tr_texts, tr_labels = _extract(dcfg["splits"]["train"])
    val_key = dcfg["splits"].get("val")
    te_key = dcfg["splits"].get("test")

    te_texts, te_labels = _extract(te_key)
    val_texts, val_labels = _extract(val_key)

    # PubMedQA and datasets with no pre-split validation/test: carve from train
    if not te_texts and tr_texts:
        tr_texts, te_texts, tr_labels, te_labels = train_test_split(
            tr_texts, tr_labels, test_size=0.15, random_state=42, stratify=tr_labels
        )
    if not val_texts and tr_texts:
        tr_texts, val_texts, tr_labels, val_labels = train_test_split(
            tr_texts, tr_labels, test_size=0.15, random_state=42, stratify=tr_labels
        )

    # Quick-run: subsample training set
    if MAX_TRAIN and len(tr_texts) > MAX_TRAIN:
        idx = np.random.RandomState(42).choice(len(tr_texts), MAX_TRAIN, replace=False)
        tr_texts = [tr_texts[i] for i in idx]
        tr_labels = [tr_labels[i] for i in idx]

    elapsed = time.time() - t0
    print(
        f"    {dataset_name}: train={len(tr_texts)}  "
        f"val={len(val_texts)}  test={len(te_texts)}  "
        f"({elapsed:.1f}s)  [{_now()}]"
    )

    result = (tr_texts, tr_labels, val_texts, val_labels, te_texts, te_labels)
    _DATA_CACHE[dataset_name] = result
    return result


class TokenisedDataset(Dataset):
    """Pre-tokenises the full list of texts once; avoids per-batch overhead."""

    def __init__(self, texts: list, labels: list, tokenizer, max_len: int = MAX_LEN):
        enc = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        self.inp = enc
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return {k: v[i] for k, v in self.inp.items()}, self.labels[i]


def _build_loaders(tr_texts, tr_labels, val_texts, val_labels, te_texts, te_labels, tokenizer):
    pin = torch.cuda.is_available()
    tr_ds = TokenisedDataset(tr_texts, tr_labels, tokenizer)
    val_ds = TokenisedDataset(val_texts, val_labels, tokenizer)
    te_ds = TokenisedDataset(te_texts, te_labels, tokenizer)
    tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
    val_ldr = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    return tr_ldr, val_ldr, te_ldr


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NOISE INJECTION
# ══════════════════════════════════════════════════════════════════════════════


def inject_uniform_noise(labels: list, noise_rate: float, num_classes: int, seed: int = 42) -> list:
    """Symmetric uniform noise: flip each label to a random different class
    with probability η.  Returns a new list (original unchanged)."""
    if noise_rate <= 0.0:
        return list(labels)
    rng = np.random.RandomState(seed)
    noisy = list(labels)
    mask = rng.rand(len(labels)) < noise_rate
    for i in np.where(mask)[0]:
        choices = [c for c in range(num_classes) if c != labels[i]]
        noisy[i] = int(rng.choice(choices))
    print(f"    [Uniform noise η={noise_rate:.1f}]  {mask.sum()}/{len(labels)} labels flipped")
    return noisy


def inject_classdep_noise(labels: list, noise_rate: float, num_classes: int, seed: int = 42) -> list:
    """Asymmetric (cyclic) noise: class c → (c+1) % C with probability η."""
    if noise_rate <= 0.0:
        return list(labels)
    rng = np.random.RandomState(seed)
    noisy = list(labels)
    mask = rng.rand(len(labels)) < noise_rate
    for i in np.where(mask)[0]:
        noisy[i] = (labels[i] + 1) % num_classes
    print(f"    [Class-dep noise η={noise_rate:.1f}]  {mask.sum()}/{len(labels)} labels flipped")
    return noisy


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING HARNESS
# ══════════════════════════════════════════════════════════════════════════════


def _train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total = 0.0
    for batch, labels in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(**batch).logits
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(len(loader), 1)


def _evaluate(model, loader):
    model.eval()
    preds_all, labels_all, loss_total = [], [], 0.0
    cce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch, labels in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            loss_total += cce(logits, labels.to(DEVICE)).item()
            preds_all.extend(logits.argmax(dim=-1).cpu().tolist())
            labels_all.extend(labels.tolist())
    acc = accuracy_score(labels_all, preds_all)
    loss = loss_total / max(len(loader), 1)
    return acc, loss, preds_all, labels_all


def train_and_evaluate(
    dataset_name: str,
    loss_name: str,
    loss_fn: nn.Module,
    tr_texts: list,
    tr_labels: list,
    val_texts: list,
    val_labels: list,
    te_texts: list,
    te_labels: list,
    n_epochs: int = N_EPOCHS,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Fine-tunes a pretrained BERT-family model with any given loss function.

    Returns a result dict containing:
        best_val_acc, best_test_acc, history (epoch-by-epoch), confusion matrix.
    """
    set_seed(seed)
    dcfg = DATASETS[dataset_name]
    model_name = dcfg["model"]
    C = dcfg["num_classes"]

    t0 = time.time()
    if verbose:
        print(f"  [{dataset_name}][{loss_name}][seed={seed}] start {_now()}  model={model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=C, ignore_mismatched_sizes=True
    ).to(DEVICE)

    tr_ldr, val_ldr, te_ldr = _build_loaders(tr_texts, tr_labels, val_texts, val_labels, te_texts, te_labels, tokenizer)

    total_steps = len(tr_ldr) * n_epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(DEVICE)

    history = {k: [] for k in ("epoch", "train_loss", "val_acc", "val_loss", "test_acc", "test_loss")}
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_preds = []

    pbar = tqdm(range(1, n_epochs + 1), desc=f"  {dataset_name}/{loss_name}", unit="ep", leave=False)
    for epoch in pbar:
        ep_t0 = time.time()
        tr_loss = _train_epoch(model, tr_ldr, optimizer, scheduler, loss_fn)
        val_acc, val_loss, _, _ = _evaluate(model, val_ldr)
        te_acc, te_loss, te_preds, _ = _evaluate(model, te_ldr)

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        history["test_acc"].append(te_acc)
        history["test_loss"].append(te_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = te_acc
            best_preds = te_preds

        ep_sec = time.time() - ep_t0
        pbar.set_postfix(tr=f"{tr_loss:.3f}", val=f"{val_acc:.3f}", te=f"{te_acc:.3f}", spe=f"{ep_sec:.1f}s")

    elapsed = time.time() - t0
    if verbose:
        print(
            f"  [{dataset_name}][{loss_name}][seed={seed}] "
            f"best_val={best_val_acc:.4f}  best_test={best_test_acc:.4f}  "
            f"total={elapsed:.0f}s  end {_now()}"
        )

    cm = confusion_matrix(te_labels, best_preds) if best_preds else None

    return {
        "dataset": dataset_name,
        "loss": loss_name,
        "model": model_name,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "history": history,
        "confusion": cm,
        "seed": seed,
        "elapsed_s": elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════


def _savefig(name: str) -> None:
    path = RESULTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


# ── Figure 2 reproduction ─────────────────────────────────────────────────────


def plot_figure2_style(
    histories: dict,
    dataset_name: str,
    q_values: list = Q_SWEEP,
    noise_rates: list = None,
    fname_prefix: str = "figure2",
) -> None:
    """Reproduces the 2×3 Figure 2 layout from the paper:
      Top row   : test accuracy vs epochs  (3 cols = 3 noise rates)
      Bottom row: validation loss vs epochs (log-scale)
    Each curve = one q value; colours match the original paper."""
    if noise_rates is None:
        noise_rates = [0.0, 0.2, 0.6]

    COLOURS = {0.0: "#FF8C00", 0.4: "#2CA02C", 0.8: "#D62728", 1.0: "#1F77B4"}
    n_cols = len(noise_rates)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))

    for col, eta in enumerate(noise_rates):
        ax_acc = axes[0, col]
        ax_loss = axes[1, col]

        for q in q_values:
            colour = COLOURS.get(q, "#9467BD")
            key = (q, eta)
            if key not in histories:
                continue
            hist = histories[key]
            epochs = hist["epoch"]
            ax_acc.plot(epochs, hist["test_acc"], color=colour, linewidth=1.5, label=f"q = {q}")
            ax_loss.semilogy(epochs, hist["val_loss"], color=colour, linewidth=1.5, label=f"q = {q}")

        # Top row formatting
        ax_acc.set_title(f"noise rate = {eta}", fontsize=11)
        ax_acc.set_xlabel("number of epochs")
        ax_acc.set_ylabel("test accuracy" if col == 0 else "")
        ax_acc.legend(fontsize=8, loc="lower right")
        ax_acc.grid(alpha=0.3)
        ax_acc.annotate(
            f"({chr(ord('a') + col)})", xy=(0.05, 0.05), xycoords="axes fraction", fontsize=11, fontweight="bold"
        )

        # Bottom row formatting
        ax_loss.set_title(f"noise rate = {eta}", fontsize=11)
        ax_loss.set_xlabel("number of epochs")
        ax_loss.set_ylabel("validation loss" if col == 0 else "")
        ax_loss.legend(fontsize=8, loc="upper right")
        ax_loss.grid(alpha=0.3, which="both")
        ax_loss.annotate(
            f"({chr(ord('d') + col)})", xy=(0.05, 0.93), xycoords="axes fraction", fontsize=11, fontweight="bold"
        )

    plt.suptitle(
        f"[{dataset_name}]  Test accuracy & validation loss vs epochs "
        r"for $\mathcal{L}_q$ loss at different values of $q$",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    _savefig(f"{fname_prefix}_{dataset_name.replace(' ', '_')}.png")


# ── Table 1 reproduction ──────────────────────────────────────────────────────


def make_summary_table(rows: list) -> pd.DataFrame:
    """Aggregates list of result dicts → mean ± std accuracy across seeds."""
    df = pd.DataFrame(rows)
    grp = df.groupby(["dataset", "loss", "noise_type", "noise_rate"])["best_test_acc"]
    mean = grp.mean().rename("mean")
    std = grp.std(ddof=0).fillna(0.0).rename("std")
    out = pd.concat([mean, std], axis=1).reset_index()
    out["acc_str"] = out.apply(lambda r: f"{r['mean'] * 100:.2f} ± {r['std'] * 100:.2f}", axis=1)
    return out


def plot_table1_style(summary: pd.DataFrame, fname: str = "table1_style") -> None:
    """Renders a formatted accuracy table as a Matplotlib figure.
    Green shading marks the best 2 accuracies per noise-rate column."""
    for nt in summary["noise_type"].unique():
        sub = summary[summary["noise_type"] == nt]
        if sub.empty:
            continue

        acc_pivot = sub.pivot_table(index=["dataset", "loss"], columns="noise_rate", values="acc_str", aggfunc="first")

        mean_pivot = sub.pivot_table(index=["dataset", "loss"], columns="noise_rate", values="mean", aggfunc="mean")

        # Two-best colouring per column
        n_rows, n_cols = mean_pivot.shape
        cell_colours = [["#FFFFFF"] * n_cols for _ in range(n_rows)]
        for j in range(n_cols):
            col_vals = mean_pivot.values[:, j].astype(float)
            valid = ~np.isnan(col_vals)
            if valid.sum() < 1:
                continue
            ranked = np.argsort(col_vals[valid])[::-1]
            real_idx = np.where(valid)[0]
            if len(ranked) > 0:
                cell_colours[real_idx[ranked[0]]][j] = "#7FBF7F"  # best
            if len(ranked) > 1:
                cell_colours[real_idx[ranked[1]]][j] = "#C8EBC8"  # 2nd best

        row_labels = [f"{d}\n{l}" for d, l in acc_pivot.index]
        col_labels = [f"η={v:.1f}" for v in acc_pivot.columns]

        fig_h = max(5, n_rows * 0.55 + 1.5)
        fig_w = max(10, n_cols * 2.2 + 3)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=acc_pivot.values,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellcolors=cell_colours,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.3, 1.6)

        ax.set_title(
            f"Average Test Accuracy ± Std (%)  [{nt} Noise]\n"
            f"Best 2 per column highlighted  "
            f"(reproduced from Table 1, {len(SEEDS)} seed(s))",
            fontsize=11,
            pad=12,
            loc="center",
        )

        plt.tight_layout()
        _savefig(f"{fname}_{nt.lower()}.png")


def plot_table2_style(summary: pd.DataFrame, fname: str = "table2_style") -> None:
    """Table 2 equivalent: high-noise comparison across all methods and datasets."""
    high_noise = summary[summary["noise_rate"] >= 0.4].copy()
    if high_noise.empty:
        return

    pivot = high_noise.pivot_table(index=["dataset", "loss"], columns="noise_rate", values="acc_str", aggfunc="first")

    mean_pivot = high_noise.pivot_table(index=["dataset", "loss"], columns="noise_rate", values="mean", aggfunc="mean")

    n_rows, n_cols = mean_pivot.shape
    cell_colours = [["#FFFFFF"] * n_cols for _ in range(n_rows)]
    for j in range(n_cols):
        col = mean_pivot.values[:, j].astype(float)
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        ranked = np.argsort(col[valid])[::-1]
        idx = np.where(valid)[0]
        cell_colours[idx[ranked[0]]][j] = "#7FBF7F"
        cell_colours[idx[ranked[1]]][j] = "#C8EBC8"

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 2 + 3), max(5, n_rows * 0.55 + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=pivot.values,
        rowLabels=[f"{d}/{l}" for d, l in pivot.index],
        colLabels=[f"η={v:.1f}" for v in pivot.columns],
        cellcolors=cell_colours,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.3, 1.6)
    ax.set_title("Table 2 — High-Noise Regime (η ≥ 0.4) Average Accuracy (%)", fontsize=11, pad=12)
    plt.tight_layout()
    _savefig(f"{fname}.png")


# ── 2-D noise robustness curves ───────────────────────────────────────────────


def plot_noise_robustness_curves(summary: pd.DataFrame, noise_type: str = "Uniform") -> None:
    """2-D accuracy vs noise-rate per loss (one panel per dataset)."""
    sub = summary[summary["noise_type"] == noise_type]
    datasets = sub["dataset"].unique()
    if not len(datasets):
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for loss_name, grp in sub[sub["dataset"] == ds].groupby("loss"):
            g = grp.sort_values("noise_rate")
            ax.plot(g["noise_rate"] * 100, g["mean"] * 100, marker="o", linewidth=1.8, label=loss_name)
        ax.set_title(f"{ds}\n({DATASETS[ds]['domain']})", fontsize=10)
        ax.set_xlabel("Noise Rate η (%)")
        ax.set_ylabel("Test Accuracy (%)" if ax is axes[0] else "")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Robustness under {noise_type} Label Noise", fontsize=12)
    plt.tight_layout()
    _savefig(f"noise_robustness_{noise_type.lower()}.png")


# ── Confusion matrices ────────────────────────────────────────────────────────


def plot_confusion_matrices(rows: list, noise_rate: float = 0.0) -> None:
    """Grid of confusion matrices at a given noise rate (seed=42 only)."""
    subset = [
        r
        for r in rows
        if abs(r.get("noise_rate", 0.0)) < 1e-6 and r.get("seed") == 42 and r.get("confusion") is not None
    ]
    if not subset:
        return

    cols = min(4, len(subset))
    rows_n = (len(subset) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.5 * cols, 4.5 * rows_n))
    axes_flat = list(np.array(axes).flatten()) if rows_n * cols > 1 else [axes]

    for ax, r in zip(axes_flat, subset):
        cnames = DATASETS[r["dataset"]]["class_names"]
        sns.heatmap(
            r["confusion"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=cnames,
            yticklabels=cnames,
            ax=ax,
            cbar=False,
            annot_kws={"size": 7},
        )
        ax.set_title(f"[{r['dataset']}]\n{r['loss']}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)

    for ax in axes_flat[len(subset) :]:
        ax.axis("off")

    plt.suptitle("Confusion Matrices — Clean Condition (η = 0.0)", fontsize=12)
    plt.tight_layout()
    _savefig("confusion_matrices_clean.png")


# ── Training curves ───────────────────────────────────────────────────────────


def plot_training_curves(rows: list, dataset_name: str) -> None:
    """Loss + accuracy curves for all loss functions on one dataset (seed=42)."""
    subset = [
        r for r in rows if r["dataset"] == dataset_name and r.get("seed") == 42 and abs(r.get("noise_rate", 0.0)) < 1e-6
    ]
    if not subset:
        return

    n = len(subset)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, r in enumerate(subset):
        h = r["history"]
        axes[0, j].plot(h["epoch"], h["train_loss"], label="train loss")
        axes[0, j].set_title(r["loss"], fontsize=9)
        axes[0, j].set_xlabel("epoch")
        if j == 0:
            axes[0, j].set_ylabel("loss")
        axes[0, j].grid(alpha=0.3)

        axes[1, j].plot(h["epoch"], [v * 100 for v in h["val_acc"]], label="val acc", color="orange")
        axes[1, j].plot(h["epoch"], [v * 100 for v in h["test_acc"]], label="test acc", color="steelblue")
        axes[1, j].set_xlabel("epoch")
        if j == 0:
            axes[1, j].set_ylabel("accuracy (%)")
        axes[1, j].legend(fontsize=7)
        axes[1, j].grid(alpha=0.3)

    plt.suptitle(f"Training Curves — {dataset_name} (clean)", fontsize=12)
    plt.tight_layout()
    _savefig(f"training_curves_{dataset_name.replace(' ', '_')}.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BATTERY A: CLEAN BASELINE
# ══════════════════════════════════════════════════════════════════════════════


def battery_A_clean(datasets: list, n_epochs: int = N_EPOCHS) -> list:
    """All losses × all datasets with clean labels.  No noise injection."""
    print("\n" + "═" * 70)
    print("BATTERY A — Clean Baseline")
    print("═" * 70)
    t0 = time.time()
    rows = []

    for ds_name in datasets:
        dcfg = DATASETS[ds_name]
        C = dcfg["num_classes"]
        tr_t, tr_l, val_t, val_l, te_t, te_l = load_nlp_dataset(ds_name)
        losses = make_loss_registry(num_classes=C)

        for seed in SEEDS:
            for loss_name, loss_fn in losses.items():
                res = train_and_evaluate(
                    ds_name, loss_name, loss_fn, tr_t, tr_l, val_t, val_l, te_t, te_l, n_epochs=n_epochs, seed=seed
                )
                rows.append({**res, "noise_type": "Clean", "noise_rate": 0.0})

    plot_confusion_matrices(rows, noise_rate=0.0)
    for ds in datasets:
        plot_training_curves(rows, ds)

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "confusion")} for r in rows])
    df.to_csv(RESULTS_DIR / "battery_A_clean.csv", index=False)
    print(f"\n  Battery A done  {len(rows)} runs  total={time.time() - t0:.0f}s  [{_now()}]")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — BATTERY B: UNIFORM NOISE SWEEP
# ══════════════════════════════════════════════════════════════════════════════


def battery_B_uniform_noise(datasets: list, n_epochs: int = N_EPOCHS) -> list:
    """All losses × all datasets × noise rates under symmetric uniform noise."""
    print("\n" + "═" * 70)
    print("BATTERY B — Uniform Label-Noise Sweep")
    print("═" * 70)
    t0 = time.time()
    rows = []

    for ds_name in datasets:
        dcfg = DATASETS[ds_name]
        C = dcfg["num_classes"]
        tr_t, tr_l, val_t, val_l, te_t, te_l = load_nlp_dataset(ds_name)
        tokenizer = AutoTokenizer.from_pretrained(dcfg["model"])

        for eta in NOISE_RATES:
            T_oracle = make_T_uniform(C, eta) if eta > 0 else None
            losses = make_loss_registry(num_classes=C, T_oracle=T_oracle)

            # Estimate T̂ once per (dataset, noise_rate) via a short CCE warmup
            T_hat_loss = None
            if eta > 0:
                noisy_for_warm = inject_uniform_noise(tr_l, eta, C, seed=42)
                warm_ds = TokenisedDataset(tr_t, noisy_for_warm, tokenizer)
                warm_ldr = DataLoader(warm_ds, batch_size=BATCH_SIZE, shuffle=True)
                warm_mdl = AutoModelForSequenceClassification.from_pretrained(
                    dcfg["model"], num_labels=C, ignore_mismatched_sizes=True
                ).to(DEVICE)
                opt = torch.optim.AdamW(warm_mdl.parameters(), lr=LR)
                cce = nn.CrossEntropyLoss()
                for _ in range(min(2, n_epochs)):
                    for b, lbl in warm_ldr:
                        b = {k: v.to(DEVICE) for k, v in b.items()}
                        lss = cce(warm_mdl(**b).logits, lbl.to(DEVICE))
                        lss.backward()
                        opt.step()
                        opt.zero_grad()
                T_hat = estimate_T_hat(warm_mdl, warm_ldr, C)
                T_hat_loss = ForwardCorrectionLoss(T=T_hat)
                losses["ForwardThat"] = T_hat_loss
                del warm_mdl
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for seed in SEEDS:
                noisy_labels = inject_uniform_noise(tr_l, eta, C, seed=seed)
                for loss_name, loss_fn in losses.items():
                    res = train_and_evaluate(
                        ds_name,
                        loss_name,
                        loss_fn,
                        tr_t,
                        noisy_labels,
                        val_t,
                        val_l,
                        te_t,
                        te_l,
                        n_epochs=n_epochs,
                        seed=seed,
                    )
                    rows.append({**res, "noise_type": "Uniform", "noise_rate": eta})

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "confusion")} for r in rows])
    df.to_csv(RESULTS_DIR / "battery_B_uniform.csv", index=False)
    print(f"\n  Battery B done  {len(rows)} runs  total={time.time() - t0:.0f}s  [{_now()}]")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — BATTERY C: CLASS-DEPENDENT (ASYMMETRIC) NOISE
# ══════════════════════════════════════════════════════════════════════════════


def battery_C_classdep_noise(datasets: list, n_epochs: int = N_EPOCHS) -> list:
    """All losses × all datasets × noise rates under asymmetric cyclic noise."""
    print("\n" + "═" * 70)
    print("BATTERY C — Class-Dependent (Asymmetric) Noise")
    print("═" * 70)
    t0 = time.time()
    rows = []
    cd_rates = [0.1, 0.2, 0.3, 0.4]  # follows Table 1 column headers

    for ds_name in datasets:
        dcfg = DATASETS[ds_name]
        C = dcfg["num_classes"]
        tr_t, tr_l, val_t, val_l, te_t, te_l = load_nlp_dataset(ds_name)

        for eta in cd_rates:
            T_oracle = make_T_classdep(C, eta)
            losses = make_loss_registry(num_classes=C, T_oracle=T_oracle)

            for seed in SEEDS:
                noisy_labels = inject_classdep_noise(tr_l, eta, C, seed=seed)
                for loss_name, loss_fn in losses.items():
                    res = train_and_evaluate(
                        ds_name,
                        loss_name,
                        loss_fn,
                        tr_t,
                        noisy_labels,
                        val_t,
                        val_l,
                        te_t,
                        te_l,
                        n_epochs=n_epochs,
                        seed=seed,
                    )
                    rows.append({**res, "noise_type": "ClassDep", "noise_rate": eta})

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "confusion")} for r in rows])
    df.to_csv(RESULTS_DIR / "battery_C_classdep.csv", index=False)
    print(f"\n  Battery C done  {len(rows)} runs  total={time.time() - t0:.0f}s  [{_now()}]")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — BATTERY D: q-SENSITIVITY SWEEP  (reproduces Figure 2)
# ══════════════════════════════════════════════════════════════════════════════


def battery_D_q_sweep(datasets: list, n_epochs: int = N_EPOCHS) -> list:
    """GCE with q ∈ {0.0, 0.4, 0.8, 1.0} × noise rates {0.0, 0.2, 0.6}.
    Saves epoch-by-epoch history and generates the Figure 2 style plots."""
    print("\n" + "═" * 70)
    print("BATTERY D — q-Sensitivity Sweep (Figure 2 Reproduction)")
    print("═" * 70)
    t0 = time.time()
    rows = []
    fig2_noise = [0.0, 0.2, 0.6]

    for ds_name in datasets:
        dcfg = DATASETS[ds_name]
        C = dcfg["num_classes"]
        tr_t, tr_l, val_t, val_l, te_t, te_l = load_nlp_dataset(ds_name)

        histories: dict = {}

        for eta in fig2_noise:
            noisy_labels = inject_uniform_noise(tr_l, eta, C, seed=42) if eta > 0 else list(tr_l)

            for q in Q_SWEEP:
                loss_name = f"GCE(q={q})"
                loss_fn = GCELoss(q=q)
                res = train_and_evaluate(
                    ds_name,
                    loss_name,
                    loss_fn,
                    tr_t,
                    noisy_labels,
                    val_t,
                    val_l,
                    te_t,
                    te_l,
                    n_epochs=n_epochs,
                    seed=42,
                    verbose=True,
                )
                histories[(q, eta)] = res["history"]
                rows.append({**res, "noise_type": "Uniform", "noise_rate": eta, "q": q})

        plot_figure2_style(histories, ds_name, q_values=Q_SWEEP, noise_rates=fig2_noise, fname_prefix="figure2")

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "confusion")} for r in rows])
    df.to_csv(RESULTS_DIR / "battery_D_q_sweep.csv", index=False)
    print(f"\n  Battery D done  {len(rows)} runs  total={time.time() - t0:.0f}s  [{_now()}]")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    maybe_hf_login()
    wall_start = time.time()
    print("=" * 70)
    print("Part 3 — BERT-Style Robust NLP Classification")
    print(f"  Datasets : {list(DATASETS.keys())}")
    print(f"  Device   : {DEVICE}")
    print(f"  QUICK_RUN: {QUICK_RUN}  (epochs={N_EPOCHS}, max_train={MAX_TRAIN}, seeds={SEEDS})")
    print(f"  Start    : {_now()}")
    print("=" * 70)

    all_datasets = list(DATASETS.keys())
    all_rows: list = []

    rows_A = battery_A_clean(all_datasets)
    all_rows.extend(rows_A)

    rows_B = battery_B_uniform_noise(all_datasets)
    all_rows.extend(rows_B)

    rows_C = battery_C_classdep_noise(all_datasets)
    all_rows.extend(rows_C)

    rows_D = battery_D_q_sweep(all_datasets)
    all_rows.extend(rows_D)

    # Aggregate and plot summary tables
    structured = [
        {**r, "noise_type": r.get("noise_type", "Clean"), "noise_rate": r.get("noise_rate", 0.0)}
        for r in all_rows
        if "best_test_acc" in r
    ]
    summary = make_summary_table(structured)
    summary.to_csv(RESULTS_DIR / "summary_all.csv", index=False)

    plot_table1_style(summary, fname="table1_style")
    plot_table2_style(summary, fname="table2_style")
    plot_noise_robustness_curves(summary, noise_type="Uniform")
    plot_noise_robustness_curves(summary, noise_type="ClassDep")

    total = time.time() - wall_start
    print("\n" + "═" * 70)
    print(f"All done!  Total wall time = {total / 60:.1f} min   [{_now()}]")
    print(f"Results in  {RESULTS_DIR}/")
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f"  {f.name}")
    print("═" * 70)


if __name__ == "__main__":
    main()
