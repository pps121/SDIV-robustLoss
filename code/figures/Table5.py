"""
Part 4 — Zero-Shot Robust Loss Evaluation on Medical Vision
============================================================
Research goal: evaluate robustness of loss functions under label noise, using
fixed pretrained vision-language models (no model fine-tuning).

What this script does:
1) Compute zero-shot logits once per (dataset, model) on clean test images.
2) Inject synthetic label noise into the *evaluation labels*.
3) Compare robust losses (CCE, MAE, GCE, TruncGCE, SCE, SDIV, ForwardT, ForwardThat)
   and accuracy degradation under noisy labels.
4) Produce publication-ready CSV + plots (noise curves, q-sweep, confusion).

Datasets:
- PathMNIST (9 classes)
- DermaMNIST (7 classes)

Models (non-generative, CLIP-style):
- CLIP:       openai/clip-vit-base-patch32
- PLIP:       vinid/plip
- MedSigLIP:  google/medsiglip-448 (gated)
- BiomedCLIP: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (open_clip)

Run mode:
- QUICK_RUN=True -> fast debug
- QUICK_RUN=False -> fuller sweep

Notes:
- This is intentionally NOT generative VLM evaluation.
- This is also intentionally NOT training/fine-tuning; model signal is fixed and
  robustness is studied at the metric/loss level under label corruption.
"""

from __future__ import annotations

import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
if "COLAB_RELEASE_TAG" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from medmnist.info import INFO
import medmnist
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Runtime guards
# -----------------------------------------------------------------------------


def _version_tuple(v: str) -> Tuple[int, int]:
    parts = v.split("+")[0].split(".")
    return int(parts[0]), int(parts[1])


if _version_tuple(torch.__version__) < (2, 6):
    warnings.warn(
        "torch>=2.6.0 is recommended (security + HF compatibility).",
        stacklevel=1,
    )


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


QUICK_RUN = os.environ.get("ROBUST_NN_QUICK_RUN", "1") == "1"
SEED = int(os.environ.get("ROBUST_NN_SEED", "42"))
MAX_SAMPLES = int(os.environ.get("ROBUST_NN_MAX_SAMPLES", "1200" if QUICK_RUN else "8000"))
BATCH_SIZE = int(os.environ.get("ROBUST_NN_BATCH_SIZE", "16"))

# Same battery concept as Part 3.
NOISE_RATES_UNIFORM = [0.0, 0.2, 0.4, 0.6] if QUICK_RUN else [0.0, 0.2, 0.4, 0.6, 0.8]
NOISE_RATES_CLASSDEP = [0.1, 0.2, 0.3, 0.4]
Q_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

MODEL_KEYS_DEFAULT = ["MedSigLIP", "CLIP", "PLIP", "BiomedCLIP"] if not QUICK_RUN else ["MedSigLIP", "CLIP"]
MODEL_ONLY = os.environ.get("ROBUST_NN_MODEL_ONLY", "").strip()
MODEL_KEYS = [MODEL_ONLY] if MODEL_ONLY else MODEL_KEYS_DEFAULT

DATASET_KEYS = ["PathMNIST", "DermaMNIST"]


def _results_dir(default_subdir: str) -> Path:
    root = Path(os.environ.get("ROBUST_NN_WORKSPACE", os.getcwd())).resolve()
    sub = os.environ.get("ROBUST_NN_RESULTS_SUBDIR", default_subdir)
    p = root / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


RESULTS_DIR = _results_dir("results_multimodal_vision")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(
    f"[CONFIG] device={DEVICE} quick={QUICK_RUN} batch={BATCH_SIZE} "
    f"max_samples={MAX_SAMPLES} models={MODEL_KEYS}"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_hf_login() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        print("[HF] Authenticated via environment token.")
    except Exception as exc:
        print(f"[HF] Login skipped: {exc}")


# -----------------------------------------------------------------------------
# Robust losses (metric evaluation on fixed logits)
# -----------------------------------------------------------------------------


class CCELoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets)


class MAELoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.c = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        y = F.one_hot(targets, self.c).float()
        return (1.0 - (y * probs).sum(dim=1)).mean()


class GCELoss(nn.Module):
    def __init__(self, q: float):
        super().__init__()
        self.q = q

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        if abs(self.q) < 1e-10:
            return -torch.log(py).mean()
        return ((1.0 - py.pow(self.q)) / self.q).mean()


class TruncGCELoss(nn.Module):
    def __init__(self, q: float = 0.7, k: float = 0.5):
        super().__init__()
        self.q = q
        self.k = k

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        g = (1.0 - py.pow(self.q)) / self.q
        m = (py < self.k).float()
        return (g * m).sum() / m.sum().clamp(min=1.0)


class SCELoss(nn.Module):
    def __init__(self, num_classes: int, alpha: float = 0.1, beta: float = 1.0):
        super().__init__()
        self.c = num_classes
        self.a = alpha
        self.b = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-7)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        ce = -torch.log(py).mean()
        y = F.one_hot(targets, self.c).float().clamp(min=1e-4)
        rce = -(probs * torch.log(y)).sum(dim=1).mean()
        return self.a * ce + self.b * rce


class SDIVLoss(nn.Module):
    def __init__(self, beta: float = 0.05, lam: float = -0.8):
        super().__init__()
        self.beta = beta
        self.lam = lam

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        a = 1.0 + self.lam * (1.0 - self.beta)
        b = self.beta - self.lam * (1.0 - self.beta)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        loss = probs.pow(self.beta + 1.0).sum(dim=1) / a - (1.0 + self.beta) / (a * b) * py.pow(b)
        return loss.mean()


class ForwardCorrectionLoss(nn.Module):
    def __init__(self, T: np.ndarray):
        super().__init__()
        self.register_buffer("T", torch.tensor(T, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        corrected = probs @ self.T.t().to(logits.device)
        py = corrected[torch.arange(len(targets), device=logits.device), targets].clamp(min=1e-9)
        return -torch.log(py).mean()


# -----------------------------------------------------------------------------
# Noise helpers
# -----------------------------------------------------------------------------


def make_T_uniform(num_classes: int, eta: float) -> np.ndarray:
    T = np.full((num_classes, num_classes), eta / max(1, num_classes - 1), dtype=np.float32)
    np.fill_diagonal(T, 1.0 - eta)
    return T


def make_T_classdep(num_classes: int, eta: float) -> np.ndarray:
    T = np.eye(num_classes, dtype=np.float32) * (1.0 - eta)
    for i in range(num_classes):
        T[i, (i + 1) % num_classes] += eta
    return T


def inject_uniform_noise(labels: np.ndarray, eta: float, num_classes: int, seed: int) -> np.ndarray:
    if eta <= 0:
        return labels.copy()
    rng = np.random.RandomState(seed)
    y = labels.copy()
    mask = rng.rand(len(y)) < eta
    idx = np.where(mask)[0]
    for i in idx:
        choices = [c for c in range(num_classes) if c != y[i]]
        y[i] = int(rng.choice(choices))
    return y


def inject_classdep_noise(labels: np.ndarray, eta: float, num_classes: int, seed: int) -> np.ndarray:
    if eta <= 0:
        return labels.copy()
    rng = np.random.RandomState(seed)
    y = labels.copy()
    mask = rng.rand(len(y)) < eta
    idx = np.where(mask)[0]
    y[idx] = (y[idx] + 1) % num_classes
    return y


def estimate_T_hat(logits: np.ndarray, num_classes: int) -> np.ndarray:
    probs = torch.tensor(logits).softmax(dim=1).cpu().numpy()
    best = np.zeros(num_classes, dtype=np.float32)
    T_hat = np.eye(num_classes, dtype=np.float32)
    for c in range(num_classes):
        i = int(probs[:, c].argmax())
        if probs[i, c] > best[c]:
            best[c] = probs[i, c]
            T_hat[c] = probs[i]
    T_hat = T_hat / T_hat.sum(axis=1, keepdims=True).clip(min=1e-9)
    return T_hat


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


@dataclass
class VisionDatasetPack:
    key: str
    images: List[Image.Image]
    labels: np.ndarray
    class_names: List[str]
    prompt_tpl: str


class PILListDataset(Dataset):
    def __init__(self, images: Sequence[Image.Image]):
        self.images = list(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i: int):
        return self.images[i]


def _labels_from_info(key_lower: str) -> List[str]:
    labels = INFO[key_lower]["label"]
    return [labels[str(i)] for i in range(len(labels))]


def _load_medmnist(ds_key: str) -> VisionDatasetPack:
    cfg = {
        "PathMNIST": ("PathMNIST", "a histopathology image showing {}"),
        "DermaMNIST": ("DermaMNIST", "a dermatoscopy image showing {}"),
    }
    klass_name, prompt_tpl = cfg[ds_key]
    ctor = getattr(medmnist, klass_name)
    test_ds = ctor(split="test", download=True)

    images: List[Image.Image] = []
    labels: List[int] = []
    for i in range(len(test_ds)):
        img, y = test_ds[i]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        images.append(img.convert("RGB"))
        labels.append(int(np.asarray(y).squeeze()))

    if MAX_SAMPLES and len(images) > MAX_SAMPLES:
        idx = np.random.RandomState(SEED).choice(len(images), MAX_SAMPLES, replace=False)
        images = [images[i] for i in idx]
        labels = [labels[i] for i in idx]

    key_lower = klass_name.lower()
    class_names = _labels_from_info(key_lower)
    return VisionDatasetPack(
        key=ds_key,
        images=images,
        labels=np.asarray(labels, dtype=np.int64),
        class_names=class_names,
        prompt_tpl=prompt_tpl,
    )


# -----------------------------------------------------------------------------
# Model wrappers (non-generative)
# -----------------------------------------------------------------------------


class BaseZeroShotModel:
    def infer_logits(self, images: List[Image.Image], class_prompts: List[str]) -> np.ndarray:
        raise NotImplementedError


class ClipLikeHFModel(BaseZeroShotModel):
    def __init__(self, model_id: str):
        self.model_id = model_id
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    def infer_logits(self, images: List[Image.Image], class_prompts: List[str]) -> np.ndarray:
        out_logits: List[np.ndarray] = []
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc=f"  infer {self.model_id}", leave=False):
            batch = images[i : i + BATCH_SIZE]
            enc = self.processor(text=class_prompts, images=batch, padding=True, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.model(**enc).logits_per_image
            out_logits.append(logits.detach().float().cpu().numpy())
        return np.concatenate(out_logits, axis=0)


class MedSigLIPModel(BaseZeroShotModel):
    def __init__(self, model_id: str = "google/medsiglip-448"):
        self.model_id = model_id
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(DEVICE)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()

    def infer_logits(self, images: List[Image.Image], class_prompts: List[str]) -> np.ndarray:
        out_logits: List[np.ndarray] = []
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc=f"  infer {self.model_id}", leave=False):
            batch = images[i : i + BATCH_SIZE]
            enc = self.processor(text=class_prompts, images=batch, padding="max_length", return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.model(**enc).logits_per_image
            out_logits.append(logits.detach().float().cpu().numpy())
        return np.concatenate(out_logits, axis=0)


class BiomedCLIPModel(BaseZeroShotModel):
    def __init__(self, model_id: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        self.model_id = model_id
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "open_clip_torch is required for BiomedCLIP. Install: pip install open_clip_torch"
            ) from exc

        self.open_clip = open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            f"hf-hub:{model_id}"
        )
        self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_id}")
        self.model = self.model.to(DEVICE)
        self.model.eval()

    def infer_logits(self, images: List[Image.Image], class_prompts: List[str]) -> np.ndarray:
        # Tokenize text once for efficiency.
        txt = self.tokenizer(class_prompts, context_length=256).to(DEVICE)
        with torch.inference_mode():
            txt_feat = self.model.encode_text(txt)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        out = []
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc=f"  infer {self.model_id}", leave=False):
            batch = images[i : i + BATCH_SIZE]
            px = torch.stack([self.preprocess(im) for im in batch]).to(DEVICE)
            with torch.inference_mode():
                img_feat = self.model.encode_image(px)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                scale = self.model.logit_scale.exp()
                logits = scale * img_feat @ txt_feat.t()
            out.append(logits.detach().float().cpu().numpy())
        return np.concatenate(out, axis=0)


def build_model(model_key: str) -> BaseZeroShotModel:
    if model_key == "CLIP":
        return ClipLikeHFModel("openai/clip-vit-base-patch32")
    if model_key == "PLIP":
        return ClipLikeHFModel("vinid/plip")
    if model_key == "MedSigLIP":
        return MedSigLIPModel("google/medsiglip-448")
    if model_key == "BiomedCLIP":
        return BiomedCLIPModel("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    raise ValueError(f"Unknown model key: {model_key}")


# -----------------------------------------------------------------------------
# Batteries and evaluation
# -----------------------------------------------------------------------------


def _loss_registry(num_classes: int, T_oracle: Optional[np.ndarray], T_hat: Optional[np.ndarray], q: float = 0.7):
    reg = {
        "CCE": CCELoss(),
        "MAE": MAELoss(num_classes),
        f"GCE(q={q})": GCELoss(q),
        "TruncGCE": TruncGCELoss(q=q, k=0.5),
        "SCE": SCELoss(num_classes=num_classes),
        "SDIV": SDIVLoss(beta=0.05, lam=-0.8),
    }
    if T_oracle is not None:
        reg["ForwardT"] = ForwardCorrectionLoss(T_oracle)
    if T_hat is not None:
        reg["ForwardThat"] = ForwardCorrectionLoss(T_hat)
    return reg


def _eval_losses(logits_np: np.ndarray, y_eval_np: np.ndarray, reg: Dict[str, nn.Module]) -> Dict[str, float]:
    logits = torch.tensor(logits_np, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_eval_np, dtype=torch.long, device=DEVICE)
    out: Dict[str, float] = {}
    for name, loss_fn in reg.items():
        loss_fn = loss_fn.to(DEVICE)
        with torch.inference_mode():
            out[name] = float(loss_fn(logits, y).item())
    return out


def _battery_rows(
    dataset_key: str,
    model_key: str,
    logits_np: np.ndarray,
    y_clean: np.ndarray,
    noise_type: str,
    noise_rates: Sequence[float],
    num_classes: int,
) -> List[dict]:
    rows: List[dict] = []
    preds = logits_np.argmax(axis=1)
    clean_acc = accuracy_score(y_clean, preds)

    for eta in noise_rates:
        if noise_type == "Clean":
            y_eval = y_clean.copy()
            T_oracle = None
        elif noise_type == "Uniform":
            y_eval = inject_uniform_noise(y_clean, eta, num_classes, seed=SEED)
            T_oracle = make_T_uniform(num_classes, eta)
        elif noise_type == "ClassDep":
            y_eval = inject_classdep_noise(y_clean, eta, num_classes, seed=SEED)
            T_oracle = make_T_classdep(num_classes, eta)
        else:
            raise ValueError(noise_type)

        T_hat = estimate_T_hat(logits_np, num_classes) if eta > 0 else None
        losses = _loss_registry(num_classes, T_oracle=T_oracle, T_hat=T_hat, q=0.7)
        loss_vals = _eval_losses(logits_np, y_eval, losses)
        noisy_acc = accuracy_score(y_eval, preds)

        for loss_name, loss_val in loss_vals.items():
            rows.append(
                {
                    "dataset": dataset_key,
                    "model_key": model_key,
                    "battery": f"B-{noise_type}" if noise_type != "Clean" else "A-Clean",
                    "noise_type": noise_type,
                    "noise_rate": float(eta),
                    "loss": loss_name,
                    "loss_value": float(loss_val),
                    "acc_clean_labels": float(clean_acc),
                    "acc_noisy_labels": float(noisy_acc),
                }
            )
    return rows


def battery_d_q_sweep(dataset_key: str, model_key: str, logits_np: np.ndarray, y_clean: np.ndarray, num_classes: int) -> List[dict]:
    rows: List[dict] = []
    for eta in [0.0, 0.2, 0.6]:
        y_eval = inject_uniform_noise(y_clean, eta, num_classes, seed=SEED) if eta > 0 else y_clean.copy()
        logits = torch.tensor(logits_np, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y_eval, dtype=torch.long, device=DEVICE)
        for q in Q_SWEEP:
            with torch.inference_mode():
                v = float(GCELoss(q=q).to(DEVICE)(logits, y).item())
            rows.append(
                {
                    "dataset": dataset_key,
                    "model_key": model_key,
                    "battery": "D-q-sweep",
                    "noise_type": "Uniform",
                    "noise_rate": float(eta),
                    "loss": f"GCE(q={q})",
                    "q": float(q),
                    "loss_value": v,
                }
            )
    return rows


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def _savefig(name: str) -> None:
    p = RESULTS_DIR / name
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {p}")


def plot_noise_curves(df: pd.DataFrame, noise_type: str) -> None:
    sub = df[(df["noise_type"] == noise_type) & (~df["battery"].eq("D-q-sweep"))]
    if sub.empty:
        return

    combos = sub[["dataset", "model_key"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(combos), figsize=(6 * len(combos), 4.8), squeeze=False)
    axes = axes[0]

    for ax, (ds, mk) in zip(axes, combos):
        part = sub[(sub["dataset"] == ds) & (sub["model_key"] == mk)]
        for loss_name, g in part.groupby("loss"):
            g = g.sort_values("noise_rate")
            ax.plot(g["noise_rate"] * 100, g["loss_value"], marker="o", lw=1.7, label=loss_name)
        ax.set_title(f"{ds} | {mk}")
        ax.set_xlabel("Noise rate eta (%)")
        ax.set_ylabel("Loss value")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    plt.suptitle(f"Loss-vs-noise curves ({noise_type})", fontsize=12)
    plt.tight_layout()
    _savefig(f"noise_curves_{noise_type.lower()}.png")


def plot_q_sweep(df: pd.DataFrame) -> None:
    sub = df[df["battery"] == "D-q-sweep"]
    if sub.empty:
        return

    for (ds, mk), g in sub.groupby(["dataset", "model_key"]):
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for eta, h in g.groupby("noise_rate"):
            h = h.sort_values("q")
            ax.plot(h["q"], h["loss_value"], marker="o", lw=2, label=f"eta={eta}")
        ax.set_title(f"GCE q-sweep | {ds} | {mk}")
        ax.set_xlabel("q")
        ax.set_ylabel("GCE loss")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        _savefig(f"q_sweep_{ds}_{mk}.png".replace(" ", "_"))


def plot_confusion(dataset_key: str, model_key: str, logits_np: np.ndarray, y_clean: np.ndarray, class_names: List[str]) -> None:
    preds = logits_np.argmax(axis=1)
    cm = confusion_matrix(y_clean, preds)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(cm, cmap="Blues", annot=False, cbar=True, ax=ax)
    ax.set_title(f"Confusion | clean labels | {dataset_key} | {model_key}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    ax.set_xticklabels([str(i) for i in range(len(class_names))], rotation=0)
    ax.set_yticklabels([str(i) for i in range(len(class_names))], rotation=0)
    plt.tight_layout()
    _savefig(f"confusion_{dataset_key}_{model_key}.png".replace(" ", "_"))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    set_seed(SEED)
    maybe_hf_login()

    t0 = time.time()
    rows: List[dict] = []

    # Cache logits to avoid recomputation during batteries.
    logits_cache: Dict[Tuple[str, str], np.ndarray] = {}
    ds_cache: Dict[str, VisionDatasetPack] = {}

    for ds_key in DATASET_KEYS:
        print(f"\n[DATA] loading {ds_key}")
        pack = _load_medmnist(ds_key)
        ds_cache[ds_key] = pack
        prompts = [pack.prompt_tpl.format(c) for c in pack.class_names]

        for mk in MODEL_KEYS:
            print(f"\n[MODEL] {mk} on {ds_key}")
            model = build_model(mk)
            logits_np = model.infer_logits(pack.images, prompts)
            logits_cache[(ds_key, mk)] = logits_np

            # Battery A
            rows.extend(
                _battery_rows(
                    dataset_key=ds_key,
                    model_key=mk,
                    logits_np=logits_np,
                    y_clean=pack.labels,
                    noise_type="Clean",
                    noise_rates=[0.0],
                    num_classes=len(pack.class_names),
                )
            )

            # Battery B
            rows.extend(
                _battery_rows(
                    dataset_key=ds_key,
                    model_key=mk,
                    logits_np=logits_np,
                    y_clean=pack.labels,
                    noise_type="Uniform",
                    noise_rates=NOISE_RATES_UNIFORM,
                    num_classes=len(pack.class_names),
                )
            )

            # Battery C
            rows.extend(
                _battery_rows(
                    dataset_key=ds_key,
                    model_key=mk,
                    logits_np=logits_np,
                    y_clean=pack.labels,
                    noise_type="ClassDep",
                    noise_rates=NOISE_RATES_CLASSDEP,
                    num_classes=len(pack.class_names),
                )
            )

            # Battery D
            rows.extend(
                battery_d_q_sweep(
                    dataset_key=ds_key,
                    model_key=mk,
                    logits_np=logits_np,
                    y_clean=pack.labels,
                    num_classes=len(pack.class_names),
                )
            )

            # Confusion plot (clean labels)
            plot_confusion(ds_key, mk, logits_np, pack.labels, pack.class_names)

            # Free model memory before next model.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary_all.csv", index=False)

    # Aggregate table for reporting.
    agg = (
        df[df["battery"] != "D-q-sweep"]
        .groupby(["dataset", "model_key", "loss", "noise_type", "noise_rate"], as_index=False)
        .agg(loss_mean=("loss_value", "mean"), acc_clean=("acc_clean_labels", "mean"), acc_noisy=("acc_noisy_labels", "mean"))
    )
    agg.to_csv(RESULTS_DIR / "summary_grouped.csv", index=False)

    plot_noise_curves(df, "Uniform")
    plot_noise_curves(df, "ClassDep")
    plot_q_sweep(df)

    print("\n" + "=" * 78)
    print(f"Done in {(time.time() - t0)/60:.1f} min")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
