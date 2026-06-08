"""
robust_losses.py
================
Modular, framework-agnostic implementation of all robust loss functions
studied in the rSDNet / rRNet research line.

Each loss is a PyTorch nn.Module that accepts:
    logits  : Tensor[B, C]  — raw (pre-softmax) logits
    targets : Tensor[B]     — integer class labels in [0, C)
and returns a scalar loss.

Usage
-----
from robust_losses import SDIVLoss, GCELoss, make_loss_registry

# Drop-in replacement for nn.CrossEntropyLoss
criterion = SDIVLoss(beta=0.05, lam=-0.8)
loss = criterion(logits, labels)

# Full registry for benchmarking
registry = make_loss_registry(num_classes=10)
for name, loss_fn in registry.items():
    val = loss_fn(logits, labels)

References
----------
[1] Jana & Ghosh (2026). "rSDNet: Unified Robust Neural Learning under Label
    Noise and Adversarial Attack." arXiv:2603.17628.
[2] Ghosh & Jana (2026). "Provably robust learning of regression neural
    networks using β-divergences." arXiv:2602.08933.
[3] Ghosh, Harris, Maji, Basu, Pardo (2017). "A generalized divergence for
    statistical inference." Bernoulli 23(4A), 2746-2783.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class _RobustBase(nn.Module):
    """Base class: exposes .name and .scale_info for diagnostic tables."""
    name: str = "base"
    scale_info: str = "[0, ∞)"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Standard Cross-Entropy (baseline)
# ──────────────────────────────────────────────────────────────────────────────

class CCELoss(_RobustBase):
    """Standard Categorical Cross-Entropy (= KL divergence minimisation = MLE).

    L(y, p) = −log p_y

    This is the baseline. It is:
    - Fisher consistent ✓
    - Classification-calibrated ✓
    - Robust to label noise ✗ (unbounded gradient)
    - Robust to adversarial attacks ✗ (unbounded influence function)
    """
    name = "CCE"
    scale_info = "[0, +∞)"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Mean Absolute Error
# ──────────────────────────────────────────────────────────────────────────────

class MAELoss(_RobustBase):
    """Mean Absolute Error on softmax probabilities: 1 − p_y.

    L(y, p) = 1 − p_y

    Properties:
    - Bounded gradient: gradient is always in [0, 1] ✓
    - Theoretically has 50% breakdown point ✓
    - Classification-calibrated ✓
    - Practically collapses on hard datasets under label noise ✗
      (insufficient gradient signal for uncertain samples)
    """
    name = "MAE"
    scale_info = "[0, 1]"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        return (1.0 - py).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Generalised Cross-Entropy
# ──────────────────────────────────────────────────────────────────────────────

class GCELoss(_RobustBase):
    """Generalised Cross-Entropy: (1 − p_y^q) / q.

    Interpolates between CCE (q→0) and MAE (q=1).
    Zhang & Sabuncu (2018), NeurIPS.

    Args:
        q: Robustness parameter. q ∈ (0, 1]. Typical: q=0.7.
    """

    def __init__(self, q: float = 0.7):
        super().__init__()
        if not (0 <= q <= 1):
            raise ValueError(f"GCELoss: q must be in [0, 1], got {q}")
        self.q = q
        self.name = f"GCE(q={q})"
        self.scale_info = f"[0, {1/q if q > 0 else '+∞'}]"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        if abs(self.q) < 1e-9:
            return -torch.log(py).mean()
        return ((1.0 - py.pow(self.q)) / self.q).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Truncated GCE
# ──────────────────────────────────────────────────────────────────────────────

class TruncGCELoss(_RobustBase):
    """Truncated GCE: apply GCE only to samples where p_y < k.

    Ignores confidently-correct predictions (p_y ≥ k), reducing
    the contribution of clean easy samples and hard-noisy samples.

    Args:
        q: GCE parameter (see GCELoss).
        k: Confidence threshold. Default: 0.5.
    """

    def __init__(self, q: float = 0.7, k: float = 0.5):
        super().__init__()
        self.q = q
        self.k = k
        self.name = f"TruncGCE(q={q},k={k})"
        self.scale_info = f"[0, {1/q if q > 0 else '+∞'}]"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        loss = (1.0 - py.pow(self.q)) / (self.q + 1e-10)
        mask = (py < self.k).float()
        denom = mask.sum().clamp(min=1.0)
        return (loss * mask).sum() / denom


# ──────────────────────────────────────────────────────────────────────────────
# 5. Symmetric Cross-Entropy
# ──────────────────────────────────────────────────────────────────────────────

class SCELoss(_RobustBase):
    """Symmetric Cross-Entropy: α·CCE + β·RCE.

    Wang et al. (2019), ICCV. "Symmetric cross entropy for robust learning
    with noisy labels."

    RCE = −Σ_k p_k · log(y_k) — reverse KL (from labels to predictions).

    Args:
        alpha: Weight on CCE term. Default: 0.1.
        beta:  Weight on RCE term. Default: 1.0.
        num_classes: Number of classes C.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 1.0, num_classes: int = 10):
        super().__init__()
        self.a = alpha
        self.b = beta
        self.C = num_classes
        self.name = f"SCE(α={alpha},β={beta})"
        self.scale_info = f"≈{alpha}·CCE + {beta}·RCE"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-7)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        cce = -torch.log(py).mean()
        y_oh = F.one_hot(targets, self.C).float().clamp(min=1e-4)
        rce = -(probs * torch.log(y_oh)).sum(dim=1).mean()
        return self.a * cce + self.b * rce


# ──────────────────────────────────────────────────────────────────────────────
# 6. Density Power Divergence (DPD / TPDD-CCE)
# ──────────────────────────────────────────────────────────────────────────────

class DPDLoss(_RobustBase):
    """Density Power Divergence loss (β-divergence, λ=0 special case of S-DIV).

    L(y, p) = Σ_k p_k^{β+1}  −  (1 + 1/β) · p_y^β

    Equivalent to SDIVLoss with lam=0. This is the classification analogue
    of the β-divergence used in rRNet for regression.

    Args:
        beta: Robustness parameter β > 0. Default: 0.05.
    """
    name = "DPD"
    scale_info = "(-∞, +∞)"

    def __init__(self, beta: float = 0.05):
        super().__init__()
        if beta <= 0:
            raise ValueError(f"DPDLoss: beta must be > 0, got {beta}")
        self.beta = beta
        self.name = f"DPD(β={beta})"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        return (probs.pow(self.beta + 1).sum(dim=1)
                - (1.0 + 1.0 / self.beta) * py.pow(self.beta)).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 7. S-Divergence Loss (rSDNet) — THE KEY CONTRIBUTION
# ──────────────────────────────────────────────────────────────────────────────

class SDIVLoss(_RobustBase):
    """S-Divergence loss — the core loss of rSDNet (Jana & Ghosh 2026).

    Full 2-parameter S-divergence for classification:

        L(y, p) = (1/A) Σ_k p_k^{β+1}  −  ((1+β)/(A·B)) · p_y^B

    where:
        A = 1 + λ(1−β)   (must be > 0)
        B = β − λ(1−β)   (must be > 0)

    Special cases:
        β=0, λ=−1  →  Cross-Entropy (CCE)
        λ=0         →  DPD / β-divergence

    Theoretical guarantees (β > 0):
        1. Fisher consistent (recovers true p*(x) at population level)
        2. Classification-calibrated (Bayes-optimal predictions)
        3. Robust to uniform label noise (Theorem 3.3 in rSDNet paper)
        4. Bounded influence function under adversarial perturbation

    Robustness mechanism: gradient ∝ p_y^β — model confidence weights
    each sample's gradient. Mislabeled/adversarial samples are automatically
    down-weighted. No explicit outlier detection needed.

    Args:
        beta: Primary robustness parameter. Typical: β ∈ (0.05, 0.3).
              β=0 → CCE (no robustness).
        lam:  Secondary robustness parameter. Typical: λ ∈ (−1, −0.5).
              λ=0 → DPD. λ=−1 (with β=0) → CCE.
              Paper default: β=0.05, λ=−0.8.
    """
    name = "SDIV"
    scale_info = "(-∞, +∞)"

    def __init__(self, beta: float = 0.05, lam: float = -0.8):
        super().__init__()
        A = 1.0 + lam * (1.0 - beta)
        B = beta - lam * (1.0 - beta)
        if A <= 0 or B <= 0:
            raise ValueError(
                f"SDIVLoss constraint violated: A={A:.4f}, B={B:.4f}. "
                f"Both must be > 0 for theoretical robustness guarantees. "
                f"(β={beta}, λ={lam})"
            )
        self.beta = beta
        self.lam = lam
        self.A = A
        self.B = B
        self.name = f"SDIV(β={beta},λ={lam})"
        self.scale_info = f"(-∞,+∞) A={A:.3f} B={B:.3f}"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        loss = (probs.pow(self.beta + 1).sum(dim=1) / self.A
                - (1.0 + self.beta) / (self.A * self.B) * py.pow(self.B))
        return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# 8. Trimmed Sparse CCE
# ──────────────────────────────────────────────────────────────────────────────

class TSCCELoss(_RobustBase):
    """Trimmed Sparse CCE: drop the top trim_ratio highest-loss samples.

    Sort per-sample CCE losses; average only the lowest (1 − trim_ratio)
    fraction. The high-loss samples are presumed to be noisy.

    Args:
        trim_ratio: Fraction of highest-loss samples to discard. Default: 0.2.
    """

    def __init__(self, trim_ratio: float = 0.2):
        super().__init__()
        if not (0.0 <= trim_ratio < 1.0):
            raise ValueError(f"TSCCELoss: trim_ratio must be in [0, 1), got {trim_ratio}")
        self.trim = trim_ratio
        self.name = f"TSCCE(trim={trim_ratio})"
        self.scale_info = "[0, +∞)"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per = F.cross_entropy(logits, targets, reduction="none")
        k = max(1, int((1.0 - self.trim) * len(per)))
        return per.topk(k, largest=False).values.mean()


# ──────────────────────────────────────────────────────────────────────────────
# 9. Fractional Cross-Entropy Loss (FCL)
# ──────────────────────────────────────────────────────────────────────────────

class FCLoss(_RobustBase):
    """Fractional Cross-Entropy Loss (rSDNet companion loss).

    L(y, p) = (−log p_y)^{1−μ} / Γ(2−μ)  +  2·(1 − p_y)

    where Γ is the Gamma function.

    μ ∈ [0, 1):
        μ → 0:  approaches shifted CCE + constant
        μ → 1:  approaches MAE-style (bounded gradient floor)

    The fractional power (1−μ) of CCE softens sensitivity to large individual
    losses, while 2(1−p_y) provides a MAE-style robustness floor.

    Args:
        mu: Fractional parameter μ ∈ [0, 1). Default: 0.5.
    """

    def __init__(self, mu: float = 0.5):
        super().__init__()
        if not (0.0 <= mu < 1.0):
            raise ValueError(f"FCLoss: mu must be in [0, 1), got {mu}")
        self.mu = mu
        self._gamma_denom = math.gamma(2.0 - mu)
        self.name = f"FCL(μ={mu})"
        self.scale_info = "[0, +∞)"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        py = probs[torch.arange(len(targets), device=logits.device), targets]
        cce_frac = (-torch.log(py)).pow(1.0 - self.mu) / self._gamma_denom
        mae_term = 2.0 * (1.0 - py)
        return (cce_frac + mae_term).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 10. Forward Label Correction
# ──────────────────────────────────────────────────────────────────────────────

class ForwardCorrectionLoss(_RobustBase):
    """Label-correction loss via the noise transition matrix T.

    T[i, j] = P(observed label = j | true label = i).

    The predicted probability vector is corrected to p_corrupt = p · Tᵀ,
    and CE is applied to p_corrupt vs. the (possibly noisy) observed label.

    Patrini et al. (2017), CVPR. "Making Deep Neural Networks Robust to
    Label Noise: A Loss Correction Approach."

    Args:
        T: Noise transition matrix, shape (C, C), numpy array.
           T[i, i] = 1 − noise_rate for uniform noise.
           T must be row-stochastic (rows sum to 1).
    """
    name = "ForwardT"
    scale_info = "[0, +∞)"

    def __init__(self, T: np.ndarray):
        super().__init__()
        self.register_buffer("T", torch.tensor(T, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        T = self.T.to(logits.device)
        probs = F.softmax(logits, dim=1).clamp(min=1e-9)
        p_corrupt = (probs @ T.t()).clamp(min=1e-9)
        py = p_corrupt[torch.arange(len(targets), device=logits.device), targets]
        return -torch.log(py).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Transition matrix factories (for noise injection experiments)
# ──────────────────────────────────────────────────────────────────────────────

def make_T_uniform(num_classes: int, noise_rate: float) -> np.ndarray:
    """Oracle T for symmetric uniform noise.

    P(ỹ = j | y = i) = η/(C−1)  for j ≠ i
    P(ỹ = i | y = i) = 1 − η
    """
    T = np.full((num_classes, num_classes), noise_rate / max(num_classes - 1, 1))
    np.fill_diagonal(T, 1.0 - noise_rate)
    return T.astype(np.float32)


def make_T_classdep(num_classes: int, noise_rate: float) -> np.ndarray:
    """Oracle T for cyclic class-dependent noise: class c → (c+1) % C."""
    T = np.eye(num_classes, dtype=np.float32) * (1.0 - noise_rate)
    for i in range(num_classes):
        T[i, (i + 1) % num_classes] += noise_rate
    return T


# ──────────────────────────────────────────────────────────────────────────────
# Loss registry
# ──────────────────────────────────────────────────────────────────────────────

def make_loss_registry(
    num_classes: int,
    T_oracle: Optional[np.ndarray] = None,
    q: float = 0.7,
    beta: float = 0.05,
    lam: float = -0.8,
) -> dict[str, _RobustBase]:
    """Return the full named loss dictionary for benchmarking.

    Args:
        num_classes: Number of target classes C.
        T_oracle:    Optional noise transition matrix for ForwardCorrectionLoss.
        q:           GCE and TruncGCE parameter.
        beta:        SDIV / DPD primary robustness parameter.
        lam:         SDIV secondary robustness parameter.

    Returns:
        Dict mapping loss name (str) → loss module (nn.Module).
    """
    registry: dict[str, _RobustBase] = {
        "CCE":            CCELoss(),
        "MAE":            MAELoss(),
        f"GCE(q={q})":   GCELoss(q),
        "TruncGCE":       TruncGCELoss(q, k=0.5),
        "SCE":            SCELoss(alpha=0.1, beta=1.0, num_classes=num_classes),
        "DPD":            DPDLoss(beta),
        "SDIV":           SDIVLoss(beta, lam),
        "TSCCE":          TSCCELoss(trim_ratio=0.2),
        "FCL":            FCLoss(mu=0.5),
    }
    if T_oracle is not None:
        registry["ForwardT"] = ForwardCorrectionLoss(T_oracle)
    return registry


def print_loss_table(registry: dict[str, _RobustBase]) -> None:
    """Print a diagnostic table of loss names and expected value ranges."""
    print("\n" + "=" * 65)
    print(f"  {'Loss':<22} {'Y-axis / scale'}")
    print("  " + "-" * 61)
    for name, fn in registry.items():
        print(f"  {fn.name:<22} {fn.scale_info}")
    print("=" * 65 + "\n")
