# Robust Neural Classification via S-Divergence (rSDNet)

**Paper**: *No Unique Minimizer, No Problem: On the Consistency of Robust Neural Classifiers*  
**Theory by**: Subho Majumdar (IIM Bangalore) · Anand Deo (IIM Bangalore) · Abhik Ghosh (ISI Calcutta)  
**Experiments by**: Partha P. Saha  
**Branch**: `partha-fresh`

---

## Abstract

Neural classifiers trained by cross-entropy minimisation are highly sensitive to label noise and adversarial contamination. This repository implements **rSDNet** — a unified robust neural training framework based on the **S-divergence family** — and provides reproducible experiments on vision and clinical NLP benchmarks confirming the paper's theoretical claims. The key theoretical result is a consistency theorem requiring no identifiability assumption: empirical S-divergence minimisers converge to the population-optimal equivalence class Θ₀ under mild regularity (Theorem 1), and limit points of the training algorithm are stationary points of the empirical objective (Theorem 2).

---

## Key Results

| Dataset | CCE | SDIV (default λ=−0.8) | SDIV (optimised λ=−0.4) | Best loss |
|---------|-----|----------------------|------------------------|-----------|
| MNIST (clean) | 98.22% | 98.01% | — | FCL 98.46% |
| CIFAR-10 (clean) | 60.56% | 55.06% | — | TPDD-CCE 61.16% |
| PathMNIST (clean) | 83.02% | 82.60% | **84.11%** (β=0.05) | FCL 83.61% |
| DermaMNIST (clean) | 73.22% | 66.88%† | **73.32%** (β=0.10) | CCE 73.22% |
| Emotion NLP | 57.25% | 57.80% | — | GCE 58.20% |
| PubMedQA NLP | 56.00% | 55.33% | — | TruncGCE 58.00% |

†Default λ=−0.8 causes majority-class collapse on imbalanced DermaMNIST; optimised λ=−0.4 resolves this (see §Critical Findings).

**MNIST label-noise robustness** (η=40%): SDIV −1.5 pp vs CCE −3.6 pp.  
**PathMNIST AUNRC**: FCL 0.3321 > SCE 0.3313 > TPDD-CCE 0.3293 > SDIV 0.3279 > CCE 0.3269.

---

## Repository Structure

```
src/
  losses/robust_losses.py         # All 10 loss functions (PyTorch)
  models/vision_transformer.py    # rSDNet-ViT (vanilla Transformer encoder)

experiments/
  vit_medmnist/run_vit.py         # ← CANONICAL experiment runner (vision)
  bert_nlp/run_bert.py            # BERT NLP experiments
  multimodal/run_multimodal.py    # CLIP / MedSigLIP zero-shot
  plotting/generate_plots.py      # Publication-quality figure generation

notebooks/
  vit_exploration.ipynb           # Interactive ViT exploration
  bert_exploration.ipynb          # Interactive BERT exploration

results/
  paper/
    experiment_section.tex        # Full LaTeX experiment section (8 tables)
    tables/                       # tab_setup, tab_clean_*, tab_noise_*, tab_fgsm,
    │                             #   tab_sdiv_surface, tab_nlp
    csvs/                         # Verified result CSVs (ground-truth numbers)
    figures/                      # Publication PNG figures
  visualizations/                 # Interactive HTML dashboards

docs/
  paper/                          # LaTeX sources (authoritative)
  planning/master_plan.md

dataset/                          # pathmnist.npz
archive/                          # Superseded code, early writeups, stage snapshots
```

---

## Architecture (paper-exact, AAA_2026_RobustNN_Paper §4)

| Parameter | Value |
|-----------|-------|
| Embedding dimension d | 64 |
| Attention heads H | 4 |
| FFN hidden dim | 128 |
| Transformer layers L | 4 |
| Dropout | 0.10 |
| Patch size | 8×8 (images resized to 32×32 → 16 patches) |
| Optimizer | Adam (η₀ = 1×10⁻³, β₁=0.9, β₂=0.999) |
| Epochs / batch | 30 / 256 |
| SDIV default (β, λ) | (0.05, −0.80); grid-search optimal (0.05–0.10, −0.40) |

---

## Reproducing Experiments

### Prerequisites

```bash
pip install 'torch>=2.6' torchvision transformers datasets medmnist \
            scikit-learn matplotlib seaborn pandas tqdm
```

### Quick run (paper-exact defaults)

```bash
# Runs all 5 datasets with paper's d=64 config
python experiments/vit_medmnist/run_vit.py

# Medical datasets only (faster):
ROBUST_NN_DATASETS=pathmnist,dermamnist python experiments/vit_medmnist/run_vit.py
```

### Experiment Batteries

| Battery | Description | Paper §4 |
|---------|-------------|----------|
| A | Clean-data performance | Category A |
| B | Uniform label noise η ∈ {0, 0.1, 0.2, 0.3, 0.4} | Category B |
| C | FGSM adversarial ε ∈ {0, 1/255, 2/255, 4/255, 8/255} | Category C |
| C2 | PGD adversarial (10-step, same ε grid); skip via `ROBUST_NN_SKIP_PGD=1` | — |
| D | SDIV (β, λ) surface β∈{0.01,…,0.5} × λ∈{-0.8,-0.5,0.0} | Category D |
| E | Asymmetric (pair-flip) label noise (same η grid) | — |

### Reproduce larger April-2026 exploratory config (d=256)

```bash
ROBUST_NN_D_MODEL=256 ROBUST_NN_HEADS=8 ROBUST_NN_FFN=512 \
ROBUST_NN_LAYERS=6 ROBUST_NN_PATCH=4 ROBUST_NN_LR=3e-4 \
  python experiments/vit_medmnist/run_vit.py
```

---

## Loss Functions

| Name | Class | Reference |
|------|-------|-----------|
| CCE | `CCELoss` | Baseline |
| MAE | `MAELoss` | Ghosh et al. 2017 |
| GCE (q=0.7) | `GCELoss` | Zhang & Sabuncu 2018 |
| TruncGCE | `TruncGCELoss` | Rusiecki 2019 |
| SCE | `SCELoss` | Wang et al. 2019 |
| TPDD-CCE | `DPDLoss` | Basu et al. 1998 |
| TSCCE | `TSCCELoss` | Jana & Ghosh 2026 (λ=0 special case) |
| FCL (μ=0.5) | `FCLoss` | — |
| RKLD | `RKLDLoss` | Reverse KL + uniform regulariser |
| **SDIV (β, λ)** | `SDIVLoss` | **Proposed** — Jana & Ghosh 2026 |
| ForwardT | `ForwardTLoss` | Patrini et al. 2017 (oracle T) |

---

## Critical Findings

### 1. SDIV Majority-Class Collapse (DermaMNIST)
Default λ=−0.8 → A=1+λ(1−β)=0.24, amplifying the sum-term gradient 4.17×.
This causes gradient starvation on minority classes → collapse to majority (66.88%).
**Fix**: λ=−0.4 → A=0.62 → (β=0.10, λ=−0.40) achieves 73.32% = best on DermaMNIST.

### 2. DermaMNIST FGSM "Immunity" is an Artefact
SDIV/MAE/GCE stay flat at 66.88% under all adversarial ε — because they predict a
single class regardless of input. This is **not genuine adversarial robustness**.
Among genuinely discriminative losses, TSCCE (Δ=−30.7 pp) and SCE (Δ=−24.4 pp)
show the best true FGSM resilience vs CCE (Δ=−68.5 pp).

### 3. No Loss Confers FGSM Robustness on PathMNIST
All losses collapse catastrophically at ε=8/255 (CCE: 83.3%→17.6%, SDIV: 80.9%→15.9%).
Adversarial training (Madry et al. 2018) is required for genuine FGSM robustness.

### 4. Consistency Without Identifiability (Theory)
Theorem 1 proves convergence of empirical SDIV minimisers to Θ₀ = {θ : gθ(x) = p₀(x)}
without requiring Θ₀ to be a singleton — filling the key gap in prior rSDNet theory.

---

## References

- Basu et al. (1998). *Biometrika* 85:549–559.
- Ghosh et al. (2017). *Bernoulli* 23(4A):2746–2783.
- Jana & Ghosh (2026). arXiv:2603.17628. [rSDNet]
- Madry et al. (2018). ICLR.
- Patrini et al. (2017). CVPR.
- Zhang & Sabuncu (2018). NeurIPS.
- Wang et al. (2019). ICCV.
