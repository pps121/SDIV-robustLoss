# Master Execution Plan
**Date:** 2026-07-19  
**Task:** (1) Write full experiment section LaTeX, (2) Physically reorganize repository

---

## PART A — Experiment Section (LaTeX)

### Data sources confirmed (real, grounded, no approximations)
| CSV | Path |
|-----|------|
| PathMNIST noise | `plots_results/15April2026/results_15April2026/pathmnist_noise_results.csv` |
| DermaMNIST noise | `plots_results/15April2026/results_15April2026/dermamnist_noise_results.csv` |
| PathMNIST FGSM | `plots_results/15April2026/results_15April2026/pathmnist_fgsm_results.csv` |
| DermaMNIST FGSM | `plots_results/15April2026/results_15April2026/dermamnist_fgsm_results.csv` |
| PathMNIST SDIV surface | `plots_results/15April2026/results_15April2026/pathmnist_sdiv_surface.csv` |
| DermaMNIST SDIV surface | `plots_results/15April2026/results_15April2026/dermamnist_sdiv_surface.csv` |
| MNIST clean | `plots_results/30March2026/MNIST_Clean-data-performance.csv` |
| CIFAR-10 clean | `plots_results/30March2026/Clean-data performance.csv` |
| MNIST noise | `plots_results/30March2026/MNIST_Uniform label-noise.csv` |
| MNIST FGSM | `plots_results/30March2026/FGSM-adversarial-attacks.csv` |
| NLP Emotion | `plots_results/15April2026/results_15April2026/nlp_Emotion_results.csv` |
| NLP PubMedQA | `plots_results/15April2026/results_15April2026/nlp_PubMedQA_results.csv` |

### Key numbers (verified, no fabrication)

**MNIST clean (30 epochs, β=0.05, λ=−0.8, Adam):**  
SDIV 97.94%, CCE 98.19%, TDPDSCCE 98.50%, GCE 98.36%, SCE 98.18%, RKLD 97.43%, TSCCE 86.23%, FCL degenerate (9.8%)

**CIFAR-10 clean (same config):**  
CCE 60.56%, TDPDSCCE 61.16%, FCL 59.06%, SCE 59.05%, SDIV 55.06%, RKLD 55.97%, GCE 56.52%, TSCCE 52.94%

**PathMNIST clean (9-class, 30 epochs):**  
FCL 83.61%, CCE 83.02%, SCE 83.06%, SDIV 82.60%, GCE 82.24%, TSCCE 82.26%, TPDD-CCE 82.10%, TruncGCE 78.20%, MAE 78.50%

**DermaMNIST clean (7-class):**  
CCE 73.22%, FCL 72.57%, TPDD-CCE 72.32%, TSCCE 70.82%, SCE 70.22%, GCE 66.93%, TruncGCE 66.93%, SDIV 66.88%, MAE 66.88%

**MNIST noise robustness (SDIV vs CCE at η=40%):**  
SDIV 96.62% (Δ=−1.32%), CCE 94.68% (Δ=−3.57%), GCE 96.99% (Δ=−1.17%), TDPDSCCE 94.37% (Δ=−4.13%)

**PathMNIST noise robustness (SDIV vs CCE at η=40%):**  
SDIV 81.98% (Δ=−0.62%), CCE 82.41% (Δ=+0.39%), SCE 83.27%, FCL 82.99%
*Observation: robust losses remain within ~1% across 0–40% noise on PathMNIST*

**DermaMNIST FGSM (ε=8/255):**  
MAE **flat at 66.88%** (0% drop), GCE(q=0.7) **flat at 66.88%**, SDIV **flat at 66.88%**  
CCE: 72.27% → 22.69% (catastrophic drop, −68.5%)  
TPDD-CCE: 73.17% → 27.43%, FCL: 72.77% → 29.03%

**PathMNIST FGSM (ε=8/255):**  
SDIV: 80.93% → 15.89% (not immune on PathMNIST)  
CCE: 83.31% → 17.60% (also catastrophic)  
SCE: 80.15% → 11.95%, FCL: 76.88% → 7.12%
*Key contrast: DermaMNIST shows FGSM immunity for SDIV/MAE/GCE; PathMNIST does not*

**SDIV parameter surface — PathMNIST best:**  
(β=0.05, λ=−0.4): **84.11%** → best configuration  
(β=0.1, λ=−0.4): 83.69%, (β=0.02, λ=0.0): 83.68%

**SDIV parameter surface — DermaMNIST best:**  
(β=0.1, λ=−0.4): **73.32%**, (β=0.2, λ=−0.4): 72.87%, (β=0.02, λ=0.0): 72.87%

**NLP (BERT, clean):**  
Emotion: GCE 58.20%, TruncGCE 58.10%, SDIV 57.80%, FCL 57.70%, CCE 57.25%  
PubMedQA: CCE 56.00%, TSCCE 56.00%, TruncGCE 58.00%, GCE 55.33%, SDIV 55.33%

### Section structure (LaTeX)

```
\section{Experiments}
  5.1  Experimental Setup
    - Datasets, Models (ViT-rSDNet, BERT), Losses, Optimizer, Training details
    - Table 1: ViT hyperparameters
    - Table 2: Dataset statistics
  5.2  Category A — Clean-Data Performance
    - Table 3: MNIST / CIFAR-10 clean accuracy (all losses)
    - Table 4: PathMNIST / DermaMNIST clean accuracy
  5.3  Category B — Uniform Label-Noise Robustness
    - Figure ref: A1_noise_pathmnist, A2_noise_dermamnist
    - Table 5: Accuracy at η ∈ {0,10,20,30,40}% for key losses
    - AUNRC (area under noise-robustness curve) metric
  5.4  Category C — Adversarial Robustness (FGSM)
    - Figure ref: B1_fgsm_dermamnist, F3/F4 drop bars
    - Table 6: Accuracy under ε ∈ {0,1/255,2/255,4/255,8/255}
    - Key finding: DermaMNIST FGSM immunity of SDIV/MAE/GCE
  5.5  Category D — rSDNet Parameter Sensitivity (β, λ)
    - Figure ref: C1_sdiv_surface_3d, heatmaps
    - Table 7: PathMNIST/DermaMNIST surface peak (β,λ)
  5.6  NLP Experiments (BERT)
    - Table 8: Emotion and PubMedQA best accuracy
  5.7  Multimodal Zero-Shot (CLIP vs MedSigLIP)
    - Table 9: Multimodal summary
  5.8  Summary Discussion
```

### Output file
`docs/paper/experiment_section.tex`  
`docs/paper/tables/` — one .tex file per table

---

## PART B — Repository Reorganization

### Current state (inventory)
- 328 PNG files across 6+ directories (heavily duplicated)
- 40 notebooks (many are RunPod execution copies with no unique content)
- 18 Python scripts (3 are exact duplicates)
- Result CSVs triplicated across `_stage_20260415_065553/`, `_stage_20260415_104519/`, `_stage_20260415_104519 2/`
- Publication figures duplicated between `plots_results/publication/` and `plots_results/publication_final/`

### Target structure
```
Robust-NN-learning/
├── README.md                     ← rewritten, paper-style
├── pyproject.toml
├── src/
│   ├── losses/robust_losses.py   ← canonical loss functions
│   ├── models/vision_transformer.py
│   └── utils/
├── experiments/
│   ├── vit_medmnist/
│   │   ├── run_vit.py            ← Runpod_14April2026 (canonical latest)
│   │   └── configs/
│   ├── bert_nlp/
│   │   ├── run_bert.py           ← part3_BERT_Robust_NLP_Experiments.py
│   │   └── configs/
│   └── multimodal/
│       ├── run_multimodal.py     ← part4_Multimodal_Vision_Robust_Experiments.py
│       └── configs/
├── notebooks/
│   ├── vit_exploration.ipynb     ← Partha_VisionTr_rSDNet.ipynb (canonical)
│   ├── bert_exploration.ipynb    ← Partha_BERT_Robust_NLP.ipynb
│   └── multimodal_exploration.ipynb
├── results/
│   ├── paper/
│   │   ├── figures/              ← ONLY publication_final/ Fxx_*.png files
│   │   ├── tables/               ← generated .tex tables
│   │   └── csvs/                 ← canonical CSVs (single copy)
│   ├── visualizations/           ← the 4 .html interactive files
│   └── archive/                  ← everything else (raw, duplicates)
│       ├── 30March2026/
│       ├── 12-14April2026_raw/
│       └── stage_snapshots/
├── docs/
│   ├── planning/master_plan.md   ← this file
│   ├── figure_guide.md
│   └── experiment_map.md
├── assets/
│   └── gifs/                     ← the 5 animated GIFs
└── archive/
    ├── [Abhik]Paper1/            ← collaborator's separate paper
    ├── early_writeups/           ← Partha_WriteUp.md, Research_Understanding etc
    └── legacy_scripts/           ← 12April/13April RunPod .py duplicates
```

### What moves where (physical moves, no deletion)

| From | To | Reason |
|------|----|--------|
| `code/Runpod_14April2026_RobustNN_Experiments.py` | `experiments/vit_medmnist/run_vit.py` | Latest canonical ViT runner |
| `code/robust_losses.py` | `src/losses/robust_losses.py` | Core library |
| `code/vision_transformer.py` | `src/models/vision_transformer.py` | Core library |
| `code/part3_BERT_Robust_NLP_Experiments.py` | `experiments/bert_nlp/run_bert.py` | Canonical BERT runner |
| `code/part4_Multimodal_Vision_Robust_Experiments.py` | `experiments/multimodal/run_multimodal.py` | Canonical multimodal |
| `code/generate_publication_final.py` | `experiments/plotting/generate_plots.py` | Final plot script |
| `code/Partha_VisionTr_rSDNet.ipynb` | `notebooks/vit_exploration.ipynb` | Canonical ViT notebook |
| `code/Partha_BERT_Robust_NLP.ipynb` | `notebooks/bert_exploration.ipynb` | Canonical BERT notebook |
| `plots_results/publication_final/*.png` | `results/paper/figures/` | Paper-ready figures |
| `plots_results/15April2026/results_15April2026/*.csv` | `results/paper/csvs/` | Canonical results CSVs |
| `visualizations/*.html` | `results/visualizations/` | Interactive dashboards |
| `assets/*.gif` | `assets/gifs/` | Animation assets |
| All `_stage_20260415_*` dirs | `results/archive/stage_snapshots/` | Duplicated raw snapshots |
| `plots_results/30March2026/` | `results/archive/30March2026/` | Early experiment outputs |
| `[Abhik]Paper 1 (Reproducible codes)/` | `archive/abhik_paper1/` | Collaborator's separate work |
| `Partha_WriteUp.md/html`, `Research_Understanding_Writeup.*` | `archive/early_writeups/` | Drafts, not final |
| `code/12April2026_*.py`, `code/Runpod_12April2026_*.py`, `code/Runpod_13April2026_*.py` | `archive/legacy_scripts/` | Superseded by 14April version |
| `code/[error]*`, `code/Untitled*.ipynb`, `code/mixed_draft.ipynb` | `archive/legacy_scripts/` | Error/scratch files |
| `code/30March2026_*.ipynb`, `code/rRNet.ipynb`, `code/rSDNet.ipynb` | `archive/legacy_scripts/` | Early March experiments |

---

## Execution Plan (Parts and Sequence)

**Part 1** — Write LaTeX experiment section (`docs/paper/experiment_section.tex`)  
**Part 2** — Write LaTeX tables (clean data, noise, FGSM, SDIV surface, NLP)  
**Part 3** — Create new directory structure (`src/`, `experiments/`, `notebooks/`, `results/`, `archive/`)  
**Part 4** — Execute physical file moves (no deletion, archive everything)  
**Part 5** — Verify all canonical files are in place, run quick import check  
**Part 6** — Rewrite README.md with paper-style structure  
**Part 7** — Git commit with clear message  

---

## MANDATORY STOP

**Awaiting explicit user approval before any Part 1–7 execution.**
