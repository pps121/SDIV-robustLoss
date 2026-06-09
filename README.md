# Robust Neural Learning via S-Divergence

This repository studies robust neural learning with S-Divergence losses across three settings: **(1) Vision Transformers on medical imaging, (2) BERT-based clinical NLP, and (3) zero-shot multimodal medical classification**. It contains experiment code, curated results, raw outputs, and browser-based interactive visualizations.

## What is in this repository?

The repository extends robust-loss ideas from rSDNet to transformer-based settings and organizes the work into three experiment families:

1. **ViT on MedMNIST image classification**
   - datasets: PathMNIST, DermaMNIST
   - robustness settings: label noise and FGSM adversarial perturbations
2. **BERT on clinical NLP**
   - robustness under noisy supervision in text classification
3. **CLIP / MedSigLIP zero-shot evaluation**
   - medical-domain multimodal evaluation

## Main findings

- On **PathMNIST label-noise experiments**, several robust losses remain stable, while **MAE can collapse strongly** at intermediate noise rates.
- On **DermaMNIST FGSM experiments**, **SDIV, MAE, and GCE** remain nearly invariant across the tested perturbation budgets, while standard cross-entropy degrades substantially.
- In **zero-shot medical evaluation**, **MedSigLIP** outperforms generic CLIP on the reported datasets.

## Repository map

```text
robustNN-transformers/
├── README.md
├── code/                          # model, loss, and experiment scripts
├── visualizations/                # browser-based interactive demos
├── assets/                        # README figures / GIFs
├── plots_results/                 # generated plots (currently mixed: curated + raw)
├── result_BERT/                   # BERT experiment outputs
├── results_multimodal_vision/     # multimodal experiment outputs
├── docs/                          # repository and experiment documentation
└── pyproject.toml
```

Important files:

- `code/robust_losses.py` — robust loss implementations
- `code/vision_transformer.py` — Vision Transformer model
- `code/Runpod_14April2026_RobustNN_Experiments.py` — main ViT experiment runner
- `code/part3_BERT_Robust_NLP_Experiments.py` — BERT experiment runner
- `code/part4_Multimodal_Vision_Robust_Experiments.py` — multimodal experiment runner
- `code/generate_publication_plots.py` — static figure generation
- `code/generate_gifs.py` — GIF generation for README/demo assets

For a clearer overview of how these pieces fit together, see:
- `docs/repository_reorganization_plan.md`

## Reproducing the main experiments

### 1. Installation

```bash
git clone https://github.com/pps121/robustNN-transformers.git
cd robustNN-transformers

conda create -n robustnn python=3.11 -y
conda activate robustnn

pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets medmnist scikit-learn matplotlib seaborn pandas tqdm open_clip_torch imageio Pillow numpy
```

### 2. Vision Transformer experiments (PathMNIST / DermaMNIST)

```bash
# quick run
python3 code/Runpod_14April2026_RobustNN_Experiments.py

# fuller run
ROBUST_NN_QUICK_RUN=0 \
ROBUST_NN_VIT_EPOCHS=100 \
ROBUST_NN_DATASETS=pathmnist,dermamnist \
python3 code/Runpod_14April2026_RobustNN_Experiments.py
```

### 3. BERT experiments

```bash
export HF_TOKEN="your_token_here"
python3 code/part3_BERT_Robust_NLP_Experiments.py
```

### 4. Multimodal zero-shot experiments

```bash
python3 code/part4_Multimodal_Vision_Robust_Experiments.py
```

### 5. Regenerate publication plots

```bash
python3 code/generate_publication_plots.py
python3 code/generate_gifs.py
```

## Core result summary

### ViT on medical imaging

#### PathMNIST label-noise robustness

| Loss | η=0% | η=10% | η=20% | η=30% | η=40% |
|:---|:---:|:---:|:---:|:---:|:---:|
| CCE | 83.0 | 81.0 | 81.1 | 82.0 | 82.4 |
| MAE | 78.5 | 48.8 | 42.7 | 44.0 | 72.4 |
| GCE(q=0.7) | 82.2 | 83.0 | 81.5 | 81.2 | 81.6 |
| SDIV | 82.6 | 82.7 | 81.9 | 81.0 | 82.0 |
| FCL | 83.6 | 83.2 | 82.3 | 83.3 | 83.0 |

#### DermaMNIST FGSM robustness

| Loss | ε=0 | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|:---|:---:|:---:|:---:|:---:|:---:|
| CCE | 72.3 | 61.6 | 52.2 | 39.9 | 22.7 |
| MAE | 66.9 | 66.9 | 66.9 | 66.9 | 66.9 |
| GCE(q=0.7) | 66.9 | 66.9 | 66.9 | 66.9 | 66.9 |
| SDIV | 66.9 | 66.9 | 66.9 | 66.9 | 66.9 |
| FCL | 72.8 | 65.3 | 59.1 | 47.6 | 29.0 |

### Multimodal zero-shot evaluation

| Dataset | MedSigLIP | CLIP |
|---|:---:|:---:|
| PathMNIST | 23.0 | 18.2 |
| DermaMNIST | 19.9 | 13.7 |

## Interactive visualizations

Open these directly in the browser:

- `visualizations/index.html`
- `visualizations/vit_layer_explorer.html`
- `visualizations/sdiv_loss_surface.html`
- `visualizations/robustness_dashboard.html`

These demos are intended as explanatory companions to the experiments, not replacements for the main quantitative results.

## Current cleanup direction

The repository is being reorganized to better separate:
- curated paper figures
- raw outputs
- exploratory artifacts
- reusable implementation code
- interactive demos

The cleanup plan is documented in:
- `docs/repository_reorganization_plan.md`

## Citation

```bibtex
@misc{saha2026robustnn,
  title   = {Robust Neural Learning via S-Divergence: Extending rSDNet to Vision Transformers and BERT},
  author  = {Saha, Partha Pratim},
  year    = {2026},
  url     = {https://github.com/pps121/robustNN-transformers},
  note    = {Extension of Jana \& Ghosh (2026), arXiv:2603.17628}
}
```

## Author

**Extension work (Parts 2–4), code, experiments, and visualizations:**  
Partha Pratim Saha

**Foundational papers (rSDNet / rRNet / S-divergence theory):**  
Suryasis Jana and Abhik Ghosh
