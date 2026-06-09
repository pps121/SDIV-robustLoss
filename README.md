# Robust Neural Learning via S-Divergence

This repository evaluates S-Divergence-based robust training in three transformer-oriented settings: **(1) Vision Transformers on medical imaging, (2) BERT-based clinical NLP, and (3) zero-shot multimodal medical classification**. It includes experiment scripts, result artifacts, curated documentation, and browser-based interactive visualizations.

The repository is intended as an empirical extension of robust-loss ideas from prior S-Divergence / rSDNet work to transformer-based settings. Where theoretical properties such as Fisher consistency or Bayes optimality are discussed, those should be understood as properties established in the cited prior literature rather than proved within this repository.

## Scope of the repository

The work is organized into three experiment families:

1. **Vision Transformer experiments on medical imaging**
   - datasets: PathMNIST, DermaMNIST
   - evaluation settings: label noise and FGSM perturbations
2. **BERT experiments on clinical NLP**
   - robustness-oriented text classification experiments under noisy supervision
3. **CLIP / MedSigLIP zero-shot evaluation**
   - medical-domain multimodal comparison on reported datasets

This repository is strongest as an experiment and results repository. It should not be read as establishing universal robustness claims beyond the datasets, architectures, and perturbation settings explicitly evaluated here.

## Main observations from the reported experiments

- In the reported **PathMNIST label-noise experiments**, several robust losses remain comparatively stable across the tested noise levels, while **MAE shows a substantial drop** at intermediate noise rates.
- In the reported **DermaMNIST FGSM experiments**, **SDIV, MAE, and GCE** show no observed accuracy drop across the tested epsilon values in the stored summaries, while standard cross-entropy degrades substantially.
- In the reported **zero-shot medical evaluation**, **MedSigLIP** outperforms generic CLIP on the included datasets.

These observations should be interpreted as results for the reported experimental setup, not as general guarantees.

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

Key files:

- `code/robust_losses.py` — robust loss implementations used in the repository
- `code/vision_transformer.py` — Vision Transformer model
- `code/Runpod_14April2026_RobustNN_Experiments.py` — main ViT experiment runner
- `code/part3_BERT_Robust_NLP_Experiments.py` — BERT experiment runner
- `code/part4_Multimodal_Vision_Robust_Experiments.py` — multimodal experiment runner
- `code/generate_publication_plots.py` — static figure generation
- `code/generate_gifs.py` — GIF generation for README/demo assets

Documentation:
- `docs/repository_reorganization_plan.md`
- `docs/figure_guide.md`

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

## Compact result summary

### ViT on medical imaging

#### PathMNIST label-noise results

| Loss | η=0% | η=10% | η=20% | η=30% | η=40% |
|:---|:---:|:---:|:---:|:---:|:---:|
| CCE | 83.0 | 81.0 | 81.1 | 82.0 | 82.4 |
| MAE | 78.5 | 48.8 | 42.7 | 44.0 | 72.4 |
| GCE(q=0.7) | 82.2 | 83.0 | 81.5 | 81.2 | 81.6 |
| SDIV | 82.6 | 82.7 | 81.9 | 81.0 | 82.0 |
| FCL | 83.6 | 83.2 | 82.3 | 83.3 | 83.0 |

#### DermaMNIST FGSM results

| Loss | ε=0 | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|:---|:---:|:---:|:---:|:---:|:---:|
| CCE | 72.3 | 61.6 | 52.2 | 39.9 | 22.7 |
| MAE | 66.9 | 66.9 | 66.9 | 66.9 | 66.9 |
| GCE(q=0.7) | 66.9 | 66.9 | 66.9 | 66.9 | 66.9 |
| SDIV | 66.9 | 66.9 | 66.9 | 66.9 | 66.9 |
| FCL | 72.8 | 65.3 | 59.1 | 47.6 | 29.0 |

### Multimodal zero-shot summary

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

These visualizations are intended as explanatory companions to the experiments. Some are direct summaries of reported experimental results, while others are conceptual or illustrative views designed to help readers interpret model behavior. They should therefore be read together with the code, reported outputs, and figure documentation rather than as standalone evidence.

## Current cleanup direction

The repository is being reorganized to better separate:
- curated paper figures
- raw outputs
- exploratory artifacts
- reusable implementation code
- interactive demos

See:
- `docs/repository_reorganization_plan.md`
- `docs/figure_guide.md`

## Citation

If you use this repository, please cite the relevant empirical or theoretical sources appropriately.

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
