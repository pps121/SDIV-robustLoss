# Figure Guide

This guide identifies which figures are central to the repository's scientific story and which are supplementary or exploratory.

## Main figures to foreground

These are the figures that should be easiest for a reader to find.

### Figure 1 — PathMNIST / DermaMNIST label-noise robustness
**Purpose:** show how losses behave as noise increases.

Recommended canonical files:
- `plots_results/publication/A3_noise_combined.png`
- `plots_results/publication/A1_noise_pathmnist.png`
- `plots_results/publication/A2_noise_dermamnist.png`

### Figure 2 — DermaMNIST FGSM robustness
**Purpose:** show adversarial robustness under increasing perturbation budget.

Recommended canonical files:
- `plots_results/publication/B1_fgsm_dermamnist.png`
- `plots_results/publication/B2_fgsm_drop_bar.png`

### Figure 3 — S-Divergence parameter landscape
**Purpose:** show how performance changes over the `(beta, lambda)` grid.

Recommended canonical files:
- `plots_results/publication/C1_sdiv_surface_3d.png`
- `plots_results/publication/C2_sdiv_heatmap.png`

### Figure 4 — Clean / robust trade-off summary
**Purpose:** summarize the balance between standard accuracy and robustness.

Recommended canonical files:
- `plots_results/publication/D1_dual_frontier.png`
- `plots_results/publication/D2_summary_multipanel.png`

---

## Supplementary figures

These are useful, but should not dominate the README.

### Early experiments
- `plots_results/publication/E1_early_experiments.png`

### Additional summaries
- `plots_results/publication/F1_clean_vs_noise_bar.png`

These should be presented as supporting validation or supplementary material.

---

## Suggested future curated folder split

The repository would be clearer if figures were eventually separated into:

```text
results/
├── paper_figures/
├── supplementary_figures/
└── exploratory_figures/
```

### paper_figures
Contains only the smallest set of figures needed to tell the main story.

### supplementary_figures
Contains additional supporting results that are useful but not central.

### exploratory_figures
Contains drafts, early tests, parameter variations, and alternative visual styles.

---

## Naming recommendation

Over time, rename key figures to a more paper-like convention, for example:

- `fig1_noise_combined.png`
- `fig2_fgsm_dermamnist.png`
- `fig3_sdiv_surface.png`
- `fig4_frontier.png`
- `figS1_early_experiments.png`

This would make the repository easier to cite and discuss.

---

## Recommended README policy

The README should show only:
- one main label-noise figure
- one FGSM figure
- one parameter figure
- one overall summary figure

All additional figures should be linked from docs or results folders rather than fully expanded in the main landing page.
