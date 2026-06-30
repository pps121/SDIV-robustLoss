#!/usr/bin/env python3
"""
generate_all_publication_plots.py
==================================
Master script: regenerates ALL 15 publication-ready figures in one pass.

Academic style:
  - Clean DejaVu Sans font, no Unicode symbols
  - Legends: upper-right, white background, thin border
  - All plots note "seed=42, single run"
  - Consistent COLORS / MARKERS across figures

Run from repo root:
    cd /Volumes/Research/Subho_IIM/Robust-NN-learning
    python code/generate_all_publication_plots.py
"""

import os, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = "plots_results/15April2026/results_15April2026"
OUT_DIR  = "plots_results/publication_final"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global matplotlib style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.6,
    "figure.dpi":         150,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
})

LEGEND_KW = dict(loc="upper right", framealpha=1.0, edgecolor="#aaaaaa",
                 fancybox=False, fontsize=8.5)

COLORS = {
    "CCE":        "#1f1f1f",
    "FCL":        "#2ca02c",
    "GCE(q=0.7)": "#1f77b4",
    "MAE":        "#ff7f0e",
    "SCE":        "#7f7f7f",
    "TPDD-CCE":   "#17becf",
    "TSCCE":      "#e377c2",
    "TruncGCE":   "#9467bd",
    "ForwardT":   "#bcbd22",
    "SDIV":       "#d62728",
}
MARKERS = {
    "CCE":"o","FCL":"s","GCE(q=0.7)":"D","MAE":"^","SCE":"v",
    "TPDD-CCE":"P","TSCCE":"h","TruncGCE":"X","ForwardT":"*","SDIV":"o",
}
LINE_STYLES = {
    "CCE":"-","FCL":"--","GCE(q=0.7)":"-.","MAE":":","SCE":"-.",
    "TPDD-CCE":"--","TSCCE":"-.","TruncGCE":":","ForwardT":"--","SDIV":"-",
}


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# F1 – PathMNIST Noise Robustness
# ══════════════════════════════════════════════════════════════════════════════
def fig_pathmnist_noise():
    df = pd.read_csv(f"{DATA_DIR}/pathmnist_noise_results.csv")
    losses = [l for l in COLORS if l in df.loss.unique()]
    fig, ax = plt.subplots(figsize=(8, 5))
    for loss in losses:
        sub = df[df.loss == loss].sort_values("noise_rate")
        ax.plot(sub.noise_rate*100, sub.accuracy*100,
                color=COLORS[loss], marker=MARKERS.get(loss,"o"),
                ls=LINE_STYLES.get(loss,"-"),
                lw=2 if loss=="SDIV" else 1.4,
                ms=7 if loss=="SDIV" else 5, label=loss,
                zorder=5 if loss=="SDIV" else 3)
    ax.set_xlabel("Label Noise Rate (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_title("PathMNIST: Noise Robustness Across All Loss Functions\n"
                 "(seed=42, single run; noise varies 0% to 40%)")
    ax.legend(**LEGEND_KW)
    savefig(fig, "F1_pathmnist_noise_all_losses.png")


# ══════════════════════════════════════════════════════════════════════════════
# F2 – DermaMNIST Noise Robustness
# ══════════════════════════════════════════════════════════════════════════════
def fig_dermamnist_noise():
    df = pd.read_csv(f"{DATA_DIR}/dermamnist_noise_results.csv")
    majority = 66.88
    losses = [l for l in COLORS if l in df.loss.unique()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(majority, color="#999999", lw=1.2, ls=":",
               label=f"Majority-class baseline ({majority:.1f}%)")
    for loss in losses:
        sub = df[df.loss == loss].sort_values("noise_rate")
        flat = (sub.accuracy*100).round(1).eq(round(majority,1)).all()
        ax.plot(sub.noise_rate*100, sub.accuracy*100,
                color=COLORS[loss], marker=MARKERS.get(loss,"o"),
                ls=LINE_STYLES.get(loss,"-"),
                lw=2 if loss=="SDIV" else 1.4,
                ms=7 if loss=="SDIV" else 5,
                alpha=0.45 if flat else 1.0,
                label=f"{loss} (degenerate)" if flat else loss,
                zorder=5 if loss=="SDIV" else 3)
    ax.set_xlabel("Label Noise Rate (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_title("DermaMNIST: Noise Robustness — Default SDIV Parameters\n"
                 "(seed=42, single run; faded lines = degenerate majority-class output)")
    ax.legend(**LEGEND_KW, ncol=2)
    savefig(fig, "F2_dermamnist_noise_all_losses.png")


# ══════════════════════════════════════════════════════════════════════════════
# F3 – PathMNIST FGSM Adversarial Robustness
# ══════════════════════════════════════════════════════════════════════════════
def fig_pathmnist_fgsm():
    df = pd.read_csv(f"{DATA_DIR}/pathmnist_fgsm_results.csv")
    eps_labels = ["0", "1/255", "2/255", "4/255", "8/255"]
    losses = [l for l in COLORS if l in df.loss.unique()]
    fig, ax = plt.subplots(figsize=(8, 5))
    for loss in losses:
        sub = df[df.loss == loss].sort_values("epsilon")
        ax.plot(range(len(sub)), sub.accuracy*100,
                color=COLORS[loss], marker=MARKERS.get(loss,"o"),
                ls=LINE_STYLES.get(loss,"-"),
                lw=2 if loss=="SDIV" else 1.4,
                ms=7 if loss=="SDIV" else 5, label=loss,
                zorder=5 if loss=="SDIV" else 3)
    ax.set_xticks(range(len(eps_labels)))
    ax.set_xticklabels(eps_labels)
    ax.set_xlabel("FGSM Perturbation Budget (epsilon)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("PathMNIST: FGSM Adversarial Robustness — All Loss Functions\n"
                 "(seed=42, single run; all methods degrade under adversarial attack)")
    ax.legend(**LEGEND_KW, ncol=2)
    savefig(fig, "F3_pathmnist_fgsm_all_losses.png")


# ══════════════════════════════════════════════════════════════════════════════
# F4 – DermaMNIST FGSM Adversarial Robustness
# ══════════════════════════════════════════════════════════════════════════════
def fig_dermamnist_fgsm():
    df = pd.read_csv(f"{DATA_DIR}/dermamnist_fgsm_results.csv")
    majority = 66.88
    eps_labels = ["0", "1/255", "2/255", "4/255", "8/255"]
    losses = [l for l in COLORS if l in df.loss.unique()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(majority, color="#999999", lw=1.2, ls=":",
               label=f"Majority-class baseline ({majority:.1f}%)")
    for loss in losses:
        sub = df[df.loss == loss].sort_values("epsilon")
        accs = (sub.accuracy*100).values
        flat = np.ptp(accs) < 0.5
        ax.plot(range(len(accs)), accs,
                color=COLORS[loss], marker=MARKERS.get(loss,"o"),
                ls=LINE_STYLES.get(loss,"-"),
                lw=2 if loss=="SDIV" else 1.4,
                ms=7 if loss=="SDIV" else 5,
                alpha=0.45 if flat else 1.0,
                label=f"{loss} (degenerate)" if flat else loss,
                zorder=5 if loss=="SDIV" else 3)
    ax.set_xticks(range(len(eps_labels)))
    ax.set_xticklabels(eps_labels)
    ax.set_xlabel("FGSM Perturbation Budget (epsilon)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("DermaMNIST: FGSM Adversarial Robustness\n"
                 "(seed=42, single run; default SDIV is degenerate on this imbalanced dataset)")
    ax.legend(**LEGEND_KW, ncol=2)
    savefig(fig, "F4_dermamnist_fgsm_all_losses.png")


# ══════════════════════════════════════════════════════════════════════════════
# F5 – PathMNIST FGSM Accuracy Drop Bar
# ══════════════════════════════════════════════════════════════════════════════
def fig_pathmnist_fgsm_drop():
    df = pd.read_csv(f"{DATA_DIR}/pathmnist_fgsm_results.csv")
    base  = df[df.epsilon.eq(0.0)].set_index("loss")["accuracy"]
    worst = df[df.epsilon.eq(df.epsilon.max())].set_index("loss")["accuracy"]
    losses = [l for l in COLORS if l in base.index and l in worst.index]
    drops  = {l: (base[l] - worst[l])*100 for l in losses}
    drops  = dict(sorted(drops.items(), key=lambda x: x[1], reverse=True))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (loss, drop) in enumerate(drops.items()):
        ax.barh(loss, drop, color=COLORS.get(loss,"#888"), edgecolor="white", lw=0.5)
        ax.text(drop+0.3, i, f"-{drop:.1f} pp", va="center", fontsize=8.5)
    ax.set_xlabel("Test Accuracy Drop (percentage points): epsilon=0 to epsilon=8/255")
    ax.set_title("PathMNIST: Total FGSM Accuracy Drop by Loss Function\n"
                 "(seed=42, single run; larger bar = less adversarially robust)")
    ax.invert_yaxis()
    savefig(fig, "F5_pathmnist_fgsm_drop_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# F6 – DermaMNIST FGSM Accuracy Drop Bar
# ══════════════════════════════════════════════════════════════════════════════
def fig_dermamnist_fgsm_drop():
    df = pd.read_csv(f"{DATA_DIR}/dermamnist_fgsm_results.csv")
    base  = df[df.epsilon.eq(0.0)].set_index("loss")["accuracy"]
    worst = df[df.epsilon.eq(df.epsilon.max())].set_index("loss")["accuracy"]
    losses = [l for l in COLORS if l in base.index and l in worst.index]
    flat = set()
    drops = {}
    for l in losses:
        sub = df[df.loss == l].sort_values("epsilon")
        if np.ptp((sub.accuracy*100).values) < 0.5:
            flat.add(l)
        drops[l] = (base[l] - worst[l])*100
    drops = dict(sorted(drops.items(), key=lambda x: x[1], reverse=True))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (loss, drop) in enumerate(drops.items()):
        hatch = "//" if loss in flat else None
        ax.barh(loss, abs(drop), color=COLORS.get(loss,"#888"),
                edgecolor="white", lw=0.5, hatch=hatch)
        note = " (degenerate)" if loss in flat else ""
        ax.text(abs(drop)+0.1, i, f"{-drop:.1f} pp{note}", va="center", fontsize=8.5)
    ax.set_xlabel("Test Accuracy Drop (percentage points): epsilon=0 to epsilon=8/255")
    ax.set_title("DermaMNIST: FGSM Total Accuracy Drop by Loss Function\n"
                 "(seed=42, single run; hatched bars = degenerate majority-class output)")
    ax.invert_yaxis()
    hp = mpatches.Patch(facecolor="white", edgecolor="#555", hatch="//",
                        label="Degenerate (majority-class collapse)")
    ax.legend(handles=[hp], **LEGEND_KW)
    savefig(fig, "F6_dermamnist_fgsm_drop_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# F7 – PathMNIST Clean Accuracy Bar (all losses ranked)
# ══════════════════════════════════════════════════════════════════════════════
def fig_pathmnist_clean_bar():
    df = pd.read_csv(f"{DATA_DIR}/pathmnist_noise_results.csv")
    clean = df[df.noise_rate.eq(0.0)].sort_values("accuracy", ascending=False)
    sdiv_df = pd.read_csv(f"{DATA_DIR}/pathmnist_sdiv_surface.csv")
    best = sdiv_df.loc[sdiv_df.accuracy.idxmax()]
    sdiv_label = f"SDIV-opt\n(b={best.beta},l={best.lam})"
    losses = list(clean.loss) + [sdiv_label]
    accs   = list(clean.accuracy*100) + [best.accuracy*100]
    cols   = [COLORS.get(l,"#888") for l in clean.loss] + ["#d62728"]
    paired = sorted(zip(accs, losses, cols), reverse=True)
    accs, losses, cols = zip(*paired)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(losses)), accs, color=cols, edgecolor="white", lw=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, acc+0.05,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(losses)))
    ax.set_xticklabels(losses, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(min(accs)-2, max(accs)+1.5)
    ax.set_title("PathMNIST: Clean (0% Noise) Test Accuracy — All Loss Functions\n"
                 "(seed=42, single run; SDIV-opt uses best parameters from grid search)")
    savefig(fig, "F7_pathmnist_clean_accuracy_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# F8 – PathMNIST SDIV Parameter Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig_pathmnist_sdiv_heatmap():
    df = pd.read_csv(f"{DATA_DIR}/pathmnist_sdiv_surface.csv")
    betas = sorted(df.beta.unique())
    lams  = sorted(df.lam.unique())
    grid  = np.full((len(betas), len(lams)), np.nan)
    for _, row in df.iterrows():
        grid[betas.index(row.beta), lams.index(row.lam)] = row.accuracy*100
    cmap = LinearSegmentedColormap.from_list("rg", ["#b2182b","#f7f7f7","#1a7837"])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap=cmap, aspect="auto",
                   vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    plt.colorbar(im, ax=ax, label="Test Accuracy (%)")
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([str(l) for l in lams])
    ax.set_yticks(range(len(betas))); ax.set_yticklabels([str(b) for b in betas])
    ax.set_xlabel("Lambda"); ax.set_ylabel("Beta")
    ax.set_title("PathMNIST: S-Divergence Parameter Sensitivity (beta vs. lambda)\n"
                 "(seed=42, single run; orange star = best configuration)")
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    for bi in range(len(betas)):
        for li in range(len(lams)):
            if not np.isnan(grid[bi,li]):
                ax.text(li, bi, f"{grid[bi,li]:.1f}", ha="center", va="center",
                        fontsize=8.5,
                        fontweight="bold" if (bi,li)==best else "normal",
                        color="white" if grid[bi,li]<np.nanmean(grid) else "black")
    ax.plot(best[1], best[0], marker="*", ms=14, color="#ff7f0e",
            markeredgecolor="white", markeredgewidth=1, zorder=10)
    savefig(fig, "F8_pathmnist_sdiv_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# F9 – DermaMNIST SDIV Parameter Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig_dermamnist_sdiv_heatmap():
    df = pd.read_csv(f"{DATA_DIR}/dermamnist_sdiv_surface.csv")
    majority = 66.88
    betas = sorted(df.beta.unique())
    lams  = sorted(df.lam.unique())
    grid  = np.full((len(betas), len(lams)), np.nan)
    for _, row in df.iterrows():
        grid[betas.index(row.beta), lams.index(row.lam)] = row.accuracy*100
    cmap = LinearSegmentedColormap.from_list("rg", ["#b2182b","#f7f7f7","#1a7837"])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap=cmap, aspect="auto",
                   vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    plt.colorbar(im, ax=ax, label="Test Accuracy (%)")
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([str(l) for l in lams])
    ax.set_yticks(range(len(betas))); ax.set_yticklabels([str(b) for b in betas])
    ax.set_xlabel("Lambda"); ax.set_ylabel("Beta")
    ax.set_title("DermaMNIST: S-Divergence Parameter Sensitivity (beta vs. lambda)\n"
                 "(seed=42, single run; lambda=-0.8 causes majority-class collapse)")
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    for bi in range(len(betas)):
        for li in range(len(lams)):
            if not np.isnan(grid[bi,li]):
                degen = abs(grid[bi,li] - majority) < 0.5
                ax.text(li, bi, f"{grid[bi,li]:.1f}", ha="center", va="center",
                        fontsize=8.5,
                        fontweight="bold" if (bi,li)==best else "normal",
                        color="#cc0000" if degen else "black")
    ax.plot(best[1], best[0], marker="*", ms=14, color="#ff7f0e",
            markeredgecolor="white", markeredgewidth=1, zorder=10)
    savefig(fig, "F9_dermamnist_sdiv_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# F10 – Lambda Convergence Map (DermaMNIST)
# ══════════════════════════════════════════════════════════════════════════════
def fig_lambda_convergence_map():
    df = pd.read_csv(f"{DATA_DIR}/dermamnist_sdiv_surface.csv")
    majority = 66.88
    betas = sorted(df.beta.unique())
    lams  = sorted(df.lam.unique())
    grid  = np.full((len(betas), len(lams)), np.nan)
    for _, row in df.iterrows():
        grid[betas.index(row.beta), lams.index(row.lam)] = row.accuracy*100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: line plot
    ax = axes[0]
    bc_arr = plt.cm.viridis(np.linspace(0.1, 0.9, len(betas)))
    for bi, (beta, bc) in enumerate(zip(betas, bc_arr)):
        xs = [lams[li] for li in range(len(lams)) if not np.isnan(grid[bi,li])]
        ys = [grid[bi,li] for li in range(len(lams)) if not np.isnan(grid[bi,li])]
        ax.plot(xs, ys, color=bc, marker="o", ms=5, label=f"beta={beta}")
    ax.axhline(majority, color="#cc0000", lw=1.2, ls="--",
               label=f"Majority baseline ({majority:.1f}%)")
    ax.axvspan(min(lams)-0.05, -0.6, alpha=0.08, color="#cc0000")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("DermaMNIST: SDIV Accuracy by Lambda\n"
                 "(degenerate zone: lambda below -0.6)")
    ax.legend(**LEGEND_KW)

    # Right: convergence map
    ax2 = axes[1]
    degen = np.abs(grid - majority) < 0.5
    bg    = np.where(degen, 0.0, 1.0)
    cmap2 = LinearSegmentedColormap.from_list("dg", ["#f4cccc","#d9f7d9"])
    ax2.imshow(bg, cmap=cmap2, aspect="auto", vmin=0, vmax=1,
               extent=[-0.5, len(lams)-0.5, -0.5, len(betas)-0.5], origin="lower")
    for bi in range(len(betas)):
        for li in range(len(lams)):
            if not np.isnan(grid[bi,li]):
                color = "#cc0000" if degen[bi,li] else "#1a5c1a"
                ax2.text(li, bi, f"{grid[bi,li]:.1f}", ha="center", va="center",
                         fontsize=9, color=color,
                         fontweight="bold" if degen[bi,li] else "normal")
    ax2.set_xticks(range(len(lams))); ax2.set_xticklabels([str(l) for l in lams])
    ax2.set_yticks(range(len(betas))); ax2.set_yticklabels([str(b) for b in betas])
    ax2.set_xlabel("Lambda"); ax2.set_ylabel("Beta")
    ax2.set_title("DermaMNIST: Convergence Map\n"
                  "(red = degenerate, green = real learning)")
    r = mpatches.Patch(color="#f4cccc", label="Degenerate (majority-class collapse)")
    g = mpatches.Patch(color="#d9f7d9", label="Real learning (above baseline)")
    ax2.legend(handles=[r,g], **LEGEND_KW)

    fig.suptitle("Key Finding: Lambda Controls Convergence Regime in SDIV on Imbalanced Data\n"
                 "(seed=42, single run)", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "F10_lambda_convergence_map.png")


# ══════════════════════════════════════════════════════════════════════════════
# F11 – PathMNIST AUNRC Ranking
# ══════════════════════════════════════════════════════════════════════════════
def fig_aunrc_ranking():
    df = pd.read_csv(f"{DATA_DIR}/pathmnist_noise_results.csv")
    losses = [l for l in COLORS if l in df.loss.unique()]
    aunrcs = {}
    for loss in losses:
        sub = df[df.loss==loss].sort_values("noise_rate")
        accs = sub.accuracy.values*100
        if len(accs) >= 2:
            aunrcs[loss] = np.trapz(accs, dx=10.0) / (10.0*(len(accs)-1))
    aunrcs = dict(sorted(aunrcs.items(), key=lambda x: x[1], reverse=True))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (loss, val) in enumerate(aunrcs.items()):
        ax.barh(loss, val, color=COLORS.get(loss,"#888"), edgecolor="white", lw=0.5)
        ax.text(val+0.05, i, f"{val:.2f}%", va="center", fontsize=9)
    ax.set_xlabel("AUNRC — Area Under Noise-Robustness Curve (%)")
    ax.set_title("PathMNIST: Noise-Robustness Ranking (AUNRC)\n"
                 "(seed=42, single run; AUNRC = average accuracy across noise levels 0-40%)")
    ax.invert_yaxis()
    savefig(fig, "F11_pathmnist_aunrc_ranking.png")


# ══════════════════════════════════════════════════════════════════════════════
# F12 – NLP BERT Results
# ══════════════════════════════════════════════════════════════════════════════
def fig_nlp_bert():
    df_em  = pd.read_csv(f"{DATA_DIR}/nlp_Emotion_results.csv")
    df_pub = pd.read_csv(f"{DATA_DIR}/nlp_PubMedQA_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, title in zip(axes, [df_em, df_pub],
                              ["BERT on Emotion (6-class NLP)",
                               "BERT on PubMedQA (3-class NLP)"]):
        df = df.sort_values("best_acc", ascending=False).reset_index(drop=True)
        cols = [COLORS.get(l,"#888") for l in df.loss]
        bars = ax.bar(range(len(df)), df.best_acc*100, color=cols, edgecolor="white", lw=0.5)
        for bar, acc in zip(bars, df.best_acc*100):
            ax.text(bar.get_x()+bar.get_width()/2, acc+0.02,
                    f"{acc:.2f}%", ha="center", va="bottom", fontsize=8.5)
        sdiv_rank = (list(df.loss).index("SDIV")+1) if "SDIV" in list(df.loss) else "N/A"
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(list(df.loss), rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Best Test Accuracy (%)")
        ax.set_ylim(df.best_acc.min()*100-1, df.best_acc.max()*100+1.5)
        ax.set_title(f"{title}\n(SDIV rank: {sdiv_rank} of {len(df)})")
    fig.suptitle("NLP Results: BERT Fine-Tuned with Different Loss Functions\n"
                 "(seed=42, single run; SDIV is competitive but not consistently top in NLP)",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "F12_nlp_bert_results.png")


# ══════════════════════════════════════════════════════════════════════════════
# F13 – SDIV 3D Parameter Surface
# ══════════════════════════════════════════════════════════════════════════════
def fig_sdiv_3d_surface():
    fig = plt.figure(figsize=(14, 5))
    datasets = [
        ("pathmnist",  "PathMNIST",  pd.read_csv(f"{DATA_DIR}/pathmnist_sdiv_surface.csv")),
        ("dermamnist", "DermaMNIST", pd.read_csv(f"{DATA_DIR}/dermamnist_sdiv_surface.csv")),
    ]
    for idx, (ds, ds_label, df) in enumerate(datasets):
        ax = fig.add_subplot(1, 2, idx+1, projection="3d")
        betas = sorted(df.beta.unique())
        lams  = sorted(df.lam.unique())
        B, L  = np.meshgrid(betas, lams, indexing="ij")
        Z     = np.full_like(B, np.nan, dtype=float)
        for _, row in df.iterrows():
            Z[betas.index(row.beta), lams.index(row.lam)] = row.accuracy*100
        ax.plot_surface(B, L, Z, cmap="RdYlGn", alpha=0.85, edgecolor="none")
        best = df.loc[df.accuracy.idxmax()]
        ax.scatter([best.beta],[best.lam],[best.accuracy*100],
                   color="#ff7f0e", s=60, zorder=10,
                   label=f"Best: b={best.beta}, l={best.lam}")
        ax.set_xlabel("Beta", labelpad=6)
        ax.set_ylabel("Lambda", labelpad=6)
        ax.set_zlabel("Accuracy (%)", labelpad=6)
        ax.set_title(f"{ds_label}\n(best: {best.accuracy*100:.1f}%)", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("S-Divergence (beta, lambda) Parameter Surface\n"
                 "(seed=42, single run; orange dot = optimal configuration)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    savefig(fig, "F13_sdiv_3d_surface.png")


# ══════════════════════════════════════════════════════════════════════════════
# F14 – Master 6-panel Summary Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def fig_master_summary():
    df_pnoise = pd.read_csv(f"{DATA_DIR}/pathmnist_noise_results.csv")
    df_dnoise = pd.read_csv(f"{DATA_DIR}/dermamnist_noise_results.csv")
    df_pfgsm  = pd.read_csv(f"{DATA_DIR}/pathmnist_fgsm_results.csv")
    df_psurf  = pd.read_csv(f"{DATA_DIR}/pathmnist_sdiv_surface.csv")
    df_dsurf  = pd.read_csv(f"{DATA_DIR}/dermamnist_sdiv_surface.csv")
    majority  = 66.88

    fig = plt.figure(figsize=(17, 10))
    axes = [fig.add_subplot(2, 3, i+1) for i in range(6)]
    losses_all = list(COLORS.keys())
    eps_labels = ["0","1/255","2/255","4/255","8/255"]

    # (a) PathMNIST noise
    ax = axes[0]
    for loss in [l for l in losses_all if l in df_pnoise.loss.unique()]:
        sub = df_pnoise[df_pnoise.loss==loss].sort_values("noise_rate")
        ax.plot(sub.noise_rate*100, sub.accuracy*100,
                color=COLORS[loss], marker=MARKERS.get(loss,"o"), ms=4,
                ls=LINE_STYLES.get(loss,"-"),
                lw=2 if loss=="SDIV" else 1.2, label=loss)
    ax.set_xlabel("Noise Rate (%)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) PathMNIST: Noise Robustness")
    ax.legend(**{**LEGEND_KW, "fontsize": 7}, ncol=2)

    # (b) DermaMNIST noise
    ax = axes[1]
    ax.axhline(majority, color="#999", lw=1, ls=":", label="Majority baseline")
    for loss in [l for l in losses_all if l in df_dnoise.loss.unique()]:
        sub = df_dnoise[df_dnoise.loss==loss].sort_values("noise_rate")
        flat = (sub.accuracy*100).round(1).eq(round(majority,1)).all()
        ax.plot(sub.noise_rate*100, sub.accuracy*100,
                color=COLORS[loss], marker=MARKERS.get(loss,"o"), ms=4,
                ls=LINE_STYLES.get(loss,"-"),
                lw=2 if loss=="SDIV" else 1.2,
                alpha=0.35 if flat else 1.0, label=loss)
    ax.set_xlabel("Noise Rate (%)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) DermaMNIST: Noise\n(faded = degenerate)")
    ax.legend(**{**LEGEND_KW, "fontsize": 7}, ncol=2)

    # (c) PathMNIST FGSM
    ax = axes[2]
    for loss in [l for l in losses_all if l in df_pfgsm.loss.unique()]:
        sub = df_pfgsm[df_pfgsm.loss==loss].sort_values("epsilon")
        ax.plot(range(len(sub)), sub.accuracy*100,
                color=COLORS[loss], marker=MARKERS.get(loss,"o"), ms=4,
                ls=LINE_STYLES.get(loss,"-"),
                lw=2 if loss=="SDIV" else 1.2, label=loss)
    ax.set_xticks(range(len(eps_labels)))
    ax.set_xticklabels(eps_labels, fontsize=8)
    ax.set_xlabel("FGSM epsilon"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("(c) PathMNIST: FGSM Robustness")
    ax.legend(**{**LEGEND_KW, "fontsize": 7}, ncol=2)

    # (d) PathMNIST SDIV heatmap
    ax = axes[3]
    betas = sorted(df_psurf.beta.unique())
    lams  = sorted(df_psurf.lam.unique())
    grid  = np.full((len(betas),len(lams)), np.nan)
    for _,row in df_psurf.iterrows():
        grid[betas.index(row.beta), lams.index(row.lam)] = row.accuracy*100
    cmap = LinearSegmentedColormap.from_list("rg",["#b2182b","#f7f7f7","#1a7837"])
    im = ax.imshow(grid, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.75)
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([str(l) for l in lams], fontsize=7)
    ax.set_yticks(range(len(betas))); ax.set_yticklabels([str(b) for b in betas], fontsize=7)
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.plot(best[1], best[0], marker="*", ms=12, color="#ff7f0e",
            markeredgecolor="white", markeredgewidth=1)
    for bi in range(len(betas)):
        for li in range(len(lams)):
            if not np.isnan(grid[bi,li]):
                ax.text(li, bi, f"{grid[bi,li]:.0f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if grid[bi,li]<np.nanmean(grid) else "black")
    ax.set_xlabel("Lambda"); ax.set_ylabel("Beta")
    ax.set_title(f"(d) PathMNIST SDIV Grid\n(best={grid[best]:.1f}%)")

    # (e) DermaMNIST convergence map
    ax = axes[4]
    betas = sorted(df_dsurf.beta.unique())
    lams  = sorted(df_dsurf.lam.unique())
    grid  = np.full((len(betas),len(lams)), np.nan)
    for _,row in df_dsurf.iterrows():
        grid[betas.index(row.beta), lams.index(row.lam)] = row.accuracy*100
    degen = np.abs(grid - majority) < 0.5
    bg    = np.where(degen, 0.0, 1.0)
    cmap2 = LinearSegmentedColormap.from_list("dg",["#f4cccc","#d9f7d9"])
    ax.imshow(bg, cmap=cmap2, aspect="auto", vmin=0, vmax=1,
              extent=[-0.5,len(lams)-0.5,-0.5,len(betas)-0.5], origin="lower")
    for bi in range(len(betas)):
        for li in range(len(lams)):
            if not np.isnan(grid[bi,li]):
                ax.text(li, bi, f"{grid[bi,li]:.1f}", ha="center", va="center",
                        fontsize=8,
                        color="#cc0000" if degen[bi,li] else "#1a5c1a",
                        fontweight="bold" if degen[bi,li] else "normal")
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([str(l) for l in lams], fontsize=7)
    ax.set_yticks(range(len(betas))); ax.set_yticklabels([str(b) for b in betas], fontsize=7)
    ax.set_xlabel("Lambda"); ax.set_ylabel("Beta")
    ax.set_title("(e) DermaMNIST Convergence Map\n(red=degenerate, green=learning)")

    # (f) AUNRC ranking
    ax = axes[5]
    losses_aunrc = [l for l in COLORS if l in df_pnoise.loss.unique()]
    aunrcs = {}
    for loss in losses_aunrc:
        sub = df_pnoise[df_pnoise.loss==loss].sort_values("noise_rate")
        accs = sub.accuracy.values*100
        if len(accs) >= 2:
            aunrcs[loss] = np.trapz(accs, dx=10.0)/(10.0*(len(accs)-1))
    aunrcs = dict(sorted(aunrcs.items(), key=lambda x: x[1], reverse=True))
    for i,(loss,val) in enumerate(aunrcs.items()):
        ax.barh(loss, val, color=COLORS.get(loss,"#888"), edgecolor="white", lw=0.4)
    ax.set_xlabel("AUNRC (%)")
    ax.set_title("(f) PathMNIST AUNRC\nNoise-Robustness Ranking")
    ax.invert_yaxis()

    fig.suptitle("Robust Neural Learning via S-Divergence — Complete Benchmark Summary\n"
                 "(seed=42, single run; SDIV-opt = best beta/lambda from grid search)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    savefig(fig, "F14_master_summary_6panel.png")


# ══════════════════════════════════════════════════════════════════════════════
# F15 – DermaMNIST Clean Accuracy Bar
# ══════════════════════════════════════════════════════════════════════════════
def fig_dermamnist_clean_bar():
    df = pd.read_csv(f"{DATA_DIR}/dermamnist_noise_results.csv")
    clean = df[df.noise_rate.eq(0.0)].sort_values("accuracy", ascending=False)
    majority = 66.88
    losses = list(clean.loss); accs = list(clean.accuracy*100)
    cols   = [COLORS.get(l,"#888") for l in losses]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(losses)), accs, color=cols, edgecolor="white", lw=0.5)
    ax.axhline(majority, color="#cc0000", lw=1.2, ls="--",
               label=f"Majority-class baseline ({majority:.1f}%)")
    for bar, acc, loss in zip(bars, accs, losses):
        is_degen = abs(acc - majority) < 0.5
        ax.text(bar.get_x()+bar.get_width()/2, acc+0.1,
                f"{acc:.1f}%" + (" (deg.)" if is_degen else ""),
                ha="center", va="bottom", fontsize=8.5,
                color="#cc0000" if is_degen else "black")
    ax.set_xticks(range(len(losses)))
    ax.set_xticklabels(losses, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(majority-2, max(accs)+2)
    ax.set_title("DermaMNIST: Clean (0% Noise) Test Accuracy — All Loss Functions\n"
                 "(seed=42, single run; red dashes = majority-class floor)")
    ax.legend(**LEGEND_KW)
    savefig(fig, "F15_dermamnist_clean_accuracy_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("Generating all 15 publication-quality plots ...")
    print(f"Output: {OUT_DIR}")
    print("="*60)
    fig_pathmnist_noise()
    fig_dermamnist_noise()
    fig_pathmnist_fgsm()
    fig_dermamnist_fgsm()
    fig_pathmnist_fgsm_drop()
    fig_dermamnist_fgsm_drop()
    fig_pathmnist_clean_bar()
    fig_pathmnist_sdiv_heatmap()
    fig_dermamnist_sdiv_heatmap()
    fig_lambda_convergence_map()
    fig_aunrc_ranking()
    fig_nlp_bert()
    fig_sdiv_3d_surface()
    fig_master_summary()
    fig_dermamnist_clean_bar()
    print("="*60)
    print("All 15 figures saved successfully.")
