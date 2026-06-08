"""
generate_publication_plots.py
==============================
Generates ALL publication-quality figures for the robustNN-transformers paper.
Uses ONLY real experimental CSV data — no simulated or placeholder values.

Output: plots_results/publication/  (PNG @ 300 DPI + PDF vector)

Run:
    python3 code/generate_publication_plots.py

Requirements: numpy, pandas, matplotlib (all in requirements-dev.txt)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# ── Publication style ─────────────────────────────────────────────────────────
rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9.5,
    "ytick.labelsize":    9.5,
    "legend.fontsize":    9,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
})

# ── Colorblind-safe palette (Wong 2011, 8 colors) ────────────────────────────
PALETTE = {
    "CCE":          "#000000",   # black
    "MAE":          "#E69F00",   # orange
    "GCE(q=0.7)":  "#56B4E9",   # sky blue
    "TruncGCE":     "#009E73",   # green
    "SCE":          "#F0E442",   # yellow (darker border)
    "TPDD-CCE":     "#0072B2",   # blue
    "SDIV":         "#D55E00",   # vermillion  ← KEY LOSS
    "TSCCE":        "#CC79A7",   # pink
    "FCL":          "#999999",   # grey
    "ForwardT":     "#44AA99",   # teal
}
LINESTYLES = {
    "CCE":          (0, ()),          # solid
    "MAE":          (0, (5, 1)),      # dashed
    "GCE(q=0.7)":  (0, (3, 1, 1, 1)),# dashdot
    "TruncGCE":     (0, (5, 2)),
    "SCE":          (0, (1, 1)),      # dotted
    "TPDD-CCE":     (0, (5, 1, 1, 1, 1, 1)),
    "SDIV":         (0, ()),          # solid (thick)
    "TSCCE":        (0, (3, 2)),
    "FCL":          (0, (4, 1, 2, 1)),
    "ForwardT":     (0, (2, 1)),
}
LINEWIDTHS = {k: 2.8 if k == "SDIV" else 1.6 for k in PALETTE}
MARKERS    = {k: ("D" if k == "SDIV" else "o") for k in PALETTE}
MARKER_SZ  = {k: (7 if k == "SDIV" else 5) for k in PALETTE}

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
DATA15  = ROOT / "plots_results" / "15April2026" / "results_15April2026"
DATA30  = ROOT / "plots_results" / "30March2026"
MULTIMOD= ROOT / "results_multimodal_vision"
OUTDIR  = ROOT / "plots_results" / "publication"
OUTDIR.mkdir(parents=True, exist_ok=True)

def savefig(fig, stem):
    for ext in ("png", "pdf"):
        p = OUTDIR / f"{stem}.{ext}"
        fig.savefig(p)
        print(f"  Saved: {p.name}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# A1. Label Noise Robustness — PathMNIST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[A1] Noise robustness — PathMNIST")
df_pn = pd.read_csv(DATA15 / "pathmnist_noise_results.csv")
noise_rates = sorted(df_pn["noise_rate"].unique())

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for loss, grp in df_pn.groupby("loss"):
    grp = grp.sort_values("noise_rate")
    c = PALETTE.get(loss, "#888888")
    ls_key = loss
    ax.plot(
        grp["noise_rate"] * 100,
        grp["accuracy"] * 100,
        color=c,
        linestyle=LINESTYLES.get(ls_key, (0, ())),
        linewidth=LINEWIDTHS.get(ls_key, 1.6),
        marker=MARKERS.get(ls_key, "o"),
        markersize=MARKER_SZ.get(ls_key, 5),
        label=loss,
        zorder=(10 if loss == "SDIV" else 5),
    )

ax.set_xlabel("Label Noise Rate η (%)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("PathMNIST: Label Noise Robustness (ViT, seed=42)", fontweight="bold")
ax.set_xticks([0, 10, 20, 30, 40])
ax.legend(ncol=2, loc="lower left", fontsize=8.5)
# Annotate SDIV
sdiv_row = df_pn[df_pn["loss"] == "SDIV"].sort_values("noise_rate")
ax.annotate("SDIV", xy=(sdiv_row.iloc[0]["noise_rate"]*100, sdiv_row.iloc[0]["accuracy"]*100),
            xytext=(5, 82.8), fontsize=8, color=PALETTE["SDIV"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE["SDIV"], lw=1.2))
fig.tight_layout()
savefig(fig, "A1_noise_pathmnist")

# ─────────────────────────────────────────────────────────────────────────────
# A2. Label Noise Robustness — DermaMNIST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[A2] Noise robustness — DermaMNIST")
df_dn = pd.read_csv(DATA15 / "dermamnist_noise_results.csv")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for loss, grp in df_dn.groupby("loss"):
    grp = grp.sort_values("noise_rate")
    c = PALETTE.get(loss, "#888888")
    ax.plot(grp["noise_rate"]*100, grp["accuracy"]*100,
            color=c, linestyle=LINESTYLES.get(loss, (0,())),
            linewidth=LINEWIDTHS.get(loss, 1.6),
            marker=MARKERS.get(loss, "o"), markersize=MARKER_SZ.get(loss, 5),
            label=loss, zorder=(10 if loss == "SDIV" else 5))

ax.set_xlabel("Label Noise Rate η (%)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("DermaMNIST: Label Noise Robustness (ViT, seed=42)", fontweight="bold")
ax.set_xticks([0, 10, 20, 30, 40])
ax.legend(ncol=2, loc="lower left", fontsize=8.5)
fig.tight_layout()
savefig(fig, "A2_noise_dermamnist")

# ─────────────────────────────────────────────────────────────────────────────
# A3. Side-by-side comparison (paper figure)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[A3] Noise robustness — combined figure")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

for ax, (df, title) in zip(axes, [
    (df_pn, "PathMNIST (9-class histopathology)"),
    (df_dn, "DermaMNIST (7-class dermatoscopy)")
]):
    for loss, grp in df.groupby("loss"):
        grp = grp.sort_values("noise_rate")
        c = PALETTE.get(loss, "#888888")
        ax.plot(grp["noise_rate"]*100, grp["accuracy"]*100,
                color=c, linestyle=LINESTYLES.get(loss, (0,())),
                linewidth=LINEWIDTHS.get(loss, 1.6),
                marker=MARKERS.get(loss, "o"), markersize=MARKER_SZ.get(loss, 5),
                label=loss, zorder=(10 if loss == "SDIV" else 5))
    ax.set_xlabel("Label Noise Rate η (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks([0, 10, 20, 30, 40])

# Shared legend below
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
fig.suptitle("Robust Loss Functions: Label Noise Robustness Comparison", fontweight="bold", fontsize=13)
fig.tight_layout(rect=[0, 0.07, 1, 0.97])
savefig(fig, "A3_noise_combined")

# ─────────────────────────────────────────────────────────────────────────────
# B1. FGSM Adversarial Robustness — DermaMNIST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[B1] FGSM adversarial — DermaMNIST")
df_fgsm = pd.read_csv(DATA15 / "dermamnist_fgsm_results.csv")
EPS_LABELS = ["0", "1/255", "2/255", "4/255", "8/255"]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for loss, grp in df_fgsm.groupby("loss"):
    grp = grp.sort_values("epsilon")
    c = PALETTE.get(loss, "#888888")
    ax.plot(range(len(grp)), grp["accuracy"]*100,
            color=c, linestyle=LINESTYLES.get(loss, (0,())),
            linewidth=LINEWIDTHS.get(loss, 1.6),
            marker=MARKERS.get(loss, "o"), markersize=MARKER_SZ.get(loss, 5),
            label=loss, zorder=(10 if loss == "SDIV" else 5))

ax.set_xticks(range(5))
ax.set_xticklabels(EPS_LABELS)
ax.set_xlabel("FGSM Perturbation Budget ε (× 1/255)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("DermaMNIST: FGSM Adversarial Robustness (ViT, seed=42)", fontweight="bold")
ax.legend(ncol=2, loc="lower left", fontsize=8.5)

# Shade region of SDIV stability
sdiv_fgsm = df_fgsm[df_fgsm["loss"]=="SDIV"].sort_values("epsilon")
ax.axhline(y=sdiv_fgsm["accuracy"].mean()*100, color=PALETTE["SDIV"],
           lw=0.8, alpha=0.3, linestyle=":")
ax.annotate("SDIV flat\n(FGSM invariant)", xy=(3, 66.9), xytext=(1.2, 60),
            fontsize=8, color=PALETTE["SDIV"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE["SDIV"], lw=1.2))
fig.tight_layout()
savefig(fig, "B1_fgsm_dermamnist")

# ─────────────────────────────────────────────────────────────────────────────
# B2. FGSM accuracy drop bar chart
# ─────────────────────────────────────────────────────────────────────────────
print("\n[B2] FGSM accuracy drop bar")
eps_min = df_fgsm["epsilon"].min()
eps_max = df_fgsm["epsilon"].max()
acc_clean  = df_fgsm[df_fgsm["epsilon"]==eps_min].set_index("loss")["accuracy"]
acc_attack = df_fgsm[df_fgsm["epsilon"]==eps_max].set_index("loss")["accuracy"]
drop = ((acc_clean - acc_attack) * 100).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
colors = [PALETTE.get(l, "#888888") for l in drop.index]
bars = ax.barh(drop.index, drop.values, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Accuracy Drop (percentage points) — ε=0 → ε=8/255")
ax.set_title("DermaMNIST: Total FGSM Accuracy Drop by Loss Function", fontweight="bold")
ax.axvline(0, color="black", lw=0.8)
for bar, val, loss in zip(bars, drop.values, drop.index):
    label = f"−{val:.1f}pp" if val > 0.1 else "0 pp (invariant ✓)"
    color = "white" if val > 10 else "black"
    ax.text(bar.get_width() + 0.3 if val < 1 else bar.get_width() - 0.5,
            bar.get_y() + bar.get_height()/2,
            label, va="center", ha="left" if val < 1 else "right",
            fontsize=8.5, color="black" if val < 1 else "white", fontweight="bold")
fig.tight_layout()
savefig(fig, "B2_fgsm_drop_bar")

# ─────────────────────────────────────────────────────────────────────────────
# C1. S-Divergence Surface — PathMNIST (3D publication)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[C1] S-divergence 3D surface — PathMNIST")
df_surf_p = pd.read_csv(DATA15 / "pathmnist_sdiv_surface.csv")
df_surf_d = pd.read_csv(DATA15 / "dermamnist_sdiv_surface.csv")

fig = plt.figure(figsize=(13, 5))
for idx, (df_s, title, col) in enumerate([
    (df_surf_p, "PathMNIST", "Blues_r"),
    (df_surf_d, "DermaMNIST", "Oranges_r"),
], start=1):
    ax = fig.add_subplot(1, 2, idx, projection="3d")
    betas = sorted(df_s["beta"].unique())
    lams  = sorted(df_s["lam"].unique())
    # Build grid (fill missing with nan)
    Z = np.full((len(betas), len(lams)), np.nan)
    for _, row in df_s.iterrows():
        bi = betas.index(row["beta"])
        li = lams.index(row["lam"])
        Z[bi, li] = row["accuracy"] * 100
    B, L = np.meshgrid(betas, lams, indexing="ij")
    # Mask NaN for surface
    from matplotlib import cm
    surf = ax.plot_surface(B, L, Z, cmap=col, edgecolor="none", alpha=0.92,
                           vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    # Best point
    best_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
    ax.scatter([betas[best_idx[0]]], [lams[best_idx[1]]], [np.nanmax(Z)],
               color="#D55E00", s=80, zorder=10, label=f"Best: {np.nanmax(Z):.1f}%")
    ax.set_xlabel("β (beta)", labelpad=8)
    ax.set_ylabel("λ (lambda)", labelpad=8)
    ax.set_zlabel("Accuracy (%)", labelpad=8)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.view_init(elev=28, azim=-55)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, label="Accuracy (%)")
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("S-Divergence Parameter Sensitivity: Accuracy over (β, λ) Grid", fontweight="bold", fontsize=12)
fig.tight_layout()
savefig(fig, "C1_sdiv_surface_3d")

# ─────────────────────────────────────────────────────────────────────────────
# C2. S-Divergence Heatmap — both datasets
# ─────────────────────────────────────────────────────────────────────────────
print("\n[C2] S-divergence heatmap")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (df_s, title) in zip(axes, [
    (df_surf_p, "PathMNIST"), (df_surf_d, "DermaMNIST")
]):
    betas = sorted(df_s["beta"].unique())
    lams  = sorted(df_s["lam"].unique())
    Z = np.full((len(betas), len(lams)), np.nan)
    for _, row in df_s.iterrows():
        bi = betas.index(row["beta"]); li = lams.index(row["lam"])
        Z[bi, li] = row["accuracy"] * 100
    im = ax.imshow(Z, cmap="viridis", aspect="auto",
                   vmin=np.nanmin(Z), vmax=np.nanmax(Z), origin="lower")
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l:.1f}" for l in lams], rotation=45)
    ax.set_yticks(range(len(betas))); ax.set_yticklabels([str(b) for b in betas])
    ax.set_xlabel("λ (lambda)"); ax.set_ylabel("β (beta)")
    ax.set_title(f"{title}: Accuracy (%) over (β, λ) grid", fontweight="bold")
    # Annotate each cell
    for bi in range(len(betas)):
        for li in range(len(lams)):
            if not np.isnan(Z[bi, li]):
                ax.text(li, bi, f"{Z[bi,li]:.1f}", ha="center", va="center",
                        fontsize=7.5, color="white" if Z[bi,li] < np.nanmean(Z) else "black")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")

# Star best point on each
for ax, df_s in zip(axes, [df_surf_p, df_surf_d]):
    betas = sorted(df_s["beta"].unique())
    lams  = sorted(df_s["lam"].unique())
    best = df_s.loc[df_s["accuracy"].idxmax()]
    bi, li = betas.index(best["beta"]), lams.index(best["lam"])
    ax.plot(li, bi, "r*", markersize=14, label=f"Best: {best['accuracy']*100:.1f}%")
    ax.legend(fontsize=8)

fig.suptitle("S-Divergence Parameter Grid — Heatmap", fontweight="bold", fontsize=12)
fig.tight_layout()
savefig(fig, "C2_sdiv_heatmap")

# ─────────────────────────────────────────────────────────────────────────────
# D1. Dual frontier — clean vs noisy
# ─────────────────────────────────────────────────────────────────────────────
print("\n[D1] Dual frontier scatter")
# PathMNIST η=30%
clean_acc_p  = df_pn[df_pn["noise_rate"]==0.0].set_index("loss")["accuracy"] * 100
noisy_acc_p  = df_pn[df_pn["noise_rate"]==0.3].set_index("loss")["accuracy"] * 100
clean_acc_d  = df_dn[df_dn["noise_rate"]==0.0].set_index("loss")["accuracy"] * 100
noisy_acc_d  = df_dn[df_dn["noise_rate"]==0.3].set_index("loss")["accuracy"] * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (c_acc, n_acc, title) in zip(axes, [
    (clean_acc_p, noisy_acc_p, "PathMNIST (η=30%)"),
    (clean_acc_d, noisy_acc_d, "DermaMNIST (η=30%)"),
]):
    common = c_acc.index.intersection(n_acc.index)
    for loss in common:
        c = PALETTE.get(loss, "#888888")
        ax.scatter(c_acc[loss], n_acc[loss], color=c, s=(120 if loss=="SDIV" else 70),
                   zorder=(10 if loss=="SDIV" else 5),
                   marker=("*" if loss=="SDIV" else "o"),
                   edgecolors="black", linewidths=0.6, label=loss)
        ax.annotate(loss, (c_acc[loss], n_acc[loss]), fontsize=7.2,
                    xytext=(4, 3), textcoords="offset points")
    # Ideal line (clean=noisy)
    lim = (min(c_acc.min(), n_acc.min())-1, max(c_acc.max(), n_acc.max())+1)
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.4, label="Clean=Noisy (ideal)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Clean Accuracy (η=0%) [%]")
    ax.set_ylabel("Noisy Accuracy (η=30%) [%]")
    ax.set_title(f"Clean-Robust Frontier: {title}", fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, ncol=2)

fig.suptitle("Dual Frontier: Clean Accuracy vs Robustness Under Label Noise", fontweight="bold", fontsize=12)
fig.tight_layout()
savefig(fig, "D1_dual_frontier")

# ─────────────────────────────────────────────────────────────────────────────
# D2. Summary multi-panel (paper main figure)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[D2] Summary multi-panel")
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# Panel 1: PathMNIST noise
ax1 = fig.add_subplot(gs[0, 0])
for loss, grp in df_pn.groupby("loss"):
    grp = grp.sort_values("noise_rate")
    ax1.plot(grp["noise_rate"]*100, grp["accuracy"]*100,
             color=PALETTE.get(loss,"#888"), linewidth=LINEWIDTHS.get(loss,1.6),
             marker=MARKERS.get(loss,"o"), markersize=4,
             linestyle=LINESTYLES.get(loss,(0,())),
             label=loss, zorder=10 if loss=="SDIV" else 5)
ax1.set_xlabel("η (%)"); ax1.set_ylabel("Accuracy (%)")
ax1.set_title("(a) PathMNIST — Noise", fontweight="bold")
ax1.set_xticks([0,10,20,30,40])

# Panel 2: DermaMNIST noise
ax2 = fig.add_subplot(gs[0, 1])
for loss, grp in df_dn.groupby("loss"):
    grp = grp.sort_values("noise_rate")
    ax2.plot(grp["noise_rate"]*100, grp["accuracy"]*100,
             color=PALETTE.get(loss,"#888"), linewidth=LINEWIDTHS.get(loss,1.6),
             marker=MARKERS.get(loss,"o"), markersize=4,
             linestyle=LINESTYLES.get(loss,(0,())),
             label=loss, zorder=10 if loss=="SDIV" else 5)
ax2.set_xlabel("η (%)"); ax2.set_ylabel("Accuracy (%)")
ax2.set_title("(b) DermaMNIST — Noise", fontweight="bold")
ax2.set_xticks([0,10,20,30,40])

# Panel 3: FGSM
ax3 = fig.add_subplot(gs[0, 2])
for loss, grp in df_fgsm.groupby("loss"):
    grp = grp.sort_values("epsilon")
    ax3.plot(range(len(grp)), grp["accuracy"]*100,
             color=PALETTE.get(loss,"#888"), linewidth=LINEWIDTHS.get(loss,1.6),
             marker=MARKERS.get(loss,"o"), markersize=4,
             linestyle=LINESTYLES.get(loss,(0,())),
             label=loss, zorder=10 if loss=="SDIV" else 5)
ax3.set_xticks(range(5)); ax3.set_xticklabels(EPS_LABELS, fontsize=8)
ax3.set_xlabel("FGSM ε"); ax3.set_ylabel("Accuracy (%)")
ax3.set_title("(c) DermaMNIST — FGSM", fontweight="bold")

# Panel 4: S-DIV heatmap PathMNIST
ax4 = fig.add_subplot(gs[1, 0])
betas = sorted(df_surf_p["beta"].unique())
lams  = sorted(df_surf_p["lam"].unique())
Zp = np.full((len(betas), len(lams)), np.nan)
for _, row in df_surf_p.iterrows():
    Zp[betas.index(row["beta"]), lams.index(row["lam"])] = row["accuracy"]*100
im4 = ax4.imshow(Zp, cmap="viridis", aspect="auto", origin="lower",
                 vmin=np.nanmin(Zp), vmax=np.nanmax(Zp))
ax4.set_xticks(range(len(lams))); ax4.set_xticklabels([f"{l:.1f}" for l in lams], rotation=45, fontsize=8)
ax4.set_yticks(range(len(betas))); ax4.set_yticklabels(betas, fontsize=8)
ax4.set_xlabel("λ (lambda)"); ax4.set_ylabel("β (beta)")
ax4.set_title("(d) PathMNIST — (β,λ) Grid", fontweight="bold")
plt.colorbar(im4, ax=ax4, label="Accuracy (%)")

# Panel 5: Drop bar
ax5 = fig.add_subplot(gs[1, 1])
drop_sorted = drop.sort_values(ascending=False)
cols = [PALETTE.get(l, "#888") for l in drop_sorted.index]
ax5.barh(range(len(drop_sorted)), drop_sorted.values, color=cols, edgecolor="white")
ax5.set_yticks(range(len(drop_sorted))); ax5.set_yticklabels(drop_sorted.index, fontsize=8)
ax5.set_xlabel("Accuracy Drop (pp)")
ax5.set_title("(e) FGSM Total Drop (ε=0→8/255)", fontweight="bold")

# Panel 6: Dual frontier
ax6 = fig.add_subplot(gs[1, 2])
common_p = clean_acc_p.index.intersection(noisy_acc_p.index)
for loss in common_p:
    c = PALETTE.get(loss, "#888")
    ax6.scatter(clean_acc_p[loss], noisy_acc_p[loss], color=c,
                s=100 if loss=="SDIV" else 60,
                marker="*" if loss=="SDIV" else "o",
                edgecolors="black", linewidths=0.5,
                label=loss, zorder=10 if loss=="SDIV" else 5)
    ax6.annotate(loss[:6], (clean_acc_p[loss], noisy_acc_p[loss]),
                 fontsize=6.5, xytext=(3,2), textcoords="offset points")
lim = (76, 86)
ax6.plot(lim, lim, "k--", lw=0.8, alpha=0.4)
ax6.set_xlim(lim); ax6.set_ylim((76,86))
ax6.set_xlabel("Clean Acc (%)"); ax6.set_ylabel("Noisy Acc (η=30%) (%)")
ax6.set_title("(f) PathMNIST — Clean-Robust Frontier", fontweight="bold")

# Shared legend at bottom
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.01), framealpha=0.9)
fig.suptitle("Robust Neural Learning via S-Divergence: Complete Benchmark Summary",
             fontweight="bold", fontsize=13)
savefig(fig, "D2_summary_multipanel")

# ─────────────────────────────────────────────────────────────────────────────
# E1. MNIST + early experiments (30March data)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[E1] Early MNIST/FGSM experiments")
try:
    df_fgsm_mnist = pd.read_csv(DATA30 / "FGSM-adversarial-attacks.csv")
    df_clean_mnist = pd.read_csv(DATA30 / "Clean-data performance.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # MNIST FGSM
    ax = axes[0]
    for loss, grp in df_fgsm_mnist.groupby("loss"):
        grp = grp.sort_values("epsilon")
        c = PALETTE.get(loss.upper(), PALETTE.get(loss, "#888"))
        ax.plot(range(len(grp)), grp["accuracy"]*100,
                color=c, linewidth=2.0, marker="o", markersize=5, label=loss)
    ax.set_xticks(range(5)); ax.set_xticklabels(EPS_LABELS)
    ax.set_xlabel("FGSM ε"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("MNIST: FGSM Adversarial Robustness", fontweight="bold")
    ax.legend(fontsize=8.5)

    # CIFAR-10 clean comparison
    ax = axes[1]
    df_clean_sorted = df_clean_mnist.sort_values("accuracy", ascending=False)
    colors_c = [PALETTE.get(l, "#888") for l in df_clean_sorted["loss"]]
    bars = ax.bar(range(len(df_clean_sorted)), df_clean_sorted["accuracy"]*100,
                  color=colors_c, edgecolor="white")
    ax.set_xticks(range(len(df_clean_sorted)))
    ax.set_xticklabels(df_clean_sorted["loss"], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("CIFAR-10: Clean-Label Accuracy by Loss Function", fontweight="bold")
    ax.set_ylim(0, max(df_clean_sorted["accuracy"]*100)+5)

    fig.suptitle("Early Experiments: MNIST & CIFAR-10 Benchmarks", fontweight="bold", fontsize=12)
    fig.tight_layout()
    savefig(fig, "E1_early_experiments")
except Exception as e:
    print(f"  Skipped E1: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# F1. Complete loss function comparison — bar chart at η=0% and η=40%
# ─────────────────────────────────────────────────────────────────────────────
print("\n[F1] Clean vs heavy noise bar comparison")
losses_ordered = ["CCE","MAE","GCE(q=0.7)","TruncGCE","SCE","TPDD-CCE","SDIV","TSCCE","FCL","ForwardT"]
df_p0  = df_pn[df_pn["noise_rate"]==0.0].set_index("loss")["accuracy"] * 100
df_p4  = df_pn[df_pn["noise_rate"]==0.4].set_index("loss")["accuracy"] * 100

losses_available = [l for l in losses_ordered if l in df_p0.index]
x = np.arange(len(losses_available))
w = 0.38

fig, ax = plt.subplots(figsize=(11, 5))
bars0 = ax.bar(x - w/2, [df_p0.get(l, np.nan) for l in losses_available], w,
               label="Clean (η=0%)", color=[PALETTE.get(l,"#888") for l in losses_available],
               edgecolor="white", linewidth=0.7, alpha=0.9)
bars4 = ax.bar(x + w/2, [df_p4.get(l, np.nan) for l in losses_available], w,
               label="Noisy (η=40%)", color=[PALETTE.get(l,"#888") for l in losses_available],
               edgecolor="black", linewidth=0.7, alpha=0.55, hatch="//")

ax.set_xticks(x); ax.set_xticklabels(losses_available, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Test Accuracy (%)"); ax.set_ylim(35, 90)
ax.set_title("PathMNIST: Clean vs Heavy Noise (η=40%) Comparison", fontweight="bold")
ax.legend(fontsize=10)

# Annotate SDIV
sidx = losses_available.index("SDIV")
ax.annotate("★ SDIV", xy=(sidx+w/2, df_p4.get("SDIV",0)),
            xytext=(sidx+1.2, df_p4.get("SDIV",0)+1.5),
            fontsize=8.5, color=PALETTE["SDIV"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE["SDIV"]))
fig.tight_layout()
savefig(fig, "F1_clean_vs_noise_bar")

print(f"\n✅ All publication plots saved to: {OUTDIR}")
print(f"   Files: {len(list(OUTDIR.glob('*.png')))} PNG, {len(list(OUTDIR.glob('*.pdf')))} PDF")
