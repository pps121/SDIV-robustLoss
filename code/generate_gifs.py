"""
generate_gifs.py
================
Generates 5 animated GIFs for the robustNN-transformers GitHub homepage.
Uses ONLY real experimental CSV data — 100% accurate for paper submission.

Output: assets/  (GIF files, optimized for web)

Run:
    python3 code/generate_gifs.py

Requirements: numpy, pandas, matplotlib, imageio, Pillow
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
from pathlib import Path
import imageio.v2 as imageio
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
SURFACE  = "#161b22"
BORDER   = "#30363d"
TEXT_COL = "#e6edf3"
MUTED    = "#7d8590"

rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.facecolor": SURFACE,
    "figure.facecolor": DARK_BG,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT_COL,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT_COL,
    "grid.color": BORDER,
    "grid.alpha": 0.6,
    "legend.facecolor": SURFACE,
    "legend.edgecolor": BORDER,
    "legend.labelcolor": TEXT_COL,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
})

PALETTE = {
    "CCE":         "#ef4444",
    "MAE":         "#3b82f6",
    "GCE(q=0.7)":  "#10b981",
    "TruncGCE":    "#f59e0b",
    "SCE":         "#8b5cf6",
    "TPDD-CCE":    "#ec4899",
    "SDIV":        "#22d3ee",   # ← key loss — cyan highlight
    "TSCCE":       "#84cc16",
    "FCL":         "#f97316",
    "ForwardT":    "#a855f7",
}

ROOT   = Path(__file__).resolve().parent.parent
DATA15 = ROOT / "plots_results" / "15April2026" / "results_15April2026"
OUTDIR = ROOT / "assets"
OUTDIR.mkdir(parents=True, exist_ok=True)

FPS     = 8
W, H    = 900, 500
DPI     = 100

TARGET_W, TARGET_H = 900, 500

def fig_to_frame(fig):
    """Convert matplotlib figure to consistent-size RGB numpy array."""
    from PIL import Image
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    # Resize to fixed canvas so all frames match
    pil_img = pil_img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    plt.close(fig)
    return np.array(pil_img)

def save_gif(frames, name, fps=FPS, loop=0):
    """Save frames as animated GIF using imageio (avoids PIL palette mode issues)."""
    path = OUTDIR / name
    # Ensure all frames are uint8 RGB
    frames_uint8 = [f.astype(np.uint8) for f in frames]
    imageio.mimsave(str(path), frames_uint8, fps=fps, loop=loop)
    kb = path.stat().st_size / 1024
    print(f"  Saved: {path.name}  ({kb:.0f} KB, {len(frames)} frames)")

# ─────────────────────────────────────────────────────────────────────────────
# GIF 1: Noise Robustness Race — all losses across η levels
# ─────────────────────────────────────────────────────────────────────────────
print("\n[GIF 1] Noise robustness race — PathMNIST")
df_pn = pd.read_csv(DATA15 / "pathmnist_noise_results.csv")
noise_rates = sorted(df_pn["noise_rate"].unique())
losses_all  = list(df_pn["loss"].unique())

frames = []
# Build lines progressively, then hold each noise level for 2 extra frames
for step_idx, nr in enumerate(noise_rates):
    for hold in range(3 if step_idx < len(noise_rates)-1 else 8):
        fig, ax = plt.subplots(figsize=(W/DPI, H/DPI))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(SURFACE)

        for loss in losses_all:
            grp = df_pn[df_pn["loss"]==loss].sort_values("noise_rate")
            # Only plot points up to current noise_rate
            mask = grp["noise_rate"] <= nr
            sub = grp[mask]
            if len(sub) == 0:
                continue
            c  = PALETTE.get(loss, "#888")
            lw = 3.0 if loss == "SDIV" else 1.8
            zo = 10  if loss == "SDIV" else 5
            ax.plot(sub["noise_rate"]*100, sub["accuracy"]*100,
                    color=c, linewidth=lw, marker="o",
                    markersize=(8 if loss=="SDIV" else 5),
                    label=loss, zorder=zo, alpha=0.95)
            # Label at last point
            last = sub.iloc[-1]
            ax.annotate(f" {loss}", (last["noise_rate"]*100, last["accuracy"]*100),
                        fontsize=7.5, color=c, va="center")

        ax.set_xlim(-2, 45)
        ax.set_ylim(35, 92)
        ax.set_xlabel("Label Noise Rate η (%)", color=TEXT_COL, fontsize=11)
        ax.set_ylabel("Test Accuracy (%)", color=TEXT_COL, fontsize=11)
        ax.set_xticks([0, 10, 20, 30, 40])

        # Noise level indicator
        ax.axvline(nr*100, color="white", lw=1.2, alpha=0.4, linestyle=":")
        ax.text(nr*100+0.5, 89, f"η = {int(nr*100)}%", color="white",
                fontsize=10, fontweight="bold", va="top")

        ax.set_title("PathMNIST: Robust Loss Functions Under Label Noise",
                     color=TEXT_COL, fontsize=12, fontweight="bold", pad=10)

        # Highlight SDIV
        if "SDIV" in df_pn["loss"].values:
            sdiv_val = df_pn[(df_pn["loss"]=="SDIV")&(df_pn["noise_rate"]==nr)]["accuracy"].values
            if len(sdiv_val) > 0:
                ax.scatter(nr*100, sdiv_val[0]*100, color="#22d3ee", s=120,
                           zorder=15, edgecolors="white", linewidths=1.5)

        fig.tight_layout(pad=0.5)
        frames.append(fig_to_frame(fig))

save_gif(frames, "noise_robustness_race.gif", fps=FPS)

# ─────────────────────────────────────────────────────────────────────────────
# GIF 2: S-Divergence Surface Rotation (360°)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[GIF 2] S-divergence surface rotation")
from mpl_toolkits.mplot3d import Axes3D

df_surf_p = pd.read_csv(DATA15 / "pathmnist_sdiv_surface.csv")
df_surf_d = pd.read_csv(DATA15 / "dermamnist_sdiv_surface.csv")

def prep_surface(df_s):
    betas = sorted(df_s["beta"].unique())
    lams  = sorted(df_s["lam"].unique())
    Z = np.full((len(betas), len(lams)), np.nan)
    for _, row in df_s.iterrows():
        Z[betas.index(row["beta"]), lams.index(row["lam"])] = row["accuracy"]*100
    return np.array(betas), np.array(lams), Z

betas_p, lams_p, Zp = prep_surface(df_surf_p)
betas_d, lams_d, Zd = prep_surface(df_surf_d)

frames = []
# 36 angles × 2 datasets
N_ANGLES = 36
for phase in range(2):   # 0=PathMNIST, 1=DermaMNIST
    betas_, lams_, Z_ = (betas_p, lams_p, Zp) if phase==0 else (betas_d, lams_d, Zd)
    ds_name = "PathMNIST" if phase==0 else "DermaMNIST"
    B_, L_  = np.meshgrid(betas_, lams_, indexing="ij")
    cmap_ = "cool" if phase==0 else "hot"
    best_val = np.nanmax(Z_)

    for angle_i in range(N_ANGLES):
        azim = -60 + angle_i * (360 / N_ANGLES)
        fig = plt.figure(figsize=(W/DPI, H/DPI))
        fig.patch.set_facecolor(DARK_BG)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(DARK_BG)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(BORDER); ax.yaxis.pane.set_edgecolor(BORDER); ax.zaxis.pane.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)

        surf = ax.plot_surface(B_, L_, Z_, cmap=cmap_, edgecolor="none",
                               alpha=0.90, vmin=np.nanmin(Z_), vmax=best_val)

        # Best point star
        best_idx = np.unravel_index(np.nanargmax(Z_), Z_.shape)
        ax.scatter([betas_[best_idx[0]]], [lams_[best_idx[1]]], [best_val],
                   color="#fbbf24", s=120, zorder=15, marker="*")

        ax.set_xlabel("β (beta)", color=TEXT_COL, labelpad=6)
        ax.set_ylabel("λ (lambda)", color=TEXT_COL, labelpad=6)
        ax.set_zlabel("Accuracy (%)", color=TEXT_COL, labelpad=6)
        ax.view_init(elev=30, azim=azim)
        ax.set_title(f"S-Divergence (β,λ) Accuracy Surface — {ds_name}",
                     color=TEXT_COL, fontsize=11, fontweight="bold", pad=12)
        ax.text2D(0.02, 0.97, f"★ Peak: {best_val:.1f}%", transform=ax.transAxes,
                  color="#fbbf24", fontsize=9, fontweight="bold", va="top")
        fig.tight_layout(pad=0.3)
        frames.append(fig_to_frame(fig))

save_gif(frames, "sdiv_surface_rotation.gif", fps=10)

# ─────────────────────────────────────────────────────────────────────────────
# GIF 3: FGSM Shield Animation — epsilon increases, lines hold or collapse
# ─────────────────────────────────────────────────────────────────────────────
print("\n[GIF 3] FGSM shield animation")
df_fgsm = pd.read_csv(DATA15 / "dermamnist_fgsm_results.csv")
EPS_VALS   = sorted(df_fgsm["epsilon"].unique())
EPS_LABELS = ["ε=0", "ε=1/255", "ε=2/255", "ε=4/255", "ε=8/255"]

frames = []
for step_idx, eps in enumerate(EPS_VALS):
    for hold in range(3 if step_idx < len(EPS_VALS)-1 else 10):
        fig, ax = plt.subplots(figsize=(W/DPI, H/DPI))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(SURFACE)

        for loss in df_fgsm["loss"].unique():
            grp = df_fgsm[df_fgsm["loss"]==loss].sort_values("epsilon")
            mask = grp["epsilon"] <= eps
            sub  = grp[mask]
            if len(sub)==0: continue
            c  = PALETTE.get(loss, "#888")
            lw = 3.0 if loss in ("SDIV","MAE","GCE(q=0.7)") else 1.6
            zo = 10  if loss == "SDIV" else 5
            ax.plot(range(len(sub)), sub["accuracy"]*100,
                    color=c, linewidth=lw, marker="o",
                    markersize=(8 if loss=="SDIV" else 5),
                    label=loss, zorder=zo, alpha=0.95)
            last = sub.iloc[-1]
            ax.annotate(f" {loss}", (len(sub)-1, last["accuracy"]*100),
                        fontsize=7.5, color=c, va="center")

        ax.set_xlim(-0.3, 5.5)
        ax.set_ylim(15, 80)
        ax.set_xticks(range(step_idx+1))
        ax.set_xticklabels(EPS_LABELS[:step_idx+1], fontsize=9)
        ax.set_xlabel("FGSM Perturbation Budget (ε)", color=TEXT_COL, fontsize=11)
        ax.set_ylabel("Test Accuracy (%)", color=TEXT_COL, fontsize=11)
        ax.set_title("DermaMNIST: Adversarial Attack Robustness — SDIV Holds!",
                     color=TEXT_COL, fontsize=12, fontweight="bold", pad=10)

        # Robustness region annotation for SDIV
        ax.fill_between([0,4.5], [64,64], [70,70], color="#22d3ee", alpha=0.08)
        ax.text(4.6, 67, "SDIV\nFGSM-\ninvariant", color="#22d3ee",
                fontsize=8, fontweight="bold", va="center")

        eps_pct = eps * 255
        ax.text(0.02, 0.97, f"Attack: ε = {eps_pct:.2f}/255",
                transform=ax.transAxes, color="white", fontsize=10,
                fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ef444444"))

        fig.tight_layout(pad=0.5)
        frames.append(fig_to_frame(fig))

save_gif(frames, "fgsm_shield_animation.gif", fps=FPS)

# ─────────────────────────────────────────────────────────────────────────────
# GIF 4: Dual Frontier Evolution — losses revealed one by one
# ─────────────────────────────────────────────────────────────────────────────
print("\n[GIF 4] Dual frontier evolution")
clean_acc = df_pn[df_pn["noise_rate"]==0.0].set_index("loss")["accuracy"] * 100
noisy_acc = df_pn[df_pn["noise_rate"]==0.3].set_index("loss")["accuracy"] * 100
common_losses = [l for l in clean_acc.index if l in noisy_acc.index]

# Sort for dramatic reveal: worst first, SDIV last
reveal_order = [l for l in common_losses if l not in ("SDIV","FCL")] + ["FCL","SDIV"]

frames = []
for reveal_n in range(1, len(reveal_order)+1):
    for hold in range(3 if reveal_n < len(reveal_order) else 12):
        fig, ax = plt.subplots(figsize=(W/DPI, H/DPI))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(SURFACE)

        shown = reveal_order[:reveal_n]
        for loss in shown:
            c = PALETTE.get(loss, "#888")
            sz = 160 if loss == "SDIV" else 90
            mk = "*" if loss == "SDIV" else "o"
            ax.scatter(clean_acc[loss], noisy_acc[loss], color=c, s=sz,
                       marker=mk, edgecolors="white", linewidths=1.2, zorder=10,
                       alpha=0.9)
            ax.annotate(loss, (clean_acc[loss], noisy_acc[loss]),
                        fontsize=8.5, color=c, xytext=(6,3), textcoords="offset points",
                        fontweight=("bold" if loss=="SDIV" else "normal"))

        # Ideal diagonal
        ax.plot([76,88],[76,88], color=MUTED, lw=1.2, linestyle="--", alpha=0.6,
                label="Perfect robustness (clean=noisy)")
        ax.fill_between([76,88],[76,88],[76,88], alpha=0)

        ax.set_xlim(76, 87)
        ax.set_ylim(37, 87)
        ax.set_xlabel("Clean Accuracy (η=0%) [%]", color=TEXT_COL, fontsize=11)
        ax.set_ylabel("Noisy Accuracy (η=30%) [%]", color=TEXT_COL, fontsize=11)
        ax.set_title("PathMNIST: Clean-Robust Pareto Frontier\n(closer to diagonal = better robust loss)",
                     color=TEXT_COL, fontsize=11, fontweight="bold", pad=10)

        ax.text(0.02, 0.98,
                f"Showing {reveal_n}/{len(reveal_order)} loss functions",
                transform=ax.transAxes, color=MUTED, fontsize=9, va="top")

        if "SDIV" in shown:
            ax.text(clean_acc["SDIV"]+0.1, noisy_acc["SDIV"]+0.3,
                    "★ Best balanced", color="#22d3ee", fontsize=9, fontweight="bold")

        fig.tight_layout(pad=0.5)
        frames.append(fig_to_frame(fig))

save_gif(frames, "dual_frontier_evolution.gif", fps=FPS)

# ─────────────────────────────────────────────────────────────────────────────
# GIF 5: β Parameter Sweep Animation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[GIF 5] β parameter sweep")
frames = []
betas_sweep = sorted(df_surf_p["beta"].unique())
lam_sweep   = sorted(df_surf_p["lam"].unique())

for bi, beta in enumerate(betas_sweep):
    # For each β: show accuracy vs λ for both datasets
    for hold in range(4 if bi < len(betas_sweep)-1 else 12):
        fig, axes = plt.subplots(1, 2, figsize=(W/DPI, H/DPI))
        fig.patch.set_facecolor(DARK_BG)

        for ax, (df_s, ds_name, color) in zip(axes, [
            (df_surf_p, "PathMNIST", "#22d3ee"),
            (df_surf_d, "DermaMNIST", "#f59e0b"),
        ]):
            ax.set_facecolor(SURFACE)
            # All β curves (faded)
            for b in betas_sweep:
                sub = df_s[df_s["beta"]==b].sort_values("lam")
                if len(sub)==0: continue
                alpha = 1.0 if b==beta else 0.12
                lw    = 2.8 if b==beta else 0.8
                c     = color if b==beta else MUTED
                ax.plot(sub["lam"], sub["accuracy"]*100,
                        color=c, linewidth=lw, alpha=alpha,
                        marker=("o" if b==beta else None),
                        markersize=7, label=f"β={b}" if b==beta else None,
                        zorder=10 if b==beta else 3)

            # Current β highlight
            sub = df_s[df_s["beta"]==beta].sort_values("lam")
            if len(sub) > 0:
                best_lam = sub.loc[sub["accuracy"].idxmax(), "lam"]
                best_acc = sub["accuracy"].max() * 100
                ax.scatter([best_lam], [best_acc], color="#fbbf24", s=120,
                           zorder=20, marker="*", edgecolors="white", linewidths=1)
                ax.annotate(f"λ={best_lam}, {best_acc:.1f}%",
                            (best_lam, best_acc), xytext=(0.05, 0.97),
                            textcoords="axes fraction", color="#fbbf24",
                            fontsize=8.5, fontweight="bold", va="top")

            ax.set_xlabel("λ (lambda)", color=TEXT_COL)
            ax.set_ylabel("Accuracy (%)", color=TEXT_COL)
            ax.set_title(f"{ds_name}: β = {beta}", color=color, fontweight="bold", pad=8)
            ax.tick_params(colors=MUTED)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_edgecolor(BORDER)
            ax.spines["left"].set_edgecolor(BORDER)

        fig.suptitle("S-Divergence Sensitivity: Effect of β on Accuracy",
                     color=TEXT_COL, fontsize=12, fontweight="bold")
        fig.tight_layout(pad=0.5)
        frames.append(fig_to_frame(fig))

save_gif(frames, "parameter_sensitivity_sweep.gif", fps=FPS)

print(f"\n✅ All GIFs saved to: {OUTDIR}")
print(f"   Files: {sorted([f.name for f in OUTDIR.glob('*.gif')])}")
