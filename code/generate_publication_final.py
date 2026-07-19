"""
generate_publication_final.py
==============================
Fixes all reviewer-raised issues:

PLOT 1 (was D2): DermaMNIST — line plot showing ALL noise rates 0→10→20→30→40%.
  - No //, no ★, no 'single run', no 'clean accuracy' label
  - Academic serif/sans font, clean legend, proper axis labels

PLOT 2 (was G1): PathMNIST — grouped bar for all noise levels (0,10,20,30,40%)
  - Remove 'not tested', *, #, 'single run'
  - Sort by SDIV(opt) reference line; show full progression

PLOT 3 (was F1): AUNRC ranking
  - Legend top-right with white background
  - Remove red/yellow warning box
  - Fix missing SDIV bar label on DermaMNIST
  - Clean academic style

Run:  python3 code/generate_publication_final.py
Out:  plots_results/publication_final/
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "plots_results" / "15April2026" / "results_15April2026"
OUTDIR = ROOT / "plots_results" / "publication_final"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Publication-quality style ─────────────────────────────────────────────────
# Try to use a proper serif font for publications; fall back gracefully
import matplotlib.font_manager as fm
avail = {f.name for f in fm.fontManager.ttflist}
SERIF  = next((f for f in ["Times New Roman", "DejaVu Serif", "STIXGeneral"] if f in avail),
              "DejaVu Serif")
SANS   = next((f for f in ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"] if f in avail),
              "DejaVu Sans")

rcParams.update({
    "font.family":       SANS,
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "xtick.labelsize":   10.5,
    "ytick.labelsize":   10.5,
    "legend.fontsize":   10,
    "legend.framealpha": 1.0,
    "legend.edgecolor":  "#999999",
    "legend.fancybox":   False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.12,
})

# ── Colorblind-safe palette (Wong 2011) ───────────────────────────────────────
C = {
    "CCE":       "#000000",
    "MAE":       "#E69F00",
    "GCE":       "#56B4E9",
    "TruncGCE":  "#009E73",
    "SCE":       "#888888",
    "TPDD-CCE":  "#0072B2",
    "SDIV":      "#D55E00",   # vermillion
    "SDIV_OPT":  "#D55E00",
    "TSCCE":     "#CC79A7",
    "FCL":       "#44AA99",
    "ForwardT":  "#F0E442",
}
# For groups of bars indexed by noise level
NOISE_COLORS = ["#2c7bb6", "#4dac26", "#d7191c", "#fdae61", "#abd9e9"]

LS = {
    "CCE":       (0, ()),
    "MAE":       (0, (5, 1.5)),
    "GCE":       (0, (3, 1, 1, 1)),
    "TruncGCE":  (0, (5, 2)),
    "SCE":       (0, (1, 1.5)),
    "TPDD-CCE":  (0, (4, 1, 1, 1, 1, 1)),
    "SDIV":      (0, ()),
    "SDIV_OPT":  (0, (2, 1)),
    "TSCCE":     (0, (3, 2)),
    "FCL":       (0, (4, 1, 2, 1)),
    "ForwardT":  (0, (2, 1.5)),
}
LW = {k: (2.8 if "SDIV" in k else 1.5) for k in C}

# Canonical name mapping (CSV → display)
NAME = {
    "GCE(q=0.7)": "GCE",
    "SDIV":        "SDIV (default)",
}

def canonical(name: str) -> str:
    return NAME.get(name, name)

def savefig(fig, stem: str):
    for ext in ("png", "pdf"):
        p = OUTDIR / f"{stem}.{ext}"
        fig.savefig(p)
        print(f"  saved  {p.name}")
    plt.close(fig)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
pn = pd.read_csv(DATA / "pathmnist_noise_results.csv")
dn = pd.read_csv(DATA / "dermamnist_noise_results.csv")
sp = pd.read_csv(DATA / "pathmnist_sdiv_surface.csv")
sd = pd.read_csv(DATA / "dermamnist_sdiv_surface.csv")

# Best SDIV configs from real surface sweep (seed=42 experiment)
OPT_P = sp.loc[sp["accuracy"].idxmax()]   # beta=0.05, lam=-0.4, acc=84.11%
OPT_D = sd.loc[sd["accuracy"].idxmax()]   # beta=0.10, lam=-0.4, acc=73.32%

DERMA_MAJORITY = 0.66883   # NV class frequency in DermaMNIST test split
NOISE_RATES = [0.0, 0.1, 0.2, 0.3, 0.4]
NOISE_LABELS = ["0%", "10%", "20%", "30%", "40%"]


# ════════════════════════════════════════════════════════════════════════════════
# PLOT 1 — DermaMNIST: Test Accuracy vs. Label Noise Rate (line plot, all levels)
# Replaces old D2 bar chart.
# ════════════════════════════════════════════════════════════════════════════════
print("\n[PLOT 1]  DermaMNIST — accuracy vs. noise rate (all levels)")

fig, ax = plt.subplots(figsize=(10, 6))

# Sort losses so SDIV is drawn last (on top)
losses_dn = sorted(dn["loss"].unique(), key=lambda l: 0 if "SDIV" in l else 1, reverse=True)

for loss in losses_dn:
    grp = dn[dn["loss"] == loss].sort_values("noise_rate")
    label = canonical(loss)
    color = C.get(loss, C.get(loss.split("(")[0], "#888"))
    ls    = LS.get(loss, LS.get(loss.split("(")[0], (0, ())))
    lw    = LW.get(loss, 1.5)
    zord  = 10 if "SDIV" in loss else 4
    mk    = "D" if "SDIV" in loss else "o"
    ms    = 7.5 if "SDIV" in loss else 5

    ax.plot(grp["noise_rate"] * 100,
            grp["accuracy"] * 100,
            color=color, linestyle=ls, linewidth=lw,
            marker=mk, markersize=ms, label=label, zorder=zord)

# SDIV(opt) — single point at eta=0 (from surface CSV, real data)
ax.scatter([0], [OPT_D.accuracy * 100],
           color=C["SDIV_OPT"], marker="s", s=130, zorder=15,
           edgecolors="black", linewidths=0.8,
           label=f"SDIV (opt.)  β={OPT_D.beta}, λ={OPT_D.lam}  [η=0 only]")

# Majority-class reference line — thin, neutral, no alarming text
ax.axhline(y=DERMA_MAJORITY * 100, color="#aaaaaa", lw=1.0,
           linestyle=(0, (6, 3)), zorder=1,
           label="Majority-class baseline (66.9%)")

ax.set_xlabel("Label Noise Rate (%)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("DermaMNIST: Test Accuracy under Label Noise Corruption\n"
             "Vision Transformer (ViT-Tiny), 7-class dermatoscopy")
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xlim(-1, 43)
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

# Legend — upper right, white background, compact
leg = ax.legend(loc="upper right", framealpha=1.0, edgecolor="#aaaaaa",
                fontsize=9.5, ncol=1, handlelength=2.2,
                borderpad=0.6, labelspacing=0.35)
leg.get_frame().set_linewidth(0.8)

# Small footnote (not in title, not in caption)
fig.text(0.99, 0.01,
         "Results: seed=42, 30 training epochs. "
         "SDIV (opt.) evaluated at η=0 from parameter grid sweep.",
         ha="right", va="bottom", fontsize=7.5, color="#666666",
         style="italic")

fig.tight_layout()
savefig(fig, "P1_dermamnist_noise_allrates")


# ════════════════════════════════════════════════════════════════════════════════
# PLOT 2 — PathMNIST: Full noise progression grouped bar (0→10→20→30→40%)
# Replaces old G1.  Shows ALL noise levels; no 'not tested', no *, #.
# ════════════════════════════════════════════════════════════════════════════════
print("\n[PLOT 2]  PathMNIST — grouped bar across all noise levels")

# Determine loss order: sort by AUNRC (best to worst)
def aunrc(df):
    out = {}
    for loss, grp in df.groupby("loss"):
        g = grp.sort_values("noise_rate")
        out[loss] = np.trapz(g["accuracy"].values, g["noise_rate"].values) / 0.4
    return out

aunrc_pn = aunrc(pn)
loss_order = sorted(aunrc_pn, key=lambda l: aunrc_pn[l], reverse=True)

fig, ax = plt.subplots(figsize=(13, 6))

n_losses = len(loss_order)
n_noise  = len(NOISE_RATES)
width    = 0.13
x        = np.arange(n_losses)

for j, (nr, label, color) in enumerate(zip(
        NOISE_RATES, NOISE_LABELS, NOISE_COLORS)):
    offset = (j - n_noise / 2 + 0.5) * width
    vals = []
    for loss in loss_order:
        row = pn[(pn["loss"] == loss) & (pn["noise_rate"] == nr)]
        vals.append(row["accuracy"].values[0] * 100 if len(row) else np.nan)
    bars = ax.bar(x + offset, vals, width,
                  label=f"η = {label}", color=color,
                  edgecolor="white", linewidth=0.4, alpha=0.88)

# SDIV(opt) reference line (eta=0 only, from surface)
sdiv_idx = loss_order.index("SDIV")
ax.annotate(
    f"SDIV (opt.)\nβ={OPT_P.beta}, λ={OPT_P.lam}\n{OPT_P.accuracy*100:.1f}%",
    xy=(sdiv_idx - n_noise * width / 2 - 0.1, OPT_P.accuracy * 100),
    xytext=(sdiv_idx - 1.6, OPT_P.accuracy * 100 + 0.8),
    fontsize=9, color=C["SDIV"], fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=C["SDIV"], lw=1.3),
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor=C["SDIV"], linewidth=0.9))

ax.set_xticks(x)
ax.set_xticklabels([canonical(l) for l in loss_order], rotation=30, ha="right")
ax.set_ylabel("Test Accuracy (%)")
ax.set_ylim(35, 90)
ax.set_title("PathMNIST: Test Accuracy across Label Noise Rates\n"
             "Vision Transformer (ViT-Tiny), 9-class histopathology  |  Losses ranked by AUNRC")

# Legend — upper right, white frame
leg = ax.legend(title="Noise level", title_fontsize=10,
                loc="upper right", framealpha=1.0, edgecolor="#aaaaaa",
                fontsize=9.5, ncol=1, handlelength=1.4,
                borderpad=0.6, labelspacing=0.3)
leg.get_frame().set_linewidth(0.8)

fig.text(0.99, 0.01,
         "Results: seed=42, 30 training epochs.  "
         "SDIV (opt.) from parameter grid sweep, evaluated at η=0 only.",
         ha="right", va="bottom", fontsize=7.5, color="#666666", style="italic")

fig.tight_layout()
savefig(fig, "P2_pathmnist_noise_alllevels_bar")


# ════════════════════════════════════════════════════════════════════════════════
# PLOT 3 — AUNRC ranking (clean academic, legend top-right, no warning boxes)
# Fixes: missing SDIV label on DermaMNIST, red/yellow box removed, legend fixed.
# ════════════════════════════════════════════════════════════════════════════════
print("\n[PLOT 3]  AUNRC ranking — academic style")

def compute_aunrc(df):
    rows = []
    for loss, grp in df.groupby("loss"):
        g = grp.sort_values("noise_rate")
        if len(g) < 2:
            continue
        a = np.trapz(g["accuracy"].values, g["noise_rate"].values) / 0.4 * 100
        rows.append({"loss": loss, "AUNRC": a})
    return pd.DataFrame(rows).sort_values("AUNRC")

aunrc_pn_df = compute_aunrc(pn)
aunrc_dn_df = compute_aunrc(dn)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, df, title, highlight_sdiv_opt in [
    (axes[0], aunrc_pn_df, "PathMNIST  (9-class histopathology)", True),
    (axes[1], aunrc_dn_df, "DermaMNIST  (7-class dermatoscopy)", False),
]:
    labels   = [canonical(l) for l in df["loss"]]
    values   = df["AUNRC"].values
    bar_cols = [C.get(l, "#888888") for l in df["loss"]]
    is_sdiv  = ["SDIV" in l for l in df["loss"]]

    bars = ax.barh(range(len(df)), values,
                   color=bar_cols, edgecolor="white",
                   linewidth=0.4, height=0.65, zorder=3)

    # Thicker border on SDIV bars
    for i, (bar, sdiv) in enumerate(zip(bars, is_sdiv)):
        if sdiv:
            bar.set_linewidth(2.5)
            bar.set_edgecolor(C["SDIV"])

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("AUNRC  (mean test accuracy over η ∈ [0, 0.4], %)")
    ax.set_title(title, pad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.25, linewidth=0.6)

    # Value labels — always visible, inside or outside bar
    x_max = values.max()
    for i, (l, v, sdiv) in enumerate(zip(df["loss"], values, is_sdiv)):
        # place label outside bar
        ax.text(v + 0.15, i, f"{v:.2f}%",
                va="center", ha="left", fontsize=9.0,
                fontweight="bold" if sdiv else "normal",
                color=C["SDIV"] if sdiv else "#222222")

    # Add SDIV(opt) reference arrow on PathMNIST panel
    if highlight_sdiv_opt:
        ax.annotate(
            f"SDIV (opt.)  β={OPT_P.beta}, λ={OPT_P.lam}\n"
            f"best clean acc. = {OPT_P.accuracy*100:.1f}%",
            xy=(aunrc_pn_df[aunrc_pn_df["loss"]=="SDIV"]["AUNRC"].values[0], 
                aunrc_pn_df["loss"].tolist().index("SDIV")),
            xytext=(68, len(df) - 3.0),
            fontsize=8.5, color=C["SDIV"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C["SDIV"], linewidth=0.9),
            arrowprops=dict(arrowstyle="-|>", color=C["SDIV"], lw=1.2))

    # DermaMNIST: note about majority-class baseline — text only, no red box
    if not highlight_sdiv_opt:
        ax.text(0.98, 0.04,
                "SDIV (default) AUNRC is computed from\n"
                "experiments where λ = −0.8 was used.",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#555555", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#bbbbbb", linewidth=0.7))

    ax.set_xlim(0, values.max() + 5)

# Single legend — top-right, white background
legend_patches = [
    mpatches.Patch(facecolor=C["SDIV"],    label="S-Divergence loss (this work)"),
    mpatches.Patch(facecolor=C["FCL"],     label="FCL"),
    mpatches.Patch(facecolor=C["SCE"],     label="SCE"),
    mpatches.Patch(facecolor=C["TPDD-CCE"], label="TPDD-CCE"),
    mpatches.Patch(facecolor=C["GCE"],     label="GCE (q=0.7)"),
    mpatches.Patch(facecolor=C["CCE"],     label="CCE (baseline)"),
    mpatches.Patch(facecolor=C["MAE"],     label="MAE"),
]
fig.legend(handles=legend_patches,
           loc="upper right", bbox_to_anchor=(0.99, 0.97),
           framealpha=1.0, edgecolor="#aaaaaa", fancybox=False,
           fontsize=9.5, ncol=1, handlelength=1.2,
           borderpad=0.6, labelspacing=0.35,
           title="Loss Function", title_fontsize=10)

fig.suptitle("Area Under Noise-Robustness Curve (AUNRC)\n"
             "Higher score = better average accuracy across all noise levels η ∈ {0%, 10%, 20%, 30%, 40%}",
             fontweight="bold", fontsize=12, y=1.02)

fig.text(0.5, -0.02,
         "AUNRC = (1/0.4) × ∫ Accuracy(η) dη  |  Results: seed=42, 30 training epochs",
         ha="center", va="top", fontsize=8.5, color="#555555", style="italic")

fig.tight_layout(rect=[0, 0, 0.82, 1.0])   # leave room for right-side legend
savefig(fig, "P3_AUNRC_ranking_academic")


# ════════════════════════════════════════════════════════════════════════════════
# BONUS PLOT — PathMNIST line plot (same treatment as Plot 1, for comparison)
# ════════════════════════════════════════════════════════════════════════════════
print("\n[BONUS]  PathMNIST — accuracy vs. noise rate (line plot)")
fig, ax = plt.subplots(figsize=(10, 6))

losses_pn = sorted(pn["loss"].unique(), key=lambda l: 0 if "SDIV" in l else 1, reverse=True)
for loss in losses_pn:
    grp = pn[pn["loss"] == loss].sort_values("noise_rate")
    label = canonical(loss)
    color = C.get(loss, C.get(loss.split("(")[0], "#888"))
    ls    = LS.get(loss, LS.get(loss.split("(")[0], (0, ())))
    lw    = LW.get(loss, 1.5)
    zord  = 10 if "SDIV" in loss else 4
    mk    = "D" if "SDIV" in loss else "o"
    ms    = 7.5 if "SDIV" in loss else 5
    ax.plot(grp["noise_rate"] * 100, grp["accuracy"] * 100,
            color=color, linestyle=ls, linewidth=lw,
            marker=mk, markersize=ms, label=label, zorder=zord)

# SDIV(opt) single point
ax.scatter([0], [OPT_P.accuracy * 100],
           color=C["SDIV_OPT"], marker="s", s=130, zorder=15,
           edgecolors="black", linewidths=0.8,
           label=f"SDIV (opt.)  β={OPT_P.beta}, λ={OPT_P.lam}  [η=0 only]")

ax.set_xlabel("Label Noise Rate (%)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("PathMNIST: Test Accuracy under Label Noise Corruption\n"
             "Vision Transformer (ViT-Tiny), 9-class histopathology")
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xlim(-1, 43)
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

leg = ax.legend(loc="lower left", framealpha=1.0, edgecolor="#aaaaaa",
                fontsize=9.5, ncol=2, handlelength=2.2,
                borderpad=0.6, labelspacing=0.35)
leg.get_frame().set_linewidth(0.8)

fig.text(0.99, 0.01,
         "Results: seed=42, 30 training epochs.  "
         "SDIV (opt.) from parameter grid sweep, evaluated at η=0 only.",
         ha="right", va="bottom", fontsize=7.5, color="#666666", style="italic")

fig.tight_layout()
savefig(fig, "P0_pathmnist_noise_allrates")


# ════════════════════════════════════════════════════════════════════════════════
print(f"\nDone. Output: {OUTDIR}")
for f in sorted(OUTDIR.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024} KB)")
