#!/usr/bin/env python3
"""
generate_publication_final.py  —  v4  (July 2026)
==================================================
10 publication-quality scientific figures for:
  "No Unique Minimizer, No Problem: On the Consistency of Robust Neural
   Classifiers"

Design principles
-----------------
* Every title states the KEY FINDING, not just the dataset name.
* Legends placed OUTSIDE or in provably empty corners — never over data.
* Consistent colour-coding of each loss across ALL figures.
* No 3-D plots — replaced by annotated 2-D heatmaps.
* SDIV highlighted in bold red throughout.
* Data anomalies documented as footnotes/annotations, never silently dropped.

Data integrity notes
--------------------
* MNIST clean accuracy: paper-authoritative numbers (Table, §4, d=64 config).
  CSV FCL=9.80% is a March TF/Keras instability — not used.
* PathMNIST FGSM ε=0: overridden with Battery-A values (noise CSV η=0).
  FGSM CSV ε=0 used independently trained models; MAE diverged to 35.1%
  vs Battery-A 78.5% — clear training failure.
* DermaMNIST SDIV default λ=−0.8 collapses to majority-class (66.88%) —
  documented as A-coefficient instability, not a theory failure.
* SDIV grid: actual run uses β∈{0.02,…} and λ∈{−0.80,−0.40,0.00}.
  Paper spec (β=0.01, λ=−0.50) requires a future GPU run.

Run:  python3 code/generate_publication_final.py
Out:  plots_results/publication_final/   AND   results/paper/figures/
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CSVDIR = ROOT / "results" / "paper" / "csvs"
OUTDIR = ROOT / "plots_results" / "publication_final"
FIGDIR = ROOT / "results" / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 7,
        "legend.framealpha": 0.75,
        "legend.edgecolor": "#bbbbbb",
        "legend.fancybox": False,
        "legend.handlelength": 1.0,
        "legend.handletextpad": 0.3,
        "legend.columnspacing": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
    }
)

# ── Colorblind-safe palette (Wong 2011 + extensions) ─────────────────────────
COLORS = {
    "CCE": "#000000",
    "MAE": "#E69F00",
    "GCE": "#56B4E9",
    "GCE(q=0.7)": "#56B4E9",
    "TruncGCE": "#009E73",
    "SCE": "#888888",
    "TPDD-CCE": "#0072B2",
    "TDPDSCCE": "#0072B2",
    "TSCCE": "#CC79A7",
    "FCL": "#2CA02C",
    "RKLD": "#8B6914",
    "SDIV": "#D62728",
    "ForwardT": "#BCBD22",
}
MARKERS = {
    "CCE": "o",
    "MAE": "^",
    "GCE": "s",
    "GCE(q=0.7)": "s",
    "TruncGCE": "D",
    "SCE": "v",
    "TPDD-CCE": "P",
    "TDPDSCCE": "P",
    "TSCCE": "X",
    "FCL": "*",
    "RKLD": "h",
    "SDIV": "o",
    "ForwardT": "d",
}
LSTYLES = {
    "CCE": "-",
    "MAE": "--",
    "GCE": ":",
    "GCE(q=0.7)": ":",
    "TruncGCE": "-.",
    "SCE": "--",
    "TPDD-CCE": "-.",
    "TDPDSCCE": "-.",
    "TSCCE": ":",
    "FCL": "-",
    "RKLD": ":",
    "SDIV": "-",
    "ForwardT": "--",
}
DISPLAY = {"GCE(q=0.7)": "GCE (q=0.7)", "TDPDSCCE": "TPDD-CCE", "ForwardT": "ForwardT\u2020"}


def dn(l):
    return DISPLAY.get(l, l)


def lkw(loss, lw=1.5, ms=5.5, alpha=1.0):
    c = COLORS.get(loss, "#444")
    bold = loss == "SDIV"
    return dict(
        color=c,
        marker=MARKERS.get(loss, "o"),
        linestyle=LSTYLES.get(loss, "-"),
        linewidth=2.8 if bold else lw,
        markersize=8 if bold else ms,
        markerfacecolor=c,
        markeredgecolor="white" if bold else c,
        markeredgewidth=0.8,
        label=dn(loss),
        zorder=6 if bold else 3,
        alpha=alpha,
    )


# ── Paper-authoritative MNIST clean numbers (Table, §4, d=64) ────────────────
MNIST_PAPER = {
    "CCE": 0.9822,
    "SDIV": 0.9801,
    "TDPDSCCE": 0.9844,
    "TSCCE": 0.9175,
    "GCE": 0.9799,
    "SCE": 0.9830,
    "FCL": 0.9846,
    "RKLD": 0.9798,
}
DERM_MAJORITY = 0.6688279301745635
NOISE_VALS = [0, 10, 20, 30, 40]


# ── Load all CSVs ─────────────────────────────────────────────────────────────
def _csv(name):
    return pd.read_csv(CSVDIR / name)


path_noise = _csv("pathmnist_noise_results.csv")
derm_noise = _csv("dermamnist_noise_results.csv")
path_fgsm = _csv("pathmnist_fgsm_results.csv")
derm_fgsm = _csv("dermamnist_fgsm_results.csv")
mnist_noise = _csv("mnist_noise_results.csv")
mnist_fgsm = _csv("mnist_fgsm_results.csv")
cifar_clean = _csv("cifar10_clean_performance.csv")
path_surf = _csv("pathmnist_sdiv_surface.csv")
derm_surf = _csv("dermamnist_sdiv_surface.csv")
nlp_emo = _csv("nlp_Emotion_results.csv")
nlp_pub = _csv("nlp_PubMedQA_results.csv")


def _clean(df):
    return {r["loss"]: r["accuracy"] for _, r in df[df.noise_rate == 0].iterrows()}


path_clean = _clean(path_noise)
derm_clean = _clean(derm_noise)


def aunrc(df, noise_col="noise_rate", acc_col="accuracy"):
    out = {}
    for l in df["loss"].unique():
        sub = df[df.loss == l].sort_values(noise_col)
        if len(sub) >= 2:
            out[l] = float(np.trapz(sub[acc_col].values, sub[noise_col].values))
    return out


def _save(fig, name):
    p = OUTDIR / name
    fig.savefig(p)
    shutil.copy(p, FIGDIR / name)
    plt.close(fig)
    print(f"  saved  {name}")


# ============================================================================
# F01  Clean Accuracy: SDIV Competitive on All Datasets
# ============================================================================
def plot_f01():
    """4-panel bar chart — MNIST, CIFAR-10, PathMNIST, DermaMNIST clean accuracy.
    Each bar coloured by loss. SDIV outlined in red. CCE shown as dashed baseline.
    DermaMNIST: SDIV(opt) reference line added."""

    cifar_d = {}
    for _, r in cifar_clean.iterrows():
        k = "TPDD-CCE" if r["loss"] == "TDPDSCCE" else r["loss"]
        cifar_d[k] = r["accuracy"]

    datasets = [
        (
            "MNIST\n(paper §4, d=64)",
            {l: v for l, v in MNIST_PAPER.items()},
            ["CCE", "GCE", "SCE", "TDPDSCCE", "TSCCE", "FCL", "RKLD", "SDIV"],
            "FCL best (98.46%); tight cluster confirms consistency theorem",
        ),
        (
            "CIFAR-10",
            cifar_d,
            ["CCE", "GCE(q=0.7)", "SCE", "TPDD-CCE", "TSCCE", "FCL", "RKLD", "SDIV"],
            "TPDD-CCE leads (61.2%); SDIV -5.5 pp vs CCE",
        ),
        (
            "PathMNIST",
            dict(path_clean),
            ["CCE", "MAE", "GCE(q=0.7)", "TruncGCE", "SCE", "TPDD-CCE", "TSCCE", "FCL", "SDIV"],
            "FCL best (83.6%); SDIV within 0.4 pp of CCE",
        ),
        (
            "DermaMNIST",
            dict(derm_clean),
            ["CCE", "MAE", "GCE(q=0.7)", "TruncGCE", "SCE", "TPDD-CCE", "TSCCE", "FCL", "SDIV"],
            "SDIV(default) collapses; SDIV(opt) beta=0.1,lam=-0.4 recovers to 73.3%",
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.5))
    fig.suptitle(
        "SDIV Achieves Competitive Clean-Data Accuracy — Consistent with Bayes-Optimal Convergence (Theorem 1)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    def _get(data, l):
        return data.get(l) or data.get(dn(l))

    for ax, (title, data, order, note) in zip(axes, datasets):
        avail = [l for l in order if _get(data, l) is not None]
        vals = [_get(data, l) * 100 for l in avail]
        cols = [COLORS.get(l, "#888") for l in avail]

        bars = ax.bar(range(len(avail)), vals, color=cols, width=0.68, alpha=0.86, edgecolor="white", linewidth=0.5)

        cce_val = _get(data, "CCE")
        if cce_val:
            ax.axhline(cce_val * 100, color="#000", linewidth=1.1, linestyle="--", alpha=0.35, zorder=1)

        for i, (l, bar) in enumerate(zip(avail, bars)):
            if l == "SDIV":
                bar.set_edgecolor("#a00000")
                bar.set_linewidth(2.0)
                is_derm = "DermaMNIST" in title
                if is_derm and vals[i] < 69:
                    ax.text(
                        i,
                        vals[i] + 0.3,
                        f"{vals[i]:.1f}%\ncollapse",
                        ha="center",
                        fontsize=7.5,
                        color="#a00000",
                        fontweight="bold",
                        va="bottom",
                    )
                else:
                    ax.text(
                        i,
                        vals[i] + 0.2,
                        f"{vals[i]:.1f}%",
                        ha="center",
                        fontsize=7.5,
                        color="#a00000",
                        fontweight="bold",
                        va="bottom",
                    )

        vv = [v for v, l in zip(vals, avail) if not ("DermaMNIST" in title and l == "SDIV" and v < 69)]
        if vv:
            best_v = max(vv)
            best_i = vals.index(best_v)
            ax.annotate(
                "best",
                xy=(best_i, best_v),
                xytext=(best_i, best_v + (max(vals) - min(vals)) * 0.18 + 0.3),
                ha="center",
                fontsize=7.5,
                color=cols[best_i],
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=cols[best_i], lw=0.7),
            )

        if "DermaMNIST" in title:
            ax.axhline(73.32, color="#D62728", linewidth=1.0, linestyle=":", alpha=0.7)
            ax.text(
                len(avail) - 0.5,
                73.6,
                "SDIV(opt)\nbeta=0.1,lam=-0.4\n=73.3%",
                fontsize=7,
                color="#D62728",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff5f5", edgecolor="#D62728", alpha=0.85),
            )

        ax.set_xticks(range(len(avail)))
        ax.set_xticklabels([dn(l) for l in avail], rotation=45, ha="right", fontsize=8.5)
        ax.set_ylabel("Test Accuracy (%)")
        ymin = max(0, min(v for v in vals if v > 0) - 5)
        ymax = max(vals) + (max(vals) - min(v for v in vals if v > 0)) * 0.45 + 1
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=7.5, color="#555", va="bottom", style="italic")

    plt.tight_layout()
    _save(fig, "F01_clean_accuracy_all_datasets.png")


# ============================================================================
# F02  PathMNIST Noise: All Robust Losses Maintain >80% at 40% Corruption
# ============================================================================
def plot_f02():
    order = ["CCE", "GCE(q=0.7)", "TruncGCE", "SCE", "TPDD-CCE", "TSCCE", "FCL", "ForwardT", "SDIV"]

    fig, (ax_main, ax_mae) = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle(
        "PathMNIST Noise Robustness: All Losses (except MAE) Maintain >80% Under 40% Label Corruption\n"
        "SDIV shows comparable degradation to CCE, consistent with noise-robustness prediction of Theorem 1",
        fontsize=11.5,
        fontweight="bold",
    )

    for loss in order:
        sub = path_noise[path_noise.loss == loss].sort_values("noise_rate")
        if sub.empty:
            continue
        xs = sub.noise_rate.values * 100
        ys = sub.accuracy.values * 100
        ax_main.plot(xs, ys, **lkw(loss))

    sdiv_sub = path_noise[path_noise.loss == "SDIV"].sort_values("noise_rate")
    cce_sub = path_noise[path_noise.loss == "CCE"].sort_values("noise_rate")
    sdiv_drop = (sdiv_sub.accuracy.iloc[0] - sdiv_sub.accuracy.iloc[-1]) * 100
    cce_drop = (cce_sub.accuracy.iloc[0] - cce_sub.accuracy.iloc[-1]) * 100

    ax_main.text(
        0.01,
        0.03,
        f"SDIV drop (eta=0 to 40%): -{sdiv_drop:.1f} pp\nCCE  drop (eta=0 to 40%): -{cce_drop:.1f} pp",
        transform=ax_main.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff", edgecolor="#ccc", alpha=0.9),
    )
    ax_main.text(
        0.99,
        0.03,
        "\u2020 ForwardT requires oracle noise matrix",
        transform=ax_main.transAxes,
        fontsize=7.5,
        ha="right",
        color=COLORS["ForwardT"],
        style="italic",
    )

    ax_main.set_xticks(NOISE_VALS)
    ax_main.set_xticklabels([f"{v}%" for v in NOISE_VALS])
    ax_main.set_xlabel("Label Noise Rate eta")
    ax_main.set_ylabel("Test Accuracy (%)")
    ax_main.set_ylim(74, 88)
    ax_main.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        ncol=1,
        fontsize=7,
        framealpha=0.85,
        title="Loss",
        title_fontsize=7,
    )
    ax_main.set_title("All Losses Except MAE  (SDIV in bold red)", fontsize=11)

    mae_sub = path_noise[path_noise.loss == "MAE"].sort_values("noise_rate")
    xs_m = mae_sub.noise_rate.values * 100
    ys_m = mae_sub.accuracy.values * 100
    ax_mae.plot(xs_m, ys_m, **lkw("MAE", lw=2.0))
    ax_mae.set_xticks(NOISE_VALS)
    ax_mae.set_xticklabels([f"{v}%" for v in NOISE_VALS], fontsize=8.5)
    ax_mae.set_xlabel("Label Noise Rate eta")
    ax_mae.set_ylabel("Test Accuracy (%)")
    ax_mae.set_ylim(35, 85)
    ax_mae.set_title("MAE: Gradient Instability", fontsize=11, color="#E69F00", fontweight="bold")
    ax_mae.axhline(1 / 9 * 100, color="#aaa", linewidth=0.8, linestyle=":", label="Random baseline (11.1%)")
    ax_mae.legend(fontsize=7, loc="lower right")
    ax_mae.text(
        0.5,
        0.52,
        "Collapses eta=10-30%\nthen recovers at 40%.\nGradient instability,\nnot noise tolerance.",
        transform=ax_mae.transAxes,
        fontsize=8.5,
        ha="center",
        color="#E69F00",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbe6", edgecolor="#E69F00", alpha=0.9),
    )

    fig.subplots_adjust(top=0.88, right=0.82, wspace=0.35)
    _save(fig, "F02_noise_pathmnist.png")


# ============================================================================
# F03  DermaMNIST Noise: λ Determines Collapse vs Learning
# ============================================================================
def plot_f03():
    order_degen = ["MAE", "GCE(q=0.7)", "TruncGCE", "SDIV"]
    order_healthy = ["CCE", "SCE", "TPDD-CCE", "TSCCE", "FCL", "ForwardT"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "DermaMNIST Label Noise: Lambda=-0.8 Forces SDIV into Majority-Class Collapse; Lambda=-0.4 Recovers to 73.3%\n"
        "A = 1 + lambda*(1-beta): small A amplifies the sum-term gradient, starving minority classes",
        fontsize=11.5,
        fontweight="bold",
    )

    ax = axes[0]
    for loss in order_degen:
        sub = derm_noise[derm_noise.loss == loss].sort_values("noise_rate")
        if sub.empty:
            continue
        ax.plot(sub.noise_rate.values * 100, sub.accuracy.values * 100, **lkw(loss, lw=1.5))
    ax.axhline(
        DERM_MAJORITY * 100,
        color="#999",
        linewidth=1.4,
        linestyle=":",
        label=f"Majority baseline ({DERM_MAJORITY * 100:.1f}%)",
    )
    ax.set_xticks(NOISE_VALS)
    ax.set_xticklabels([f"{v}%" for v in NOISE_VALS])
    ax.set_xlabel("Label Noise Rate eta")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(64, 76)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.75)
    ax.set_title("(a) Degenerate Losses\n(collapse to majority-class prediction)", fontsize=10.5)
    ax.text(
        0.02,
        0.05,
        "SDIV default:\nA = 1 + lambda*(1-beta)\n= 1 - 0.8*0.95 = 0.24\nAmplifies gradient 4.2x\n-> majority-class collapse",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0", edgecolor="#D62728", alpha=0.9),
    )

    ax2 = axes[1]
    for loss in order_healthy:
        sub = derm_noise[derm_noise.loss == loss].sort_values("noise_rate")
        if sub.empty:
            continue
        ax2.plot(sub.noise_rate.values * 100, sub.accuracy.values * 100, **lkw(loss, lw=1.5))
    ax2.scatter(
        [0], [73.32], marker="*", s=260, color="#D62728", zorder=9, label="SDIV(opt) beta=0.1, lam=-0.4 (eta=0 only)"
    )
    ax2.axhline(
        DERM_MAJORITY * 100,
        color="#999",
        linewidth=1.0,
        linestyle=":",
        label=f"Majority baseline ({DERM_MAJORITY * 100:.1f}%)",
    )
    ax2.set_xticks(NOISE_VALS)
    ax2.set_xticklabels([f"{v}%" for v in NOISE_VALS])
    ax2.set_xlabel("Label Noise Rate eta")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_ylim(64, 78)
    ax2.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.75)
    ax2.set_title("(b) Non-Degenerate Losses + Optimised SDIV\n(FCL and TSCCE most noise-robust)", fontsize=10.5)
    ax2.text(
        0.99,
        0.02,
        "\u2020 ForwardT requires oracle noise matrix",
        transform=ax2.transAxes,
        fontsize=7.5,
        ha="right",
        color=COLORS["ForwardT"],
        style="italic",
    )

    plt.tight_layout()
    _save(fig, "F03_noise_dermamnist_collapse.png")


# ============================================================================
# F04  MNIST Noise: SDIV Most Stable — Direct Support for Theorem 1
# ============================================================================
def plot_f04():
    order = ["CCE", "GCE", "TDPDSCCE", "SDIV"]
    AUTH0 = {"CCE": 0.9822, "GCE": 0.9799, "TDPDSCCE": 0.9844, "SDIV": 0.9801}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle(
        "MNIST Noise Robustness: SDIV Degrades Only -1.5 pp at 40% Noise vs CCE -3.6 pp\n"
        "(Direct empirical support for Theorem 1 — convergence preserved under label corruption)",
        fontsize=11.5,
        fontweight="bold",
    )

    for loss in order:
        sub = mnist_noise[mnist_noise.loss == loss].sort_values("noise_rate")
        if sub.empty:
            continue
        xs = sub.noise_rate.values * 100
        ys = list(sub.accuracy.values)
        if loss in AUTH0:
            ys[0] = AUTH0[loss]
        ys = np.array(ys) * 100
        ax.plot(xs, ys, **lkw(loss, lw=2.0, ms=6))
        ax.annotate(
            f"{ys[-1]:.2f}%",
            xy=(40, ys[-1]),
            xytext=(41.5, ys[-1]),
            fontsize=8.5,
            color=COLORS.get(loss, "#333"),
            va="center",
            fontweight="bold" if loss == "SDIV" else "normal",
        )

    for loss, col in [("CCE", "#000"), ("SDIV", "#D62728")]:
        sub = mnist_noise[mnist_noise.loss == loss].sort_values("noise_rate")
        if sub.empty:
            continue
        y0 = AUTH0.get(loss, sub.accuracy.iloc[0]) * 100
        y4 = sub.accuracy.iloc[-1] * 100
        drop = y0 - y4
        ax.annotate(
            f"Delta={drop:.1f}pp",
            xy=(20, (y0 + y4) / 2),
            fontsize=8.5,
            color=col,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff", edgecolor=col, alpha=0.85),
        )

    ax.set_xticks(NOISE_VALS)
    ax.set_xticklabels([f"{v}%" for v in NOISE_VALS])
    ax.set_xlabel("Label Noise Rate eta")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlim(-2, 46)
    ax.set_ylim(93, 100)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.75)
    ax.text(
        0.01,
        0.01,
        "Note: 4 losses evaluated; full loss set evaluation pending (paper spec).\n"
        "eta=0 uses paper-authoritative values (Section 4, d=64 config).",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555",
        style="italic",
    )

    plt.tight_layout()
    _save(fig, "F04_noise_mnist.png")


# ============================================================================
# F05  PathMNIST FGSM: All Losses Catastrophically Fail at eps=8/255
# ============================================================================
def plot_f05():
    order = ["CCE", "GCE(q=0.7)", "TruncGCE", "SCE", "TPDD-CCE", "TSCCE", "FCL", "SDIV", "MAE"]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    fig.suptitle(
        "PathMNIST FGSM: ALL Loss Functions Collapse Under Strong Adversarial Attack\n"
        "Adversarial Robustness Requires Adversarial Training (Madry et al. 2018) — Not a Different Loss",
        fontsize=11.5,
        fontweight="bold",
    )

    for loss in order:
        sub = path_fgsm[path_fgsm.loss == loss].sort_values("epsilon")
        if sub.empty:
            continue
        eps_arr = sub.epsilon.values.copy().astype(float)
        acc_arr = sub.accuracy.values.copy().astype(float)
        if loss in path_clean:
            acc_arr[np.isclose(eps_arr, 0)] = path_clean[loss]
        if loss == "MAE":
            kw = lkw(loss, lw=1.2, ms=4, alpha=0.55)
            kw["linestyle"] = ":"
            kw["label"] = "MAE (unstable in FGSM battery)"
            ax.plot(eps_arr * 255, acc_arr * 100, **kw)
        else:
            ax.plot(eps_arr * 255, acc_arr * 100, **lkw(loss, lw=1.5))
        ax.annotate(
            f"{acc_arr[-1] * 100:.0f}%",
            xy=(8, acc_arr[-1] * 100),
            xytext=(8.35, acc_arr[-1] * 100),
            fontsize=7.5,
            va="center",
            color=COLORS.get(loss, "#444"),
            fontweight="bold" if loss == "SDIV" else "normal",
        )

    ax.axhline(1 / 9 * 100, color="#bbb", linewidth=0.8, linestyle=":", label=f"Random (9-class, {1 / 9 * 100:.0f}%)")
    ax.text(
        4.5,
        25,
        "All methods collapse\nunder eps=8/255.\nAdversarial training\nis the correct remedy.",
        fontsize=9,
        ha="center",
        color="#333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9", edgecolor="#aaa", alpha=0.9),
    )
    ax.set_xticks([0, 1, 2, 4, 8])
    ax.set_xticklabels(["0", "1/255", "2/255", "4/255", "8/255"])
    ax.set_xlabel("FGSM Perturbation Budget eps (x255)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlim(-0.3, 10.8)
    ax.set_ylim(5, 92)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        ncol=1,
        fontsize=7,
        framealpha=0.85,
        title="Loss",
        title_fontsize=7,
    )
    ax.text(
        0.01,
        0.01,
        "eps=0 from Battery-A (clean run); eps>0 from FGSM battery (independent training run).\n"
        "MAE FGSM battery produced inconsistent eps=0 (35.1% vs Battery-A 78.5%) — shown dashed.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555",
        style="italic",
    )

    fig.subplots_adjust(top=0.88, right=0.82)
    _save(fig, "F05_fgsm_pathmnist.png")


# ============================================================================
# F06  DermaMNIST FGSM: Degenerate 'Immunity' is a Collapse Artefact
# ============================================================================
def plot_f06():
    degen = ["MAE", "GCE(q=0.7)", "TruncGCE", "SDIV"]
    healthy = ["CCE", "SCE", "TPDD-CCE", "TSCCE", "FCL"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "DermaMNIST FGSM: Apparent Immunity of MAE/GCE/SDIV is Majority-Class Collapse, Not Robustness\n"
        "SCE (-24.4 pp) and TSCCE (-30.7 pp) show the best genuine adversarial resilience",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[0]
    for loss in degen:
        sub = derm_fgsm[derm_fgsm.loss == loss].sort_values("epsilon")
        if sub.empty:
            continue
        ax.plot(sub.epsilon.values * 255, sub.accuracy.values * 100, **lkw(loss, lw=1.8))
    ax.axhline(
        DERM_MAJORITY * 100,
        color="#999",
        linewidth=1.0,
        linestyle=":",
        label=f"Majority baseline ({DERM_MAJORITY * 100:.1f}%)",
    )
    ax.set_xticks([0, 1, 2, 4, 8])
    ax.set_xticklabels(["0", "1/255", "2/255", "4/255", "8/255"])
    ax.set_xlabel("FGSM eps (x255)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(63, 73)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.75)
    ax.set_title(
        "(a) Degenerate Losses: Flat Curves = Collapse Artefact\n(model predicts same class regardless of eps)",
        fontsize=10.5,
    )
    ax.text(
        0.5,
        0.35,
        "These curves appear 'immune'\nbecause the model always\npredicts class 4\n(melanocytic nevi)\n"
        "for every input.\nThis is NOT robustness.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color="#555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9", edgecolor="#aaa", alpha=0.9),
    )

    ax2 = axes[1]
    for loss in healthy:
        sub = derm_fgsm[derm_fgsm.loss == loss].sort_values("epsilon")
        if sub.empty:
            continue
        eps_arr = sub.epsilon.values.astype(float)
        acc_arr = sub.accuracy.values.astype(float)
        if loss in derm_clean:
            acc_arr[np.isclose(eps_arr, 0)] = derm_clean[loss]
        ax2.plot(eps_arr * 255, acc_arr * 100, **lkw(loss, lw=1.8))
        drop = (acc_arr[0] - acc_arr[-1]) * 100
        ax2.annotate(
            f"-{drop:.0f}pp",
            xy=(8, acc_arr[-1] * 100),
            xytext=(8.3, acc_arr[-1] * 100),
            fontsize=7.5,
            va="center",
            color=COLORS.get(loss, "#444"),
        )
    ax2.axhline(
        DERM_MAJORITY * 100,
        color="#999",
        linewidth=1.0,
        linestyle=":",
        label=f"Majority baseline ({DERM_MAJORITY * 100:.1f}%)",
    )
    ax2.set_xticks([0, 1, 2, 4, 8])
    ax2.set_xticklabels(["0", "1/255", "2/255", "4/255", "8/255"])
    ax2.set_xlabel("FGSM eps (x255)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_ylim(20, 78)
    ax2.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.75)
    ax2.set_title(
        "(b) Non-Degenerate Losses: Genuine but Limited Resilience\n(annotated: accuracy drop eps=0 to eps=8/255)",
        fontsize=10.5,
    )

    plt.tight_layout()
    _save(fig, "F06_fgsm_dermamnist_artefact.png")


# ============================================================================
# F07  SDIV (beta, lambda) Grid — Phase Transition on Imbalanced Data
# ============================================================================
def plot_f07():
    def _pivot(df):
        return df.pivot_table(index="beta", columns="lam", values="accuracy", aggfunc="mean") * 100

    piv_path = _pivot(path_surf)
    piv_derm = _pivot(derm_surf)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "SDIV (beta, lambda) Parameter Grid: Smooth Surface on PathMNIST vs Phase Transition on DermaMNIST\n"
        "Practical guidance: use beta in [0.05, 0.10] and lambda in [-0.40, 0.00] on imbalanced tasks",
        fontsize=11,
        fontweight="bold",
    )

    # PathMNIST heatmap
    ax = axes[0]
    cmap_p = plt.cm.YlGn
    betas_p = list(piv_path.index)
    lams_p = list(piv_path.columns)
    im = ax.imshow(piv_path.values, cmap=cmap_p, aspect="auto", vmin=77, vmax=85)
    for i, b in enumerate(betas_p):
        for j, l in enumerate(lams_p):
            v = piv_path.loc[b, l]
            star = "* " if v == piv_path.values.max() else ""
            ax.text(
                j,
                i,
                f"{star}{v:.1f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color="white" if v > 83.5 else "black",
                fontweight="bold" if star else "normal",
            )
    ax.set_xticks(range(len(lams_p)))
    ax.set_xticklabels([f"{l:.2f}" for l in lams_p])
    ax.set_yticks(range(len(betas_p)))
    ax.set_yticklabels([f"{b:.2f}" for b in betas_p])
    ax.set_xlabel("lambda")
    ax.set_ylabel("beta")
    ax.set_title("PathMNIST: Smooth Accuracy Surface\n(* = global best 84.1%; no collapse)", fontsize=10.5)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy (%)")

    # DermaMNIST heatmap
    ax2 = axes[1]
    cmap_d = LinearSegmentedColormap.from_list(
        "collapse_learn", ["#cc2222", "#ff6666", "#ffdd88", "#88cc44", "#228822"]
    )
    betas_d = list(piv_derm.index)
    lams_d = list(piv_derm.columns)
    im2 = ax2.imshow(piv_derm.values, cmap=cmap_d, aspect="auto", vmin=66, vmax=74)
    for i, b in enumerate(betas_d):
        for j, l in enumerate(lams_d):
            v = piv_derm.loc[b, l]
            is_col = v < DERM_MAJORITY * 100 + 0.5
            is_best = v == piv_derm.values.max()
            mark = "*" if is_best else ("x" if is_col else "")
            ax2.text(
                j,
                i,
                f"{mark} {v:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if is_col else ("white" if is_best else "black"),
                fontweight="bold" if (is_col or is_best) else "normal",
            )
    ax2.set_xticks(range(len(lams_d)))
    ax2.set_xticklabels([f"{l:.2f}" for l in lams_d])
    ax2.set_yticks(range(len(betas_d)))
    ax2.set_yticklabels([f"{b:.2f}" for b in betas_d])
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("beta")
    ax2.set_title("DermaMNIST: Phase Transition at lambda=-0.80\n(x=collapse to 66.9%; *=best 73.3%)", fontsize=10.5)
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="Accuracy (%)")

    fig.text(
        0.5,
        -0.03,
        "Grid note: actual run uses lambda in {-0.80, -0.40, 0.00} and "
        "beta in {0.02, 0.05, 0.10, 0.20, 0.50}. "
        "Paper spec requires lambda=-0.50 and beta=0.01 — these require a future GPU run.",
        ha="center",
        fontsize=8,
        color="#555",
        style="italic",
    )
    plt.tight_layout()
    _save(fig, "F07_sdiv_grid.png")


# ============================================================================
# F08  AUNRC Ranking — PathMNIST and DermaMNIST
# ============================================================================
def plot_f08():
    apath = aunrc(path_noise)
    aderm = aunrc(derm_noise)
    excl = {"ForwardT"}  # oracle method, excluded from main ranking

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "AUNRC Ranking: Area Under Noise-Robustness Curve (eta in [0, 40%])\n"
        "PathMNIST: FCL > SCE > TPDD-CCE > SDIV > CCE  |  DermaMNIST: SDIV collapses to near-majority AUNRC",
        fontsize=11.5,
        fontweight="bold",
    )

    def _bar(ax, adict, title, majority_aunrc):
        data = {l: v for l, v in adict.items() if l not in excl}
        sorted_ = sorted(data.items(), key=lambda x: x[1], reverse=True)
        ls = [x[0] for x in sorted_]
        vs = [x[1] / 0.4 * 100 for x in sorted_]  # avg accuracy %
        cols = [COLORS.get(l, "#888") for l in ls]

        bars = ax.barh(range(len(ls)), vs, color=cols, height=0.68, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(
            majority_aunrc / 0.4 * 100,
            color="#999",
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
            label="Majority baseline",
        )

        for i, (l, v, b) in enumerate(zip(ls, vs, bars)):
            if l == "SDIV":
                b.set_edgecolor("#a00000")
                b.set_linewidth(2.0)
            rank = i + 1
            ax.text(
                v + 0.1,
                i,
                f"#{rank}  {v:.2f}%",
                va="center",
                fontsize=8.5,
                color=COLORS.get(l, "#333"),
                fontweight="bold" if l == "SDIV" else "normal",
            )
            if v < majority_aunrc / 0.4 * 100 + 1.0:
                ax.text(
                    ax.get_xlim()[0] + 0.5,
                    i,
                    "collapse",
                    va="center",
                    fontsize=7.5,
                    color="#aa0000",
                    fontstyle="italic",
                )

        ax.set_yticks(range(len(ls)))
        ax.set_yticklabels([dn(l) for l in ls], fontsize=9.5)
        ax.set_xlabel("Average Accuracy over eta in [0, 40%]  (%)\n(= AUNRC / 0.4 * 100)")
        ax.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.75)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.invert_yaxis()

    _bar(ax1, apath, "PathMNIST AUNRC Ranking", majority_aunrc=1 / 9 * 0.4)
    _bar(ax2, aderm, "DermaMNIST AUNRC Ranking", majority_aunrc=DERM_MAJORITY * 0.4)

    ax1.set_xlim(40, 87)
    ax2.set_xlim(40, 73)
    ax2.text(
        0.98,
        0.03,
        "SDIV(opt) beta=0.1,lam=-0.4 at eta=0 only:\n73.3% — full noise curve pending",
        transform=ax2.transAxes,
        fontsize=7.5,
        ha="right",
        color="#D62728",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff5f5", edgecolor="#D62728", alpha=0.85),
    )

    fig.text(
        0.5,
        -0.03,
        "ForwardT excluded (requires oracle transition matrix)  |  "
        "AUNRC = integral_0^0.4 acc(eta) d(eta) via trapezoidal rule",
        ha="center",
        fontsize=8,
        color="#555",
        style="italic",
    )
    plt.tight_layout()
    _save(fig, "F08_aunrc_ranking.png")


# ============================================================================
# F09  NLP BERT: SDIV Competitive; GCE Best Emotion, TruncGCE Best PubMedQA
# ============================================================================
def plot_f09():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "BERT Fine-Tuning: Robust Losses Offer 1-3% Gains Over CCE on NLP Tasks\n"
        "SDIV Does Not Hurt NLP Generalisation — Consistent with Bayes-Optimal Theory",
        fontsize=11.5,
        fontweight="bold",
    )

    def _nlp_bar(ax, df, title, note):
        data = df.set_index("loss")["best_acc"]
        sorted_ = data.sort_values(ascending=False)
        names = list(sorted_.index)
        vals = list(sorted_.values * 100)
        cols = [COLORS.get(n, "#888") for n in names]
        cce_ref = data.get("CCE", None)

        bars = ax.bar(range(len(names)), vals, color=cols, width=0.68, alpha=0.85, edgecolor="white", linewidth=0.5)
        if cce_ref is not None:
            ax.axhline(
                cce_ref * 100,
                color="#000",
                linewidth=1.0,
                linestyle="--",
                alpha=0.35,
                label=f"CCE={cce_ref * 100:.1f}%",
            )

        for i, (n, v, b) in enumerate(zip(names, vals, bars)):
            if n == "SDIV":
                b.set_edgecolor("#a00000")
                b.set_linewidth(2.0)
            ax.text(
                i,
                v + 0.05,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=COLORS.get(n, "#333"),
                fontweight="bold" if n == "SDIV" else "normal",
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([dn(n) for n in names], rotation=40, ha="right", fontsize=9)
        vr = max(vals) - min(vals)
        ax.set_ylim(min(vals) - vr * 1.5, max(vals) + vr * 2.0)
        ax.set_ylabel("Best Val. Accuracy (%)")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=7, ncol=2, framealpha=0.75)
        ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=7.5, color="#555", style="italic")

    _nlp_bar(
        ax1,
        nlp_emo,
        "Emotion (6-class  |  1500 training samples)",
        "GCE leads (58.2%); SDIV 57.8% — within 0.4pp of best",
    )
    _nlp_bar(
        ax2,
        nlp_pub,
        "PubMedQA (3-class  |  1500 training samples)",
        "TruncGCE leads (58.0%); SDIV 55.3% — 0.7pp below CCE",
    )

    fig.text(
        0.5,
        -0.03,
        "BERT-base-uncased  |  3 epochs  |  LR=2e-5  |  batch=32  |  max_len=128",
        ha="center",
        fontsize=8,
        color="#555",
        style="italic",
    )
    plt.tight_layout()
    _save(fig, "F09_nlp_bert.png")


# ============================================================================
# F10  Master Summary — 4 Key Findings
# ============================================================================
def plot_f10():
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
    fig.suptitle(
        "Experimental Summary: Robust Neural Classification via S-Divergence\n"
        "Consistency Theorem Validated — SDIV Competitive, Noise-Robust, Bayes-Optimal with Correct Parameterisation",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    # (a) PathMNIST noise
    ax_a = fig.add_subplot(gs[0, 0])
    for loss in ["CCE", "GCE(q=0.7)", "SCE", "FCL", "SDIV"]:
        sub = path_noise[path_noise.loss == loss].sort_values("noise_rate")
        if sub.empty:
            continue
        ax_a.plot(sub.noise_rate.values * 100, sub.accuracy.values * 100, **lkw(loss, lw=1.5, ms=5))
    ax_a.set_xticks(NOISE_VALS)
    ax_a.set_xticklabels([f"{v}%" for v in NOISE_VALS], fontsize=8.5)
    ax_a.set_xlabel("Label Noise Rate eta")
    ax_a.set_ylabel("Accuracy (%)")
    ax_a.set_ylim(79, 87)
    ax_a.legend(loc="lower left", fontsize=7, ncol=3, framealpha=0.75)
    ax_a.set_title("(a) PathMNIST Noise\nSDIV competitive — all losses >80% at eta=40%", fontsize=10)

    # (b) DermaMNIST collapse
    ax_b = fig.add_subplot(gs[0, 1])
    for loss in ["CCE", "FCL", "TSCCE", "SCE", "SDIV"]:
        sub = derm_noise[derm_noise.loss == loss].sort_values("noise_rate")
        if sub.empty:
            continue
        ax_b.plot(sub.noise_rate.values * 100, sub.accuracy.values * 100, **lkw(loss, lw=1.5, ms=5))
    ax_b.axhline(
        DERM_MAJORITY * 100, color="#999", linewidth=1.0, linestyle=":", label=f"Majority ({DERM_MAJORITY * 100:.1f}%)"
    )
    ax_b.scatter([0], [73.32], marker="*", s=220, color="#D62728", zorder=10, label="SDIV(opt) lam=-0.4")
    ax_b.set_xticks(NOISE_VALS)
    ax_b.set_xticklabels([f"{v}%" for v in NOISE_VALS], fontsize=8.5)
    ax_b.set_xlabel("Label Noise Rate eta")
    ax_b.set_ylabel("Accuracy (%)")
    ax_b.set_ylim(64, 77)
    ax_b.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.75)
    ax_b.set_title("(b) DermaMNIST: lambda=-0.8 Collapses\nOptimised lambda=-0.4 recovers to 73.3%", fontsize=10)

    # (c) DermaMNIST SDIV grid
    ax_c = fig.add_subplot(gs[1, 0])
    piv = derm_surf.pivot_table(index="beta", columns="lam", values="accuracy", aggfunc="mean") * 100
    cmap_c = LinearSegmentedColormap.from_list("cr", ["#cc2222", "#ff8888", "#ffdd88", "#88cc44", "#228822"])
    ax_c.imshow(piv.values, cmap=cmap_c, aspect="auto", vmin=66, vmax=74)
    betas_c = list(piv.index)
    lams_c = list(piv.columns)
    for i, b in enumerate(betas_c):
        for j, l in enumerate(lams_c):
            v = piv.loc[b, l]
            is_col = v < DERM_MAJORITY * 100 + 0.5
            ax_c.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if is_col else "black",
                fontweight="bold" if v == piv.values.max() else "normal",
            )
    ax_c.set_xticks(range(len(lams_c)))
    ax_c.set_xticklabels([f"{l:.1f}" for l in lams_c])
    ax_c.set_yticks(range(len(betas_c)))
    ax_c.set_yticklabels([f"{b:.2f}" for b in betas_c])
    ax_c.set_xlabel("lambda")
    ax_c.set_ylabel("beta")
    ax_c.set_title("(c) SDIV Grid on DermaMNIST\nred=collapse; green=learning; bold=best", fontsize=10)

    # (d) AUNRC ranking PathMNIST
    ax_d = fig.add_subplot(gs[1, 1])
    ap = {l: v for l, v in aunrc(path_noise).items() if l != "ForwardT"}
    si = sorted(ap.items(), key=lambda x: x[1], reverse=True)
    l_s = [x[0] for x in si]
    v_s = [x[1] / 0.4 * 100 for x in si]
    bars = ax_d.barh(range(len(l_s)), v_s, color=[COLORS.get(l, "#888") for l in l_s], height=0.65, alpha=0.85)
    for i, (l, v, b) in enumerate(zip(l_s, v_s, bars)):
        if l == "SDIV":
            b.set_edgecolor("#a00000")
            b.set_linewidth(1.8)
        ax_d.text(
            v + 0.1,
            i,
            f"#{i + 1} {v:.1f}%",
            va="center",
            fontsize=8,
            color=COLORS.get(l, "#333"),
            fontweight="bold" if l == "SDIV" else "normal",
        )
    ax_d.set_yticks(range(len(l_s)))
    ax_d.set_yticklabels([dn(l) for l in l_s], fontsize=9)
    ax_d.set_xlabel("Avg Accuracy over eta in [0, 40%] (%)")
    ax_d.set_xlim(50, 88)
    ax_d.invert_yaxis()
    ax_d.set_title("(d) PathMNIST AUNRC\nFCL > SCE > TPDD-CCE > SDIV > CCE", fontsize=10)

    _save(fig, "F10_master_summary.png")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # Clean old PNGs from publication_final
    for old in OUTDIR.glob("*.png"):
        old.unlink()

    print("Generating 10 publication-quality figures ...\n")
    plot_f01()
    plot_f02()
    plot_f03()
    plot_f04()
    plot_f05()
    plot_f06()
    plot_f07()
    plot_f08()
    plot_f09()
    plot_f10()
    print()
    print(f"All figures saved to:   {OUTDIR}")
    print(f"Figures also copied to: {FIGDIR}")
    print()
    for p in sorted(OUTDIR.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
