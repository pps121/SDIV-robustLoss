#!/usr/bin/env python3
"""
code/figures/common_style.py
============================
Shared color palettes, dataset definitions, matplotlib styling parameters,
and CSV loading utilities across all figure generation scripts.
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Paths
FIGURES_DIR = Path(__file__).resolve().parent
CODE_DIR = FIGURES_DIR.parent
ROOT_DIR = CODE_DIR.parent
CSV_DIR = ROOT_DIR / "results" / "paper" / "csvs"
OUTPUT_DIR = ROOT_DIR / "results" / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe color palette (Wong 2011 + extensions)
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

DISPLAY_NAMES = {
    "GCE(q=0.7)": "GCE (q=0.7)",
    "TDPDSCCE": "TPDD-CCE",
    "ForwardT": "ForwardT\u2020",
}


def get_display_name(loss_name: str) -> str:
    return DISPLAY_NAMES.get(loss_name, loss_name)


def get_line_kwargs(loss_name: str, lw: float = 1.5, ms: float = 5.5, alpha: float = 1.0) -> dict:
    color = COLORS.get(loss_name, "#444444")
    is_sdiv = loss_name == "SDIV"
    return dict(
        color=color,
        marker=MARKERS.get(loss_name, "o"),
        linestyle=LSTYLES.get(loss_name, "-"),
        linewidth=2.8 if is_sdiv else lw,
        markersize=8.0 if is_sdiv else ms,
        markerfacecolor=color,
        markeredgecolor="white" if is_sdiv else color,
        markeredgewidth=0.8,
        label=get_display_name(loss_name),
        zorder=6 if is_sdiv else 3,
        alpha=alpha,
    )


def setup_matplotlib_style():
    matplotlib.use("Agg")
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
