#!/usr/bin/env python3
"""
code/figures/figure8_sdiv_surface.py
====================================
Figure 8: SDIV Parameter Surface (beta, lambda) Response Grid

Paper Context:
--------------
- Evaluates SDIV across parameter grid points: beta in {0.02, 0.05, 0.10, 0.20, 0.50} and lambda in {-0.80, -0.40, 0.00, +0.20}.
- Demonstrates phase transition:
    * lambda = -0.80 (Default): Degenerate region (majority-class collapse on DermaMNIST due to A = 0.24 gradient starvation).
    * lambda >= -0.40 (Optimal): Learning region (#1 Best accuracy 84.11% on PathMNIST, 73.32% on DermaMNIST).
- Clean axis labels and formatting without nan text.

Outputs:
--------
- results/paper/figures/F07_sdiv_grid.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from common_style import (
    CSV_DIR,
    OUTPUT_DIR,
    setup_matplotlib_style,
)

DERM_MAJORITY = 0.6688279301745635


def generate_figure8():
    setup_matplotlib_style()
    path_surf = pd.read_csv(CSV_DIR / "pathmnist_sdiv_surface.csv")
    derm_surf = pd.read_csv(CSV_DIR / "dermamnist_sdiv_surface.csv")

    def _pivot(df):
        return df.pivot_table(index="beta", columns="lam", values="accuracy", aggfunc="mean") * 100

    piv_path = _pivot(path_surf)
    piv_derm = _pivot(derm_surf)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "SDIV (beta, lambda) Parameter Surface: Phase Transition from Degenerate (lam=-0.80) to Optimal (lam >= -0.40)\n"
        "Guidance: Optimal region lam in [-0.40, 0.00] & beta in [0.05, 0.10] achieves #1 top accuracy (PathMNIST 84.1%, DermaMNIST 73.3%)",
        fontsize=11.5,
        fontweight="bold",
        y=1.01,
    )

    def _draw_heatmap(ax, piv, title, is_derm=False):
        betas = list(piv.index)
        lams = list(piv.columns)
        
        if is_derm:
            cmap = LinearSegmentedColormap.from_list(
                "collapse_learn", ["#d9534f", "#f0ad4e", "#f7f7f7", "#5cb85c", "#337ab7"]
            )
            vmin, vmax = 66.0, 74.0
        else:
            cmap = plt.cm.YlGn
            vmin, vmax = 77.0, 85.0

        data_vals = piv.values.copy()
        im = ax.imshow(data_vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        for i, b in enumerate(betas):
            for j, l in enumerate(lams):
                v = piv.loc[b, l]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="#888888")
                else:
                    is_col = is_derm and (v < DERM_MAJORITY * 100 + 0.5)
                    is_best = v == piv.values.max()
                    
                    if is_best:
                        text_str = f"#1 Best\n{v:.2f}%"
                        tc = "white" if (is_col or v > 83.5 or is_derm) else "#D62728"
                        ax.text(j, i, text_str, ha="center", va="center", fontsize=8.5, color=tc, fontweight="bold")
                    elif is_col:
                        ax.text(j, i, f"Collapse\n{v:.1f}%", ha="center", va="center", fontsize=8, color="white")
                    else:
                        tc = "white" if (v > 83.5 or (is_derm and v > 72.5)) else "#222222"
                        ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=8.5, color=tc)

        ax.set_xticks(range(len(lams)))
        ax.set_xticklabels([f"{l:+.2f}" for l in lams], fontsize=9.5, fontweight="bold")
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([f"{b:.2f}" for b in betas], fontsize=9.5, fontweight="bold")
        
        ax.set_xlabel("Mixing Parameter lambda  (lam)", fontsize=10.5, fontweight="bold", labelpad=8)
        ax.set_ylabel("Divergence Exponent beta  (beta)", fontsize=10.5, fontweight="bold", labelpad=8)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        
        # Draw vertical line separating Degenerate (lam = -0.80) from Optimal (lam >= -0.40)
        ax.axvline(0.5, color="#cc0000", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(0.0, 1.03, "Degenerate\n(lam = -0.80)", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#cc0000", fontweight="bold")
        ax.text(1.5, 1.03, "Optimal Learning Region  (lam >= -0.40)", transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5, color="#007700", fontweight="bold")
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cbar.ax.set_ylabel("Accuracy (%)", fontsize=9.5)

    _draw_heatmap(axes[0], piv_path, "PathMNIST Response Surface\n(#1 Peak = 84.11% at beta=0.05, lam=-0.40)", is_derm=False)
    _draw_heatmap(axes[1], piv_derm, "DermaMNIST Response Surface\n(#1 Peak = 73.32% at beta=0.10, lam=-0.40)", is_derm=True)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "F07_sdiv_grid.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated Figure 8: {output_path}")


if __name__ == "__main__":
    generate_figure8()
