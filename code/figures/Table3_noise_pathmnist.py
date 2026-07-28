#!/usr/bin/env python3
"""
code/figures/figure3_noise_pathmnist.py
========================================
Figure 3: PathMNIST Uniform Label Noise Robustness (eta = 0% to 40%)

Ground-Truth Empirical Data Sources:
------------------------------------
- results/paper/csvs/pathmnist_noise_results.csv

Scientific Context & Takeaways:
--------------------------------
- Evaluates 9 robust loss functions + ForwardT under symmetric label noise up to 40% on PathMNIST.
- All robust losses maintain >80% accuracy across noise levels (ceiling effect due to pathology patch structures).
- MAE is an extreme outlier that collapses to 42.69% at eta=20% due to gradient instability.
- Legend is placed completely OUTSIDE the canvas on the right.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from common_style import (
    CSV_DIR,
    OUTPUT_DIR,
    COLORS,
    get_line_kwargs,
    setup_matplotlib_style,
)

NOISE_VALS = [0, 10, 20, 30, 40]


def generate_figure3():
    setup_matplotlib_style()
    path_noise = pd.read_csv(CSV_DIR / "pathmnist_noise_results.csv")
    order = ["SDIV", "CCE", "GCE(q=0.7)", "TruncGCE", "SCE", "TPDD-CCE", "TSCCE", "FCL", "ForwardT"]

    fig, (ax_main, ax_mae) = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle(
        "PathMNIST Noise Robustness: All Losses (except MAE) Maintain >80% Under 40% Label Corruption\n"
        "Grounded in Real CSV Measurements (SDIV maintains 81.98% accuracy at eta=40%)",
        fontsize=11.5,
        fontweight="bold",
    )

    for loss in order:
        sub = path_noise[path_noise.loss == loss].sort_values("noise_rate")
        if sub.empty:
            continue
        xs = sub.noise_rate.values * 100
        ys = sub.accuracy.values * 100
        ax_main.plot(xs, ys, **get_line_kwargs(loss, lw=2.4 if loss=="SDIV" else 1.5, ms=7 if loss=="SDIV" else 5))

    sdiv_sub = path_noise[path_noise.loss == "SDIV"].sort_values("noise_rate")
    cce_sub = path_noise[path_noise.loss == "CCE"].sort_values("noise_rate")
    sdiv_drop = (sdiv_sub.accuracy.iloc[0] - sdiv_sub.accuracy.iloc[-1]) * 100
    cce_drop = (cce_sub.accuracy.iloc[0] - cce_sub.accuracy.iloc[-1]) * 100

    ax_main.text(
        0.01,
        0.03,
        f"SDIV drop (eta=0 to 40%): -{sdiv_drop:.2f} pp\nCCE  drop (eta=0 to 40%): -{cce_drop:.2f} pp",
        transform=ax_main.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9),
    )

    ax_main.set_xticks(NOISE_VALS)
    ax_main.set_xticklabels([f"{v}%" for v in NOISE_VALS])
    ax_main.set_xlabel("Label Noise Rate eta")
    ax_main.set_ylabel("Test Accuracy (%)")
    ax_main.set_ylim(74, 88)

    # Position legend OUTSIDE canvas on the right
    ax_main.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        ncol=1,
        fontsize=8,
        framealpha=0.9,
        title="Loss Function",
        title_fontsize=8.5,
    )
    ax_main.set_title("All Losses Except MAE (SDIV in Bold Red)", fontsize=10.5, fontweight="bold")

    mae_sub = path_noise[path_noise.loss == "MAE"].sort_values("noise_rate")
    xs_m = mae_sub.noise_rate.values * 100
    ys_m = mae_sub.accuracy.values * 100
    ax_mae.plot(xs_m, ys_m, **get_line_kwargs("MAE", lw=2.0))
    ax_mae.set_xticks(NOISE_VALS)
    ax_mae.set_xticklabels([f"{v}%" for v in NOISE_VALS], fontsize=8.5)
    ax_mae.set_xlabel("Label Noise Rate eta")
    ax_mae.set_ylabel("Test Accuracy (%)")
    ax_mae.set_ylim(35, 85)
    ax_mae.set_title("MAE Outlier\n(Gradient Instability)", fontsize=10.5, color="#E69F00", fontweight="bold")
    ax_mae.axhline(1 / 9 * 100, color="#aaaaaa", linewidth=0.8, linestyle=":", label="Random (11.1%)")
    ax_mae.legend(fontsize=7, loc="lower right")

    fig.subplots_adjust(top=0.88, right=0.80, wspace=0.35)
    output_path = OUTPUT_DIR / "F02_noise_pathmnist.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated Figure 3: {output_path}")


if __name__ == "__main__":
    generate_figure3()
