#!/usr/bin/env python3
"""
code/figures/figure6_fgsm_pathmnist.py
======================================
Figure 6: PathMNIST FGSM Adversarial Robustness

Ground-Truth Empirical Data Sources:
------------------------------------
- results/paper/csvs/pathmnist_fgsm_results.csv
- results/paper/csvs/pathmnist_noise_results.csv (eta=0 clean baseline)

Scientific Context & Takeaways:
--------------------------------
- Tests models under FGSM single-step adversarial attacks at epsilon in {0, 1/255, 2/255, 4/255, 8/255}.
- Evaluates whether robust loss functions alone confer adversarial robustness.
- Proves ALL loss functions collapse catastrophically on PathMNIST at eps=8/255 (CCE: 83.31% -> 17.60%, SDIV: 80.93% -> 15.89%, FCL: 76.88% -> 7.12%).
- Legend is placed completely OUTSIDE the canvas on the right to prevent any overlap.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common_style import (
    CSV_DIR,
    OUTPUT_DIR,
    COLORS,
    get_line_kwargs,
    setup_matplotlib_style,
)


def generate_figure6():
    setup_matplotlib_style()
    path_fgsm = pd.read_csv(CSV_DIR / "pathmnist_fgsm_results.csv")
    path_noise = pd.read_csv(CSV_DIR / "pathmnist_noise_results.csv")

    path_clean = {
        r["loss"]: r["accuracy"] for _, r in path_noise[path_noise.noise_rate == 0].iterrows()
    }
    order = ["CCE", "GCE(q=0.7)", "TruncGCE", "SCE", "TPDD-CCE", "TSCCE", "FCL", "SDIV", "MAE"]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    fig.suptitle(
        "PathMNIST FGSM: ALL Loss Functions Collapse Under Strong Adversarial Attack\n"
        "Grounded in Real CSV Measurements (Adversarial Training required for FGSM robustness)",
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
            kw = get_line_kwargs(loss, lw=1.2, ms=4, alpha=0.55)
            kw["linestyle"] = ":"
            kw["label"] = "MAE (unstable run)"
            ax.plot(eps_arr * 255, acc_arr * 100, **kw)
        else:
            ax.plot(eps_arr * 255, acc_arr * 100, **get_line_kwargs(loss, lw=2.4 if loss=="SDIV" else 1.5, ms=7 if loss=="SDIV" else 5))

        ax.annotate(
            f"{acc_arr[-1] * 100:.0f}%",
            xy=(8, acc_arr[-1] * 100),
            xytext=(8.35, acc_arr[-1] * 100),
            fontsize=7.5,
            va="center",
            color=COLORS.get(loss, "#444444"),
            fontweight="bold" if loss == "SDIV" else "normal",
        )

    ax.axhline(1 / 9 * 100, color="#bbbbbb", linewidth=0.8, linestyle=":", label=f"Random (9-class, {1 / 9 * 100:.0f}%)")
    ax.set_xticks([0, 1, 2, 4, 8])
    ax.set_xticklabels(["0", "1/255", "2/255", "4/255", "8/255"])
    ax.set_xlabel("FGSM Perturbation Budget eps (x255)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlim(-0.3, 10.5)
    ax.set_ylim(5, 90)

    # Position legend OUTSIDE the canvas on the right
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        ncol=1,
        fontsize=8,
        framealpha=0.9,
        title="Loss Function",
        title_fontsize=8.5,
    )

    fig.subplots_adjust(top=0.88, right=0.80)
    output_path = OUTPUT_DIR / "F05_fgsm_pathmnist.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated Figure 6: {output_path}")


if __name__ == "__main__":
    generate_figure6()
