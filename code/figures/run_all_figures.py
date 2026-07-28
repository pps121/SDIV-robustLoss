#!/usr/bin/env python3
"""
code/figures/run_all_figures.py
===============================
Master script to sequentially execute figure1.py through figure10.py.
Outputs all publication figures to results/paper/figures/.
"""

import sys
from pathlib import Path

# Add current directory to path
FIGURES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIGURES_DIR))

from figure1_clean_accuracy import generate_figure1
from figure2_noise_mnist import generate_figure2
from figure3_noise_pathmnist import generate_figure3
from figure4_noise_dermamnist import generate_figure4
from figure5_aunrc_ranking import generate_figure5
from figure6_fgsm_pathmnist import generate_figure6
from figure7_fgsm_dermamnist import generate_figure7
from figure8_sdiv_surface import generate_figure8
from figure9_nlp_bert import generate_figure9
from figure10_master_summary import generate_figure10


def main():
    print("==================================================================")
    print(" Robust Neural Classification — Publication Figure Generation Suite")
    print("==================================================================\n")

    generators = [
        ("Figure 1 (Clean Accuracy)", generate_figure1),
        ("Figure 2 (MNIST Label Noise)", generate_figure2),
        ("Figure 3 (PathMNIST Label Noise)", generate_figure3),
        ("Figure 4 (DermaMNIST Label Noise)", generate_figure4),
        ("Figure 5 (AUNRC Ranking)", generate_figure5),
        ("Figure 6 (PathMNIST FGSM Attack)", generate_figure6),
        ("Figure 7 (DermaMNIST FGSM Attack)", generate_figure7),
        ("Figure 8 (SDIV Parameter Grid)", generate_figure8),
        ("Figure 9 (BERT Fine-Tuning NLP)", generate_figure9),
        ("Figure 10 (Master Summary)", generate_figure10),
    ]

    for name, func in generators:
        print(f"Generating {name}...")
        func()

    print("\nAll 10 paper figures successfully generated in results/paper/figures/ !")


if __name__ == "__main__":
    main()
