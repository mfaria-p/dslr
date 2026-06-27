#!/usr/bin/env python3
"""scatter_plot.py — scatter of the two most similar features.

Question answered: what are the two features that are similar?
Answer: Astronomy and Defense Against the Dark Arts are almost perfectly
correlated, so their points fall on a straight line.
"""

import os
import sys
import signal
import pandas as pd
import matplotlib.pyplot as plt

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
COLORS = {
    "Gryffindor": "red",
    "Hufflepuff": "gold",
    "Ravenclaw": "blue",
    "Slytherin": "green",
}

FEATURE_X = "Astronomy"
FEATURE_Y = "Defense Against the Dark Arts"


def main():
    path = sys.argv[1] if len(sys.argv) == 2 else "datasets/dataset_train.csv"
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(9, 7))
    for house in HOUSES:
        sub = df[df["Hogwarts House"] == house]
        ax.scatter(sub[FEATURE_X], sub[FEATURE_Y],
                   s=8, alpha=0.5, color=COLORS[house], label=house)

    ax.set_xlabel(FEATURE_X)
    ax.set_ylabel(FEATURE_Y)
    ax.set_title(f"{FEATURE_X} vs {FEATURE_Y}")
    ax.legend()
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(os.path.join("plots", "scatter_plot.png"), dpi=100)

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
