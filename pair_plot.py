#!/usr/bin/env python3
"""pair_plot.py — scatter-plot matrix of all course features, colored by house.

Question answered: from this visualization, which features will you use for
the logistic regression? (Keep features that SEPARATE the houses; drop the
ones that overlap completely or that duplicate another feature.)
"""

import os
import sys
import signal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Same color-per-house mapping as histogram.py, so the plots stay consistent.
HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
COLORS = {
    "Gryffindor": "red",
    "Hufflepuff": "gold",
    "Ravenclaw": "blue",
    "Slytherin": "green",
}


def main():
    path = sys.argv[1] if len(sys.argv) == 2 else "datasets/dataset_train.csv"
    df = pd.read_csv(path)

    courses = list(df.columns[6:])          # the 13 numeric course features

    # seaborn builds the whole NxN grid: scatter plots off-diagonal,
    # per-feature histograms on the diagonal. hue colors points by house.
    grid = sns.pairplot(
        df,
        vars=courses,
        hue="Hogwarts House",
        hue_order=HOUSES,        # fix the order so colors map consistently
        palette=COLORS,          # same house->color mapping as histogram.py
        diag_kind="hist",
        plot_kws={"s": 8, "alpha": 0.4},
        diag_kws={"element": "step", "fill": False},  # outlines, all visible
        height=1.8,          # bigger cells -> more legible (larger window)
    )
    grid.figure.suptitle("Pair plot of Hogwarts courses by house", y=1.01)

    os.makedirs("plots", exist_ok=True)
    grid.savefig(os.path.join("plots", "pair_plot.png"), dpi=90)

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
