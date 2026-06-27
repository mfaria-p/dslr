#!/usr/bin/env python3
"""histogram.py — overlaid score distributions per house, for every course.

Question answered: which Hogwarts course has a homogeneous score distribution
between all four houses? (i.e. the four houses overlap almost completely.)
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


def main():
    path = sys.argv[1] if len(sys.argv) == 2 else "datasets/dataset_train.csv"
    df = pd.read_csv(path)

    # Course columns are the numeric features (everything after 'Best Hand').
    courses = list(df.columns[6:])

    # Lay all courses out in a grid so we can compare shapes at a glance.
    ncols = 4
    nrows = (len(courses) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9))
    axes = axes.flatten()

    for i, course in enumerate(courses):
        ax = axes[i]
        for house in HOUSES:
            # scores for this house in this course, missing values dropped
            scores = df[df["Hogwarts House"] == house][course].dropna()
            ax.hist(scores, bins=20, alpha=0.5, color=COLORS[house], label=house)
        ax.set_title(course, fontsize=9)
        ax.tick_params(labelsize=6)

    # Hide any unused subplot cells (13 courses, 16 grid slots -> 3 empty).
    for j in range(len(courses), len(axes)):
        axes[j].axis("off")

    # One shared legend for the whole figure.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    fig.tight_layout()

    # Save a copy into plots/ (created if missing), then display interactively.
    os.makedirs("plots", exist_ok=True)
    fig.savefig(os.path.join("plots", "histogram.png"), dpi=100)

    # Let the OS handle Ctrl-C so it kills the GUI loop instantly, instead of
    # waiting for the next window event to deliver the interrupt to Python.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl-C: exit quietly instead of dumping a traceback.
        print("\nInterrupted.")
        sys.exit(0)
