#!/usr/bin/env python3
"""correlation_ranking.py — rank every pair of numeric features by how
strongly they are linearly correlated (Pearson's r).

Exploration helper: this is how you discover which features are redundant
(|r| near 1) without guessing.
"""

import sys
from itertools import combinations
import pandas as pd


def main():
    path = sys.argv[1] if len(sys.argv) == 2 else "datasets/dataset_train.csv"
    df = pd.read_csv(path)

    # numeric course features only
    courses = [c for c in df.columns[6:] if pd.api.types.is_numeric_dtype(df[c])]

    # pandas computes the full correlation matrix (pairwise-complete by default)
    corr = df[courses].corr()

    # flatten the upper triangle into (|r|, r, a, b) and sort strongest first
    pairs = [(abs(corr.loc[a, b]), corr.loc[a, b], a, b)
             for a, b in combinations(courses, 2)]
    pairs.sort(reverse=True)

    print(f"{'|r|':>6}  {'r':>7}   feature pair")
    print("-" * 60)
    for abs_r, r, a, b in pairs:
        print(f"{abs_r:6.3f}  {r:7.3f}   {a}  /  {b}")


if __name__ == "__main__":
    main()
