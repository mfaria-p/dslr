#!/usr/bin/env python3
"""accuracy.py — compare predictions against the ground truth.

Usage: python accuracy.py houses.csv datasets/dataset_truth.csv

Uses scikit-learn's accuracy_score, the same metric the defense uses.
(sklearn here only *measures* the result; it is not part of the model.)
"""

import sys
import pandas as pd
from sklearn.metrics import accuracy_score


def main():
    if len(sys.argv) != 3:
        print("Usage: python accuracy.py <predictions.csv> <truth.csv>")
        sys.exit(1)

    pred = pd.read_csv(sys.argv[1])
    truth = pd.read_csv(sys.argv[2])

    # Align on Index so row order can never cause a silent mismatch.
    merged = truth.merge(pred, on="Index", suffixes=("_true", "_pred"))
    if len(merged) != len(truth):
        print(f"warning: matched {len(merged)} of {len(truth)} rows by Index")

    y_true = merged["Hogwarts House_true"]
    y_pred = merged["Hogwarts House_pred"]

    acc = accuracy_score(y_true, y_pred)
    correct = (y_true == y_pred).sum()
    total = len(merged)

    print(f"Correct: {correct} / {total}")
    print(f"Accuracy: {acc:.4f}  ({acc * 100:.2f}%)")
    print("PASS (>= 98%)" if acc >= 0.98 else "BELOW TARGET (< 98%)")


if __name__ == "__main__":
    main()
