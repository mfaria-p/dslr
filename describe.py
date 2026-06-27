#!/usr/bin/env python3
"""describe.py — print summary statistics for the numeric features of a CSV.

Reimplements the core of pandas' DataFrame.describe() by hand.
Forbidden: describe(), mean(), std(), min(), max(), percentile(), etc.
We may use pandas ONLY to read the file into memory.
"""

import sys
import pandas as pd


# ---------------------------------------------------------------------------
# Statistic helpers. Each takes a clean list of floats (no missing values)
# and returns one number. Everything is a plain loop — no shortcuts.
# ---------------------------------------------------------------------------

def count_(values):
    n = 0
    for _ in values:
        n += 1
    return n


def mean_(values):
    total = 0.0
    for v in values:
        total += v
    return total / count_(values)


def std_(values):
    """Sample standard deviation, dividing by (N - 1) like pandas does."""
    n = count_(values)
    if n < 2:
        return float("nan")
    m = mean_(values)
    squared_diff_sum = 0.0
    for v in values:
        squared_diff_sum += (v - m) ** 2
    return (squared_diff_sum / (n - 1)) ** 0.5


def min_(values):
    smallest = values[0]
    for v in values:
        if v < smallest:
            smallest = v
    return smallest


def max_(values):
    largest = values[0]
    for v in values:
        if v > largest:
            largest = v
    return largest


def percentile_(values, p):
    """p-th percentile via linear interpolation (numpy/pandas default)."""
    ordered = sorted(values)
    n = count_(ordered)
    if n == 1:
        return ordered[0]
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= n:
        return ordered[lo]
    return ordered[lo] + frac * (ordered[lo + 1] - ordered[lo])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

STATS = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]


def clean_column(series):
    """Return the column's values as floats, dropping missing/non-numeric."""
    cleaned = []
    for v in series:
        if v != v:                 # NaN test: NaN != NaN
            continue
        try:
            cleaned.append(float(v))
        except (TypeError, ValueError):
            continue
    return cleaned


def compute(values):
    return {
        "Count": count_(values),
        "Mean": mean_(values),
        "Std": std_(values),
        "Min": min_(values),
        "25%": percentile_(values, 25),
        "50%": percentile_(values, 50),
        "75%": percentile_(values, 75),
        "Max": max_(values),
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: describe.py <dataset.csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    results = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue               # skip House, names, Birthday, Best Hand
        values = clean_column(df[col])
        if len(values) == 0:
            continue
        results[col] = compute(values)

    if not results:
        print("No numeric features found.")
        return

    features = list(results.keys())
    col_width = 15

    header = " " * 7 + "".join(f"{f[:col_width-1]:>{col_width}}" for f in features)
    print(header)

    for stat in STATS:
        row = f"{stat:<7}"
        for f in features:
            row += f"{results[f][stat]:>{col_width}.3f}"
        print(row)


if __name__ == "__main__":
    main()
