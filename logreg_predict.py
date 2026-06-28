#!/usr/bin/env python3
"""logreg_predict.py — predict houses for a test set using saved weights.

Usage: python logreg_predict.py dataset_test.csv weights.json
Output: houses.csv  (Index,Hogwarts House)
"""

import sys
import json
import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py dataset_test.csv weights.json")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    with open(sys.argv[2]) as f:
        model = json.load(f)

    features = model["features"]
    houses = model["houses"]
    means = pd.Series(model["means"])
    stds = pd.Series(model["stds"])

    # --- Recreate the EXACT training transformation ----------------------
    raw = df[features].astype(float)
    raw = raw.fillna(means)              # impute with TRAINING means
    Xs = (raw - means) / stds            # scale with TRAINING means/stds
    X = Xs.to_numpy()
    X = np.hstack([np.ones((X.shape[0], 1)), X])   # same bias column

    # --- Probability from each house's classifier, shape (m, 4) ----------
    probs = np.column_stack([
        sigmoid(X @ np.array(model["thetas"][house])) for house in houses
    ])

    # --- Pick the most confident house per student -----------------------
    best = probs.argmax(axis=1)          # index of the max prob in each row
    predictions = [houses[i] for i in best]

    # --- Write houses.csv exactly as the subject specifies ---------------
    out = pd.DataFrame({"Index": range(len(predictions)),
                        "Hogwarts House": predictions})
    out.to_csv("houses.csv", index=False)
    print(f"wrote houses.csv ({len(predictions)} predictions)")


if __name__ == "__main__":
    main()
