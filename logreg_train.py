#!/usr/bin/env python3
"""logreg_train.py — train a one-vs-all logistic regression with gradient
descent, and save the learned parameters.

Usage: python logreg_train.py dataset_train.csv
Output: weights.json  (theta per house + the scaling stats predict needs)
"""

import sys
import json
import numpy as np
import pandas as pd

# Features kept after the visualization step:
#  - dropped Arithmancy and Care of Magical Creatures (homogeneous / no signal)
#  - dropped Defense Against the Dark Arts (redundant with Astronomy, r = -1)
FEATURES = [
    "Astronomy", "Herbology", "Divination", "Muggle Studies", "Ancient Runes",
    "History of Magic", "Transfiguration", "Potions", "Charms", "Flying",
]
HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

LEARNING_RATE = 0.5
EPOCHS = 3000


def sigmoid(z):
    """g(z) = 1 / (1 + e^-z), squashes any real number into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))


def cost(X, y, theta):
    """Log-loss J(theta). The tiny 1e-15 keeps log() away from log(0)."""
    h = sigmoid(X @ theta)
    eps = 1e-15
    return -np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))


def train_one(X, y, alpha, epochs, label=""):
    """Gradient descent for a single binary classifier.
    X: (m, n+1) feature matrix WITH a leading bias column of ones.
    y: (m,)     0/1 labels for this house (label/ground-truth for each student).
    Returns the learned theta vector (n+1,)."""
    m, n = X.shape
    theta = np.zeros(n)                      # start all features weights at 0
    for i in range(epochs):
        h = sigmoid(X @ theta)               # predictions, shape (m,): one score per student
        gradient = (X.T @ (h - y)) / m       # ∂J/∂θ = mean((h - y) * x)
        theta -= alpha * gradient            # step downhill
        # Report the cost periodically so we can watch it decrease (converge).
        if i % 500 == 0 or i == epochs - 1:
            print(f"  [{label}] epoch {i:5d}  cost = {cost(X, y, theta):.6f}")
    return theta


def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py dataset_train.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    # --- Prepare X -------------------------------------------------------
    raw = df[FEATURES].astype(float)

    # 1) impute missing values with each feature's (training) mean
    means = raw.mean()
    raw = raw.fillna(means)

    # 2) standardize: (x - mean) / std, so every feature has mean 0, std 1
    stds = raw.std()
    Xs = (raw - means) / stds
    X = Xs.to_numpy()

    # 3) prepend a bias column of ones -> lets theta[0] act as the intercept
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # --- Train one classifier per house (one-vs-all) ---------------------
    thetas = {}
    for house in HOUSES:
        y = (df["Hogwarts House"] == house).to_numpy().astype(float)  # 1 vs rest
        thetas[house] = train_one(X, y, LEARNING_RATE, EPOCHS, house).tolist()

    # --- Save everything predict will need -------------------------------
    model = {
        "features": FEATURES,
        "houses": HOUSES,
        "means": means.to_dict(),   # for the SAME imputation + scaling at predict
        "stds": stds.to_dict(),
        "thetas": thetas,           # one weight vector per house
    }
    with open("weights.json", "w") as f:
        json.dump(model, f, indent=2)
    print("saved -> weights.json")


if __name__ == "__main__":
    main()
