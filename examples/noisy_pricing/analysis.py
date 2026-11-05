"""Prewritten estimator, evaluated in a new isolated process for each slot.

MODEL and RIDGE are prepended by the owner from the frozen public protocol.
No family/seed, simulator import, hidden label, previous fit, or oracle constant.
"""
import csv
import json
from pathlib import Path
import numpy as np

assert not Path("data/1/eval.csv").exists()
with open("data/0/train.csv", newline="") as f:
    training = list(csv.DictReader(f))
with open("data/1/features.csv", newline="") as f:
    evaluation = list(csv.DictReader(f))
assert set(training[0]) == {"id", "x", "price", "demand"}
assert set(evaluation[0]) == {"id", "context_id", "x", "price"}


def features(row):
    t, x = (float(row["price"]) - 40) / 20, float(row["x"])
    return [1, t, x] if MODEL == "linear" else [1, t, x, t*x, t*t, x*x]


X = np.array([features(row) for row in training], dtype=float)
y = np.array([float(row["demand"]) for row in training])
# Minimize SUM squared residuals + RIDGE * sum(beta[j]**2, j>0).
penalty = np.eye(X.shape[1]) * RIDGE
penalty[0, 0] = 0
beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
with open("model.json", "w") as f:
    json.dump(dict(model=MODEL, ridge_sum_loss=RIDGE, coefficients=beta.tolist(), n=len(y)), f)
with open("predictions.csv", "w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["id", "predicted_mean_demand", "predicted_revenue"])
    for row in evaluation:
        mean = float(np.array(features(row)) @ beta)
        writer.writerow([row["id"], mean, float(row["price"]) * mean])
print(json.dumps(dict(model=MODEL, n=len(y), external_agent_calls=0)))
