"""Prewritten price-only OLS; this exact source becomes a registered notebook cell.

No simulator law, expected coefficients, oracle values, or external-agent calls.
"""
import csv
import json
import math
from pathlib import Path

with open("data/0/train.csv", newline="") as stream:
    training = list(csv.DictReader(stream))
with open("data/1/features.csv", newline="") as stream:
    features = list(csv.DictReader(stream))
assert set(training[0]) == {"id", "price", "demand"}
assert set(features[0]) == {"id", "population", "price"}
assert not Path("data/1/eval.csv").exists()

x = [float(row["price"]) for row in training]
y = [float(row["demand"]) for row in training]
mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
sxx = math.fsum((v - mx) ** 2 for v in x)
if sxx == 0:
    raise ValueError("Two distinct training prices are required")
slope = math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx
intercept = my - slope * mx
with open("predictions.csv", "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=[
        "id", "predicted_mean_demand", "predicted_revenue", "fitted_intercept", "fitted_slope"
    ], lineterminator="\n")
    writer.writeheader()
    for row in features:
        price = float(row["price"])
        mean = intercept + slope * price
        writer.writerow(dict(id=row["id"], predicted_mean_demand=mean,
                             predicted_revenue=price * mean,
                             fitted_intercept=intercept, fitted_slope=slope))
print(json.dumps(dict(model="price-only OLS with intercept", n=len(x),
                      intercept=intercept, slope=slope, external_agent_calls=0)))
