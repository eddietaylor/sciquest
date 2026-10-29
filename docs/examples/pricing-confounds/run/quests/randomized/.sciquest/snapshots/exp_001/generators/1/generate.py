"""Owner-side, noiseless teaching world. Never staged into the estimator notebook."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PRICES = (30, 40)


def generate(output: Path, kind: str, seed: int = 42) -> None:
    rng = random.Random(seed)
    rows = []
    if kind == "evaluation":
        # Uniform Z population; these are expectations, not individual outcomes.
        for price in PRICES:
            mean = sum(100 - 2 * price + 40 * z for z in (0, 1)) / 2
            rows.append([f"evaluation_{price}", "balanced_population", price, mean, price * mean])
        fields = ["id", "population", "price", "mean_demand", "expected_revenue"]
    else:
        # Eight units in either arm. Randomize price assignment within each Z
        # stratum, retaining exactly two units per price/stratum cell.
        for z in (0, 1):
            prices = [30 + 10 * z] * 4 if kind == "observational" else [30, 30, 40, 40]
            rng.shuffle(prices)
            for price in prices:
                rows.append([price, 100 - 2 * price + 40 * z])
        rng.shuffle(rows)
        rows = [[f"{kind}_unit{unit}", *values] for unit, values in enumerate(rows)]
        fields = ["id", "price", "demand"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["observational", "randomized", "evaluation"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output, args.kind, args.seed)
