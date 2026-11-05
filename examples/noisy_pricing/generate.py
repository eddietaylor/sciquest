"""Owner-only synthetic DGP. Standalone for isolated SciQuest generator replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from pathlib import Path

FAMILIES = ("affine", "quadratic", "interaction", "threshold")
TRAIN_PRICES = (20, 30, 40, 50, 60)
EVAL_PRICES = tuple(range(20, 61, 5))


def stream(family: str, seed: int, component: str) -> random.Random:
    identity = f"sciquest-noisy-pricing-v1/{family}/{seed}/{component}"
    return random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big"))


def parameters(family: str, seed: int) -> dict:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    r = stream(family, seed, "world")
    return dict(base=r.uniform(108, 120), slope=r.uniform(40, 52), context=r.uniform(2, 6),
                latent=r.uniform(12, 20), noise=r.uniform(6, 10),
                curvature=r.uniform(6, 12) if family == "quadratic" else 0.0,
                interaction=r.uniform(5, 9) if family == "interaction" else 0.0,
                hinge=r.uniform(12, 20) if family == "threshold" else 0.0)


def mean_demand(p: float, x: float, world: dict) -> float:
    """Conditional intervention mean given observed x, not the hidden unit z."""
    t = (p - 40) / 20
    return (world["base"] - world["slope"] * t + world["context"] * x
            - world["curvature"] * t * t + world["interaction"] * x * t
            - world["hinge"] * max(t, 0))


def assignment_probabilities(z: int) -> list[float]:
    weights = [math.exp(1.5 * (2 * z - 1) * (p - 40) / 20) for p in TRAIN_PRICES]
    return [w / sum(weights) for w in weights]


def generate(output: Path, family: str, seed: int, kind: str, training_units: int = 160,
             evaluation_contexts: int = 64) -> None:
    world = parameters(family, seed)
    rows = []
    if kind == "evaluation":
        r = stream(family, seed, "evaluation")
        for i in range(evaluation_contexts):
            x, z, e = r.uniform(-1, 1), r.randrange(2), r.uniform(-1, 1)
            # One latent type and disturbance per potential unit, shared over
            # prices. All evaluation units are independent of training units.
            for p in EVAL_PRICES:
                mu = mean_demand(p, x, world)
                observed = mu + world["latent"] * (z - .5) + world["noise"] * e
                rows.append([f"eval_{i:04d}_{p}", f"context_{i:04d}", x, p, mu, p * mu, observed])
        fields = ["id", "context_id", "x", "price", "mean_demand", "expected_revenue", "realized_demand"]
    elif kind in ("observational", "randomized"):
        units = stream(family, seed, "training_units")
        assignment = stream(family, seed, f"assignment_{kind}")
        for i in range(training_units):
            x, z, e = units.uniform(-1, 1), units.randrange(2), units.uniform(-1, 1)
            p = (assignment.choice(TRAIN_PRICES) if kind == "randomized" else
                 assignment.choices(TRAIN_PRICES, weights=assignment_probabilities(z), k=1)[0])
            y = mean_demand(p, x, world) + world["latent"] * (z - .5) + world["noise"] * e
            rows.append([f"train_{i:04d}", x, p, y])
        fields = ["id", "x", "price", "demand"]
    else:
        raise ValueError(kind)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--kind", choices=["observational", "randomized", "evaluation"], required=True)
    p.add_argument("--training-units", type=int, default=160)
    p.add_argument("--evaluation-contexts", type=int, default=64)
    a = p.parse_args()
    generate(a.output, a.family, a.seed, a.kind, a.training_units, a.evaluation_contexts)
