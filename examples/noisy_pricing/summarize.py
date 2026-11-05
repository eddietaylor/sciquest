"""Campaign-level descriptive summaries; never treats action rows as replicates."""
from __future__ import annotations

import math
import statistics

METRICS = ("regret", "mean_demand_rmse", "individual_demand_rmse", "policy_value", "seen_price_rmse", "unseen_price_rmse")


def describe(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        return dict(n=0, mean=None, sd=None, mcse=None, p10=None, median=None, p90=None)
    # Linear interpolation, zero-index h=(n-1)*q (numpy's default 'linear').
    def quantile(q):
        h = (len(values) - 1) * q
        lo = math.floor(h)
        return values[lo] + (h - lo) * (values[math.ceil(h)] - values[lo])
    sd = statistics.stdev(values) if len(values) > 1 else None
    return dict(n=len(values), mean=statistics.fmean(values), sd=sd,
                mcse=None if sd is None else sd / math.sqrt(len(values)),
                p10=quantile(.1), median=quantile(.5), p90=quantile(.9))


def summarize(protocol: dict, outcomes: list[dict]) -> dict:
    groups, pairs = [], []
    for family in protocol["families"]:
        for baseline in protocol["baselines"]:
            entries = [r for r in outcomes if r["family"] == family and r["baseline"] == baseline["name"]]
            completed = [r for r in entries if r["status"] == "completed"]
            groups.append(dict(family=family, baseline=baseline["name"], planned=len(entries),
                               completed=len(completed), all_planned_completed=len(entries) == len(completed),
                               metrics={m: describe([r["metrics"][m] for r in completed]) for m in METRICS}))
        for left, right in protocol["paired_comparisons"]:
            entries = [r for r in outcomes if r["family"] == family]
            lookup = {(r["seed"], r["baseline"]): r for r in entries}
            differences = {m: [] for m in METRICS}
            for seed in protocol["replicate_seeds"]:
                a, b = lookup.get((seed, left)), lookup.get((seed, right))
                if a and b and a["status"] == b["status"] == "completed":
                    for m in METRICS:
                        differences[m].append(a["metrics"][m] - b["metrics"][m])
            pairs.append(dict(family=family, left=left, right=right, direction="left minus right",
                              planned_pairs=len(protocol["replicate_seeds"]),
                              metrics={m: describe(v) for m, v in differences.items()}))
    return dict(scope="Per-family campaign draws; available-case descriptive summaries with explicit denominators. No pooled headline, confidence interval, p-value, or row-level standard error.",
                planned=len(outcomes), completed=sum(r["status"] == "completed" for r in outcomes),
                groups=groups, paired_comparisons=pairs)
