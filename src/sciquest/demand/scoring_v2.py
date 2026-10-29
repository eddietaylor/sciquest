"""Versioned private v2 scores. No equation, oracle or score is an agent hint."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from sciquest.evaluation import _number, _table
from .designed_v2 import latent
from .world import evaluation_rows

VERSION = "demand-scoring-v2.1"
ALPHA = 0.1
FIELDS = ("mean_demand", "mean_sales", "lower_sales", "upper_sales")
SCORING_CONTRACT = {
    "version": VERSION,
    "prediction": "Separate RMSE of expected latent demand and capped physical sales on query-setting rows and transfer rows. Query-setting does not imply access in observational-only campaigns, or adequate finite-budget evidence.",
    "uncertainty": "Central 90% intervals for fresh standard-precision noisy sales readings, including instrument error. Expected interval score, width and coverage integrate the private finite Gaussian mixture. Coverage alone is not a proper score; no prose-based numerical calibration score.",
    "decision": "Choose maximum predicted price*mean_sales among listed decision_eligible actions at or below the public price_limit; break ties at lowest price. Equal weight to six states. Compare expected physical revenue with feasible oracle, highest-feasible and best-constant baselines. Unfiltered preference violations are separate diagnostics, not deployed violations.",
    "causal": "No causal or explanation credit from forecast/decision scores or a status label. Claim-level evidence/identification review is separate; unsupported true guesses remain unsupported.",
    "failure": "Keep failures in completion and intent-to-run results. Numeric components unavailable on failure remain null, never zero or dropped. A future study must register its failure-inclusive primary and loss scales before campaigns.",
    "aggregation": "Equal independent-world weight, paired arms; row and state counts are not independent sample sizes. No composite scientific-success score.",
}


def _cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _positive_normal(delta, sigma):
    """E[(N(delta,sigma^2))_+], including stable far-tail handling."""
    z = delta / sigma
    return max(
        0.0, sigma * math.exp(-z * z / 2) / math.sqrt(2 * math.pi) + delta * _cdf(z)
    )


def interval_expectations(low, high, means, sigma):
    if low > high or sigma <= 0 or not means:
        raise ValueError("Invalid interval or measurement distribution")
    coverage = math.fsum(
        _cdf((high - m) / sigma) - _cdf((low - m) / sigma) for m in means
    ) / len(means)
    penalty = math.fsum(
        _positive_normal(low - m, sigma) + _positive_normal(m - high, sigma)
        for m in means
    ) / len(means)
    return dict(
        coverage_90=max(0.0, min(1.0, coverage)),
        width=high - low,
        expected_interval_score_90=high - low + 2 / ALPHA * penalty,
    )


def score_predictions(world: dict, artifact: Path) -> dict:
    if world.get("spec", {}).get("version") != "designed-demand-v2":
        raise ValueError("V2 scoring requires a v2 world")
    rows = evaluation_rows(world)
    pred = _table(Path(artifact), "id")
    if set(pred) != {r["id"] for r in rows}:
        raise ValueError("Predictions must cover all sealed rows exactly")
    spec = world["spec"]
    buckets = defaultdict(list)
    states = defaultdict(list)
    inconsistency = 0
    for row in rows:
        p = {k: _number(pred[row["id"]].get(k), k) for k in FIELDS}
        if any(abs(v) > 1e6 for v in p.values()):
            raise ValueError("Predictions exceed numerical safety bounds")
        if p["mean_demand"] < 0 or not 0 <= p["mean_sales"] <= row["capacity"]:
            raise ValueError("Predicted means violate physical bounds")
        if p["lower_sales"] > p["upper_sales"]:
            raise ValueError("Reversed predictive interval")
        inconsistency += p["mean_sales"] > p["mean_demand"] + 1e-9
        means = [
            min(row["capacity"], latent(spec, row["price"], row["context"], z))
            + spec["sensor_bias"]
            for z in (-1, 0, 1)
        ]
        interval = interval_expectations(
            p["lower_sales"], p["upper_sales"], means, spec["sensor_sigma"]
        )
        v = dict(
            demand_squared_error=(p["mean_demand"] - row["expected_demand"]) ** 2,
            sales_squared_error=(p["mean_sales"] - row["expected_sales"]) ** 2,
            **interval,
        )
        buckets["all"].append(v)
        buckets["query_settings" if row["decision_eligible"] else "transfer"].append(v)
        if row["decision_eligible"]:
            states[row["state_id"]].append(
                {**row, "predicted_revenue": row["price"] * p["mean_sales"]}
            )
    prediction = {}
    for name, entries in buckets.items():
        avg = lambda key: math.fsum(e[key] for e in entries) / len(entries)
        prediction[name] = dict(
            rows=len(entries),
            demand_rmse=math.sqrt(avg("demand_squared_error")),
            sales_rmse=math.sqrt(avg("sales_squared_error")),
            **{
                key: avg(key)
                for key in ("coverage_90", "width", "expected_interval_score_90")
            },
        )
    # Public feasibility is applied BEFORE choosing; invalid unconstrained
    # preferences cannot earn a super-oracle negative constrained regret.
    evidence = []
    all_prices = sorted(
        {
            r["price"]
            for group in states.values()
            for r in group
            if r["price"] <= r["price_limit"]
        }
    )
    constants = {
        price: math.fsum(
            next(r["expected_revenue"] for r in group if r["price"] == price)
            for group in states.values()
        )
        / len(states)
        for price in all_prices
    }
    best_constant = min(constants, key=lambda price: (-constants[price], price))
    for identity, group in sorted(states.items()):
        feasible = [r for r in group if r["price"] <= r["price_limit"]]
        choose = lambda rs, key: min(rs, key=lambda r: (-r[key], r["price"]))
        selected = choose(feasible, "predicted_revenue")
        oracle = choose(feasible, "expected_revenue")
        raw = choose(group, "predicted_revenue")
        highest = max(feasible, key=lambda r: r["price"])
        constant = next(r for r in feasible if r["price"] == best_constant)
        evidence.append(
            dict(
                state_id=identity,
                selected_price=selected["price"],
                expected_revenue=selected["expected_revenue"],
                feasible_oracle_revenue=oracle["expected_revenue"],
                feasible_regret=max(
                    0.0, oracle["expected_revenue"] - selected["expected_revenue"]
                ),
                highest_feasible_regret=max(
                    0.0, oracle["expected_revenue"] - highest["expected_revenue"]
                ),
                best_constant_regret=max(
                    0.0, oracle["expected_revenue"] - constant["expected_revenue"]
                ),
                unfiltered_preference_price=raw["price"],
                unfiltered_preference_violation=raw["price"] > raw["price_limit"],
            )
        )
    mean = lambda key: math.fsum(e[key] for e in evidence) / len(evidence)
    return dict(
        version=VERSION,
        prediction=prediction,
        decision=dict(
            states=len(evidence),
            policy_value=mean("expected_revenue"),
            feasible_oracle_value=mean("feasible_oracle_revenue"),
            feasible_regret=mean("feasible_regret"),
            highest_feasible_regret=mean("highest_feasible_regret"),
            best_constant_regret=mean("best_constant_regret"),
            best_constant_price=best_constant,
            improvement_over_highest=mean("highest_feasible_regret")
            - mean("feasible_regret"),
            improvement_over_best_constant=mean("best_constant_regret")
            - mean("feasible_regret"),
            unfiltered_preference_violation_rate=mean(
                "unfiltered_preference_violation"
            ),
            deployed_violation_rate=0.0,
            feasibility_enforced_by_supervisor=True,
            evidence=evidence,
        ),
        prediction_consistency_violations=inconsistency,
        contract=SCORING_CONTRACT,
    )
