"""Private v2 physics. Public apparatus never includes these laws or parameters."""

from __future__ import annotations

import ast
import itertools
import json
import math

from sciquest.io import canonical_json, finite_number
from .designed import (
    P_GRID,
    X_GRID,
    Z_GRID,
    STRATA,
    expression,
    latent as original_latent,
    raw,
)

VERSION = "designed-demand-v2"
REFERENCES = (0.0, 60.0, 120.0)
FIELDS = {
    "version",
    "stratum",
    "expression",
    "sensor_sigma",
    "precise_sigma",
    "sensor_bias",
    "confounding_strength",
    "price_limit",
    "description",
    "scientific_challenge",
}


def latent(spec, p, x, z):
    if spec["stratum"] in ("confounded", "observational_only") and p == 1 + z / 4:
        # Stable canonical form makes historical twins bitwise identical before
        # adding the same instrument noise, not merely algebraically equivalent.
        return max(
            0.0, raw(spec["expression"], 1, x, z) + spec["confounding_strength"] * z
        )
    return original_latent(spec, p, x, z)


def validate_spec(spec):
    if (
        not isinstance(spec, dict)
        or set(spec) != FIELDS
        or spec["version"] != VERSION
        or spec["stratum"] not in STRATA
    ):
        raise ValueError("Invalid v2 world specification")
    for key in ("description", "scientific_challenge"):
        if not isinstance(spec[key], str) or not 5 <= len(spec[key]) <= 2000:
            raise ValueError("World documentation must be bounded nonempty text")
    sigma, precise, bias, strength, limit = [
        finite_number(spec[k], k)
        for k in (
            "sensor_sigma",
            "precise_sigma",
            "sensor_bias",
            "confounding_strength",
            "price_limit",
        )
    ]
    if not (
        0.5 <= sigma <= 4
        and 0.2 <= precise <= min(2, sigma)
        and abs(bias) <= 4
        and 0 <= strength <= 20
        and limit in (1.0, 1.25)
    ):
        raise ValueError(
            "V2 instrumentation, selection or operating limit outside design bounds"
        )
    expression(spec["expression"])
    if spec["stratum"] == "flat" and any(
        isinstance(n, ast.Name) and n.id == "p"
        for n in ast.walk(ast.parse(spec["expression"], mode="eval"))
    ):
        raise ValueError("Invariant control must contain no price variable")
    try:
        values = [
            latent(spec, p, x, z)
            for p, x, z in itertools.product(P_GRID, X_GRID, Z_GRID)
        ]
        if any(not 0 <= v <= 300 for v in values):
            raise ValueError("Demand outside finite validated physical range")
        means = {
            (p, x): math.fsum(latent(spec, p, x, z) for z in Z_GRID) / 3
            for p, x in itertools.product(P_GRID, X_GRID)
        }
        if any(
            means[p, x] < means[q, x] - 1e-9
            for x in X_GRID
            for p, q in zip(P_GRID, P_GRID[1:])
        ):
            raise ValueError(
                "Expected demand must be nonincreasing on the validated price grid"
            )
        spans = [means[0.5, x] - means[1.5, x] for x in X_GRID]
        if spec["stratum"] != "flat" and max(spans) < 40:
            raise ValueError(
                "V2 price response too small for a useful discovery challenge"
            )
        if spec["stratum"] == "nonlinear":
            # Departure from the best affine price response, not a single midpoint proxy.
            mean_p = math.fsum(P_GRID) / len(P_GRID)
            departures = []
            for x in X_GRID:
                mean_y = math.fsum(means[p, x] for p in P_GRID) / len(P_GRID)
                slope = math.fsum(
                    (p - mean_p) * (means[p, x] - mean_y) for p in P_GRID
                ) / math.fsum((p - mean_p) ** 2 for p in P_GRID)
                departures.append(
                    max(
                        abs(means[p, x] - mean_y - slope * (p - mean_p)) for p in P_GRID
                    )
                )
            if max(departures) < 3:
                raise ValueError(
                    "Nonlinear stratum needs a substantive non-affine mean response"
                )
        if spec["stratum"] == "interaction" and max(spans) - min(spans) < 15:
            raise ValueError(
                "Interaction stratum needs different context-specific price responses"
            )
        if spec["stratum"] == "censored" and not any(
            latent(spec, 1, x, z) > 75 for x, z in itertools.product(X_GRID, Z_GRID)
        ):
            raise ValueError(
                "Censored stratum needs hidden demand above historical capacity"
            )
    except (ArithmeticError, TypeError) as exc:
        raise ValueError("Undefined expression on the finite apparatus domain") from exc
    return json.loads(canonical_json(spec))


def draw(world, design, index, previous):
    from .world import stream, PRICES

    spec = world["spec"]
    rng = stream(world["config"]["seed"], "discovery", index)
    precision = "precise" if design["kind"] == "measure" else design["precision"]
    sigma = spec["precise_sigma"] if precision == "precise" else spec["sensor_sigma"]
    if design["kind"] == "calibrate":
        obs = dict(
            id=f"q{index:05d}",
            kind="calibrate",
            quantity="reference",
            reference=design["reference"],
            precision=precision,
            value=design["reference"] + spec["sensor_bias"] + rng.gauss(0, sigma),
            sigma=sigma,
        )
        hidden = {}
    elif design["kind"] == "measure":
        origin = next(
            r for r in previous if r["observation"]["id"] == design["observation_id"]
        )
        hidden = dict(origin["private"])
        obs = dict(
            id=f"q{index:05d}",
            kind="measure",
            quantity="latent_demand",
            price=origin["observation"]["price"],
            context=origin["observation"]["context"],
            capacity=origin["observation"]["capacity"],
            precision=precision,
            sigma=sigma,
            paired_with=design["observation_id"],
            value=hidden["demand"] + spec["sensor_bias"] + rng.gauss(0, sigma),
        )
    else:
        z = rng.choice(Z_GRID)
        x = design["context"]
        if design["kind"] == "observe":
            p = (
                1 + z / 4
                if spec["stratum"] in ("confounded", "observational_only")
                else rng.choice(PRICES)
            )
            cap = 60 if spec["stratum"] == "censored" else 180
        else:
            p, cap = design["price"], design["capacity"]
        demand = latent(spec, p, x, z)
        sales = min(cap, demand)
        hidden = dict(u=z, demand=demand, sales=sales)
        obs = dict(
            id=f"q{index:05d}",
            kind=design["kind"],
            quantity="physical_sales",
            price=p,
            context=x,
            capacity=cap,
            precision=precision,
            sigma=sigma,
            value=sales + spec["sensor_bias"] + rng.gauss(0, sigma),
        )
    if "replicates" in design:
        obs["replicates"] = design["replicates"]
    return obs, hidden


def evaluation_rows(world):
    from .world import CAPACITIES, CONTEXTS, PRICES, stream

    spec = world["spec"]
    rows = []
    for i, (x, cap, p) in enumerate(itertools.product(X_GRID, CAPACITIES, P_GRID)):
        demands = [latent(spec, p, x, z) for z in Z_GRID]
        sales = [min(cap, v) for v in demands]
        mean_d, mean_s = math.fsum(demands) / 3, math.fsum(sales) / 3
        rng = stream(world["config"]["seed"], "sealed-evaluation", i)
        rows.append(
            dict(
                id=f"eval_{i:03d}",
                state_id=f"x{x}_c{cap}",
                price=p,
                context=x,
                capacity=cap,
                split="interpolation" if 0.5 <= p <= 1.5 else "extrapolation",
                decision_eligible=int(p in PRICES and x in CONTEXTS),
                price_limit=world["config"]["price_limit"],
                expected_demand=mean_d,
                expected_sales=mean_s,
                measured_sales=rng.choice(sales)
                + spec["sensor_bias"]
                + rng.gauss(0, spec["sensor_sigma"]),
                expected_revenue=p * mean_s,
                violation=int(p > world["config"]["price_limit"]),
            )
        )
    return rows
