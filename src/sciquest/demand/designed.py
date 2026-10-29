"""Supervisor-only, bounded expression worlds. Never import this in a scientist.

Expressions are mathematical data in a small AST grammar, not executable agent
programs. The apparatus and scoring stay under supervisor control. New laws can
be composed without choosing a known family or exposing a candidate taxonomy.
"""

from __future__ import annotations

import ast
import itertools
import math
from functools import lru_cache

from sciquest.io import canonical_json, content_hash, finite_number

VERSION = "designed-demand-v1"
STRATA = (
    "flat",
    "nonlinear",
    "interaction",
    "confounded",
    "censored",
    "observational_only",
)
FUNCTIONS = {
    name: getattr(math, name) for name in ("exp", "tanh", "sin", "cos", "log1p", "sqrt")
}
FUNCTIONS.update(abs=abs, min=min, max=max)
P_GRID = (0.4, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.6)
X_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
Z_GRID = (-1, 0, 1)


@lru_cache(maxsize=1024)
def expression(source: str):
    if not isinstance(source, str) or not 1 <= len(source) <= 1024:
        raise ValueError("A bounded mathematical expression is required")
    tree = ast.parse(source, mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 140:
        raise ValueError("Expression is too complex")
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Constant,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.IfExp,
        ast.Compare,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Eq,
    )
    for node in nodes:
        if not isinstance(node, allowed):
            raise ValueError("Expression contains forbidden syntax")
        if isinstance(node, ast.Constant):
            if (
                type(node.value) not in (int, float)
                or not math.isfinite(node.value)
                or abs(node.value) > 1000
            ):
                raise ValueError("Expression constants must be finite bounded numbers")
        if isinstance(node, ast.Name) and node.id not in {"p", "x", "z", *FUNCTIONS}:
            raise ValueError("Unknown expression variable or function")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in FUNCTIONS
            or node.keywords
            or not 1 <= len(node.args) <= 3
        ):
            raise ValueError("Only allowlisted mathematical calls are accepted")
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Pow)
            and (
                not isinstance(node.right, ast.Constant)
                or type(node.right.value) is not int
                or not 0 <= node.right.value <= 4
            )
        ):
            raise ValueError("Powers require a literal integer from zero to four")
    return compile(tree, "<private-demand-expression>", "eval")


def raw(source: str, p: float, x: float, z: int) -> float:
    value = eval(
        expression(source), {"__builtins__": {}, **FUNCTIONS}, dict(p=p, x=x, z=z)
    )
    return finite_number(value, "demand expression result")


def latent(spec: dict, p: float, x: float, z: int) -> float:
    source = spec["expression"]
    value = raw(source, p, x, z)
    if spec["stratum"] in ("confounded", "observational_only"):
        # The historical price is a function of the same unit's latent state.
        # Matching twins g(p,x,z) and g(1,x,z) have identical historical demand,
        # even with paired audits, while their counterfactual responses differ.
        value += (
            spec["confounding_strength"] * z
            - raw(source, 1 + z / 4, x, z)
            + raw(source, 1, x, z)
        )
    return max(0.0, value)


def validate_spec(spec: dict) -> dict:
    fields = {
        "version",
        "stratum",
        "expression",
        "sensor_sigma",
        "precise_sigma",
        "confounding_strength",
        "description",
        "scientific_challenge",
    }
    if (
        not isinstance(spec, dict)
        or set(spec) != fields
        or spec["version"] != VERSION
        or spec["stratum"] not in STRATA
    ):
        raise ValueError("Invalid designed-world specification")
    for key in ("description", "scientific_challenge"):
        if not isinstance(spec[key], str) or not 5 <= len(spec[key]) <= 2000:
            raise ValueError("World documentation must be bounded nonempty text")
    sigma = finite_number(spec["sensor_sigma"])
    precise = finite_number(spec["precise_sigma"])
    strength = finite_number(spec["confounding_strength"])
    if (
        not 1 <= sigma <= 8
        or not 0.2 <= precise <= min(2, sigma)
        or not 0 <= strength <= 25
    ):
        raise ValueError(
            "Instrumentation or confounding parameters outside design limits"
        )
    expression(spec["expression"])
    if spec["stratum"] == "flat" and any(
        isinstance(n, ast.Name) and n.id == "p"
        for n in ast.walk(ast.parse(spec["expression"], mode="eval"))
    ):
        raise ValueError("Flat control must not contain the price variable")
    try:
        values = [
            latent(spec, p, x, z)
            for p, x, z in itertools.product(P_GRID, X_GRID, Z_GRID)
        ]
        if any(not math.isfinite(v) or not 0 <= v <= 300 for v in values):
            raise ValueError("Demand outside the validated physical range")
        means = {
            (p, x): math.fsum(latent(spec, p, x, z) for z in Z_GRID) / 3
            for p, x in itertools.product(P_GRID, X_GRID)
        }
        if not all(
            means[p, x] >= means[q, x] - 1e-9
            for x in X_GRID
            for p, q in zip(P_GRID, P_GRID[1:])
        ):
            raise ValueError(
                "This collection requires nonincreasing expected demand with price"
            )
        span = max(means[0.5, x] - means[1.5, x] for x in X_GRID)
        if spec["stratum"] == "flat" and span > 1e-9:
            raise ValueError("Flat control must have no expected price response")
        if spec["stratum"] != "flat" and span < max(8, 2 * sigma):
            raise ValueError("Price response is too small for the declared challenge")
        if spec["stratum"] == "censored" and max(means.values()) < 75:
            raise ValueError(
                "Censored challenge needs demand above historical capacity"
            )
        if (
            spec["stratum"] == "nonlinear"
            and max(
                abs(means[0.5, x] + means[1.5, x] - 2 * means[1.0, x]) for x in X_GRID
            )
            < 1
        ):
            raise ValueError("Nonlinear challenge needs meaningful curvature")
        if spec["stratum"] == "interaction":
            responses = [means[0.5, x] - means[1.5, x] for x in X_GRID]
            if max(responses) - min(responses) < 3:
                raise ValueError(
                    "Interaction challenge needs context-dependent price response"
                )
    except (ArithmeticError, TypeError) as exc:
        raise ValueError("Expression is undefined on the apparatus domain") from exc
    return __import__("json").loads(canonical_json(spec))


def design_report(spec: dict, seed: int) -> dict:
    """Private apparatus checks; no scientist or held-out campaign is run here."""
    from .world import stream

    spec = validate_spec(spec)
    checks = []
    for p, x, cap in ((0.5, -1, 60), (1, 0, 180), (1.5, 1, 180)):
        d = [latent(spec, p, x, z) for z in Z_GRID]
        sales = [min(cap, v) for v in d]
        mean = math.fsum(sales) / 3
        variance = (
            math.fsum((v - mean) ** 2 for v in sales) / 3 + spec["sensor_sigma"] ** 2
        )
        rng = stream(seed, f"world-validation-{p}-{x}")
        sample = [
            rng.choice(sales) + rng.gauss(0, spec["sensor_sigma"]) for _ in range(1200)
        ]
        error = abs(math.fsum(sample) / len(sample) - mean)
        passed = error <= 7 * math.sqrt(variance / len(sample)) + 1e-10
        if not passed:
            raise ValueError("Independent sample-moment apparatus check failed")
        checks.append(
            dict(price=p, context=x, capacity=cap, passed=passed, mean_error=error)
        )
    twin = None
    if spec["stratum"] == "observational_only":
        max_error = max(
            abs(
                latent(spec, 1 + z / 4, x, z)
                - max(
                    0,
                    raw(spec["expression"], 1, x, z) + spec["confounding_strength"] * z,
                )
            )
            for x, z in itertools.product(X_GRID, Z_GRID)
        )
        twin = dict(
            historical_equivalence_max_error=max_error, causal_response_differs=True
        )
        if max_error > 1e-10:
            raise ValueError("Observational twin construction failed")
    return dict(
        validated=True,
        spec_hash=content_hash(spec),
        moment_checks=checks,
        observational_twin=twin,
        scope="Finite apparatus/evaluation domain, deterministic replay and sample moments. Signal checks do not establish finite-budget learnability or industry validity.",
    )


def draw(world: dict, design: dict, index: int, previous: list[dict]):
    from .world import PRICES, stream

    spec = world["spec"]
    rng = stream(world["config"]["seed"], "discovery", index)
    if design["kind"] == "measure":
        origin = next(
            r for r in previous if r["observation"]["id"] == design["observation_id"]
        )
        hidden = origin["private"]
        return dict(
            id=f"q{index:05d}",
            kind="measure",
            price=origin["observation"]["price"],
            context=origin["observation"]["context"],
            capacity=origin["observation"]["capacity"],
            sigma=spec["precise_sigma"],
            quantity="latent_demand",
            value=hidden["demand"] + rng.gauss(0, spec["precise_sigma"]),
            paired_with=design["observation_id"],
        ), dict(hidden)
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
    d = latent(spec, p, x, z)
    sales = min(cap, d)
    sigma = (
        spec["precise_sigma"]
        if design["precision"] == "precise"
        else spec["sensor_sigma"]
    )
    obs = dict(
        id=f"q{index:05d}",
        kind=design["kind"],
        price=p,
        context=x,
        capacity=cap,
        sigma=sigma,
        quantity="physical_sales",
        value=sales + rng.gauss(0, sigma),
    )
    if "replicates" in design:
        obs["replicates"] = design["replicates"]
    return obs, dict(u=z, demand=d, sales=sales)


def evaluation_rows(world: dict) -> list[dict]:
    from .world import CAPACITIES, stream

    spec, rows = world["spec"], []
    for i, (x, cap, p) in enumerate(
        itertools.product(X_GRID, CAPACITIES, (0.4, 0.625, 0.875, 1.125, 1.375, 1.6))
    ):
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
                decision_eligible=int(0.5 <= p <= 1.5),
                price_limit=world["config"]["price_limit"],
                expected_demand=mean_d,
                expected_sales=mean_s,
                measured_sales=rng.choice(sales) + rng.gauss(0, spec["sensor_sigma"]),
                expected_revenue=p * mean_s,
                violation=int(p > world["config"]["price_limit"]),
            )
        )
    return rows
