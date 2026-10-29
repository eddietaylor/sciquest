"""Transparent synthetic structural models. Never pass a private world to an agent."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import secrets
from dataclasses import asdict, dataclass, field

from sciquest.io import canonical_json, finite_number

VERSION = "demand-laboratory-v1"
FAMILIES = ("null", "linear", "exponential", "segmented", "other")
SCENARIOS = (
    "clean",
    "confounded",
    "censored",
    "noisy",
    "observational_only",
    "misspecified",
)
PRICES = (0.5, 0.75, 1.0, 1.25, 1.5)
CONTEXTS = (-1.0, 0.0, 1.0)
CAPACITIES = (60.0, 180.0)


def stream(seed: int, purpose: str, index: int = 0) -> random.Random:
    digest = hashlib.sha256(canonical_json([seed, purpose, index]).encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def catalog() -> list[dict]:
    models = []
    for family in FAMILIES[:-1]:
        sensitivities = (
            (0,) if family == "null" else (20, 40) if family == "linear" else (0.4, 0.8)
        )
        for baseline, context, sensitivity in itertools.product(
            (90, 110), (0, 10), sensitivities
        ):
            models.append(
                dict(
                    family=family,
                    baseline=baseline,
                    context_effect=context,
                    sensitivity=sensitivity,
                )
            )
    return models


@dataclass(frozen=True)
class DemandConfig:
    scenario: str = "clean"
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    budget: int = 32
    family: str | None = None
    price_limit: float = 1.5
    world_spec: dict | None = None

    def __post_init__(self):
        if self.scenario not in SCENARIOS:
            raise ValueError("Unknown demand-lab scenario")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("Seed must be an unsigned 32-bit integer")
        if (
            isinstance(self.budget, bool)
            or not isinstance(self.budget, int)
            or not 1 <= self.budget <= 10000
        ):
            raise ValueError("Budget must be between 1 and 10000 cost units")
        if self.family is not None and self.family not in FAMILIES[:-1]:
            raise ValueError("Unknown candidate mechanism family")
        if not 0.5 <= finite_number(self.price_limit, "price_limit") <= 1.5:
            raise ValueError("price_limit must be in [0.5, 1.5]")
        if self.world_spec is not None:
            if not isinstance(self.world_spec, dict):
                raise ValueError("World specification must be a mapping")
            if self.world_spec.get("version") == "designed-demand-v2":
                from .designed_v2 import validate_spec
            else:
                from .designed import validate_spec

            object.__setattr__(self, "world_spec", validate_spec(self.world_spec))
            if self.family is not None or (self.scenario == "observational_only") != (
                self.world_spec["stratum"] == "observational_only"
            ):
                raise ValueError(
                    "Designed-world access must match its apparatus configuration"
                )
            if (
                self.world_spec["version"] == "designed-demand-v2"
                and self.price_limit != self.world_spec["price_limit"]
            ):
                raise ValueError(
                    "V2 session price limit must match its admitted public apparatus"
                )


def private_world(config: DemandConfig) -> dict:
    if config.world_spec is not None:
        return dict(
            config=asdict(config), model=dict(family="designed"), spec=config.world_spec
        )
    rng = stream(config.seed, "world")
    family = config.family or rng.choice(FAMILIES[:-1])
    possible = [m for m in catalog() if m["family"] == family]
    model = dict(rng.choice(possible))
    if config.scenario == "misspecified":
        model.update(family="logistic", sensitivity=0.8)
    return dict(config=asdict(config), model=model)


def public_contract(config: DemandConfig) -> dict:
    """Positive allowlist of apparatus controls and measurement semantics only.

    This must not depend on the mechanism, seed, nuisance process or noise law.
    Capabilities are observable properties of the apparatus, not explanations.
    """
    contract = dict(
        environment_version="demand-black-box-v2",
        question="How do demand and sales respond to changes in price and context? Develop and test your own explanations.",
        prices=list(PRICES),
        contexts=list(CONTEXTS),
        capacities=list(CAPACITIES),
        budget=config.budget,
        interventions_available=config.scenario != "observational_only",
        costs={"observe": 1, "intervene": 2, "precise_surcharge": 2, "measure": 5},
        price_limit=config.price_limit,
        measurements={
            "observe": "Request a sales reading at a specified context; historical price and capacity are reported with the reading.",
            "intervene": "Set price, context and capacity; receive a sales reading for a fresh unit.",
            "replicate": "Repeat a previous action on a fresh unit.",
            "measure": "Request a demand reading on the same unit as a previous sales reading. These are paired measurements, not independent units.",
            "precision": "Sales readings offer standard or precise instrumentation. Instrument errors must be learned from observations.",
        },
        forecasts="Submit expected demand, expected sales, and central 90% prediction intervals for future sales readings at supplied evaluation features.",
        scoring="Demand/sales RMSE, interval coverage and interval score; price times expected sales determines decisions scored by revenue, regret and price-limit violations. No supplied hypothesis catalog or automated correctness grade for an open-ended explanation.",
        prediction_bounds={
            "absolute_max": 1000000,
            "mean_demand_min": 0,
            "mean_sales": "[0, capacity]",
        },
    )
    if config.world_spec and config.world_spec["version"] == "designed-demand-v2":
        from .designed_v2 import REFERENCES

        contract["environment_version"] = "demand-black-box-v3"
        contract["calibration_references"] = list(REFERENCES)
        contract["costs"]["calibrate"] = 1
        contract["measurements"][
            "calibrate"
        ] = "Measure a known reference quantity using standard or precise instrumentation. Supply reference and precision. This is an instrument reading, not a customer or demand observation."
    return contract


def assisted_contract(config: DemandConfig) -> dict:
    """Owner-only specification for the archived, assisted design benchmark."""
    return {
        "environment_version": VERSION,
        "question": "Which demand mechanism explains sales, and how does demand respond to price interventions?",
        "estimands": [
            "expected latent demand under do(price)",
            "expected physical sales under do(price)",
            "future noisy sales measurements",
        ],
        "families": list(FAMILIES),
        "candidate_models": catalog(),
        "prices": list(PRICES),
        "contexts": list(CONTEXTS),
        "capacities": list(CAPACITIES),
        "budget": config.budget,
        "interventions_available": config.scenario != "observational_only",
        "historical_assignment": (
            "endogenous"
            if config.scenario in ("confounded", "observational_only")
            else "randomized"
        ),
        "historical_capacity": 60 if config.scenario == "censored" else 180,
        "sensor_sigma": 12 if config.scenario == "noisy" else 3,
        "precise_sensor_sigma": 1,
        "costs": {"observe": 1, "intervene": 2, "precise_surcharge": 2, "measure": 5},
        "price_limit": config.price_limit,
        "nuisance_model": "u uniformly distributed over -1,0,1. Randomized history has additive 8*u. Endogenous history sets price=1+u/4 and adds 20*u - mechanism_effect(1+u/4,context); all candidate families have identical historical distributions at fixed baseline/context parameters.",
        "evaluation_scope": "Fresh private observations, unseen prices/contexts, and a finite decision grid. No real-market validation or multi-period inventory simulation.",
        "scoring": {
            "mechanisms": "Five-class Brier sum and natural-log loss; zero truth probability gives undefined finite log loss plus an explicit flag.",
            "forecasts": "Demand and sales RMSE; separate interpolation/extrapolation demand RMSE. Central 90% future noisy-sales interval coverage and interval score (width + 20 times outside distance).",
            "decisions": "Maximize price times predicted expected sales within each eligible state; lowest-price tie break. Protected expected revenue determines value and unconstrained finite-grid regret. Operating price-limit violations are scored separately, never silently repaired.",
            "prediction_bounds": {
                "absolute_max": 1000000,
                "mean_demand_min": 0,
                "mean_sales": "[0, capacity]",
            },
            "aggregation": "No combined score for scientific understanding and decision utility.",
        },
        "interval_nominal_coverage": 0.9,
        "recovery_confidence_threshold": 0.95,
        "observation_model": "Physical demand is nonnegative and sales are capacity-capped. Independent Gaussian sensor error is then added; measured values may exceed physical bounds. Audits remeasure the SAME unit's latent demand and are correlated with its earlier sales measurement.",
    }


def effect(model: dict, price: float, context: float) -> float:
    p = price - 1
    family, s, base = model["family"], model["sensitivity"], model["baseline"]
    if family == "null":
        return 0.0
    if family == "linear":
        return -s * p
    if family == "exponential":
        return base * math.expm1(-s * p)
    if family == "segmented":
        weight = 0.2 + 0.6 * (context + 1) / 2
        return base * (
            weight * math.exp(-s * 0.25 * p) + (1 - weight) * math.exp(-s * 2 * p) - 1
        )
    if family == "logistic":
        return base * (2 / (1 + math.exp(8 * s * p)) - 1)
    raise ValueError("Unknown structural mechanism")


def demand(
    model: dict, price: float, context: float, u: int, endogenous: bool
) -> float:
    nuisance = 20 * u - effect(model, 1 + u / 4, context) if endogenous else 8 * u
    return max(
        0.0,
        model["baseline"]
        + model["context_effect"] * context
        + effect(model, price, context)
        + nuisance,
    )


def components(
    model: dict,
    price: float,
    context: float,
    capacity: float,
    endogenous: bool,
    quantity: str = "sales",
) -> list[float]:
    values = [demand(model, price, context, u, endogenous) for u in (-1, 0, 1)]
    return [min(capacity, v) for v in values] if quantity == "sales" else values


def density(value: float, means: list[float], sigma: float) -> float:
    return math.fsum(
        math.exp(-0.5 * ((value - m) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        for m in means
    ) / len(means)


def bounded_probability(value: float) -> float:
    """Clamp only floating-point roundoff in mathematically normalized sums."""
    if not math.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12:
        raise ValueError("Predictive probability outside the unit interval")
    return min(1.0, max(0.0, value))


def event_probability(threshold: float, means: list[float], sigma: float) -> float:
    return math.fsum(
        0.5 * (1 + math.erf((threshold - m) / (sigma * math.sqrt(2)))) for m in means
    ) / len(means)


def normalize_action(
    action: dict, public: dict, previous: list[dict]
) -> tuple[dict, int]:
    if not isinstance(action, dict):
        raise ValueError("Action must be a mapping")
    kind = action.get("kind")
    if not isinstance(kind, str):
        raise ValueError(
            "action.kind must be a string naming an available apparatus action"
        )
    if kind == "calibrate":
        if "calibration_references" not in public:
            raise ValueError("action.kind: calibration is unavailable")
        allowed = {"kind", "reference", "precision"}
        if set(action) - allowed:
            raise ValueError(
                "action.calibrate unknown fields: "
                + ", ".join(sorted(set(action) - allowed))
                + "; allowed fields: "
                + ", ".join(sorted(allowed))
            )
        reference = finite_number(action.get("reference"), "reference")
        precision = action.get("precision", "standard")
        if reference not in public["calibration_references"] or precision not in (
            "standard",
            "precise",
        ):
            raise ValueError("Invalid calibration reference or precision")
        return dict(kind=kind, reference=reference, precision=precision), public[
            "costs"
        ]["calibrate"] + (
            public["costs"]["precise_surcharge"] if precision == "precise" else 0
        )
    if kind not in ("observe", "intervene", "replicate", "measure"):
        raise ValueError("Unknown demand action")
    allowed = (
        {"kind", "context", "price", "capacity", "precision"}
        if kind == "intervene"
        else (
            {"kind", "context", "precision"}
            if kind == "observe"
            else {"kind", "observation_id"}
        )
    )
    if set(action) - allowed:
        raise ValueError(
            "action."
            + kind
            + " unknown fields: "
            + ", ".join(sorted(set(action) - allowed))
            + "; allowed fields: "
            + ", ".join(sorted(allowed))
        )
    if kind in ("replicate", "measure"):
        matches = [
            r
            for r in previous
            if r["observation"]["id"] == action.get("observation_id")
        ]
        if len(matches) != 1:
            raise ValueError("Action needs an existing observation_id")
        origin = matches[0]
        if origin["design"]["kind"] == "measure":
            raise ValueError("Replicate or audit an original experimental unit")
        if kind == "measure" and origin["design"]["kind"] == "calibrate":
            raise ValueError("Calibration has no customer demand unit to audit")
        if kind == "replicate":
            design, cost = normalize_action(
                {k: v for k, v in origin["design"].items() if k != "replicates"},
                public,
                previous,
            )
            return {**design, "replicates": action["observation_id"]}, cost
        return {"kind": "measure", "observation_id": action["observation_id"]}, public[
            "costs"
        ]["measure"]
    if kind == "intervene" and not public["interventions_available"]:
        raise ValueError("Only observational access is available")
    context = finite_number(action.get("context", 0), "context")
    if context not in CONTEXTS:
        raise ValueError("Context is outside the query strata")
    precision = action.get("precision", "standard")
    if precision not in ("standard", "precise"):
        raise ValueError("Unknown measurement precision")
    design = dict(kind=kind, context=context, precision=precision)
    if kind == "intervene":
        price = finite_number(action.get("price"), "price")
        capacity = finite_number(action.get("capacity", 180), "capacity")
        if price not in PRICES or capacity not in CAPACITIES:
            raise ValueError("Intervention outside permitted action space")
        design.update(price=price, capacity=capacity)
    elif "price" in action or "capacity" in action:
        raise ValueError(
            "Historical price and capacity cannot be set by an observation request"
        )
    return design, public["costs"][kind] + (
        public["costs"]["precise_surcharge"] if precision == "precise" else 0
    )


def draw(
    world: dict, design: dict, index: int, private_history: list[dict]
) -> tuple[dict, dict]:
    if "spec" in world:
        if world["spec"]["version"] == "designed-demand-v2":
            from .designed_v2 import draw as designed_draw
        else:
            from .designed import draw as designed_draw

        return designed_draw(world, design, index, private_history)
    config = DemandConfig(**world["config"])
    public = assisted_contract(config)
    rng = stream(config.seed, "discovery", index)
    endogenous = public["historical_assignment"] == "endogenous"
    if design["kind"] == "measure":
        origin = next(
            r
            for r in private_history
            if r["observation"]["id"] == design["observation_id"]
        )
        hidden = origin["private"]
        obs = dict(
            id=f"q{index:05d}",
            kind="measure",
            price=origin["observation"]["price"],
            context=origin["observation"]["context"],
            capacity=origin["observation"]["capacity"],
            sigma=1,
            quantity="latent_demand",
            value=hidden["demand"] + rng.gauss(0, 1),
            paired_with=design["observation_id"],
        )
        return obs, dict(hidden)
    u = rng.choice((-1, 0, 1))
    context = design["context"]
    if design["kind"] == "observe":
        price = 1 + u / 4 if endogenous else rng.choice(PRICES)
        capacity = public["historical_capacity"]
    else:
        price, capacity = design["price"], design["capacity"]
    latent = demand(world["model"], price, context, u, endogenous)
    sales = min(capacity, latent)
    sigma = 1 if design["precision"] == "precise" else public["sensor_sigma"]
    obs = dict(
        id=f"q{index:05d}",
        kind=design["kind"],
        price=price,
        context=context,
        capacity=capacity,
        sigma=sigma,
        quantity="physical_sales",
        value=sales + rng.gauss(0, sigma),
    )
    if "replicates" in design:
        obs["replicates"] = design["replicates"]
    return obs, dict(u=u, demand=latent, sales=sales)


def evaluation_rows(world: dict) -> list[dict]:
    if "spec" in world:
        if world["spec"]["version"] == "designed-demand-v2":
            from .designed_v2 import evaluation_rows as designed_rows
        else:
            from .designed import evaluation_rows as designed_rows

        return designed_rows(world)
    config = DemandConfig(**world["config"])
    public = assisted_contract(config)
    endog = public["historical_assignment"] == "endogenous"
    rows = []
    for i, (x, cap, p) in enumerate(
        itertools.product(
            (-1, -0.5, 0, 0.5, 1), CAPACITIES, (0.4, 0.625, 0.875, 1.125, 1.375, 1.6)
        )
    ):
        d = components(world["model"], p, x, cap, endog, "demand")
        s = [min(cap, v) for v in d]
        rng = stream(config.seed, "sealed-evaluation", i)
        measured = rng.choice(s) + rng.gauss(0, public["sensor_sigma"])
        rows.append(
            dict(
                id=f"eval_{i:03d}",
                state_id=f"x{x}_c{cap}",
                price=p,
                context=x,
                capacity=cap,
                split="interpolation" if 0.5 <= p <= 1.5 else "extrapolation",
                decision_eligible=int(0.5 <= p <= 1.5),
                price_limit=config.price_limit,
                expected_demand=math.fsum(d) / 3,
                expected_sales=math.fsum(s) / 3,
                measured_sales=measured,
                expected_revenue=p * math.fsum(s) / 3,
                violation=int(p > config.price_limit),
            )
        )
    return rows


EVAL_FEATURES = (
    "id",
    "state_id",
    "price",
    "context",
    "capacity",
    "split",
    "decision_eligible",
    "price_limit",
)
