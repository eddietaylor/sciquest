"""Prospective, supervisor-only identities and seed coupling for v2 physics.

These identities do not amend a bank, certify admission or migrate old sessions.
See docs/research-next-phase/SIMULATOR_IDENTITY.txt for their mathematical scope.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from sciquest.io import atomic_text, canonical_json, content_hash, finite_number
from .designed import P_GRID, X_GRID, Z_GRID, STRATA, expression
from .designed_v2 import FIELDS, VERSION as PHYSICS_VERSION, latent
from .policies_v2 import ResourceLimits, assignment
from .world import CAPACITIES, CONTEXTS, PRICES, DemandConfig, public_contract

VERSION = "demand-identity-v3.1"
SEED_SPACE = 2**32


def _spec(spec):
    """Public-compatible finite physics, without private admission/stratum gates.

    In particular, a reference-preserving invariant observational twin is allowed.
    Actual sessions still use DemandConfig and its existing v2 validation.
    """
    if (
        not isinstance(spec, dict)
        or set(spec) != FIELDS
        or spec["version"] != PHYSICS_VERSION
        or spec["stratum"] not in STRATA
    ):
        raise ValueError("Expected a complete v2 specification")
    for key in ("description", "scientific_challenge"):
        if not isinstance(spec[key], str) or not 5 <= len(spec[key]) <= 2000:
            raise ValueError("Expected bounded specification documentation")
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
        and limit in (1, 1.25)
    ):
        raise ValueError("Invalid v2 apparatus parameters")
    expression(spec["expression"])
    return json.loads(canonical_json(spec))


def _number(value):
    number = finite_number(value)
    return 0.0 if number == 0 else number


def _demand(spec, price, context, z):
    value = _number(latent(spec, price, context, z))
    if not 0 <= value <= 300:
        raise ValueError("Demand outside finite physical domain")
    return value


def spec_commitment(spec):
    """Commit all supplied spec fields, including prose and expression syntax.

    Equality is canonical JSON identity, never mathematical law equivalence.
    Numeric JSON spellings such as 2 versus 2.0 can have different commitments.
    """
    return content_hash(dict(version=VERSION, kind="spec_commitment", spec=_spec(spec)))


def physical_marginal_payload(spec):
    """Finite marginal latent-demand laws; excludes sensors, access and prose.

    Sorted three-point multisets preserve multiplicity and forget latent labels.
    They describe each setting independently, not cross-setting potential outcomes.
    """
    spec = _spec(spec)
    return dict(
        version=VERSION,
        kind="finite_physical_marginals",
        price_grid=list(P_GRID),
        context_grid=list(X_GRID),
        latent_probabilities=[1, 1, 1],
        latent_denominator=3,
        rows=[
            dict(
                price=p,
                context=x,
                demand=sorted(_demand(spec, p, x, z) for z in Z_GRID),
            )
            for p, x in itertools.product(P_GRID, X_GRID)
        ],
    )


def physical_marginal_signature(spec):
    return content_hash(physical_marginal_payload(spec))


def _mixture(components):
    # Component tuples include discrete reported controls and joint reading means.
    # Combine exact duplicates; do not round numerics into an equivalence claim.
    counts = Counter(tuple(_number(v) for v in component) for component in components)
    return dict(
        denominator=len(components),
        components=[
            dict(values=list(values), numerator=count)
            for values, count in sorted(counts.items())
        ],
    )


def apparatus_task_payload(spec, *, observation_budget=24):
    """All available finite acquisition kernels plus public task constraints.

    Every kernel jointly represents reported price/capacity, one sales reading,
    and a same-unit precise demand audit. Conditional sensor errors are independent.
    Fresh replications redraw mixture components; repeated audits retain the unit.
    """
    spec = _spec(spec)
    if type(observation_budget) is not int or not 1 <= observation_budget <= 10000:
        raise ValueError("Invalid observation budget")
    observational = spec["stratum"] == "observational_only"
    endogenous = spec["stratum"] in ("confounded", "observational_only")
    capacity = 60 if spec["stratum"] == "censored" else 180
    # public_contract needs only these four attributes. Avoid imposing private
    # admission gates on scientifically compatible witness alternatives.
    public = public_contract(
        SimpleNamespace(
            scenario="observational_only" if observational else "clean",
            budget=observation_budget,
            price_limit=spec["price_limit"],
            world_spec=spec,
        )
    )
    bias = spec["sensor_bias"]
    kernels = []
    for precision, sigma in (
        ("standard", spec["sensor_sigma"]),
        ("precise", spec["precise_sigma"]),
    ):
        for x in CONTEXTS:
            components = []
            for z in Z_GRID:
                for p in ((1 + z / 4,) if endogenous else PRICES):
                    demand = _demand(spec, p, x, z)
                    components.append(
                        (p, capacity, min(capacity, demand) + bias, demand + bias)
                    )
            kernels.append(
                dict(
                    kind="observe",
                    context=x,
                    precision=precision,
                    mixture=_mixture(components),
                    reading_variances=[
                        _number(sigma**2),
                        _number(spec["precise_sigma"] ** 2),
                    ],
                )
            )
        if not observational:
            for p, x, cap in itertools.product(PRICES, CONTEXTS, CAPACITIES):
                components = []
                for z in Z_GRID:
                    demand = _demand(spec, p, x, z)
                    components.append((p, cap, min(cap, demand) + bias, demand + bias))
                kernels.append(
                    dict(
                        kind="intervene",
                        price=p,
                        context=x,
                        capacity=cap,
                        precision=precision,
                        mixture=_mixture(components),
                        reading_variances=[
                            _number(sigma**2),
                            _number(spec["precise_sigma"] ** 2),
                        ],
                    )
                )
    return dict(
        version=VERSION,
        kind="available_access_apparatus_task",
        public_contract=public,
        component_fields=[
            "reported_price",
            "reported_capacity",
            "sales_mean",
            "audit_mean",
        ],
        conditional_sensor_covariance="diagonal; fresh independent errors per reading",
        unit_semantics="replicate redraws unit; audits retain original unit",
        acquisition_kernels=kernels,
        calibration=[
            dict(
                reference=ref,
                precision=precision,
                mean=_number(ref + bias),
                variance=_number(sigma**2),
            )
            for ref in public["calibration_references"]
            for precision, sigma in (
                ("standard", spec["sensor_sigma"]),
                ("precise", spec["precise_sigma"]),
            )
        ],
    )


def apparatus_task_signature(spec, *, observation_budget=24):
    return content_hash(
        apparatus_task_payload(spec, observation_budget=observation_budget)
    )


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{label} must be nonempty bounded text")
    return value


def campaign_seed(root_seed, world_id, repetition):
    """Same world/repetition shares a seed across arms; repetitions never collide.

    A 32-bit cyclic permutation guarantees distinct seeds for repetitions 0..2^32-1
    of one root/world. Different worlds/roots can collide in the finite seed space;
    a study must check its complete allocation, not infer global uniqueness here.
    """
    if type(root_seed) is not int or not 0 <= root_seed < 2**256:
        raise ValueError("Root seed must be an unsigned 256-bit integer")
    _identifier(world_id, "World/coupling ID")
    if type(repetition) is not int or not 0 <= repetition < SEED_SPACE:
        raise ValueError("Repetition must be an unsigned 32-bit integer")
    offset = int(content_hash([VERSION, "seed", root_seed, world_id])[:8], 16)
    return (offset + repetition) % SEED_SPACE


def create_session(
    path,
    *,
    spec,
    world_id,
    repetition,
    root_seed,
    comparison_policy,
    resource_limits: ResourceLimits,
    observation_budget=24,
    coupling_id=None,
):
    """Create a NEW session only; no bank migration, model call or old-state update.

    The caller freezes common resource limits/budget across arms. This helper
    cannot certify that separate calls use the same limits. Identity metadata is
    supervisor-only; a parent study must commit it in its allocation/protocol.
    coupling_id defaults to world_id; explicit shared IDs deliberately couple
    different access tasks without collapsing their distinct task identities.
    """
    from .session import DemandSession

    spec = _spec(spec)
    _identifier(world_id, "World ID")
    coupling_id = (
        world_id if coupling_id is None else _identifier(coupling_id, "Coupling ID")
    )
    if not isinstance(resource_limits, ResourceLimits):
        raise ValueError("Explicit ResourceLimits are required")
    assigned = assignment(comparison_policy, resource_limits)
    seed = campaign_seed(root_seed, coupling_id, repetition)
    task_signature = apparatus_task_signature(
        spec, observation_budget=observation_budget
    )
    config = DemandConfig(
        scenario=(
            "observational_only" if spec["stratum"] == "observational_only" else "clean"
        ),
        seed=seed,
        budget=observation_budget,
        price_limit=spec["price_limit"],
        world_spec=spec,
    )
    record = dict(
        version=VERSION,
        world_id=world_id,
        coupling_id=coupling_id,
        repetition=repetition,
        root_seed_commitment=content_hash([VERSION, "root_seed", root_seed]),
        campaign_seed=seed,
        spec_commitment=spec_commitment(spec),
        physical_marginal_signature=physical_marginal_signature(spec),
        apparatus_task_signature=task_signature,
        comparison_assignment=assigned,
        resource_limits=asdict(resource_limits),
        observation_budget=observation_budget,
        coupling="common seed across arms/tasks sharing coupling_id at equal query index; fresh seed per repetition",
    )
    session = DemandSession.create(
        Path(path),
        config,
        comparison_policy=comparison_policy,
        resource_limits=resource_limits,
    )
    record_path = session.path / ".demand/identity-v3.json"
    atomic_text(record_path, canonical_json(record) + "\n", exclusive=True)
    record_path.chmod(0o600)
    return session
