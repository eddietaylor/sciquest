"""OWNER ONLY: fixed development tasks and cheap simulator diagnostics.

Never stage this module, its specs, private reports or baseline answers in a
scientist workspace. Nothing here calls an LLM or writes/migrates artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median

from sciquest.io import content_hash, finite_number
from .admission import admission_report
from .designed_v2 import evaluation_rows
from .identity_v3 import (
    apparatus_task_signature,
    campaign_seed,
    physical_marginal_signature,
    spec_commitment,
)
from .session import observation_fields
from .world import DemandConfig, draw, normalize_action, public_contract

VERSION = "diagnostic-tasks-v3.1"
BASELINE_ROOT_SEED = 2026102401
TASK_ROOT_SEED = 20261024
BASELINE_REPETITIONS = 256
SCOPE = (
    "Three fixed development access tasks from TWO physical constructions/family "
    "clusters, not a random world sample. Access twins, noise repetitions and policy "
    "arms are dependent design units, not additional independent worlds. No scientific "
    "generalization, policy-superiority, whole-law recovery or industry-validity claim."
)
CONSTRUCTION_RULE = (
    "Freeze the exact coefficient table, instrumentation and metadata returned by "
    "task_definitions(). Use its flat law once and its identical responsive law with "
    "confounded/intervention versus observational-only access. No candidate sampling, "
    "subject-performance selection or parameter retry. If any existing admission "
    "gate fails, stop and preserve that failure; do not relabel it as admitted. "
    "Canonical outputs here supersede preliminary candidate names/hashes/seed tables."
)
BASELINE_ASSUMPTIONS = (
    "Endpoint fit receives ONLY the public apparatus contract and public readings. "
    "Two calibration readings estimate one stable additive instrument bias; ten "
    "independent fresh-unit standard readings estimate endpoint means. A straight "
    "line between endpoint means is used at each query context. This approximation "
    "is not a validated likelihood, causal explanation, uncertainty interval or "
    "test of price invariance. Clipping the fitted high-capacity mean to a lower "
    "capacity approximates E[min(C,D)] by min(C,E[D]); these differ under censoring. "
    "No monotonicity constraint, private coefficients or private noise law is fitted. "
    "The separate private evaluator computes expected regret after predictions."
)


def task_definitions():
    """Return new mutable records; no caller can mutate canonical definitions."""
    common = dict(
        version="designed-demand-v2",
        description="Prospective diagnostic construction v1; no subject-performance selection.",
        scientific_challenge="Development task for a bounded inquiry-guidance diagnostic; not a generalization sample.",
    )
    flat = dict(
        common,
        stratum="flat",
        expression="88+31*x+9*x**2+(5+2*x)*z",
        price_limit=1.25,
        confounding_strength=0,
        sensor_sigma=2.3,
        precise_sigma=0.65,
        sensor_bias=-1.4,
    )
    responsive = dict(
        common,
        stratum="confounded",
        expression="111+43*x+11*x**2-(112-37*x+17*x**2)*(p-1)+18*(p-1)**2+(6+2*x)*z",
        price_limit=1.25,
        confounding_strength=9,
        sensor_sigma=2.7,
        precise_sigma=0.7,
        sensor_bias=1.9,
    )
    return [
        dict(
            task_id="diagnostic-v3-flat",
            family_id="diagnostic-v3-flat-family",
            coupling_id="diagnostic-v3-flat-family",
            spec=flat,
        ),
        dict(
            task_id="diagnostic-v3-confounded",
            family_id="diagnostic-v3-responsive-family",
            coupling_id="diagnostic-v3-responsive-family",
            spec=responsive,
        ),
        dict(
            task_id="diagnostic-v3-observational-only",
            family_id="diagnostic-v3-responsive-family",
            coupling_id="diagnostic-v3-responsive-family",
            spec={**responsive, "stratum": "observational_only"},
        ),
    ]


def _public(spec):
    return public_contract(
        DemandConfig(
            scenario=(
                "observational_only"
                if spec["stratum"] == "observational_only"
                else "clean"
            ),
            seed=0,
            budget=24,
            price_limit=spec["price_limit"],
            world_spec=spec,
        )
    )


def build_tasks(*, root_seed=TASK_ROOT_SEED):
    """Return exact full admission reports and identities; propagate gate failures.

    The reports use repetition-zero family-coupled seeds. The caller separately
    freezes every actual pilot repetition/arm allocation and its common resources.
    """
    definitions = task_definitions()
    tasks = []
    for task in definitions:
        spec = task["spec"]
        seed = campaign_seed(root_seed, task["coupling_id"], 0)
        tasks.append(
            dict(
                task,
                world_id=task["task_id"],
                admission_seed=seed,
                spec_commitment=spec_commitment(spec),
                physical_marginal_signature=physical_marginal_signature(spec),
                apparatus_task_signature=apparatus_task_signature(spec),
                public_contract=_public(spec),
                admission_report=admission_report(spec, seed),
            )
        )
    return dict(
        version=VERSION,
        scope=SCOPE,
        construction_rule=CONSTRUCTION_RULE,
        construction_commitment=content_hash([VERSION, CONSTRUCTION_RULE, definitions]),
        root_seed=root_seed,
        tasks=tasks,
        physical_family_count=2,
        access_task_count=3,
        seed_rule="campaign_seed(root_seed,coupling_id,repetition); shared across policies and access twins",
    )


def endpoint_schedule(public):
    """Public-only fixed 12-query/22-cost schedule, or unavailable for denied access."""
    if not public["interventions_available"]:
        return None
    contexts = sorted(public["contexts"])
    if len(contexts) != 3:
        raise ValueError(
            "This declared endpoint schedule requires three public contexts"
        )
    references = sorted(public["calibration_references"])
    actions = [
        dict(kind="calibrate", reference=ref, precision="standard")
        for ref in (references[0], references[-1])
    ]
    for x in (contexts[0], contexts[0], contexts[1], contexts[2], contexts[2]):
        for p in (min(public["prices"]), max(public["prices"])):
            actions.append(
                dict(
                    kind="intervene",
                    price=p,
                    context=x,
                    capacity=max(public["capacities"]),
                    precision="standard",
                )
            )
    return actions


def fit_endpoint(public, observations):
    """Fit/select using public readings only; private scoring is a separate API.

    Returns one price for each public query-context/capacity decision state.
    Unknown observation annotations (e.g. private demand, sigma, expression) fail.
    """
    if not public["interventions_available"]:
        raise ValueError("Endpoint fit is unavailable without intervention access")
    allowed = set(observation_fields(public))
    if any(not isinstance(obs, dict) or set(obs) - allowed for obs in observations):
        raise ValueError("Endpoint fit accepts public observation fields only")
    calibrations = [
        finite_number(o["value"]) - finite_number(o["reference"])
        for o in observations
        if o["kind"] == "calibrate"
    ]
    if len(calibrations) != 2:
        raise ValueError("Endpoint fit requires its two calibration readings")
    bias = mean(calibrations)
    cells = defaultdict(list)
    for obs in observations:
        if obs["kind"] == "intervene":
            if obs["capacity"] != max(public["capacities"]):
                raise ValueError("Endpoint fit requires the public maximum capacity")
            cells[obs["context"], obs["price"]].append(
                finite_number(obs["value"]) - bias
            )
        elif obs["kind"] != "calibrate":
            raise ValueError("Endpoint fit requires calibration/intervention readings")
    low, high = min(public["prices"]), max(public["prices"])
    feasible = [p for p in public["prices"] if p <= public["price_limit"]]
    if not feasible or low == high:
        raise ValueError("Endpoint fit needs feasible actions and distinct endpoints")
    selections, context_fits = [], []
    for x in public["contexts"]:
        if not cells[x, low] or not cells[x, high]:
            raise ValueError("Endpoint readings must cover all public contexts")
        left, right = mean(cells[x, low]), mean(cells[x, high])
        context_fits.append(
            dict(
                context=x,
                low_price=low,
                high_price=high,
                low_readings=len(cells[x, low]),
                high_readings=len(cells[x, high]),
                low_mean=left,
                high_mean=right,
                slope=(right - left) / (high - low),
            )
        )
        for cap in public["capacities"]:
            values = []
            for p in feasible:
                fitted = max(
                    0.0,
                    min(
                        max(public["capacities"]),
                        left + (right - left) * (p - low) / (high - low),
                    ),
                )
                values.append((p, p * min(cap, fitted)))
            selected = min(values, key=lambda item: (-item[1], item[0]))[0]
            selections.append(dict(context=x, capacity=cap, selected_price=selected))
    return dict(estimated_bias=bias, context_fits=context_fits, selections=selections)


def _groups(spec):
    rows = evaluation_rows(
        dict(spec=spec, config=dict(seed=0, price_limit=spec["price_limit"]))
    )
    groups = defaultdict(list)
    for row in rows:
        if row["decision_eligible"] and row["price"] <= row["price_limit"]:
            groups[row["context"], row["capacity"]].append(row)
    return groups


def evaluate_selections(spec, selections):
    """Private evaluator only: expected physical revenue/regret over six states."""
    groups = _groups(spec)
    if len(selections) != len(groups):
        raise ValueError("Selections must cover every decision state exactly")
    selected = {
        (row["context"], row["capacity"]): row["selected_price"] for row in selections
    }
    if set(selected) != set(groups):
        raise ValueError("Selections must cover every decision state exactly")
    evidence = []
    for state, rows in sorted(groups.items()):
        matches = [r for r in rows if r["price"] == selected[state]]
        if len(matches) != 1:
            raise ValueError("Selected price is not a public feasible action")
        value = matches[0]["expected_revenue"]
        oracle = max(row["expected_revenue"] for row in rows)
        evidence.append(
            dict(
                context=state[0],
                capacity=state[1],
                selected_price=selected[state],
                policy_value=value,
                feasible_oracle_value=oracle,
                regret=max(0.0, oracle - value),
            )
        )
    return dict(
        mean_regret=mean(row["regret"] for row in evidence),
        policy_value=mean(row["policy_value"] for row in evidence),
        evidence=evidence,
    )


def no_observation_diagnostics(spec):
    """Public highest-feasible versus PRIVATE best-constant oracle diagnostic.

    Best-constant oracle is not an attainable no-data scientist: it uses protected
    truth, but restricts its action to one price across all six decision states.
    """
    public = _public(spec)
    feasible = [p for p in public["prices"] if p <= public["price_limit"]]

    def constant(price):
        return [
            dict(context=x, capacity=cap, selected_price=price)
            for x in public["contexts"]
            for cap in public["capacities"]
        ]

    highest = max(feasible)
    evaluated = {p: evaluate_selections(spec, constant(p)) for p in feasible}
    best = min(feasible, key=lambda p: (-evaluated[p]["policy_value"], p))
    return dict(
        highest_feasible=dict(
            selected_price=highest,
            **evaluated[highest],
            observations=0,
            cost=0,
            truth_used_to_select=False,
        ),
        best_constant_oracle=dict(
            selected_price=best,
            **evaluated[best],
            observations=0,
            cost=0,
            truth_used_to_select=True,
        ),
    )


def _summary(values):
    values = [v for v in values if v is not None]
    return dict(
        available=len(values),
        mean_regret=mean(values) if values else None,
        median_regret=median(values) if values else None,
        minimum_regret=min(values) if values else None,
        maximum_regret=max(values) if values else None,
    )


def run_baselines(
    *, tasks=None, root_seed=BASELINE_ROOT_SEED, repetitions=BASELINE_REPETITIONS
):
    """Return reproducible owner-only raw simulator diagnostics; never run models.

    Default seeds are 256 declared repetitions of each family under a baseline-only
    root distinct from task admission. Denied endpoint access yields explicit null
    results and zero queries, never private counterfactual access or zero regret.
    """
    if type(repetitions) is not int or not 1 <= repetitions <= BASELINE_REPETITIONS:
        raise ValueError("Use 1..256 declared baseline repetitions")
    bundle = build_tasks() if tasks is None else tasks
    definitions = task_definitions()
    expected_commitment = content_hash([VERSION, CONSTRUCTION_RULE, definitions])
    if (
        bundle.get("version") != VERSION
        or bundle.get("construction_commitment") != expected_commitment
        or [
            {k: task[k] for k in ("task_id", "family_id", "coupling_id", "spec")}
            for task in bundle["tasks"]
        ]
        != definitions
    ):
        raise ValueError("Baseline tasks differ from the fixed canonical construction")
    outputs = []
    for task in bundle["tasks"]:
        spec, public = task["spec"], _public(task["spec"])
        schedule = endpoint_schedule(public)
        diagnostics = no_observation_diagnostics(spec)
        raw = []
        for repetition in range(repetitions):
            seed = campaign_seed(root_seed, task["coupling_id"], repetition)
            record = dict(
                repetition=repetition,
                seed=seed,
                highest_feasible_regret=diagnostics["highest_feasible"]["mean_regret"],
                best_constant_oracle_regret=diagnostics["best_constant_oracle"][
                    "mean_regret"
                ],
            )
            if schedule is None:
                record["endpoint_fit"] = dict(
                    status="unavailable_no_intervention_access",
                    regret=None,
                    cost=0,
                    queries=0,
                    readings=[],
                    fit=None,
                    evaluation=None,
                )
            else:
                world = dict(
                    spec=spec, config=dict(seed=seed, price_limit=spec["price_limit"])
                )
                history, readings, spent = [], [], 0
                for index, action in enumerate(schedule, 1):
                    design, cost = normalize_action(action, public, history)
                    if index > 12 or spent + cost > 24:
                        raise ValueError(
                            "Baseline exceeds declared apparatus resources"
                        )
                    observation, private = draw(world, design, index, history)
                    public_observation = {
                        k: v
                        for k, v in observation.items()
                        if k in observation_fields(public)
                    }
                    readings.append(
                        dict(action=action, cost=cost, observation=public_observation)
                    )
                    history.append(
                        dict(design=design, observation=observation, private=private)
                    )
                    spent += cost
                if len(readings) != 12 or spent != 22:
                    raise ValueError(
                        "Declared endpoint baseline must cost22 with12 queries"
                    )
                fitted = fit_endpoint(public, [r["observation"] for r in readings])
                evaluated = evaluate_selections(spec, fitted["selections"])
                record["endpoint_fit"] = dict(
                    status="completed",
                    regret=evaluated["mean_regret"],
                    cost=spent,
                    queries=len(readings),
                    readings=readings,
                    fit=fitted,
                    evaluation=evaluated,
                )
            raw.append(record)
        outputs.append(
            dict(
                task_id=task["task_id"],
                family_id=task["family_id"],
                coupling_id=task["coupling_id"],
                spec_commitment=spec_commitment(spec),
                physical_marginal_signature=physical_marginal_signature(spec),
                apparatus_task_signature=apparatus_task_signature(spec),
                no_observation_diagnostics=diagnostics,
                raw_repetitions=raw,
                aggregate=dict(
                    highest_feasible=_summary(
                        [r["highest_feasible_regret"] for r in raw]
                    ),
                    best_constant_oracle=_summary(
                        [r["best_constant_oracle_regret"] for r in raw]
                    ),
                    endpoint_fit=_summary([r["endpoint_fit"]["regret"] for r in raw]),
                ),
            )
        )
    return dict(
        version=VERSION,
        scope=SCOPE,
        construction_commitment=bundle["construction_commitment"],
        baseline_root_seed=root_seed,
        repetitions=repetitions,
        declared_repetitions=list(range(repetitions)),
        assumptions_and_limits=BASELINE_ASSUMPTIONS,
        reporting="Descriptive simulator-only checks, not subject outcomes or independent-world inference. No dropping unavailable tasks.",
        tasks=outputs,
    )
