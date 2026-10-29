"""Inspectable experimental-design baselines; all receive only public state.

The four strategies share the same finite predictive inference backend to isolate
experimental-design choices. They are operational baselines, not full philosophies.
"""

from __future__ import annotations

import itertools
import json
import math
from functools import lru_cache
from pathlib import Path

import nbformat

from sciquest.io import canonical_json

from .world import (
    bounded_probability,
    FAMILIES,
    catalog,
    components,
    demand,
    density,
    event_probability,
)

POLICIES = (
    "standard_empirical",
    "strong_inference",
    "bayesian",
    "popperian_falsificationist",
)


def particles():
    result = []
    candidates = catalog()
    for m in candidates:
        count = sum(n["family"] == m["family"] for n in candidates)
        result.append(dict(model=m, label=m["family"], prior=0.9 / 4 / count))
    # Explicit broad alternative predictive prior, not a distribution over every
    # unknown mechanism. Calibration outside this finite approximation is tested.
    broad = list(itertools.product((30, 100, 170), (-10, 10), (-100, 0, 100)))
    for b, x, s in broad:
        result.append(
            dict(
                model=dict(
                    family="linear", baseline=b, context_effect=x, sensitivity=s
                ),
                label="other",
                prior=0.1 / len(broad),
            )
        )
    return result


def _log_normal(y, mu, sigma):
    return -0.5 * ((y - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))


def _log_sum(values):
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


def infer(view):
    models = particles()
    logs = [math.log(m["prior"]) for m in models]
    originals = [
        r["observation"]
        for r in view["history"]
        if r["observation"]["kind"] != "measure"
    ]
    endog = view["contract"]["historical_assignment"] == "endogenous"
    for i, particle in enumerate(models):
        m = particle["model"]
        for obs in originals:
            audits = [
                r["observation"]
                for r in view["history"]
                if r["observation"].get("paired_with") == obs["id"]
            ]
            us = (
                [int(round(4 * (obs["price"] - 1)))]
                if endog and obs["kind"] == "observe"
                else [-1, 0, 1]
            )
            joint = []
            for u in us:
                latent = demand(m, obs["price"], obs["context"], u, endog)
                value = _log_normal(
                    obs["value"], min(obs["capacity"], latent), obs["sigma"]
                )
                value += math.fsum(
                    _log_normal(a["value"], latent, a["sigma"]) for a in audits
                )
                joint.append(value)
            logs[i] += _log_sum(joint) - math.log(len(us))
    maximum = max(logs)
    weights = [math.exp(v - maximum) for v in logs]
    total = math.fsum(weights)
    return models, [w / total for w in weights]


def family_probabilities(models, weights):
    return {
        f: math.fsum(w for m, w in zip(models, weights) if m["label"] == f)
        for f in FAMILIES
    }


def predict(models, weights, public, row):
    p, x, c = float(row["price"]), float(row["context"]), float(row["capacity"])
    endog = public["historical_assignment"] == "endogenous"
    sigma = public["sensor_sigma"]
    ds = [components(m["model"], p, x, c, endog, "demand") for m in models]
    ss = [[min(c, v) for v in d] for d in ds]
    mean_d = math.fsum(w * math.fsum(d) / 3 for w, d in zip(weights, ds))
    mean_s = math.fsum(w * math.fsum(s) / 3 for w, s in zip(weights, ss))

    def quantile(prob):
        lo = -12 * sigma
        hi = c + 12 * sigma
        for _ in range(42):
            mid = (lo + hi) / 2
            cdf = math.fsum(
                w * event_probability(mid, means, sigma)
                for w, means in zip(weights, ss)
            )
            if cdf < prob:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return dict(
        mean_demand=mean_d,
        mean_sales=min(c, max(0, mean_s)),
        lower_sales=quantile(0.05),
        upper_sales=quantile(0.95),
    )


@lru_cache(maxsize=256)
def _information_grid(public_text, action_text):
    public = json.loads(public_text)
    action = json.loads(action_text)
    models = particles()
    sigma = (
        public["precise_sensor_sigma"]
        if action["precision"] == "precise"
        else public["sensor_sigma"]
    )
    endog = public["historical_assignment"] == "endogenous"
    means = [
        components(
            m["model"], action["price"], action["context"], action["capacity"], endog
        )
        for m in models
    ]
    lo = min(min(v) for v in means) - 7 * sigma
    hi = max(max(v) for v in means) + 7 * sigma
    bins = max(96, math.ceil((hi - lo) / (sigma / 2)))
    width = (hi - lo) / bins
    return (
        width,
        [
            [density(lo + (i + 0.5) * width, m, sigma) for m in means]
            for i in range(bins)
        ],
        means,
    )


def choose_action(view, policy, models, weights):
    public = view["contract"]
    if not public["interventions_available"]:
        return dict(
            kind="observe",
            context=(-1, 0, 1)[len(view["history"]) % 3],
            precision="standard",
        ), "Observe additional strata while preserving causal non-identifiability."
    if policy == "standard_empirical" and view["remaining"] >= 2:
        design = [
            (0.5, -1, 180),
            (1.5, -1, 180),
            (0.5, 1, 180),
            (1.5, 1, 180),
            (1, 0, 60),
            (1, 0, 180),
        ]
        p, x, c = design[len(view["history"]) % len(design)]
        return dict(
            kind="intervene", price=p, context=x, capacity=c, precision="standard"
        ), "Prespecified balanced empirical price/context/capacity design."
    choices = []
    for p, x, c, precision in itertools.product(
        (0.5, 1, 1.5), (-1, 1), (60, 180), ("standard", "precise")
    ):
        action = dict(
            kind="intervene", price=p, context=x, capacity=c, precision=precision
        )
        cost = 2 + (2 if precision == "precise" else 0)
        if cost > view["remaining"]:
            continue
        if policy == "bayesian":
            width, grid, means = _information_grid(
                canonical_json(public), canonical_json(action)
            )
        else:
            means = [
                components(
                    m["model"], p, x, c, public["historical_assignment"] == "endogenous"
                )
                for m in models
            ]
        sigma = 1 if precision == "precise" else public["sensor_sigma"]
        averages = [math.fsum(m) / 3 for m in means]
        if policy == "bayesian":
            information = 0
            for densities in grid:
                mixture = math.fsum(w * d for w, d in zip(weights, densities))
                if mixture > 0:
                    information += width * math.fsum(
                        w * d * math.log(d / mixture)
                        for w, d in zip(weights, densities)
                        if w > 0 and d > 0
                    )
            score = max(0, information) / cost
        elif policy == "strong_inference":
            active = [v for v, w in zip(averages, weights) if w > 1e-4]
            score = (max(active) - min(active)) / (sigma * cost)
        else:
            leader = max(range(len(weights)), key=weights.__getitem__)
            alternatives = [
                (weights[i], abs(averages[i] - averages[leader]))
                for i in range(len(weights))
                if models[i]["label"] != models[leader]["label"]
            ]
            score = math.fsum(w * d for w, d in alternatives) / (sigma * cost)
            desired = 0.5 if len(view["history"]) % 2 == 0 else 1.5
            if p == desired:
                score *= 1.1
        choices.append((score, -cost, p, x, c, precision, action))
    if not choices:
        return dict(
            kind="observe", context=0, precision="standard"
        ), "Use the remaining observational budget; no affordable intervention remains."
    chosen = max(choices, key=lambda c: c[:6])
    rationales = {
        "bayesian": "Finite-mixture mutual information per cost unit, approximated by deterministic quadrature with bin width at most half a sensor standard deviation.",
        "strong_inference": "Maximize the normalized predictive disagreement among currently supported rival particles per cost.",
        "popperian_falsificationist": "Target the leading particle with rival-weighted diagnostic alternatives and alternate perturbation direction.",
    }
    return chosen[-1], rationales[policy]


def query_plan(view, action, models, weights, rationale):
    public = view["contract"]
    endog = public["historical_assignment"] == "endogenous"
    # For historical requests, the registered event is marginal over the known
    # historical assignment, before the realized price/u is observed.
    if action["kind"] == "observe":
        values = []
        for m in models:
            if endog:
                values.append(
                    [
                        min(
                            public["historical_capacity"],
                            demand(m["model"], 1 + u / 4, action["context"], u, True),
                        )
                        for u in (-1, 0, 1)
                    ]
                )
            else:
                values.append(
                    [
                        min(
                            public["historical_capacity"],
                            demand(m["model"], p, action["context"], u, False),
                        )
                        for p, u in itertools.product(public["prices"], (-1, 0, 1))
                    ]
                )
    else:
        values = [
            components(
                m["model"],
                action["price"],
                action["context"],
                action["capacity"],
                endog,
            )
            for m in models
        ]
    probs = family_probabilities(models, weights)
    means = {}
    likelihood = {}
    conditional = {}
    for f in FAMILIES:
        members = [i for i, m in enumerate(models) if m["label"] == f]
        total = math.fsum(weights[i] for i in members)
        ws = (
            [weights[i] / total for i in members]
            if total
            else [1 / len(members)] * len(members)
        )
        conditional[f] = list(zip(members, ws))
        means[f] = math.fsum(
            w * math.fsum(values[i]) / len(values[i]) for i, w in conditional[f]
        )
    threshold = (min(means.values()) + max(means.values())) / 2
    sigma = 1 if action.get("precision") == "precise" else public["sensor_sigma"]
    for f in FAMILIES:
        probability = math.fsum(
            w * event_probability(threshold, values[i], sigma)
            for i, w in conditional[f]
        )
        probability = bounded_probability(probability)
        likelihood[f] = dict(supported=probability, contradicted=1 - probability)
    return dict(
        question="Which rivals predict the outcome of this budgeted measurement?",
        hypothesis="The observed measurement meets the declared discrimination threshold.",
        scope="This synthetic discovery query; a single binary registered event.",
        assumptions="Finite predictive particle approximation, Gaussian measurement error and the public structural nuisance model. Other is a broad predictive approximation, not exhaustive model coverage.",
        data_access="Public contract and earlier observations only; private world and final labels unavailable.",
        stopping_rule="One requested measurement. Final stopping is separately registered before sealed prediction.",
        action="rival_test",
        confirmatory_status="exploratory",
        rationale=rationale,
        predictions=[
            dict(id="p1", metric="observation", operator="<=", threshold=threshold)
        ],
        rivals=[
            dict(
                id=f,
                claim=f"{f} candidate predictive family",
                predictions={"p1": means[f]},
            )
            for f in FAMILIES
        ],
        discriminating_prediction_id="p1",
        belief_model=dict(
            prediction_id="p1",
            prior=probs,
            likelihood=likelihood,
            assumptions="Conditional finite-particle predictive CDFs. Priors are recomputed from all previous raw observations once, using the original particle prior; paired audits share a latent unit. The recorded binary update is distinct from conditioning the design backend on the full measured value.",
        ),
        severe_test=dict(
            prediction_id="p1",
            failure_action="revise",
            auxiliary_assumptions=[
                "Sensor calibration and nuisance distribution remain stable."
            ],
            diagnostic_rationale="A risky outcome comparison can expose the current leading explanation; severity is not certified.",
        ),
    )


def make_decision(view, models, weights):
    probs = family_probabilities(models, weights)
    public = view["contract"]
    unidentified = (
        not public["interventions_available"]
        and public["historical_assignment"] == "endogenous"
    )
    best = max(probs, key=probs.get)
    status = (
        "not_identifiable"
        if unidentified
        else "model_inadequate"
        if best == "other" and probs[best] > 0.8
        else "selected"
        if best != "other" and probs[best] >= 0.95
        else "inconclusive"
    )
    d = dict(
        status=status,
        reason="Constructed historical equivalence prevents causal identification."
        if unidentified
        else "Finite predictive model posterior and registered confidence criterion; no guarantee outside the approximation.",
        mechanism_probabilities=probs,
    )
    if status == "selected":
        d["selected_family"] = best
        total = probs[best]
        d["parameters"] = {
            k: math.fsum(
                w * m["model"][k] / total
                for m, w in zip(models, weights)
                if m["label"] == best
            )
            for k in ("baseline", "context_effect", "sensitivity")
        }
    return d


def final_artifacts(view, models, weights, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    index = 1 if view["history"] else 0
    checkpoint = dict(models=models, weights=weights, public=view["contract"])
    prediction_code = f"""import csv,json
from pathlib import Path
from sciquest.demand.reference import predict
bundle=json.loads(Path('model.json').read_text())
rows=list(csv.DictReader(open('data/{index}/features.csv')))
assert 'expected_demand' not in rows[0] and 'measured_sales' not in rows[0]
with open('predictions.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['id','mean_demand','mean_sales','lower_sales','upper_sales']);w.writeheader()
    for row in rows:w.writerow(dict(id=row['id'],**predict(bundle['models'],bundle['weights'],bundle['public'],row)))
"""
    reload_nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(prediction_code)]
    )
    code = (
        "from pathlib import Path\nPath('model.json').write_text("
        + repr(canonical_json(checkpoint))
        + ")\nPath('reload.ipynb').write_text("
        + repr(nbformat.writes(reload_nb))
        + ")\n"
        + prediction_code
    )
    path = directory / "notebook.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), path
    )
    # Registered rival expectations for the final error statistic are computed
    # entirely from the declared predictive catalog, never from private truth.
    features = view["evaluation_features"]
    expected = [
        predict(models, weights, view["contract"], r)["mean_demand"] for r in features
    ]
    endog = view["contract"]["historical_assignment"] == "endogenous"
    errors = []
    for m in models:
        errors.append(
            math.sqrt(
                math.fsum(
                    (
                        p
                        - math.fsum(
                            components(
                                m["model"],
                                float(r["price"]),
                                float(r["context"]),
                                float(r["capacity"]),
                                endog,
                                "demand",
                            )
                        )
                        / 3
                    )
                    ** 2
                    for p, r in zip(expected, features)
                )
                / len(features)
            )
        )
    family_errors = {
        f: math.fsum(errors[i] for i, m in enumerate(models) if m["label"] == f)
        / sum(m["label"] == f for m in models)
        for f in FAMILIES
    }
    threshold = (min(family_errors.values()) + max(family_errors.values())) / 2
    plan = query_plan(
        view,
        dict(
            kind="intervene", price=0.5, context=-1, capacity=180, precision="standard"
        ),
        models,
        weights,
        "Final held-out prediction test after discovery is frozen.",
    )
    plan.update(
        question="Does the frozen predictor meet the registered held-out demand-error threshold?",
        action="hypothesis_test",
        confirmatory_status="confirmatory",
        predictions=[
            dict(id="p1", metric="demand_rmse", operator="<=", threshold=threshold)
        ],
        model_export=dict(
            artifacts=["model.json"],
            reload_notebook="reload.ipynb",
            absolute_tolerance=0,
        ),
    )
    for rival in plan["rivals"]:
        rival["predictions"]["p1"] = family_errors[rival["id"]]
    for f in FAMILIES:
        members = [i for i, m in enumerate(models) if m["label"] == f]
        mass = math.fsum(weights[i] for i in members)
        ws = (
            [weights[i] / mass for i in members]
            if mass
            else [1 / len(members)] * len(members)
        )
        probability = math.fsum(
            w for i, w in zip(members, ws) if errors[i] <= threshold
        )
        probability = bounded_probability(probability)
        plan["belief_model"]["likelihood"][f] = dict(
            supported=probability, contradicted=1 - probability
        )
    plan["belief_model"]["assumptions"] = (
        "Conditional on the finite predictive particle approximation and posterior over raw discovery observations. The event is held-out expected-demand RMSE <= threshold; exact particle expectations determine its likelihood. Other remains a nonexhaustive predictive approximation."
    )
    return path, plan


def run_reference(session, policy=None):
    view = session.view()
    if view["information_mode"] != "assisted_v1":
        raise ValueError(
            "Reference strategies require an explicitly assisted session; they know the simulator catalog"
        )
    policy = policy or view["method_stack"]["primary_method"]
    if policy not in POLICIES:
        raise ValueError("Unknown demand reference strategy")
    if policy != view["method_stack"]["primary_method"]:
        raise ValueError("Reference strategy must match the registered session method")
    while view["phase"] == "collecting":
        models, weights = infer(view)
        probs = family_probabilities(models, weights)
        n = len(view["history"])
        restricted = not view["contract"]["interventions_available"]
        confident = max(probs.values()) >= 0.97 and n >= 6
        if (
            view["remaining"] == 0
            or restricted
            and n >= 4
            or policy != "standard_empirical"
            and confident
        ):
            view = session.stop(make_decision(view, models, weights), view["revision"])
            break
        action, why = choose_action(view, policy, models, weights)
        session.query(
            dict(
                revision=view["revision"],
                action=action,
                plan=query_plan(view, action, models, weights, why),
            )
        )
        view = session.view()
    if view["phase"] == "completed":
        return view["evaluation"]
    models, weights = infer(view)
    # Preparing the native quest changes the supervisor revision, not the frozen
    # observations, decision, or inference state.
    session.prepare_quest()
    view = session.view()
    notebook, plan = final_artifacts(
        view, models, weights, session.path / "reference_submission"
    )
    return session.finish(notebook, plan)
