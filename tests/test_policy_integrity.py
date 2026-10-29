"""Adversarial philosophy contracts and managed multi-policy workflows.

Identical analytic fixtures test configuration and evidence handling, not which
philosophy is better, qualitative coding, or autonomous research performance.
"""

import math
import random
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, localcontext

import nbformat
import pytest

from sciquest.agent import build_agent_prompt
from sciquest.core import create_quest, run_next
from sciquest.dashboard import _policy_evidence_panel, build_dashboard
from sciquest.io import content_hash, read_yaml, write_yaml
from sciquest.logic import run_logic_check
from sciquest.methods.ledger import preregister_experiment
from sciquest.methods.policies import (
    policy_evidence,
    policy_requirements,
    validate_policy_plan,
)
from sciquest.methods.prompt_builder import build_method_prompt_addendum
from sciquest.methods.registry import MethodRegistry
from sciquest.methods.resolver import (
    _history_signals,
    auto_recommend_method_stack,
    resolve_method_stack,
)
from sciquest.methods.schema import METHOD_PHASES, MethodProfile
from sciquest.state import get_record, record
from sciquest.workflow import execute_registered, stage_research_context


def extensions():
    return {
        "rivals": [
            {"id": "h0", "claim": "The low-error observation process"},
            {"id": "h1", "claim": "The high-error observation process"},
        ],
        "belief_model": {
            "prediction_id": "p1",
            "prior": {"h0": 0.5, "h1": 0.5},
            "likelihood": {
                "h0": {"supported": 0.8, "contradicted": 0.2},
                "h1": {"supported": 0.2, "contradicted": 0.8},
            },
            "assumptions": "One binary likelihood event for the whole evaluation. Analytic conditional fixtures check algebra, not empirical calibration.",
        },
        "severe_test": {
            "prediction_id": "p1",
            "failure_action": "replicate",
            "auxiliary_assumptions": [
                "Measurements use a stable scale and known fixture values."
            ],
            "diagnostic_rationale": "A nonzero error contradicts the registered exact fixture prediction; measurement and auxiliaries need separate review.",
        },
        "discriminating_prediction_id": "p1",
    }


def policy_plan():
    p = extensions()
    p["predictions"] = [
        {"id": "p1", "metric": "error", "operator": "<=", "threshold": 0.1}
    ]
    p["rivals"][0]["predictions"] = {"p1": 0}
    p["rivals"][1]["predictions"] = {"p1": 3.5}
    return p


@pytest.fixture
def prepare(tmp_path):
    def make(method, name="case", outcome="supported", quest=None):
        q = quest or create_quest(
            tmp_path,
            {
                "hero_statement": "Check a registered observation rule",
                "initial_hypothesis": "Error is at most 0.1",
                "method_stack": method,
            },
            slug=name,
        )
        (q / "data/raw/train.csv").write_text("id,x,actual\nt1,1,2\nt2,2,4\n")
        suffix = name.replace("-", "_")
        ys = {"supported": (6, 8), "contradicted": (9, 12), "inconclusive": (0, 0)}[
            outcome
        ]
        (q / "data/raw/eval.csv").write_text(
            f"id,x,actual\n{suffix}_a,3,{ys[0]}\n{suffix}_b,4,{ys[1]}\n"
        )
        write_yaml(
            q / "data_manifest.yaml",
            {
                "status": "ready",
                "datasets": [
                    {
                        "path": "data/raw/train.csv",
                        "role": "training",
                        "provenance": "Analytic training fixture",
                    },
                    {
                        "path": "data/raw/eval.csv",
                        "role": "evaluation",
                        "input_columns": ["id", "x"],
                        "provenance": "Analytic conditional observation fixture; not empirical evidence",
                    },
                ],
            },
        )
        write_yaml(
            q / "validation.yaml",
            {
                "status": "ready",
                "metrics": [
                    {
                        "name": "error",
                        "direction": "minimize",
                        "normalization": {"type": "minmax", "min": 0, "max": 10},
                        "computation": {
                            "kind": "wape" if outcome == "inconclusive" else "mae",
                            "truth": "data/1/eval.csv",
                            "key": "id",
                            "target": "actual",
                            "prediction": "prediction",
                        },
                    }
                ],
            },
        )
        expid = run_next(q, agent_stub=True)
        exp = q / "experiments" / expid
        p = policy_plan()
        p.update(
            question="Does the proposed fixture predictor meet the registered error threshold?",
            hypothesis="The predictor has error <=0.1",
            scope="Analytic fixture rows only",
            assumptions="Known fixture measurement scale; binary observation model is specified for algebraic testing, not inferred from this fixture.",
            data_access="Training observations and evaluation x; labels hidden",
            stopping_rule="One frozen evaluation",
            action="hypothesis_test",
            confirmatory_status="exploratory",
        )
        write_yaml(exp / "experiment_plan.yaml", p)
        code = """import csv
rows=list(csv.DictReader(open('data/1/features.csv')))
assert 'actual' not in rows[0]
with open('predictions.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['id','prediction']);w.writeheader()
    for r in rows:w.writerow({'id':r['id'],'prediction':2*float(r['x'])})
"""
        nbformat.write(
            nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]),
            exp / "notebook.ipynb",
        )
        return q, expid, p

    return make


@pytest.mark.parametrize("method", MethodRegistry.default().ids())
def test_every_profile_full_managed_workflow(prepare, method):
    q, expid, p = prepare(method, name=method)
    prompt = build_agent_prompt(q)
    assert "supervisor alone" in prompt
    registration = preregister_experiment(q, expid)
    assert registration["profiles"][method]["id"] == method
    result = execute_registered(q, expid, timeout=30)
    assert result["metrics"]["error"] == 0
    assert result["prediction_outcomes"] == {"p1": "supported"}
    assert run_logic_check(q, expid)["passed"]
    html = build_dashboard(q).read_text()
    assert "Inquiry policy evidence" in html
    assert (
        "## Inquiry policy evidence"
        in (q / "experiments" / expid / "experiment_report.md").read_text()
    )
    if method == "bayesian":
        assert result["policy_evidence"]["belief_update"]["posterior"] == pytest.approx(
            {"h0": 0.8, "h1": 0.2}
        )
    elif method == "strong_inference":
        assessment = result["policy_evidence"]["rival_assessment"]["assessments"]
        assert assessment["h0"]["rule_compatibility"] == "compatible"
        assert assessment["h1"]["rule_compatibility"] == "incompatible"
    elif method == "popperian_falsificationist":
        assert (
            result["policy_evidence"]["falsification_assessment"]["status"]
            == "survived_registered_test"
        )
    else:
        assert not registration["policy_contract"]["enforced_plan_contracts"]
        assert method in registration["policy_contract"]["guidance_only_methods"]


@pytest.mark.parametrize("outcome", ["supported", "contradicted", "inconclusive"])
def test_composed_stack_obeys_each_phase_and_records_followup(
    prepare, tmp_path, outcome
):
    stack = {
        "primary_method": "standard_empirical",
        "phases": {
            "hypothesis_generation": "feyerabendian_pluralist",
            "experiment_design": "strong_inference",
            "evidence_evaluation": "bayesian",
            "stress_testing": "popperian_falsificationist",
        },
    }
    q, expid, p = prepare(stack, name=outcome, outcome=outcome)
    reg = preregister_experiment(q, expid)
    assert set(reg["policy_contract"]["enforced_plan_contracts"]) == {
        "strong_inference",
        "bayesian",
        "popperian_falsificationist",
    }
    result = execute_registered(q, expid, timeout=30)
    assert result["prediction_outcomes"]["p1"] == outcome
    pe = result["policy_evidence"]
    if outcome == "contradicted":
        assert pe["belief_update"]["posterior"] == pytest.approx({"h0": 0.2, "h1": 0.8})
        assert pe["falsification_assessment"]["next_step"] == "replicate"
    if outcome == "inconclusive":
        assert pe["belief_update"]["posterior"] == {"h0": 0.5, "h1": 0.5}
        assert pe["belief_update"]["status"] == "no_update_inconclusive"
        assert all(
            a["rule_compatibility"] == "inconclusive"
            for a in pe["rival_assessment"]["assessments"].values()
        )
    # The prepared method and profiles are frozen; changing live configuration
    # cannot rewrite the completed evidence or future feedback about this run.
    write_yaml(q / "method_stack.yaml", {"primary_method": "pragmatist"})
    assert get_record(q, "registration", expid) == reg
    folder = tmp_path / "feedback"
    folder.mkdir()
    stage_research_context(q, folder)
    assert (
        read_yaml(folder / "research_state.yaml")["completed_research"][0][
            "policy_evidence"
        ]
        == pe
    )


def test_phase_scoping_and_inactive_primary():
    stack = resolve_method_stack(
        {
            "primary_method": "standard_empirical",
            "phases": {"hypothesis_generation": "bayesian"},
        }
    )
    contract = validate_policy_plan(stack.to_dict(), {"predictions": [{"id": "p1"}]})
    assert contract["enforced_plan_contracts"] == []
    assert not policy_requirements(stack.to_dict())["required_plan_extensions"]
    stack = resolve_method_stack(
        {
            "primary_method": "standard_empirical",
            "phases": {"hypothesis_generation": "strong_inference"},
        }
    )
    p = {"predictions": [{"id": "p1"}], "rivals": policy_plan()["rivals"]}
    contract = validate_policy_plan(stack.to_dict(), p)
    assert contract["requirements"]["strong_inference"] == ["rivals"]
    stack = resolve_method_stack(
        {
            "primary_method": "bayesian",
            "phases": dict.fromkeys(METHOD_PHASES, "standard_empirical"),
        }
    )
    assert stack.methods_used == ["standard_empirical"]
    assert not policy_requirements(stack.to_dict())["required_plan_extensions"]
    prompt = build_method_prompt_addendum(stack)
    assert "Update confidence from prior to posterior" not in prompt
    assert "read or create method_ledger" not in prompt


def test_distinct_numbers_without_distinct_rule_outcomes_are_not_discriminating():
    p = policy_plan()
    for r, v in zip(p["rivals"], [2, 3]):
        r["predictions"]["p1"] = v
    with pytest.raises(ValueError, match="different outcomes"):
        validate_policy_plan({"primary_method": "strong_inference"}, p)


def test_popperian_auxiliary_placeholders_rejected():
    p = policy_plan()
    p["severe_test"]["auxiliary_assumptions"] = [None]
    with pytest.raises(ValueError, match="auxiliary"):
        validate_policy_plan({"primary_method": "popperian_falsificationist"}, p)


def test_unverified_policy_annotations_cannot_masquerade_as_supervisor_evidence():
    html = _policy_evidence_panel(
        {
            "evidence_source": "supervisor_computed",
            "policy_evidence": {
                "belief_update": {
                    "prior": "not valid evidence",
                    "posterior": {"<script>alert(1)</script>": 1},
                }
            },
        }
    )
    assert "Unverified policy annotations" in html
    assert "<script>" not in html and "Posterior" not in html


def test_bayesian_underflow_is_distinct_from_impossible_evidence():
    p = policy_plan()
    p["belief_model"]["prior"] = {"h0": 1e-300, "h1": 1.0}
    p["belief_model"]["likelihood"] = {
        "h0": {"supported": 1e-300, "contradicted": 1},
        "h1": {"supported": 0, "contradicted": 1},
    }
    contract = validate_policy_plan({"primary_method": "bayesian"}, p)
    result = policy_evidence(
        {"plan": p, "policy_contract": contract}, {"p1": "supported"}
    )["belief_update"]
    assert result["status"] == "updated" and result["posterior"] == {"h0": 1, "h1": 0}
    assert result["log_evidence_probability"] == pytest.approx(2 * math.log(1e-300))
    p["belief_model"]["likelihood"]["h0"]["supported"] = 0
    result = policy_evidence(
        {"plan": p, "policy_contract": contract}, {"p1": "supported"}
    )["belief_update"]
    assert result["status"] == "model_inadequate" and result["posterior"] is None


@pytest.mark.parametrize("impossible", [False, True])
def test_extreme_bayesian_events_through_managed_evaluation(prepare, impossible):
    q, expid, p = prepare("bayesian", name="extreme")
    p["belief_model"]["prior"] = {"h0": 1e-300, "h1": 1.0}
    p["belief_model"]["likelihood"] = {
        "h0": {"supported": 0 if impossible else 1e-300, "contradicted": 1},
        "h1": {"supported": 0, "contradicted": 1},
    }
    write_yaml(q / "experiments" / expid / "experiment_plan.yaml", p)
    preregister_experiment(q, expid)
    result = execute_registered(q, expid, timeout=30)
    update = result["policy_evidence"]["belief_update"]
    assert update["status"] == ("model_inadequate" if impossible else "updated")
    assert update["posterior"] == (None if impossible else {"h0": 1, "h1": 0})
    assert run_logic_check(q, expid)["passed"]


@pytest.mark.parametrize("inactive_primary", [False, True])
def test_guidance_phase_does_not_force_bayesian_update(prepare, inactive_primary):
    stack = (
        {
            "primary_method": "bayesian",
            "phases": dict.fromkeys(METHOD_PHASES, "standard_empirical"),
        }
        if inactive_primary
        else {
            "primary_method": "standard_empirical",
            "phases": {"hypothesis_generation": "bayesian"},
        }
    )
    q, expid, p = prepare(stack, name="scoped")
    for field in (
        "belief_model",
        "rivals",
        "severe_test",
        "discriminating_prediction_id",
    ):
        p.pop(field)
    write_yaml(q / "experiments" / expid / "experiment_plan.yaml", p)
    reg = preregister_experiment(q, expid)
    if inactive_primary:
        assert "bayesian" not in reg["profiles"]
    result = execute_registered(q, expid, timeout=30)
    assert "belief_update" not in result["policy_evidence"]
    assert run_logic_check(q, expid)["passed"]


@pytest.mark.parametrize(
    "selection",
    [
        {"primary_method": []},
        {"primary_method": ""},
        {"phases": []},
        {"mode": False},
        {"phases": {"stress_testing": None}},
        {"rationale": []},
        {"primary_methd": "bayesian"},
    ],
)
def test_ambiguous_stack_configuration_is_rejected(selection):
    with pytest.raises(ValueError):
        resolve_method_stack(selection)


def test_bayesian_updates_match_high_precision_oracle():
    rng = random.Random(21037)
    with localcontext() as ctx:
        ctx.prec = 100
        for _ in range(200):
            p = policy_plan()
            a = rng.uniform(0.01, 0.99)
            prior = {"h0": a, "h1": 1 - a}
            p["belief_model"]["prior"] = prior
            like = {h: 10 ** rng.uniform(-300, 0) for h in prior}
            p["belief_model"]["likelihood"] = {
                h: {"supported": v, "contradicted": 1 - v} for h, v in like.items()
            }
            reg = {
                "plan": p,
                "policy_contract": validate_policy_plan(
                    {"primary_method": "bayesian"}, p
                ),
            }
            actual = policy_evidence(reg, {"p1": "supported"})["belief_update"][
                "posterior"
            ]
            weights = {
                h: Decimal.from_float(prior[h]) * Decimal.from_float(like[h])
                for h in prior
            }
            total = sum(weights.values())
            for h in prior:
                assert actual[h] == pytest.approx(
                    float(weights[h] / total), rel=1e-11, abs=1e-320
                )


def test_sequential_bayes_verifies_prior_and_prevents_exact_evidence_reuse(prepare):
    q, e1, p = prepare("bayesian", name="first")
    preregister_experiment(q, e1)
    first = execute_registered(q, e1, timeout=30)
    q, e2, p = prepare("bayesian", name="second", outcome="contradicted", quest=q)
    p["belief_model"].update(
        prior=first["policy_evidence"]["belief_update"]["posterior"],
        prior_source={"experiment_id": e1},
        conditional_evidence_assumptions="Independent observation process, fresh rows and the same rival identities.",
    )
    path = q / "experiments" / e2 / "experiment_plan.yaml"
    wrong = deepcopy(p)
    wrong["belief_model"]["prior"] = {"h0": 0.5, "h1": 0.5}
    write_yaml(path, wrong)
    with pytest.raises(ValueError, match="match the source posterior"):
        preregister_experiment(q, e2)
    write_yaml(path, p)
    reg = preregister_experiment(q, e2)
    assert reg["prior_chain"]["status"] == "linked_verified"
    second = execute_registered(q, e2, timeout=30)
    assert second["policy_evidence"]["belief_update"]["posterior"] == pytest.approx(
        {"h0": 0.5, "h1": 0.5}
    )
    # Keep the same data bytes: a third likelihood multiplication is inadmissible.
    e3 = run_next(q, agent_stub=True)
    exp = q / "experiments" / e3
    nbformat.write(
        nbformat.read(q / "experiments" / e2 / "notebook.ipynb", as_version=4),
        exp / "notebook.ipynb",
    )
    p["belief_model"].update(
        prior=second["policy_evidence"]["belief_update"]["posterior"],
        prior_source={"experiment_id": e2},
    )
    write_yaml(exp / "experiment_plan.yaml", p)
    with pytest.raises(ValueError, match="reused evaluation evidence"):
        preregister_experiment(q, e3)
    assert get_record(q, "registration", e3) is None


def test_profiles_reject_invalid_shapes_duplicate_ids_and_unknown_phases(tmp_path):
    profile = asdict(MethodRegistry.default().get("standard_empirical"))
    for change in [
        {"phase_affinities": ["next_step_selection"]},
        {"experiment_design_rules": "not a list"},
        {"description": None},
        {"typo": 1},
    ]:
        with pytest.raises(ValueError):
            MethodProfile.from_dict({**profile, **change})
    minimal = {
        "id": "custom",
        "display_name": "Custom",
        "plain_language_label": "Custom",
        "description": "Local profile",
    }
    write_yaml(tmp_path / "one.yaml", minimal)
    write_yaml(tmp_path / "two.yaml", minimal)
    with pytest.raises(ValueError, match="Duplicate method"):
        MethodRegistry.from_directory(tmp_path)
    (tmp_path / "two.yaml").unlink()
    (tmp_path / "one.yaml").write_text("id: one\nid: two\n")
    with pytest.raises(ValueError, match="Duplicate YAML"):
        MethodRegistry.from_directory(tmp_path)


def test_recommendation_uses_only_comparable_authoritative_scores(tmp_path):
    q = create_quest(tmp_path, {"hero_statement": "Predict a quantity"}, slug="history")
    for i, (score, group) in enumerate([(0.9, "g"), (0.8, "other"), (0.7, "g")], 1):
        expid = f"exp_{i:03d}"
        registration = {
            "registration_hash": f"r{i}",
            "file_hashes": {},
            "data_manifest": {"datasets": []},
        }
        evaluation = {
            "aggregate_score": score,
            "comparison_group": group,
            "registration_hash": f"r{i}",
            "validation_status": "scored",
            "evidence_source": "supervisor_computed",
        }
        record(q, "registration", expid, registration)
        record(q, "evaluation", expid, evaluation)
        record(q, "completion", expid, {"evaluation_hash": content_hash(evaluation)})
    assert not _history_signals(q)["declining_scores"]
    expid = "exp_004"
    evaluation = {
        "aggregate_score": 0.6,
        "comparison_group": "g",
        "registration_hash": "r4",
        "validation_status": "scored",
        "evidence_source": "supervisor_computed",
    }
    record(
        q,
        "registration",
        expid,
        {
            "registration_hash": "r4",
            "file_hashes": {},
            "data_manifest": {"datasets": []},
        },
    )
    record(q, "evaluation", expid, evaluation)
    record(q, "completion", expid, {"evaluation_hash": content_hash(evaluation)})
    assert _history_signals(q)["declining_scores"]
    recommendation = auto_recommend_method_stack({"quest_path": q})
    assert recommendation.primary_method == "lakatosian"
    assert "does not establish" in recommendation.rationale
