"""Decision value, constraint gates, and metric-role semantics."""

import csv
import random
from copy import deepcopy

import nbformat
import pytest

from sciquest.core import create_quest, run_next
from sciquest.dashboard import build_dashboard
from sciquest.decisions import evaluate_decisions
from sciquest.evaluation import compute_artifact_metric
from sciquest.io import read_yaml, write_yaml
from sciquest.logic import run_logic_check
from sciquest.methods.ledger import preregister_experiment
from sciquest.state import read_state
from sciquest.validation import compute_aggregate_score, validate_suite
from sciquest.workflow import execute_registered, stage_research_context


def metric(kind="regret", **overrides):
    cfg = {
        "kind": kind,
        "truth": "data/0/eval.csv",
        "target": "value",
        "prediction": "predicted_value",
        "group_by": ["state"],
        "action": "price",
        "objective": "maximize",
        "tie_break": "lowest_action",
        "value_semantics": "expected_interventional",
    }
    cfg.update(overrides.pop("computation", {}))
    return dict(name=kind, computation=cfg, **overrides)


def tables():
    # Unequal group sizes test state weighting rather than row weighting.
    truth = {
        "a": {"state": "s1", "price": "50", "value": "5000", "violation": "0"},
        "b": {"state": "s1", "price": "60", "value": "5400", "violation": "1"},
        "c": {"state": "s2", "price": "1", "value": "-20", "violation": "0"},
        "d": {"state": "s2", "price": "2", "value": "-10", "violation": "0"},
        "e": {"state": "s2", "price": "3", "value": "-30", "violation": "1"},
    }
    pred = {
        k: {"predicted_value": str(v)} for k, v in zip(truth, [5000, 4800, 0, 0, 0])
    }
    return truth, pred


def test_action_selection_value_regret_and_constraint_rate():
    truth, pred = tables()
    assert (
        evaluate_decisions(truth, pred, metric("policy_value")["computation"])[0]
        == 2490
    )
    value, detail = evaluate_decisions(truth, pred, metric()["computation"])
    assert value == 205  # Mean of 400 and 10, not action-weighted.
    assert detail["selection_tie_groups"] == 1
    assert detail["trace_sample"][0]["selected_id"] == "a"
    cfg = metric("constraint_violation_rate", computation={"target": "violation"})[
        "computation"
    ]
    assert evaluate_decisions(truth, pred, cfg)[0] == 0
    cfg["tie_break"] = "highest_action"
    assert evaluate_decisions(truth, pred, cfg)[0] == 0.5
    # A different policy's claims about its reward never replace truth.
    pred["b"]["predicted_value"] = "999999999"
    assert (
        evaluate_decisions(truth, pred, metric("policy_value")["computation"])[0]
        == 2690
    )


def test_minimization_negative_values_zero_values_and_row_order():
    truth, pred = tables()
    cfg = metric(computation={"objective": "minimize"})["computation"]
    value, _ = evaluate_decisions(truth, pred, cfg)
    assert value == 205
    assert (
        evaluate_decisions(
            dict(reversed(list(truth.items()))), dict(reversed(list(pred.items()))), cfg
        )[0]
        == value
    )
    for r in truth.values():
        r["value"] = "0"
    assert evaluate_decisions(truth, pred, cfg)[0] == 0
    cfg["kind"] = "policy_value"
    assert evaluate_decisions(truth, pred, cfg)[0] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"objective": "maximise"},
        {"group_by": []},
        {"group_by": ["state", "state"]},
        {"action": ""},
        {"tie_break": "first_row"},
        {"value_semantics": "observational"},
        {"prediction": None},
        {"group_by": ["state", "price"]},
    ],
)
def test_incomplete_or_ambiguous_decision_contract_rejected(changes):
    with pytest.raises(ValueError):
        validate_suite({"metrics": [metric(computation=changes)]})


def test_direction_and_bad_evidence_fail_closed():
    with pytest.raises(ValueError, match="direction"):
        validate_suite({"metrics": [metric(direction="maximize")]})
    truth, pred = tables()
    for mutate in [
        lambda t, p: t["b"].update(price="50.0"),
        lambda t, p: t["b"].update(price="nan"),
        lambda t, p: p["b"].update(predicted_value="inf"),
        lambda t, p: t["b"].update(state=""),
        lambda t, p: p.pop("b"),
    ]:
        t, p = deepcopy(truth), deepcopy(pred)
        mutate(t, p)
        with pytest.raises(ValueError):
            evaluate_decisions(t, p, metric()["computation"])
    truth["b"]["violation"] = ".5"
    with pytest.raises(ValueError, match="binary"):
        evaluate_decisions(
            truth,
            pred,
            metric("constraint_violation_rate", computation={"target": "violation"})[
                "computation"
            ],
        )


def test_random_action_grids_against_direct_oracle():
    rng = random.Random(71)
    for objective in ["maximize", "minimize"]:
        for tie in ["lowest_action", "highest_action"]:
            for _ in range(50):
                truth, pred = {}, {}
                expected = []
                for g in range(4):
                    actual = [rng.randint(-100, 100) for _ in range(rng.randint(1, 8))]
                    prediction = [rng.randint(-2, 2) for _ in actual]
                    best = (max if objective == "maximize" else min)(prediction)
                    choices = [i for i, p in enumerate(prediction) if p == best]
                    chosen = (min if tie == "lowest_action" else max)(choices)
                    oracle = (max if objective == "maximize" else min)(actual)
                    expected.append(abs(oracle - actual[chosen]))
                    for i, (a, p) in enumerate(zip(actual, prediction)):
                        k = f"{g}-{i}"
                        truth[k] = {"state": str(g), "price": str(i), "value": str(a)}
                        pred[k] = {"predicted_value": str(p)}
                assert (
                    evaluate_decisions(
                        truth,
                        pred,
                        metric(computation={"objective": objective, "tie_break": tie})[
                            "computation"
                        ],
                    )[0]
                    == sum(expected) / 4
                )


def test_metric_roles_gates_and_undefined_diagnostics():
    suite = {
        "metrics": [
            {"name": "score"},
            {"name": "baseline", "role": "diagnostic"},
            {
                "name": "limit",
                "role": "constraint",
                "requirement": {"operator": "<=", "threshold": 0.1},
            },
        ]
    }
    passed = compute_aggregate_score(
        suite, {"score": 0.8, "baseline": 100000, "limit": 0}
    )
    assert passed["aggregate_score"] == 0.8 and set(passed["scores"]) == {"score"}
    failed = compute_aggregate_score(suite, {"score": 1, "baseline": 0, "limit": 0.2})
    assert failed["aggregate_score"] is None
    assert failed["validation_status"] == "constraint_failed"
    assert failed["constraint_outcomes"] == {"limit": "failed"}
    evidence = {
        "metrics": {"score": 0.8, "limit": 0},
        "undefined_metrics": {"baseline": "Constant rank"},
    }
    assert compute_aggregate_score(suite, evidence)["aggregate_score"] == 0.8
    evidence = {
        "metrics": {"baseline": 0, "limit": 0},
        "undefined_metrics": {"score": "Undefined primary"},
    }
    assert (
        compute_aggregate_score(suite, evidence)["validation_status"] == "inconclusive"
    )
    evidence = {
        "metrics": {"baseline": 0, "score": 1},
        "undefined_metrics": {"limit": "Undefined constraint"},
    }
    assert compute_aggregate_score(suite, evidence)["aggregate_score"] is None
    with pytest.raises(ValueError, match="Missing required"):
        compute_aggregate_score(suite, {"score": 1, "limit": 0})
    with pytest.raises(ValueError, match="conflicting"):
        compute_aggregate_score(
            suite,
            {
                "metrics": {"score": 1, "limit": 0, "baseline": 0},
                "undefined_metrics": {"baseline": "oops"},
            },
        )
    with pytest.raises(ValueError, match="weights must be zero"):
        validate_suite({"metrics": [{"name": "m", "role": "diagnostic", "weight": 1}]})
    with pytest.raises(ValueError, match="constraint requires"):
        validate_suite({"metrics": [{"name": "m", "role": "constraint"}]})
    assert (
        compute_aggregate_score(
            {"metrics": [{"name": "d", "role": "diagnostic"}]}, {"d": 2}
        )["validation_status"]
        == "evaluated"
    )


def test_trace_is_bounded_and_numerical_overflow_fails_closed():
    truth = {str(i): {"state": str(i), "price": "1", "value": "0"} for i in range(105)}
    pred = {str(i): {"predicted_value": "0"} for i in range(105)}
    value, detail = evaluate_decisions(truth, pred, metric()["computation"])
    assert value == 0 and detail["trace_truncated"] and detail["group_count"] == 105
    assert len(detail["trace_sample"]) == 100
    truth = {
        "a": {"state": "s", "price": "1", "value": "1e308"},
        "b": {"state": "s", "price": "2", "value": "-1e308"},
    }
    pred = {"a": {"predicted_value": "0"}, "b": {"predicted_value": "1"}}
    with pytest.raises(ValueError):
        evaluate_decisions(truth, pred, metric()["computation"])


def test_decision_artifacts_require_registered_complete_truth(tmp_path):
    snapshot = tmp_path / "snapshot"
    out = tmp_path / "out"
    (snapshot / "data/0").mkdir(parents=True)
    out.mkdir()
    truth, pred = tables()
    for file, rows in [
        (snapshot / "data/0/eval.csv", truth),
        (out / "predictions.csv", pred),
    ]:
        with file.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", *next(iter(rows.values())).keys()])
            w.writeheader()
            w.writerows(dict(id=k, **r) for k, r in rows.items())
    with pytest.raises(ValueError, match="registered dataset"):
        compute_artifact_metric(snapshot, out, metric(), {"file_hashes": {}})
    assert (
        compute_artifact_metric(
            snapshot, out, metric(), {"file_hashes": {"data/0/eval.csv": "hash"}}
        )
        == 205
    )


@pytest.mark.parametrize("choose_high", [False, True])
def test_managed_decisions_hidden_truth_constraints_and_reporting(
    tmp_path, choose_high
):
    q = create_quest(tmp_path, {"hero_statement": "Choose a price"}, slug="decisions")
    (q / "data/raw/train.csv").write_text("id,price,value\nt0,10,10\n")
    (q / "data/raw/eval.csv").write_text(
        "id,state,price,value,violation\na,s,50,5000,0\nb,s,60,5400,1\n"
    )
    write_yaml(
        q / "data_manifest.yaml",
        {
            "status": "ready",
            "datasets": [
                {
                    "path": "data/raw/train.csv",
                    "role": "training",
                    "provenance": "Manual training fixture",
                },
                {
                    "path": "data/raw/eval.csv",
                    "role": "evaluation",
                    "input_columns": ["id", "state", "price"],
                    "provenance": "Owner-specified expected interventional fixture",
                },
            ],
        },
    )
    metrics = [
        metric(
            "policy_value",
            role="primary",
            normalization={"type": "minmax", "min": 0, "max": 6000},
        ),
        metric(role="diagnostic"),
        metric(
            "constraint_violation_rate",
            role="constraint",
            requirement={"operator": "<=", "threshold": 0},
            computation={"target": "violation"},
        ),
    ]
    for m in metrics:
        m["computation"]["truth"] = "data/1/eval.csv"
    # Undefined constant rank is a diagnostic, so it must not suppress value.
    metrics.append(
        {
            "name": "rank",
            "role": "diagnostic",
            "computation": {
                "kind": "rank_correlation",
                "truth": "data/1/eval.csv",
                "target": "value",
                "prediction": "constant",
                "group_by": ["state"],
                "order_by": "price",
            },
        }
    )
    write_yaml(q / "validation.yaml", {"status": "ready", "metrics": metrics})
    expid = run_next(q, agent_stub=True)
    exp = q / "experiments" / expid
    write_yaml(
        exp / "experiment_plan.yaml",
        {
            "question": "Which listed price has higher value?",
            "hypothesis": "The policy meets the registered value threshold.",
            "scope": "Two-action expected-value fixture",
            "assumptions": "Owner provides known expected intervention values and violation labels.",
            "data_access": "Training and evaluation features only",
            "stopping_rule": "One evaluation",
            "action": "hypothesis_test",
            "confirmatory_status": "exploratory",
            "predictions": [
                {
                    "id": "p1",
                    "metric": "policy_value",
                    "operator": ">=",
                    "threshold": 5000,
                }
            ],
        },
    )
    code = f"""import csv
rows=list(csv.DictReader(open('data/1/features.csv')))
assert 'value' not in rows[0] and 'violation' not in rows[0]
with open('predictions.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['id','predicted_value','constant']);w.writeheader()
    for r in rows:
        score=float(r['price'])*{1 if choose_high else -1}
        w.writerow(dict(id=r['id'],predicted_value=score,constant=1))
"""
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]),
        exp / "notebook.ipynb",
    )
    preregister_experiment(q, expid)
    result = execute_registered(q, expid, timeout=30)
    assert result["metrics"]["policy_value"] == (5400 if choose_high else 5000)
    assert result["metrics"]["regret"] == (0 if choose_high else 400)
    assert result["metrics"]["constraint_violation_rate"] == int(choose_high)
    assert result["validation_status"] == (
        "constraint_failed" if choose_high else "scored"
    )
    assert result["aggregate_score"] == (None if choose_high else 5000 / 6000)
    assert result["undefined_metrics"] == {
        "rank": "Rank correlation is undefined for constant rankings"
    }
    assert result["decision_evidence"]["regret"]["trace_sample"][0]["selected_id"] == (
        "b" if choose_high else "a"
    )
    groups = read_state(q).get("score_groups", {})
    assert (groups.get(result["comparison_group"], {}).get("best_score")) == (
        None if choose_high else 5000 / 6000
    )
    assert run_logic_check(q, expid)["passed"]
    feedback = tmp_path / "feedback"
    feedback.mkdir()
    stage_research_context(q, feedback)
    history = read_yaml(feedback / "research_state.yaml")["completed_research"][0]
    assert history["constraint_outcomes"] == result["constraint_outcomes"]
    assert history["decision_evidence"]["regret"]["feedback_trace_limit"] == 5
    html = build_dashboard(q).read_text()
    assert "constraint failed" in html if choose_high else "diagnostic" in html
