"""Supervisor-only epistemic, predictive and decision evaluation."""

from __future__ import annotations

import math
from pathlib import Path

from sciquest.decisions import evaluate_decisions
from sciquest.evaluation import _number, _table

from .world import FAMILIES, evaluation_rows


def evaluate_science(
    world: dict, public: dict, decision: dict, artifact: Path, spent: int, queries: int
) -> dict:
    if world.get("spec", {}).get("version") == "designed-demand-v2":
        from .scoring_v2 import score_predictions

        return {
            **score_predictions(world, artifact),
            "cost_spent": spent,
            "query_count": queries,
            "explanation_evaluation": "Separate claim-level masked assessment required; status labels and prediction scores do not certify identification.",
        }
    rows = evaluation_rows(world)
    truth = {r["id"]: r for r in rows}
    pred = _table(artifact, "id")
    if truth.keys() != pred.keys():
        raise ValueError("Final predictions must cover every sealed evaluation row")
    errors = {"interpolation": [], "extrapolation": []}
    covered = []
    interval_scores = []
    inconsistent = 0
    parsed = {}
    for identity, row in truth.items():
        source = pred[identity]
        values = {
            key: _number(source.get(key), key)
            for key in ("mean_demand", "mean_sales", "lower_sales", "upper_sales")
        }
        if any(abs(v) > 1e6 for v in values.values()):
            raise ValueError("Predictions exceed the registered numerical safety limit")
        if (
            values["mean_demand"] < 0
            or not 0 <= values["mean_sales"] <= row["capacity"]
        ):
            raise ValueError("Predicted physical means violate demand/sales bounds")
        low, high = values["lower_sales"], values["upper_sales"]
        if low > high:
            raise ValueError("Predictive interval lower bound exceeds upper bound")
        y = row["measured_sales"]
        covered.append(float(low <= y <= high))
        interval_scores.append(
            high - low + 20 * max(0, low - y) + 20 * max(0, y - high)
        )
        errors[row["split"]].append(
            (values["mean_demand"] - row["expected_demand"]) ** 2
        )
        inconsistent += values["mean_sales"] > values["mean_demand"] + 1e-9
        parsed[identity] = {
            "predicted_revenue": str(row["price"] * values["mean_sales"])
        }
    eligible = {
        k: {key: str(value) for key, value in r.items()}
        for k, r in truth.items()
        if r["decision_eligible"]
    }
    dpred = {k: parsed[k] for k in eligible}
    cfg = dict(
        group_by=["state_id"],
        action="price",
        prediction="predicted_revenue",
        objective="maximize",
        tie_break="lowest_action",
        value_semantics="expected_interventional",
    )
    value, detail = evaluate_decisions(
        eligible, dpred, dict(cfg, kind="policy_value", target="expected_revenue")
    )
    regret, _ = evaluate_decisions(
        eligible, dpred, dict(cfg, kind="regret", target="expected_revenue")
    )
    violations, _ = evaluate_decisions(
        eligible, dpred, dict(cfg, kind="constraint_violation_rate", target="violation")
    )
    actual_family = world["model"]["family"]
    label = actual_family if actual_family in FAMILIES else "other"
    assisted = "mechanism_probabilities" in decision
    probabilities = decision.get("mechanism_probabilities", {f: 0 for f in FAMILIES})
    ptrue = probabilities[label]
    unidentified = (
        not public["interventions_available"]
        and public["historical_assignment"] == "endogenous"
    )
    selected = decision.get("selected_family")
    correct = decision["status"] == "selected" and selected == label
    parameter_errors = None
    if correct and decision.get("parameters"):
        parameter_errors = {
            key: abs(value - world["model"][key])
            for key, value in decision["parameters"].items()
        }
    result = dict(
        interventional_demand_rmse=math.sqrt(
            math.fsum(v for part in errors.values() for v in part) / len(rows)
        ),
        family_brier=math.fsum(
            (probabilities[f] - int(f == label)) ** 2 for f in FAMILIES
        ),
        family_log_loss=-math.log(ptrue) if ptrue > 0 else None,
        zero_probability_on_truth=ptrue == 0,
        selected_family_correct=correct,
        mechanism_recovered=correct
        and probabilities[selected] >= public["recovery_confidence_threshold"]
        if not unidentified
        else None,
        identifiable_under_access=not unidentified,
        correct_identifiability_abstention=unidentified
        and decision["status"] == "not_identifiable",
        unjustified_identifiability_claim=unidentified
        and decision["status"] == "selected",
        correct_model_inadequacy_abstention=label == "other"
        and decision["status"] == "model_inadequate",
        parameter_absolute_errors=parameter_errors,
        interpolation_demand_rmse=math.sqrt(
            math.fsum(errors["interpolation"]) / len(errors["interpolation"])
        ),
        extrapolation_demand_rmse=math.sqrt(
            math.fsum(errors["extrapolation"]) / len(errors["extrapolation"])
        ),
        predictive_interval_coverage_90=math.fsum(covered) / len(covered),
        mean_interval_score_90=math.fsum(interval_scores) / len(interval_scores),
        prediction_consistency_violations=inconsistent,
        policy_value=value,
        unconstrained_grid_regret=regret,
        constraint_violation_rate=violations,
        constraint_passed=violations == 0,
        decision_evidence=detail,
        cost_spent=spent,
        budget=public["budget"],
        query_count=queries,
        scope="Proper scores require repeated campaigns to assess calibration. Mechanism recovery is withheld for the constructed unidentifiable access condition. Decision regret is against the eligible finite grid, including any operating-constraint violations; no automatic scientific-validity verdict or sequential regret guarantee.",
    )
    if not assisted:
        # A free-form explanation has no automatic mapping to the owner's
        # finite labels. Do not invent a classification score or recovery claim.
        for key in (
            "family_brier",
            "family_log_loss",
            "zero_probability_on_truth",
            "selected_family_correct",
            "mechanism_recovered",
            "correct_model_inadequacy_abstention",
            "parameter_absolute_errors",
        ):
            result[key] = None
        result["explanation_evaluation"] = (
            "Requires a separately preregistered, blinded assessment; predictive accuracy alone is not mechanism recovery."
        )
        if "spec" in world:
            result["identifiable_under_access"] = None
            result["unjustified_identifiability_claim"] = None
            result["correct_identifiability_abstention"] = None
            result["scope"] = (
                "Agent-designed static expression worlds. Open-ended causal claims require the separately blinded explanation rubric. Predictive and decision scores are evaluated against private interventional truth; finite-budget identification is not automatically certified."
            )
    return result
