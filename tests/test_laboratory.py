import math

import pytest
from typer.testing import CliRunner

from sciquest.cli import app
from sciquest.laboratory import (
    LabConfig,
    PricingLab,
    POLICIES,
    _gaussian_information,
    reference_campaign,
)
from sciquest.methods.policies import validate_policy_plan, policy_evidence


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("scenario", ["elastic", "null", "censored"])
def test_reference_policies_recover_simple_known_mechanisms(policy, scenario):
    result = reference_campaign(LabConfig(scenario, 42, 16), policy)
    assert result["evaluation"]["mechanism_recovered"]
    assert result["evaluation"]["observations_used"] <= 16
    assert result["evaluation"]["intervention_prediction_rmse"] < 0.2
    assert all(step["action"]["kind"] == "intervene" for step in result["trace"])
    if scenario == "censored":
        assert all(step["action"]["capacity"] == 150 for step in result["trace"])


@pytest.mark.parametrize("policy", POLICIES)
def test_unidentifiable_questions_produce_abstention(policy):
    result = reference_campaign(LabConfig("observational_only", 13, 16), policy)
    assert result["decision"]["status"] == "not_identifiable"
    assert result["evaluation"]["correct_identifiability_abstention"]
    assert not result["evaluation"]["mechanism_recovered"]
    assert result["decision"]["posterior"] == [0.5, 0.5]


def test_model_misspecification_is_not_confused_with_posterior_confidence():
    result = reference_campaign(LabConfig("misspecified", 0, 16), "bayesian")
    assert result["decision"]["status"] == "model_inadequate"
    assert result["evaluation"]["correct_model_inadequacy_abstention"]
    assert result["trace"][-1]["model_check_failed"]


def test_campaign_reproducibility_and_budget():
    assert reference_campaign(
        LabConfig("measurement_error", 11, 16), "bayesian"
    ) == reference_campaign(LabConfig("measurement_error", 11, 16), "bayesian")
    result = reference_campaign(LabConfig("confounded", 11, 1), "strong_inference")
    assert result["decision"]["status"] == "inconclusive"
    assert result["evaluation"]["observations_used"] == 1
    lab = PricingLab(LabConfig("elastic", 1, 1))
    lab.request({"kind": "intervene", "price": 0.5, "capacity": 150})
    with pytest.raises(RuntimeError, match="budget"):
        lab.request({"kind": "observe"})
    lab = PricingLab(LabConfig("observational_only", 1, 10))
    with pytest.raises(ValueError, match="observational"):
        lab.request({"kind": "intervene"})


def test_information_integral_agrees_with_simple_limits():
    assert _gaussian_information([100, 100], [0.5, 0.5], 2) == pytest.approx(
        0, abs=1e-12
    )
    assert _gaussian_information([50, 150], [0.5, 0.5], 2) == pytest.approx(
        math.log(2), abs=1e-6
    )
    assert _gaussian_information([90, 100], [0.5, 0.5], 20) < _gaussian_information(
        [90, 100], [0.5, 0.5], 2
    )


def test_bayesian_contract_performs_numerical_update_and_detects_impossible_evidence():
    stack = {"primary_method": "bayesian", "phases": {}}
    plan = {
        "predictions": [{"id": "p1"}],
        "rivals": [
            {"id": "h0", "claim": "No effect"},
            {"id": "h1", "claim": "Price effect"},
        ],
        "belief_model": {
            "prediction_id": "p1",
            "prior": {"h0": 0.5, "h1": 0.5},
            "likelihood": {
                "h0": {"supported": 0.1, "contradicted": 0.9},
                "h1": {"supported": 0.9, "contradicted": 0.1},
            },
            "assumptions": "Specified observation model",
        },
    }
    contract = validate_policy_plan(stack, plan)
    evidence = policy_evidence(
        {"plan": plan, "policy_contract": contract}, {"p1": "supported"}
    )
    assert evidence["belief_update"]["posterior"] == pytest.approx(
        {"h0": 0.1, "h1": 0.9}
    )
    for likelihood in plan["belief_model"]["likelihood"].values():
        likelihood.update(supported=0, contradicted=1)
    evidence = policy_evidence(
        {"plan": plan, "policy_contract": contract}, {"p1": "supported"}
    )
    assert evidence["belief_update"]["status"] == "model_inadequate"
    assert evidence["belief_update"]["posterior"] is None


def test_strong_inference_rejects_rivals_with_identical_predictions():
    plan = {
        "predictions": [{"id": "p1"}],
        "discriminating_prediction_id": "p1",
        "rivals": [{"id": h, "claim": h, "predictions": {"p1": 1}} for h in ["a", "b"]],
    }
    with pytest.raises(ValueError, match="different outcomes"):
        validate_policy_plan({"primary_method": "strong_inference", "phases": {}}, plan)


def test_lab_cli_preserves_trace_without_overwriting(tmp_path):
    output = tmp_path / "campaign.yaml"
    runner = CliRunner()
    result = runner.invoke(app, ["lab", "--scenario", "null", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert runner.invoke(app, ["lab", "--output", str(output)]).exit_code != 0
