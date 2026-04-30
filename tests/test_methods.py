from pathlib import Path

import yaml
from typer.testing import CliRunner

from sciquest.agent import build_agent_prompt
from sciquest.cli import app
from sciquest.core import create_quest, run_next
from sciquest.methods.ledger import classify_result_claim, preregister_experiment
from sciquest.methods.prompt_builder import build_method_prompt_addendum, build_method_journal_template
from sciquest.methods.registry import MethodRegistry
from sciquest.methods.resolver import auto_recommend_method_stack, resolve_method_stack


def test_method_registry_loads_data_driven_profiles():
    registry = MethodRegistry.default()
    bayesian = registry.get("bayesian")

    assert bayesian.id == "bayesian"
    assert bayesian.plain_language_label == "Update confidence from evidence"
    assert "posterior" in " ".join(bayesian.journal_sections).lower()
    assert registry.get("standard_empirical").display_name == "Standard empirical loop"
    assert len(registry.list_profiles()) >= 9


def test_resolver_supports_single_method_and_phase_composition():
    single = resolve_method_stack({"primary_method": "popperian_falsificationist"})
    assert single.primary_method == "popperian_falsificationist"
    assert set(single.phases.values()) == {"popperian_falsificationist"}

    composed = resolve_method_stack({
        "primary_method": "strong_inference",
        "phases": {
            "hypothesis_generation": "feyerabendian_pluralist",
            "evidence_evaluation": "bayesian",
            "stress_testing": "popperian_falsificationist",
        },
    })
    assert composed.phases["experiment_design"] == "strong_inference"
    assert composed.phases["hypothesis_generation"] == "feyerabendian_pluralist"
    assert composed.phases["evidence_evaluation"] == "bayesian"
    assert "standard_empirical" not in composed.methods_used


def test_auto_recommend_method_stack_from_quest_metadata():
    causal = auto_recommend_method_stack({
        "hero_statement": "Estimate whether treatment changes customer retention",
        "problem_statement": "We need an intervention and counterfactual effect estimate with confounders.",
    })
    assert causal.primary_method == "causal_interventionist"
    assert causal.phases["evidence_evaluation"] == "causal_interventionist"

    ambiguous = auto_recommend_method_stack({
        "hero_statement": "Explore a vague early-stage phenomenon",
        "problem_statement": "Ambiguous topic with no clear model yet.",
    })
    assert ambiguous.primary_method == "exploratory_inductive"
    assert ambiguous.phases["hypothesis_generation"] == "abductive"


def test_method_prompt_and_journal_sections_are_method_specific():
    stack = resolve_method_stack({
        "primary_method": "strong_inference",
        "phases": {
            "hypothesis_generation": "feyerabendian_pluralist",
            "evidence_evaluation": "bayesian",
            "stress_testing": "popperian_falsificationist",
        },
    })
    prompt = build_method_prompt_addendum(stack)
    journal = build_method_journal_template(stack)

    assert "Method Stack" in prompt
    assert "rival hypotheses" in prompt.lower()
    assert "posterior" in prompt.lower()
    assert "heterodox" in prompt.lower()
    assert "Risky predictions" in journal
    assert "Priors" in journal


def test_create_quest_writes_default_method_stack_and_agent_prompt_mentions_it(tmp_path):
    quest = create_quest(tmp_path, {
        "hero_statement": "Hero",
        "problem_statement": "Problem",
        "initial_hypothesis": "Hypothesis",
    }, slug="demo")
    method_stack = yaml.safe_load((quest / "method_stack.yaml").read_text())

    assert method_stack["primary_method"] == "standard_empirical"
    assert method_stack["mode"] == "simple"
    prompt = build_agent_prompt(quest)
    assert "method_stack.yaml" in prompt
    assert "Method-aware reasoning" in prompt


def test_run_next_records_method_ledger_before_experiment(tmp_path):
    quest = create_quest(tmp_path, {
        "hero_statement": "Hero",
        "problem_statement": "Problem",
        "initial_hypothesis": "Hypothesis",
        "method_stack": {
            "primary_method": "bayesian",
            "phases": {"stress_testing": "popperian_falsificationist"},
        },
    }, slug="demo")

    exp_id = run_next(quest, agent_stub=True)
    ledger = yaml.safe_load((quest / "experiments" / exp_id / "method_ledger.yaml").read_text())
    report = (quest / "experiments" / exp_id / "experiment_report.md").read_text()

    assert ledger["experiment_id"] == exp_id
    assert ledger["pre_registration"]["primary_method"] == "bayesian"
    assert ledger["pre_registration"]["confirmatory_status"] == "confirmatory"
    assert "allowed_post_hoc_analysis_rules" in ledger["pre_registration"]
    assert "Method Ledger" in report
    assert "Priors" in report


def test_claim_classification_marks_post_hoc_and_method_violations():
    prereg = {
        "predicted_observations": ["Demand decreases as price rises"],
        "falsification_or_success_criteria": ["Law of demand pass rate >= 0.8"],
        "allowed_post_hoc_analysis_rules": ["Exploratory analyses must be labeled exploratory"],
    }
    assert classify_result_claim(prereg, {"claim_type": "confirmatory", "matched_prediction": True}) == "confirmatory_result"
    assert classify_result_claim(prereg, {"claim_type": "exploratory"}) == "exploratory_observation"
    assert classify_result_claim(prereg, {"claim_type": "post_hoc", "labeled_post_hoc": True}) == "post_hoc_reinterpretation"
    assert classify_result_claim(prereg, {"claim_type": "post_hoc", "labeled_post_hoc": False}) == "method_violation"
    assert classify_result_claim(prereg, {"claim_type": "confirmatory", "matched_prediction": False}) == "unresolved_anomaly"


def test_cli_new_accepts_simple_method_choice(tmp_path):
    runner = CliRunner()
    inputs = "Hero\nProblem\nHypothesis\n\n\n\n\ny\n"
    result = runner.invoke(app, [
        "new", "--root", str(tmp_path), "--slug", "method-demo", "--no-splash", "--method", "bayesian",
    ], input=inputs)
    assert result.exit_code == 0, result.output
    method_stack = yaml.safe_load((tmp_path / "quests" / "method-demo" / "method_stack.yaml").read_text())
    assert method_stack["primary_method"] == "bayesian"
    assert method_stack["mode"] == "simple"
