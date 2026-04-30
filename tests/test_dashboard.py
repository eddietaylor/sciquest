from pathlib import Path
import yaml
from typer.testing import CliRunner

from sciquest.cli import app
from sciquest.dashboard import build_dashboard
from sciquest.logic import run_logic_check


def make_minimal_valid_experiment(tmp_path: Path) -> Path:
    quest = tmp_path / "quests" / "demo"
    exp = quest / "experiments" / "exp_001"
    (exp / "artifacts" / "plots").mkdir(parents=True)
    (exp / "artifacts" / "diagrams").mkdir(parents=True)
    (exp / "logs").mkdir(parents=True)
    (quest / "reports").mkdir(parents=True)
    (quest / "data" / "raw").mkdir(parents=True)
    (quest / "data" / "raw" / "data.csv").write_text("price,capacity,revenue\n100,10,1000\n120,8,960\n80,12,960\n")
    (quest / "artifacts").mkdir()
    (quest / "artifacts" / "logo.png").write_bytes(b"fake-logo")
    (quest / "logs").mkdir()
    (quest / "quest.yaml").write_text("slug: demo\nhero_statement: Hero\nproblem_statement: Problem\n")
    (quest / "state.yaml").write_text("quest_status: idle\nlast_experiment: exp_001\nbest_score: 0.8\n")
    (quest / "hypotheses.yaml").write_text("hypotheses:\n- id: h001\n  claim: initial\n- id: h002\n  claim: evolved\n  experiment: exp_001\n")
    (quest / "validation.yaml").write_text("status: ready\nmetrics:\n- name: accuracy\n  description: Accuracy\n  direction: maximize\n  weight: 1\n  normalization: {type: minmax, min: 0, max: 1}\n")
    (quest / "data_manifest.yaml").write_text("status: ready\ndatasets:\n- name: synthetic_data\n  path: data/raw/data.csv\n  provenance: synthetic generated benchmark\n")
    (quest / "journal.md").write_text("# Journal\n\n## Experiment 1\n")
    (quest / "AGENTS.md").write_text("agent protocol")
    (exp / "experiment.yaml").write_text(yaml.safe_dump({
        "id": "exp_001",
        "status": "validated",
        "hypothesis_id": "h002",
        "task_type": "supervised",
        "model_architecture": "Linear probe",
        "input_features": ["price", "capacity"],
        "target_features": ["revenue"],
        "validation_technique": "Holdout counterfactual split",
        "technical_diagrams": ["artifacts/diagrams/model_architecture.svg", "artifacts/diagrams/validation_technique.svg"],
    }))
    (exp / "hypothesis.md").write_text("# h002\nTestable hypothesis")
    (exp / "notebook.ipynb").write_text('{"cells":[{"cell_type":"markdown","metadata":{},"source":["# Data preview target feature validation_results"]}],"metadata":{},"nbformat":4,"nbformat_minor":5}')
    (exp / "executed_notebook.ipynb").write_text((exp / "notebook.ipynb").read_text())
    (exp / "validation_results.yaml").write_text("metrics:\n  accuracy: 0.8\naggregate_score: 0.8\nscores:\n  accuracy:\n    raw: 0.8\n    normalized: 0.8\n    weight: 1\n")
    (exp / "experiment_report.md").write_text("# Report\n\n## Evidence\nGood graph interpretation.\n\n## Speculation\nMaybe generalizes.\n")
    (exp / "artifacts" / "plots" / "accuracy.svg").write_text("<svg><text>Accuracy</text></svg>")
    (exp / "artifacts" / "diagrams" / "model_architecture.svg").write_text("<svg><text>Model Architecture</text></svg>")
    (exp / "artifacts" / "diagrams" / "validation_technique.svg").write_text("<svg><text>Validation Technique</text></svg>")
    (exp / "artifacts" / "data_generation_script.py").write_text("# reproducible generator\n")
    return quest


def test_logic_check_requires_diagrams_and_reproducibility_artifact(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    assert run_logic_check(quest, "exp_001")["passed"]
    (quest / "experiments" / "exp_001" / "artifacts" / "data_generation_script.py").unlink()
    result = run_logic_check(quest, "exp_001")
    assert not result["passed"]
    assert any("data generation" in issue.lower() for issue in result["issues"])


def test_dashboard_builds_dynamic_experiment_page(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    out = build_dashboard(quest)
    html = out.read_text()
    assert "SciQuest Dashboard" in html
    assert "exp_001" in html
    assert "supervised" in html
    assert "price" in html and "revenue" in html
    assert "The Data" in html
    assert "Data source" in html
    assert "Synthetic" in html
    assert "Data shape" in html
    assert "3 rows × 3 columns" in html
    assert "Descriptive statistics" in html
    assert "price" in html and "100.000" in html
    assert html.index("The Data") < html.index("Task + Model")
    assert "Model Architecture" in html
    assert "Validation Technique" in html
    assert "Accuracy" in html
    assert "artifacts/plots/accuracy.svg" not in html
    assert "Graph Interpretation" in html
    assert "sciquest-logo" in html
    assert "data:image/png;base64" in html
    assert "cycle-strip" in html
    assert "Experiment Verdict" in html
    assert "Delta vs previous" in html
    assert "Main improvement" in html
    assert "Main failure" in html
    assert "Scientific verdict" in html
    assert "experiment-timeline" in html
    assert "score-delta" in html
    assert "metric-scorecards" in html
    assert "semantic-state" in html
    assert "semantic-action" in html
    assert "semantic-pass" in html
    assert "semantic-metric" in html
    assert "semantic-risk" in html
    assert "semantic-best" in html
    assert "Recommended Next Experiment" in html
    assert "model-field-list" in html
    assert "font-family:JetBrains Mono" in html


def test_dashboard_uses_clickable_single_experiment_panes(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    exp2 = quest / "experiments" / "exp_002"
    exp1 = quest / "experiments" / "exp_001"
    import shutil
    shutil.copytree(exp1, exp2)
    (exp2 / "experiment.yaml").write_text((exp2 / "experiment.yaml").read_text().replace("exp_001", "exp_002"))
    out = build_dashboard(quest)
    html = out.read_text()
    assert "data-exp-target=\"exp_001\"" in html
    assert "data-exp-target=\"exp_002\"" in html
    assert "class=\"experiment active\" id=\"exp_002\"" in html
    assert "class=\"experiment\" id=\"exp_001\"" in html
    assert "function showExperiment" in html


def test_dashboard_ignores_pending_experiment_scores_when_finding_best(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    exp2 = quest / "experiments" / "exp_002"
    exp1 = quest / "experiments" / "exp_001"
    import shutil
    shutil.copytree(exp1, exp2)
    (exp2 / "experiment.yaml").write_text((exp2 / "experiment.yaml").read_text().replace("exp_001", "exp_002"))
    (exp2 / "validation_results.yaml").write_text("aggregate_score:\nmetrics: {}\nscores: {}\n")

    out = build_dashboard(quest)
    html = out.read_text()

    assert "exp_002" in html
    assert "Best score" in html
    assert "0.800000" in html


def test_dashboard_includes_metric_definitions_and_latex_equations(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    out = build_dashboard(quest)
    html = out.read_text()
    assert "Validation Metric Definitions" in html
    assert "Weighted aggregate score" in html
    assert "\\sum_i w_i" in html
    assert "MathJax" in html
    assert "assets/mathjax/tex-mml-chtml.js" in html
    assert "cdn.jsdelivr" not in html
    assert (out.parent / "assets" / "mathjax" / "tex-mml-chtml.js").exists()
    assert "\\(" in html and "\\)" in html
    assert "WAPE" in html
    assert "RMSE" in html


def test_dashboard_includes_research_model_abstraction_section(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    out = build_dashboard(quest)
    html = out.read_text()
    assert "Research Model Abstraction" in html
    assert "Latest model architecture" in html
    assert "Linear probe" in html
    assert "state/context" in html
    assert "\\hat{y}" in html
    assert "Validation score" in html
    assert "ResearchModelAbstraction" in html


def test_dashboard_includes_operator_explanations_for_technical_phrases(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    exp = quest / "experiments" / "exp_001"
    meta = yaml.safe_load((exp / "experiment.yaml").read_text())
    meta["model_architecture"] = "Inverse-propensity-weighted ridge regression in log-demand space"
    meta["validation_technique"] = "Rank-aware objective with law-of-demand pass rate"
    (exp / "experiment.yaml").write_text(yaml.safe_dump(meta))

    out = build_dashboard(quest)
    html = out.read_text()

    assert "Explain this like I’m the operator" in html
    assert "operator-explain" in html
    assert "Inverse-propensity-weighted ridge regression" in html
    assert "Some observations are more likely to appear under certain pricing policies" in html
    assert "Log-demand space" in html
    assert "The model predicts the logarithm of demand instead of raw demand" in html
    assert "Rank-aware objective" in html
    assert "not only judged by absolute error" in html
    assert "Law-of-demand pass rate" in html
    assert "predicted demand usually decreases when price increases" in html


def test_dashboard_cli_writes_index(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["dashboard", "--root", str(tmp_path), "--quest", "demo"])
    assert result.exit_code == 0, result.output
    assert (quest / "reports" / "dashboard" / "index.html").exists()



def test_logic_check_inspects_method_ledger_and_claim_labels(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    exp = quest / "experiments" / "exp_001"
    (quest / "method_stack.yaml").write_text("mode: simple\nprimary_method: popperian_falsificationist\nphases: {}\n")
    (exp / "method_ledger.yaml").write_text(yaml.safe_dump({
        "pre_registration": {"primary_method": "popperian_falsificationist", "predicted_observations": ["accuracy > 0.9"], "profile_versions": {"popperian_falsificationist": "1.0.0"}},
        "post_experiment_classification": {"confirmatory_result": [], "exploratory_observation": [], "post_hoc_reinterpretation": [], "method_violation": [], "unresolved_anomaly": []},
    }))
    (exp / "experiment_report.md").write_text("# Report\n\n## Evidence\nEvidence exists.\n\n## Speculation\nSpeculation exists.\n\n## Post Hoc Reinterpretation\n- labeled_post_hoc: false\n- claim: rescued the hypothesis after results\n")
    result = run_logic_check(quest, "exp_001")
    assert not result["passed"]
    assert any("method violation" in issue.lower() for issue in result["issues"])


def test_logic_check_does_not_fail_older_experiments_without_method_ledger(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    assert not (quest / "experiments" / "exp_001" / "method_ledger.yaml").exists()
    result = run_logic_check(quest, "exp_001")
    assert result["passed"]
    assert any("method_ledger" in warning.lower() for warning in result["warnings"])


def test_dashboard_shows_method_stack_and_ledger_status(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    exp = quest / "experiments" / "exp_001"
    (quest / "method_stack.yaml").write_text(yaml.safe_dump({
        "mode": "simple",
        "primary_method": "bayesian",
        "phases": {"hypothesis_generation": "bayesian", "stress_testing": "popperian_falsificationist"},
        "rationale": "Need uncertainty-aware stress testing.",
    }))
    (exp / "method_ledger.yaml").write_text(yaml.safe_dump({
        "pre_registration": {"primary_method": "bayesian", "confirmatory_status": "confirmatory", "profile_versions": {"bayesian": "1.0.0"}},
        "post_experiment_classification": {"confirmatory_result": ["held"], "method_violation": []},
    }))
    out = build_dashboard(quest)
    html = out.read_text()
    assert "Scientific Method Stack" in html
    assert "bayesian" in html
    assert "popperian_falsificationist" in html
    assert "Method Ledger Status" in html
    assert "confirmatory" in html
    assert "profile version" in html.lower()
