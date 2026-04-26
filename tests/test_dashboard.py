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
    (quest / "artifacts").mkdir()
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
    assert "Model Architecture" in html
    assert "Validation Technique" in html
    assert "Accuracy" in html
    assert "<img " not in html
    assert "Graph Interpretation" in html


def test_dashboard_cli_writes_index(tmp_path):
    quest = make_minimal_valid_experiment(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["dashboard", "--root", str(tmp_path), "--quest", "demo"])
    assert result.exit_code == 0, result.output
    assert (quest / "reports" / "dashboard" / "index.html").exists()
