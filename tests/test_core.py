from pathlib import Path
import yaml
from typer.testing import CliRunner

from sciquest.cli import app
from sciquest.validation import compute_aggregate_score
from sciquest.logic import run_logic_check


def test_new_quest_initializes_required_artifacts(tmp_path):
    runner = CliRunner()
    inputs = "".join([
        "Make science reproducible\n",
        "Research programs need transparent iteration\n",
        "If experiments are journaled, reproducibility improves\n",
        "Priors are weak\n",
        "\n",
        "\n",
        "Prefer evidence quality over speed\n",
        "y\n",
    ])
    result = runner.invoke(app, ["new", "--root", str(tmp_path), "--slug", "repro-quest"], input=inputs)
    assert result.exit_code == 0, result.output
    quest = tmp_path / "quests" / "repro-quest"
    for name in ["quest.yaml", "state.yaml", "hypotheses.yaml", "validation.yaml", "data_manifest.yaml", "journal.md", "AGENTS.md"]:
        assert (quest / name).exists(), name
    assert (quest / "experiments").is_dir()
    manifest = yaml.safe_load((quest / "data_manifest.yaml").read_text())
    assert manifest["status"] == "missing_user_data"
    validation = yaml.safe_load((quest / "validation.yaml").read_text())
    assert validation["status"] == "agent_required"
    assert validation["weighting_preferences_raw"] == "Prefer evidence quality over speed"


def test_validation_weighted_sum_with_directions_and_normalization():
    suite = {
        "metrics": [
            {"name": "accuracy", "direction": "maximize", "weight": 2, "normalization": {"type": "minmax", "min": 0, "max": 1}},
            {"name": "error", "direction": "minimize", "weight": 1, "normalization": {"type": "minmax", "min": 0, "max": 10}},
            {"name": "pH", "direction": "target", "weight": 1, "normalization": {"type": "target", "target": 7, "tolerance": 2}},
        ]
    }
    results = {"metrics": {"accuracy": 0.8, "error": 2, "pH": 8}}
    scored = compute_aggregate_score(suite, results)
    assert round(scored["aggregate_score"], 3) == 0.725
    assert set(scored["scores"]) == {"accuracy", "error", "pH"}


def test_run_next_creates_experiment_stub_and_updates_state(tmp_path):
    runner = CliRunner()
    inputs = "Hero\nProblem\nHypothesis\n\nData exists elsewhere\nmetric: outcome\nweights\ny\n"
    assert runner.invoke(app, ["new", "--root", str(tmp_path), "--slug", "demo"], input=inputs).exit_code == 0
    result = runner.invoke(app, ["run-next", "--root", str(tmp_path), "--quest", "demo", "--agent-stub"])
    assert result.exit_code == 0, result.output
    exp = tmp_path / "quests" / "demo" / "experiments" / "exp_001"
    assert (exp / "notebook.ipynb").exists()
    assert (exp / "experiment.yaml").exists()
    state = yaml.safe_load((tmp_path / "quests" / "demo" / "state.yaml").read_text())
    assert state["last_experiment"] == "exp_001"
    assert state["quest_status"] == "idle"


def test_logic_check_reports_missing_alignment(tmp_path):
    quest = tmp_path / "quest"
    quest.mkdir()
    (quest / "hypotheses.yaml").write_text("hypotheses:\n- id: h001\n  claim: Plants grow faster with blue light\n")
    (quest / "validation.yaml").write_text("status: ready\nmetrics: []\n")
    (quest / "data_manifest.yaml").write_text("status: ready\ndatasets: []\n")
    issues = run_logic_check(quest)
    assert not issues["passed"]
    assert any("validation" in issue.lower() for issue in issues["issues"])
