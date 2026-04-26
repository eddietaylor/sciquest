from pathlib import Path
import yaml
from typer.testing import CliRunner

from sciquest.cli import app
from sciquest.loop import run_iteration_loop


def make_loop_quest(tmp_path: Path) -> Path:
    quest = tmp_path / "quests" / "loop-demo"
    (quest / "experiments").mkdir(parents=True)
    (quest / "reports").mkdir()
    (quest / "artifacts").mkdir()
    (quest / "logs").mkdir()
    (quest / "quest.yaml").write_text("slug: loop-demo\nhero_statement: Hero\nproblem_statement: Problem\n")
    (quest / "state.yaml").write_text("quest_status: idle\ncurrent_experiment: null\nlast_experiment: null\nlast_score: null\nbest_score: null\nfailures: []\nlock_id: null\n")
    (quest / "hypotheses.yaml").write_text("hypotheses:\n- id: h001\n  claim: initial\n")
    (quest / "validation.yaml").write_text("status: ready\nmetrics: []\n")
    (quest / "data_manifest.yaml").write_text("status: ready\ndatasets: []\n")
    (quest / "journal.md").write_text("# Journal\n")
    (quest / "AGENTS.md").write_text("protocol\n")
    return quest


def test_run_iteration_loop_defaults_to_three_completed_stub_iterations(tmp_path):
    quest = make_loop_quest(tmp_path)
    result = run_iteration_loop(quest, agent_stub=True)
    assert result["requested_iterations"] == 3
    assert result["completed_iterations"] == 3
    assert result["experiments"] == ["exp_001", "exp_002", "exp_003"]
    state = yaml.safe_load((quest / "state.yaml").read_text())
    assert state["quest_status"] == "idle"
    assert state["lock_id"] is None
    assert state["last_experiment"] == "exp_003"


def test_run_iteration_loop_stops_when_previous_iteration_fails(tmp_path, monkeypatch):
    quest = make_loop_quest(tmp_path)
    calls = []

    def fake_runner(qpath, **kwargs):
        calls.append(qpath)
        if len(calls) == 1:
            return "exp_001"
        raise RuntimeError("boom")

    result = run_iteration_loop(quest, max_iterations=3, iteration_runner=fake_runner)
    assert result["completed_iterations"] == 1
    assert result["failed"] is True
    assert "boom" in result["error"]
    assert len(calls) == 2


def test_run_loop_cli_defaults_to_three_iterations(tmp_path):
    quest = make_loop_quest(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["run-loop", "--root", str(tmp_path), "--quest", "loop-demo", "--agent-stub"])
    assert result.exit_code == 0, result.output
    assert "Completed 3/3 iterations" in result.output
    assert (quest / "experiments" / "exp_003").exists()


def test_run_loop_cli_accepts_max_iterations(tmp_path):
    quest = make_loop_quest(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["run-loop", "--root", str(tmp_path), "--quest", "loop-demo", "--max-iterations", "2", "--agent-stub"])
    assert result.exit_code == 0, result.output
    assert "Completed 2/2 iterations" in result.output
    assert (quest / "experiments" / "exp_002").exists()
    assert not (quest / "experiments" / "exp_003").exists()
