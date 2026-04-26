from typer.testing import CliRunner

from sciquest.agent import SCIQUEST_SPLASH, build_agent_prompt, build_agent_argv, parse_agent_command
from sciquest.cli import app


def test_parse_agent_command_respects_shell_quoting():
    assert parse_agent_command('python -c "print(123)"') == ["python", "-c", "print(123)"]


def test_build_agent_prompt_points_agent_at_quest_and_protocol(tmp_path):
    quest = tmp_path / "quests" / "demo"
    quest.mkdir(parents=True)
    prompt = build_agent_prompt(quest)
    assert str(quest) in prompt
    assert "AGENTS.md" in prompt
    assert "one SciQuest iteration" in prompt


def test_build_agent_argv_appends_prompt_for_hermes_query_style_command(tmp_path):
    quest = tmp_path / "quests" / "demo"
    quest.mkdir(parents=True)
    argv = build_agent_argv(quest, "hermes chat -q")
    assert argv[:3] == ["hermes", "chat", "-q"]
    assert str(quest) in argv[-1]


def test_build_agent_argv_supports_prompt_and_quest_placeholders(tmp_path):
    quest = tmp_path / "quests" / "demo"
    quest.mkdir(parents=True)
    argv = build_agent_argv(quest, "agent --cwd {quest_path} --message {prompt}")
    assert argv[:3] == ["agent", "--cwd", str(quest)]
    assert str(quest) in argv[-1]


def test_new_with_start_agent_runs_configured_agent_command(tmp_path):
    runner = CliRunner()
    marker = tmp_path / "agent-ran.txt"
    inputs = "Hero\nProblem\nHypothesis\n\n\n\nweights\ny\n"
    command = f"python -c \"from pathlib import Path; Path(r'{marker}').write_text('ran')\""
    result = runner.invoke(
        app,
        ["new", "--root", str(tmp_path), "--slug", "demo", "--start-agent", "--agent-command", command, "--no-splash"],
        input=inputs,
    )
    assert result.exit_code == 0, result.output
    assert marker.read_text() == "ran"
    assert "Agent completed" in result.output


def test_new_prints_sciquest_banner_by_default(tmp_path):
    runner = CliRunner()
    inputs = "Hero\nProblem\nHypothesis\n\n\n\n\ny\n"
    result = runner.invoke(app, ["new", "--root", str(tmp_path), "--slug", "splash-demo"], input=inputs)
    assert result.exit_code == 0, result.output
    assert "SCI-QUEST" in result.output
    assert "autonomous research framework" in result.output


def test_continue_prints_sciquest_banner_by_default(tmp_path):
    runner = CliRunner()
    inputs = "Hero\nProblem\nHypothesis\n\n\n\n\ny\n"
    assert runner.invoke(app, ["new", "--root", str(tmp_path), "--slug", "demo", "--no-splash"], input=inputs).exit_code == 0
    result = runner.invoke(app, ["continue", "--root", str(tmp_path), "--quest", "demo", "--idea", "next idea"])
    assert result.exit_code == 0, result.output
    assert "SCI-QUEST" in result.output
    assert "Quest updated with injection" in result.output


def test_sciquest_splash_constant_has_banner_text_without_literal_escape_codes():
    assert "SCI-QUEST" in SCIQUEST_SPLASH
    assert "\\033" not in SCIQUEST_SPLASH
    assert "[38;5;" not in SCIQUEST_SPLASH
