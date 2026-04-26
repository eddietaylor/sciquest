from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .core import create_quest, list_quests, run_next, continue_quest, update_state_with_validation
from .io import quest_dir, read_yaml, append_text
from .validation import validate_experiment
from .logic import run_logic_check
from .agent import NEWTON_SPLASH, launch_agent
from .dashboard import build_dashboard

app = typer.Typer(help="SciQuest autonomous research framework")
console = Console()


def _qpath(root: Path, quest: str) -> Path:
    q = quest_dir(root, quest)
    if not q.exists():
        raise typer.BadParameter(f"Quest not found: {quest}")
    return q


@app.command("new")
def new_quest(
    root: Path = typer.Option(Path.cwd(), help="Workspace root containing quests/"),
    slug: Optional[str] = typer.Option(None, help="Quest slug"),
    start_agent: bool = typer.Option(False, help="After creating the quest, launch the configured external agent"),
    agent_command: Optional[str] = typer.Option(None, help="External agent command. Also configurable via SCIQUEST_AGENT_COMMAND"),
    splash: bool = typer.Option(True, "--splash/--no-splash", help="Show the Newton/SciQuest startup splash"),
):
    """Create a new quest and initialize all artifacts."""
    if splash:
        console.print(NEWTON_SPLASH)
    answers = {
        "hero_statement": typer.prompt("Hero Statement"),
        "problem_statement": typer.prompt("Problem Statement"),
        "initial_hypothesis": typer.prompt("Initial Hypothesis (conceptual + observational)"),
        "subjective_priors": typer.prompt("Subjective Priors (optional)", default="", show_default=False),
        "core_data_description": typer.prompt("Core Data Description (optional)", default="", show_default=False),
        "validation_suite": typer.prompt("Validation Suite (optional)", default="", show_default=False),
        "validation_weighting_preferences": typer.prompt("Validation weighting preferences (optional, natural language ok)", default="", show_default=False),
    }
    if not typer.confirm("Start quest?", default=True):
        raise typer.Abort()
    qpath = create_quest(root, answers, slug)
    console.print(f"Created quest: {qpath}")
    if start_agent:
        console.print("Launching configured SciQuest agent...")
        result = launch_agent(qpath, agent_command)
        if result.returncode != 0:
            console.print(f"Agent exited with code {result.returncode}")
            raise typer.Exit(result.returncode)
        console.print("Agent completed.")


@app.command("continue")
def continue_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    root: Path = typer.Option(Path.cwd()),
    idea: Optional[str] = typer.Option(None, help="New idea to inject"),
    data: Optional[str] = typer.Option(None, help="New data note/path to inject"),
):
    """Resume an existing quest, optionally inject new ideas/data."""
    continue_quest(_qpath(root, quest), idea, data)
    console.print("Quest updated with injection.")


@app.command("list")
def list_cmd(root: Path = typer.Option(Path.cwd())):
    """List all quests."""
    for q in list_quests(root):
        console.print(f"{q['slug']}\t{q['status']}\tbest={q['best_score']}")


@app.command("status")
def status_cmd(quest: str = typer.Option(..., "--quest", "-q"), root: Path = typer.Option(Path.cwd())):
    """Show quest state: running/idle, last experiment, best score, failures."""
    state = read_yaml(_qpath(root, quest) / "state.yaml", {})
    console.print(state)


@app.command("run-next")
def run_next_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    root: Path = typer.Option(Path.cwd()),
    agent_stub: bool = typer.Option(False, help="Create deterministic scaffold without agent reasoning"),
    execute: bool = typer.Option(False, help="Execute generated notebook"),
    start_agent: bool = typer.Option(False, help="Launch the configured external agent instead of only creating a deterministic scaffold"),
    agent_command: Optional[str] = typer.Option(None, help="External agent command. Also configurable via SCIQUEST_AGENT_COMMAND"),
):
    """Execute the next iteration; primarily for agents/schedulers."""
    qpath = _qpath(root, quest)
    if start_agent:
        console.print("Launching configured SciQuest agent...")
        result = launch_agent(qpath, agent_command)
        if result.returncode != 0:
            console.print(f"Agent exited with code {result.returncode}")
            raise typer.Exit(result.returncode)
        console.print("Agent completed.")
        return
    exp_id = run_next(qpath, agent_stub=agent_stub, execute=execute)
    console.print(f"Created {exp_id}")


@app.command("validate")
def validate_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    experiment: str = typer.Option(..., "--experiment", "-e"),
    root: Path = typer.Option(Path.cwd()),
):
    """Compute validation scores for an experiment."""
    q = _qpath(root, quest)
    result = validate_experiment(q, experiment)
    update_state_with_validation(q, experiment, result)
    console.print(result)


@app.command("logic-check")
def logic_check_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    experiment: Optional[str] = typer.Option(None, "--experiment", "-e"),
    root: Path = typer.Option(Path.cwd()),
):
    """Verify scientific and logical consistency across the pipeline."""
    result = run_logic_check(_qpath(root, quest), experiment)
    console.print(result)
    if not result["passed"]:
        raise typer.Exit(1)


@app.command("dashboard")
def dashboard_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    root: Path = typer.Option(Path.cwd()),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory for static dashboard"),
):
    """Build a static dashboard for visualizing experiment iterations."""
    out = build_dashboard(_qpath(root, quest), output_dir)
    console.print(f"Dashboard written: {out}")


@app.command("journal")
def journal_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    root: Path = typer.Option(Path.cwd()),
    append: Optional[str] = typer.Option(None, help="Append a journal note"),
):
    """View or append journal entries."""
    path = _qpath(root, quest) / "journal.md"
    if append is not None:
        append_text(path, f"\n\n## Manual Note\n\n{append}\n")
    console.print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
