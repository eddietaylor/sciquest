from __future__ import annotations

from pathlib import Path
import shutil
from typing import Optional

import typer
from rich.console import Console

from .core import create_quest, list_quests, run_next, continue_quest, update_state_with_validation
from .io import quest_dir, read_yaml, append_text
from .validation import validate_experiment
from .logic import run_logic_check
from .agent import SCIQUEST_SPLASH, launch_agent
from .dashboard import build_dashboard
from .loop import run_iteration_loop
from .methods.registry import MethodRegistry
from .methods.resolver import auto_recommend_method_stack

app = typer.Typer(help="SciQuest autonomous research framework")
methods_app = typer.Typer(help="Inspect and recommend SciQuest scientific method profiles")
app.add_typer(methods_app, name="methods")
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
    splash: bool = typer.Option(True, "--splash/--no-splash", help="Show the SCI-QUEST startup banner"),
    method: Optional[str] = typer.Option(None, "--method", help="Simple mode: choose one method lens, e.g. bayesian or popperian_falsificationist"),
    method_stack_file: Optional[Path] = typer.Option(None, "--method-stack-file", help="Advanced mode: YAML file defining primary_method and phase methods"),
    auto_method: bool = typer.Option(False, "--auto-method", help="Auto-recommend a method stack from quest metadata"),
):
    """Create a new quest and initialize all artifacts."""
    if splash:
        console.print(SCIQUEST_SPLASH)
    answers = {
        "hero_statement": typer.prompt("Hero Statement"),
        "problem_statement": typer.prompt("Problem Statement"),
        "initial_hypothesis": typer.prompt("Initial Hypothesis (conceptual + observational)"),
        "subjective_priors": typer.prompt("Subjective Priors (optional)", default="", show_default=False),
        "core_data_description": typer.prompt("Core Data Description (optional)", default="", show_default=False),
        "validation_suite": typer.prompt("Validation Suite (optional)", default="", show_default=False),
        "validation_weighting_preferences": typer.prompt("Validation weighting preferences (optional, natural language ok)", default="", show_default=False),
    }
    if method_stack_file:
        answers["method_stack"] = read_yaml(method_stack_file, {})
    elif method:
        answers["method"] = method
    if auto_method:
        answers["auto_method"] = True
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
    splash: bool = typer.Option(True, "--splash/--no-splash", help="Show the SCI-QUEST banner"),
):
    """Resume an existing quest, optionally inject new ideas/data."""
    if splash:
        console.print(SCIQUEST_SPLASH)
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


@app.command("run-loop")
def run_loop_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    root: Path = typer.Option(Path.cwd()),
    max_iterations: int = typer.Option(3, "--max-iterations", "-n", help="Maximum iterations to run; default is 3"),
    agent_stub: bool = typer.Option(False, help="Use deterministic scaffold iterations instead of an external agent"),
    execute: bool = typer.Option(False, help="Execute generated stub notebooks"),
    start_agent: bool = typer.Option(False, help="Launch the configured external agent for each iteration"),
    agent_command: Optional[str] = typer.Option(None, help="External agent command. Also configurable via SCIQUEST_AGENT_COMMAND"),
    wait_timeout_seconds: float = typer.Option(0.0, help="Seconds to wait for an in-progress previous iteration; 0 means fail immediately"),
):
    """Run multiple iterations sequentially, starting the next only after the previous one completes."""
    result = run_iteration_loop(
        _qpath(root, quest),
        max_iterations=max_iterations,
        agent_stub=agent_stub,
        execute=execute,
        start_agent=start_agent,
        agent_command=agent_command,
        wait_timeout_seconds=wait_timeout_seconds,
    )
    console.print(f"Completed {result['completed_iterations']}/{result['requested_iterations']} iterations")
    if result["experiments"]:
        console.print(f"Experiments: {', '.join(result['experiments'])}")
    if result.get("failed"):
        console.print(f"Loop stopped: {result.get('error')}")
        raise typer.Exit(1)


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
    export_html: Optional[Path] = typer.Option(None, "--export-html", help="Also copy the dashboard HTML to this file for sharing/archival export"),
):
    """Build a static dashboard for visualizing experiment iterations."""
    out = build_dashboard(_qpath(root, quest), output_dir)
    console.print(f"Dashboard written: {out}")
    if export_html is not None:
        export_html.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, export_html)
        console.print(f"Dashboard HTML exported: {export_html}")


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


@methods_app.command("list")
def methods_list():
    """List available scientific method profiles."""
    registry = MethodRegistry.default()
    for profile in registry.list_profiles():
        console.print(f"{profile.id}	{profile.plain_language_label}	v{profile.profile_version}")


@methods_app.command("show")
def methods_show(method_id: str = typer.Argument(..., help="Method profile id")):
    """Show a scientific method profile."""
    registry = MethodRegistry.default()
    try:
        profile = registry.get(method_id)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print({
        "id": profile.id,
        "display_name": profile.display_name,
        "plain_language_label": profile.plain_language_label,
        "profile_version": profile.profile_version,
        "description": profile.description,
        "best_for": profile.best_for,
        "bad_for": profile.bad_for,
        "phase_affinities": profile.phase_affinities,
    })


@methods_app.command("recommend")
def methods_recommend(
    quest: Optional[str] = typer.Option(None, "--quest", "-q", help="Existing quest slug to inspect"),
    root: Path = typer.Option(Path.cwd()),
    hero: str = typer.Option("", help="Hero statement if no quest is provided"),
    problem: str = typer.Option("", help="Problem statement if no quest is provided"),
):
    """Recommend a method stack from quest text and accumulated quest history."""
    metadata = {"hero_statement": hero, "problem_statement": problem}
    if quest:
        metadata["quest_path"] = _qpath(root, quest)
    stack = auto_recommend_method_stack(metadata)
    console.print({
        "primary_method": stack.primary_method,
        "mode": stack.mode,
        "rationale": stack.rationale,
        "phases": stack.phases,
    })


if __name__ == "__main__":
    app()
