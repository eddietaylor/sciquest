from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import SCIQUEST_SPLASH
from .core import (
    continue_quest,
    create_quest,
    list_quests,
    run_next,
    update_state_with_validation,
)
from .dashboard import build_dashboard
from .io import append_text, quest_dir, read_yaml
from .logic import run_logic_check
from .loop import run_iteration_loop
from .methods.ledger import preregister_experiment
from .methods.registry import MethodRegistry
from .methods.resolver import auto_recommend_method_stack
from .state import acquire_lock, read_state, recover, release_lock
from .validation import validate_experiment
from .workflow import execute_registered, run_agent_iteration

app = typer.Typer(help="SciQuest autonomous research framework")
methods_app = typer.Typer(
    help="Inspect and recommend SciQuest scientific method profiles"
)
app.add_typer(methods_app, name="methods")
from .demand.cli import app as demand_app

app.add_typer(demand_app, name="demand")
console = Console()


def _qpath(root: Path, quest: str) -> Path:
    q = quest_dir(root, quest)
    if not q.is_dir() or not (q / "quest.yaml").is_file():
        raise typer.BadParameter(f"Quest not found: {quest}")
    return q


@app.command("new")
def new_quest(
    root: Path = typer.Option(Path("."), help="Workspace root containing quests/"),
    slug: Optional[str] = typer.Option(None, help="Quest slug"),
    start_agent: bool = typer.Option(
        False, help="After creating the quest, launch the configured external agent"
    ),
    agent_command: Optional[str] = typer.Option(
        None,
        help="External agent command. Also configurable via SCIQUEST_AGENT_COMMAND",
    ),
    splash: bool = typer.Option(
        True, "--splash/--no-splash", help="Show the SCI-QUEST startup banner"
    ),
    method: Optional[str] = typer.Option(
        None,
        "--method",
        help="Simple mode: choose one method lens, e.g. bayesian or popperian_falsificationist",
    ),
    method_stack_file: Optional[Path] = typer.Option(
        None,
        "--method-stack-file",
        help="Advanced mode: YAML file defining primary_method and phase methods",
    ),
    auto_method: bool = typer.Option(
        False, "--auto-method", help="Auto-recommend a method stack from quest metadata"
    ),
):
    """Create a new quest and initialize all artifacts."""
    if splash:
        console.print(SCIQUEST_SPLASH)
    answers = {
        "hero_statement": typer.prompt("Hero Statement"),
        "problem_statement": typer.prompt("Problem Statement"),
        "initial_hypothesis": typer.prompt(
            "Initial Hypothesis (conceptual + observational)"
        ),
        "subjective_priors": typer.prompt(
            "Subjective Priors (optional)", default="", show_default=False
        ),
        "core_data_description": typer.prompt(
            "Core Data Description (optional)", default="", show_default=False
        ),
        "validation_suite": typer.prompt(
            "Validation Suite (optional)", default="", show_default=False
        ),
        "validation_weighting_preferences": typer.prompt(
            "Validation weighting preferences (optional, natural language ok)",
            default="",
            show_default=False,
        ),
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
        exp_id = run_agent_iteration(qpath, agent_command)
        console.print(
            f"Supervisor evaluated {exp_id}."
            if exp_id
            else "Research decision recorded; no evaluated experiment is claimed."
        )


@app.command("continue")
def continue_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    root: Path = typer.Option(Path(".")),
    idea: Optional[str] = typer.Option(None, help="New idea to inject"),
    data: Optional[str] = typer.Option(None, help="New data note/path to inject"),
    splash: bool = typer.Option(
        True, "--splash/--no-splash", help="Show the SCI-QUEST banner"
    ),
):
    """Resume an existing quest, optionally inject new ideas/data."""
    if splash:
        console.print(SCIQUEST_SPLASH)
    continue_quest(_qpath(root, quest), idea, data)
    console.print("Quest updated with injection.")


@app.command("list")
def list_cmd(root: Path = typer.Option(Path("."))):
    """List all quests."""
    for q in list_quests(root):
        console.print(f"{q['slug']}\t{q['status']}\tbest={q['best_score']}")


@app.command("status")
def status_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    root: Path = typer.Option(Path(".")),
):
    """Show quest state: running/idle, last experiment, best score, failures."""
    state = read_state(_qpath(root, quest))
    console.print(state)


@app.command("run-next")
def run_next_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    root: Path = typer.Option(Path(".")),
    agent_stub: bool = typer.Option(
        False, help="Create deterministic scaffold without agent reasoning"
    ),
    execute: bool = typer.Option(False, help="Execute generated notebook"),
    start_agent: bool = typer.Option(
        False,
        help="Launch the configured external agent instead of only creating a deterministic scaffold",
    ),
    agent_command: Optional[str] = typer.Option(
        None,
        help="External agent command. Also configurable via SCIQUEST_AGENT_COMMAND",
    ),
    trusted_local: bool = typer.Option(
        False,
        help="Explicitly allow unsandboxed exploration; default workers are isolated",
    ),
    timeout_seconds: float = typer.Option(
        600, help="Wall-time budget for worker preparation/execution"
    ),
):
    """Execute the next iteration; primarily for agents/schedulers."""
    qpath = _qpath(root, quest)
    if start_agent:
        console.print("Launching configured SciQuest agent...")
        exp_id = run_agent_iteration(
            qpath, agent_command, isolated=not trusted_local, timeout=timeout_seconds
        )
        console.print(
            f"Supervisor evaluated {exp_id}."
            if exp_id
            else "Research decision recorded; no evaluated experiment is claimed."
        )
        return
    exp_id = run_next(qpath, agent_stub=agent_stub, execute=execute)
    console.print(f"Created {exp_id}")


@app.command("run-loop")
def run_loop_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    root: Path = typer.Option(Path(".")),
    max_iterations: int = typer.Option(
        3, "--max-iterations", "-n", help="Maximum iterations to run; default is 3"
    ),
    agent_stub: bool = typer.Option(
        False, help="Use deterministic scaffold iterations instead of an external agent"
    ),
    execute: bool = typer.Option(False, help="Execute generated stub notebooks"),
    start_agent: bool = typer.Option(
        False, help="Launch the configured external agent for each iteration"
    ),
    agent_command: Optional[str] = typer.Option(
        None,
        help="External agent command. Also configurable via SCIQUEST_AGENT_COMMAND",
    ),
    wait_timeout_seconds: float = typer.Option(
        0.0,
        help="Seconds to wait for an in-progress previous iteration; 0 means fail immediately",
    ),
    trusted_local: bool = typer.Option(
        False, help="Explicitly allow unsandboxed exploration"
    ),
    timeout_seconds: float = typer.Option(600, help="Per-iteration wall-time budget"),
    max_repairs: int = typer.Option(
        0,
        help="Bounded repair attempts with new experiment IDs, sharing the iteration budget",
    ),
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
        isolated=not trusted_local,
        timeout_seconds=timeout_seconds,
        max_repairs=max_repairs,
    )
    console.print(
        f"Completed {result['completed_iterations']}/{result['requested_iterations']} iterations"
    )
    if result.get("prepared_iterations"):
        console.print(
            f"Prepared {result['prepared_iterations']} drafts; these are not completed experiments"
        )
    if result["experiments"]:
        console.print(f"Experiments: {', '.join(result['experiments'])}")
    if result.get("failed"):
        console.print(f"Loop stopped: {result.get('error')}")
        raise typer.Exit(1)


@app.command("validate")
def validate_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    experiment: str = typer.Option(..., "--experiment", "-e"),
    root: Path = typer.Option(Path(".")),
):
    """Compute validation scores for an experiment."""
    q = _qpath(root, quest)
    result = validate_experiment(q, experiment)
    update_state_with_validation(q, experiment, result)
    console.print(result)


@app.command("register")
def register_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    experiment: str = typer.Option(..., "--experiment", "-e"),
    root: Path = typer.Option(Path(".")),
):
    """Freeze a concrete experiment plan, data, code, methods and metric suite."""
    qpath = _qpath(root, quest)
    owner = acquire_lock(qpath)
    try:
        result = preregister_experiment(qpath, experiment, owner=owner)
    finally:
        release_lock(qpath, owner)
    console.print(
        {"experiment": experiment, "registration_hash": result["registration_hash"]}
    )


@app.command("execute")
def execute_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    experiment: str = typer.Option(..., "--experiment", "-e"),
    root: Path = typer.Option(Path(".")),
    trusted_local: bool = typer.Option(False),
    timeout_seconds: float = typer.Option(600),
):
    """Execute registered code and compute metrics against registered evaluation truth."""
    console.print(
        execute_registered(
            _qpath(root, quest),
            experiment,
            isolated=not trusted_local,
            timeout=timeout_seconds,
        )
    )


@app.command("recover")
def recover_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    root: Path = typer.Option(Path(".")),
    lock_id: Optional[str] = typer.Option(
        None, help="Expected abandoned lock ID; live owners are never cleared"
    ),
):
    """Reset a failed quest or a verified dead local owner, preserving its evidence."""
    console.print(recover(_qpath(root, quest), lock_id))


@app.command("lab")
def lab_cmd(
    scenario: str = typer.Option("elastic"),
    policy: str = typer.Option("strong_inference"),
    seed: int = typer.Option(0),
    budget: int = typer.Option(16),
    output: Optional[Path] = typer.Option(None),
):
    """Exercise an inquiry policy in a small controlled laboratory fixture."""
    from .io import write_yaml
    from .laboratory import LabConfig, reference_campaign

    result = reference_campaign(LabConfig(scenario, seed, budget), policy)
    if output:
        write_yaml(output, result, exclusive=True)
    console.print(result)


@app.command("backup")
def backup_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    output: Path = typer.Option(...),
    root: Path = typer.Option(Path(".")),
):
    """Archive idle research history, registered inputs, records, and data with hashes."""
    from .backup import backup_quest

    console.print(backup_quest(_qpath(root, quest), output))


@app.command("logic-check")
def logic_check_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    experiment: Optional[str] = typer.Option(None, "--experiment", "-e"),
    root: Path = typer.Option(Path(".")),
):
    """Verify scientific and logical consistency across the pipeline."""
    result = run_logic_check(_qpath(root, quest), experiment)
    console.print(result)
    if not result["passed"]:
        raise typer.Exit(1)


@app.command("dashboard")
def dashboard_cmd(
    quest: str = typer.Option(..., "--quest", "-q"),
    root: Path = typer.Option(Path(".")),
    output_dir: Optional[Path] = typer.Option(
        None, help="Output directory for static dashboard"
    ),
    export_html: Optional[Path] = typer.Option(
        None,
        "--export-html",
        help="Also copy the dashboard HTML to this file for sharing/archival export",
    ),
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
    root: Path = typer.Option(Path(".")),
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
    title = Text("SciQuest Method Stack Library", style="bold cyan")
    subtitle = Text(
        "Choose the philosophy-of-science operating mode for a quest or phase.",
        style="dim",
    )
    console.print(Panel.fit(subtitle, title=title, border_style="cyan", box=box.DOUBLE))
    console.print(
        "[bold bright_black]Columns:[/bold bright_black] Method · Scientific job · Best used when · Phase affinity · Version"
    )

    table = Table(
        show_header=True,
        header_style="bold bright_white",
        box=box.ROUNDED,
        border_style="bright_black",
        row_styles=["", "dim"],
        pad_edge=True,
    )
    table.add_column("Method", style="bold cyan", no_wrap=True, ratio=2)
    table.add_column("Scientific job", style="bright_white", ratio=2)
    table.add_column("Best used when / Phase affinity", style="green", ratio=4)

    for profile in registry.list_profiles():
        best_for = (
            "; ".join(profile.best_for[:2])
            if profile.best_for
            else "General empirical work"
        )
        affinities = (
            ", ".join(profile.phase_affinities[:2])
            if profile.phase_affinities
            else "all phases"
        )
        method_cell = f"[bold cyan]{profile.id}[/bold cyan]\n[bright_blue]v{profile.profile_version}[/bright_blue]"
        from .methods.policies import method_capability

        capability = method_capability(profile.id)["implementation"]
        use_cell = f"{best_for}\n[magenta]Phases:[/magenta] {affinities}\n[dim]{capability}[/dim]"
        table.add_row(
            method_cell,
            profile.plain_language_label,
            use_cell,
        )

    console.print(table)
    console.print(
        "[dim]Tip:[/dim] [cyan]sciquest methods show bayesian[/cyan] or [cyan]sciquest new --method strong_inference[/cyan]"
    )


@methods_app.command("show")
def methods_show(method_id: str = typer.Argument(..., help="Method profile id")):
    """Show a scientific method profile."""
    registry = MethodRegistry.default()
    from .methods.policies import method_capability

    try:
        profile = registry.get(method_id)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(
        {
            "id": profile.id,
            "display_name": profile.display_name,
            "plain_language_label": profile.plain_language_label,
            "profile_version": profile.profile_version,
            "description": profile.description,
            "best_for": profile.best_for,
            "bad_for": profile.bad_for,
            "phase_affinities": profile.phase_affinities,
            "capability": method_capability(profile.id),
            "hypothesis_generation_rules": profile.hypothesis_generation_rules,
            "experiment_design_rules": profile.experiment_design_rules,
            "evidence_evaluation_rules": profile.evidence_evaluation_rules,
            "validation_rules": profile.validation_rules,
            "next_step_policy": profile.next_step_policy,
        }
    )


@methods_app.command("recommend")
def methods_recommend(
    quest: Optional[str] = typer.Option(
        None, "--quest", "-q", help="Existing quest slug to inspect"
    ),
    root: Path = typer.Option(Path(".")),
    hero: str = typer.Option("", help="Hero statement if no quest is provided"),
    problem: str = typer.Option("", help="Problem statement if no quest is provided"),
):
    """Recommend a method stack from quest text and accumulated quest history."""
    metadata = {"hero_statement": hero, "problem_statement": problem}
    if quest:
        metadata["quest_path"] = _qpath(root, quest)
    stack = auto_recommend_method_stack(metadata)
    console.print(
        {
            "primary_method": stack.primary_method,
            "mode": stack.mode,
            "rationale": stack.rationale,
            "phases": stack.phases,
        }
    )


if __name__ == "__main__":
    app()
