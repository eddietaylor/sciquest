from __future__ import annotations

import os
import shlex
import json
from pathlib import Path

from .io import read_yaml
from .methods.prompt_builder import build_method_prompt_addendum
from .methods.resolver import resolve_method_stack

SCIQUEST_SPLASH = """
███████╗ ██████╗██╗       ██████╗ ██╗   ██╗███████╗███████╗████████╗
██╔════╝██╔════╝██║      ██╔═══██╗██║   ██║██╔════╝██╔════╝╚══██╔══╝
███████╗██║     ██║█████╗██║   ██║██║   ██║█████╗  ███████╗   ██║
╚════██║██║     ██║╚════╝██║▄▄ ██║██║   ██║██╔══╝  ╚════██║   ██║
███████║╚██████╗██║      ╚██████╔╝╚██████╔╝███████╗███████║   ██║
╚══════╝ ╚═════╝╚═╝       ╚══▀▀═╝  ╚═════╝ ╚══════╝╚══════╝   ╚═╝

        SCI-QUEST  ·  autonomous research framework  ·  hypothesis → experiment → evidence
"""

NEWTON_SPLASH = SCIQUEST_SPLASH


def parse_agent_command(command: str) -> list[str]:
    """Parse an agent command using shell-like quoting without invoking a shell."""
    return shlex.split(command)


def build_agent_prompt(quest_path: Path, display_path: Path | None = None) -> str:
    """Create the handoff prompt passed to an external agent process."""
    stack = resolve_method_stack(
        read_yaml(
            quest_path / "method_stack.yaml", {"primary_method": "standard_empirical"}
        )
    )
    method_addendum = build_method_prompt_addendum(stack)
    shown = display_path or quest_path
    return f"""You are preparing one SciQuest experiment for supervisor execution.

Quest path: {shown}
Agent protocol: {shown / "AGENTS.md"}
Method stack: {shown / "method_stack.yaml"}

Read the supplied research context, then prepare exactly one SciQuest iteration.
Read research_state.yaml and relevant history notebooks. They contain previous
registered hypotheses and outcomes. Reusing evaluation feedback is exploratory;
confirmation requires fresh evaluation evidence. Do not restart from the initial
hypothesis while ignoring previous negative or inconclusive results.
Read policy_contract.yaml for the structured plan extensions required by the
selected inquiry policies. Replace all examples with justified scientific content.
Read handoff.yaml first. If mode is propose_setup, write setup_proposal.yaml with
action: propose_setup, a concrete reason, proposed evidence contract, proposed
data sources and metric definitions. Do not manufacture a confirmatory result.
If available evidence cannot support a useful experiment, you may instead write
decision.yaml with action: needs_data or stop, and a concrete reason.
Write experiment_plan.yaml and notebook.ipynb in the working directory.
The plan needs question, hypothesis, scope, assumptions, data_access, stopping_rule,
action (hypothesis_test, rival_test, replication, measurement, or data_acquisition),
confirmatory_status (exploratory or confirmatory), and predictions (id, metric,
operator, threshold). Use the existing metric names and definitions.
Training data is staged under data/INDEX/FILENAME. Evaluation features are under
data/INDEX/features.csv; hidden target labels are unavailable to you.
The notebook must emit prediction CSVs with the complete evaluation IDs and the
prediction column specified by validation.yaml. Do not write claimed scores.
The supervisor registers the plan, runs the notebook, computes metrics, and
records the outcome. You cannot declare completion or scientific confirmation.
Do not execute the notebook during preparation or alter authoritative history.
Stop after writing the plan and notebook. A new hypothesis is not required when
a replication, measurement, or rival test is scientifically more useful.

{method_addendum}"""


def resolve_agent_command(agent_command: str | None = None) -> list[str] | None:
    """Resolve command from explicit option or SCIQUEST_AGENT_COMMAND."""
    command = agent_command or os.environ.get("SCIQUEST_AGENT_COMMAND")
    if not command:
        return None
    return parse_agent_command(command)


def build_agent_argv(
    quest_path: Path,
    agent_command: str | None = None,
    *,
    display_path: Path | None = None,
) -> list[str] | None:
    """Build argv for an external agent, adding the SciQuest prompt as input.

    If the configured command contains `{prompt}` or `{quest_path}`, those
    placeholders are expanded before shell-style parsing. Otherwise the prompt is
    appended as the final argument, which works with commands like
    `hermes chat -q` and `codex exec`.
    """
    command = agent_command or os.environ.get("SCIQUEST_AGENT_COMMAND")
    if not command:
        return None
    prompt = build_agent_prompt(quest_path, display_path)
    if "{prompt}" in command or "{quest_path}" in command:
        argv = parse_agent_command(command)
        return [
            part.replace("{quest_path}", str(display_path or quest_path)).replace(
                "{prompt}", prompt
            )
            for part in argv
        ]
    argv = parse_agent_command(command)
    return [*argv, prompt]


def launch_agent(
    quest_path: Path,
    agent_command: str | None = None,
    *,
    hidden_root: Path | None = None,
    isolated: bool = True,
    timeout: float = 600,
    log_path: Path | None = None,
    on_tick=None,
) -> dict:
    """Launch an external agent with a complete SciQuest prompt.

    SciQuest does not embed a proprietary agent. It starts whatever open or local
    agent command the user configured and supplies a one-iteration protocol prompt.
    """
    from .execution import run_command

    argv = build_agent_argv(
        quest_path,
        agent_command,
        display_path=Path("/tmp/sciquest-workspace")
        if isolated
        else quest_path.resolve(),
    )
    if not argv:
        raise ValueError(
            "No agent command configured. Pass --agent-command or set SCIQUEST_AGENT_COMMAND."
        )
    result = run_command(
        argv,
        quest_path,
        log_path or quest_path / "logs/agent.log",
        hidden_root=hidden_root or quest_path,
        timeout=timeout,
        isolated=isolated,
        network=True,
        on_tick=on_tick,
    )
    usage = None
    events = 0
    with (log_path or quest_path / "logs/agent.log").open(
        encoding="utf-8", errors="replace"
    ) as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(item, dict):
                continue
            events += 1
            if item.get("type") == "turn.completed" and isinstance(
                item.get("usage"), dict
            ):
                values = item["usage"]
                if all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in values.values()
                ):
                    usage = usage or {}
                    for key, value in values.items():
                        usage[key] = usage.get(key, 0) + value
    result.update(
        backend_reported_usage=usage,
        structured_events=events,
        usage_scope="Optional backend-reported telemetry; unavailable is not zero and does not establish monetary cost",
    )
    return result
