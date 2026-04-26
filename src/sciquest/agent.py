from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

NEWTON_SPLASH = r'''
        .-"""-.
       /  .===.  \        ∑  ∫  Δ  λ  π  →
       \/ 6   6 \/       symbols gather into inquiry
       ( \  _  / )
        _`|(_) |`_
       /  `---'  \\       Newton watches the apple fall
      / / .   . \ \\      SciQuest starts the next question
     /_/|   :   |\_\\
        |   :   |        F = G·m₁m₂/r²
        |___:___|        hypothesis → experiment → evidence
          | | |
         _| | |_
        (___|___)

              SciQuest
'''


def parse_agent_command(command: str) -> list[str]:
    """Parse an agent command using shell-like quoting without invoking a shell."""
    return shlex.split(command)


def build_agent_prompt(quest_path: Path) -> str:
    """Create the handoff prompt passed to an external agent process."""
    return f"""You are operating SciQuest.

Quest path: {quest_path}
Agent protocol: {quest_path / 'AGENTS.md'}

Read every quest file first, then perform exactly one SciQuest iteration.
Do not overwrite history. Preserve failed experiments and logs.
Stop after one iteration.
"""


def resolve_agent_command(agent_command: str | None = None) -> list[str] | None:
    """Resolve command from explicit option or SCIQUEST_AGENT_COMMAND."""
    command = agent_command or os.environ.get("SCIQUEST_AGENT_COMMAND")
    if not command:
        return None
    return parse_agent_command(command)


def build_agent_argv(quest_path: Path, agent_command: str | None = None) -> list[str] | None:
    """Build argv for an external agent, adding the SciQuest prompt as input.

    If the configured command contains `{prompt}` or `{quest_path}`, those
    placeholders are expanded before shell-style parsing. Otherwise the prompt is
    appended as the final argument, which works with commands like
    `hermes chat -q` and `codex exec`.
    """
    command = agent_command or os.environ.get("SCIQUEST_AGENT_COMMAND")
    if not command:
        return None
    prompt = build_agent_prompt(quest_path)
    if "{prompt}" in command or "{quest_path}" in command:
        argv = parse_agent_command(command)
        return [part.replace("{quest_path}", str(quest_path)).replace("{prompt}", prompt) for part in argv]
    argv = parse_agent_command(command)
    return [*argv, prompt]


def launch_agent(quest_path: Path, agent_command: str | None = None) -> subprocess.CompletedProcess[str]:
    """Launch an external agent with a complete SciQuest prompt.

    SciQuest does not embed a proprietary agent. It starts whatever open or local
    agent command the user configured and supplies a one-iteration protocol prompt.
    """
    argv = build_agent_argv(quest_path, agent_command)
    if not argv:
        raise ValueError("No agent command configured. Pass --agent-command or set SCIQUEST_AGENT_COMMAND.")
    return subprocess.run(argv, text=True, capture_output=False, check=False)
