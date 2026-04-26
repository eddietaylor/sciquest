from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

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


def launch_agent(quest_path: Path, agent_command: str | None = None) -> subprocess.CompletedProcess[str]:
    """Launch an external agent with a SciQuest prompt on stdin.

    SciQuest does not embed a proprietary agent. It starts whatever open or local
    agent command the user configured, passing a complete protocol prompt through
    stdin so the command can act on the quest directory.
    """
    argv = resolve_agent_command(agent_command)
    if not argv:
        raise ValueError("No agent command configured. Pass --agent-command or set SCIQUEST_AGENT_COMMAND.")
    prompt = build_agent_prompt(quest_path)
    return subprocess.run(argv, input=prompt, text=True, capture_output=False, check=False)
