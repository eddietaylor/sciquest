from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .core import run_next
from .io import read_yaml, write_yaml, utc_now

IterationRunner = Callable[..., str]


def _quest_is_ready_for_next(qpath: Path) -> tuple[bool, str | None]:
    state = read_yaml(qpath / "state.yaml", {})
    if state.get("lock_id"):
        return False, f"quest is locked by {state['lock_id']}"
    if state.get("quest_status") == "running":
        return False, "quest_status is running"
    if state.get("quest_status") == "failed":
        return False, "quest_status is failed"
    return True, None


def run_iteration_loop(
    qpath: Path,
    max_iterations: int = 3,
    *,
    agent_stub: bool = False,
    execute: bool = False,
    start_agent: bool = False,
    agent_command: str | None = None,
    poll_seconds: float = 1.0,
    wait_timeout_seconds: float = 0.0,
    iteration_runner: IterationRunner | None = None,
) -> dict[str, Any]:
    """Run multiple SciQuest iterations sequentially.

    The next iteration starts only after the previous one has returned and the
    quest state is idle/unlocked. This is intentionally not cron; it is a
    foreground loop for autonomous multi-iteration execution.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    runner = iteration_runner or run_next
    completed: list[str] = []
    result: dict[str, Any] = {
        "requested_iterations": max_iterations,
        "completed_iterations": 0,
        "experiments": completed,
        "failed": False,
        "error": None,
    }

    for _ in range(max_iterations):
        deadline = time.monotonic() + wait_timeout_seconds if wait_timeout_seconds else None
        while True:
            ready, reason = _quest_is_ready_for_next(qpath)
            if ready:
                break
            if deadline is None or time.monotonic() >= deadline:
                result.update({"failed": True, "error": f"Previous iteration not complete: {reason}"})
                return result
            time.sleep(poll_seconds)
        try:
            if start_agent:
                from .agent import launch_agent
                proc = launch_agent(qpath, agent_command)
                if proc.returncode != 0:
                    raise RuntimeError(f"Agent exited with code {proc.returncode}")
                state = read_yaml(qpath / "state.yaml", {})
                exp_id = state.get("last_experiment") or f"agent_iteration_{len(completed)+1}"
            else:
                exp_id = runner(qpath, agent_stub=agent_stub, execute=execute)
        except Exception as exc:
            state = read_yaml(qpath / "state.yaml", {})
            failures = state.get("failures") or []
            failures.append({"time": utc_now(), "error": str(exc), "loop_iteration": len(completed) + 1})
            state["failures"] = failures
            state["quest_status"] = "failed"
            state["last_updated"] = utc_now()
            write_yaml(qpath / "state.yaml", state)
            result.update({"failed": True, "error": str(exc), "completed_iterations": len(completed)})
            return result
        completed.append(exp_id)
        result["completed_iterations"] = len(completed)
    return result
