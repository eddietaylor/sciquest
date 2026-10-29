from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .core import run_next
from .io import read_yaml, finite_number
from .state import read_state, recover, get_record
from .io import experiment_dir

IterationRunner = Callable[..., str]


def _quest_is_ready_for_next(qpath: Path) -> tuple[bool, str | None]:
    state = read_state(qpath)
    if state.get("lock_id"):
        return False, f"quest is locked by {state['lock_id']}"
    if state.get("quest_status") == "running":
        return False, "quest_status is running"
    if state.get("quest_status") == "failed":
        return False, "quest_status is failed"
    return (
        (True, None)
        if state.get("quest_status") == "idle"
        else (False, "quest_status must be idle")
    )


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
    isolated: bool = True,
    timeout_seconds: float = 600,
    max_repairs: int = 0,
) -> dict[str, Any]:
    """Run multiple SciQuest iterations sequentially.

    The next iteration starts only after the previous one has returned and the
    quest state is idle/unlocked. This is intentionally not cron; it is a
    foreground loop for autonomous multi-iteration execution.
    """
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be >= 1")
    if (
        finite_number(poll_seconds, "poll_seconds") <= 0
        or finite_number(wait_timeout_seconds, "wait_timeout_seconds") < 0
    ):
        raise ValueError("Polling must be positive and wait timeout nonnegative")
    if (
        isinstance(max_repairs, bool)
        or not isinstance(max_repairs, int)
        or not 0 <= max_repairs <= 10
    ):
        raise ValueError("max_repairs must be between 0 and 10")
    if finite_number(timeout_seconds, "timeout_seconds") <= 0:
        raise ValueError("timeout_seconds must be positive")
    runner = iteration_runner or run_next
    completed: list[str] = []
    result: dict[str, Any] = {
        "requested_iterations": max_iterations,
        "completed_iterations": 0,
        "prepared_iterations": 0,
        "repair_attempts": 0,
        "experiments": completed,
        "failed": False,
        "error": None,
    }

    for _ in range(max_iterations):
        deadline = (
            time.monotonic() + wait_timeout_seconds if wait_timeout_seconds else None
        )
        while True:
            ready, reason = _quest_is_ready_for_next(qpath)
            if ready:
                break
            if deadline is None or time.monotonic() >= deadline:
                result.update(
                    {
                        "failed": True,
                        "error": f"Previous iteration not complete: {reason}",
                    }
                )
                return result
            time.sleep(poll_seconds)
        try:
            before_ids = {path.name for path in (qpath / "experiments").glob("exp_*")}
            if start_agent:
                from .workflow import run_agent_iteration

                iteration_started = time.monotonic()
                for repair in range(max_repairs + 1):
                    try:
                        remaining = timeout_seconds - (
                            time.monotonic() - iteration_started
                        )
                        if remaining <= 0:
                            raise TimeoutError(
                                "Iteration budget exhausted before repair"
                            )
                        exp_id = run_agent_iteration(
                            qpath, agent_command, isolated=isolated, timeout=remaining
                        )
                        break
                    except Exception:
                        state = read_state(qpath)
                        if (
                            repair >= max_repairs
                            or state.get("quest_status") != "failed"
                            or state.get("lock_id")
                        ):
                            raise
                        recover(qpath)
                        result["repair_attempts"] += 1
                if exp_id is None:
                    result.update(
                        stopped=True, decision=read_state(qpath).get("last_decision")
                    )
                    return result
            else:
                exp_id = runner(qpath, agent_stub=agent_stub, execute=execute)
            if exp_id in before_ids or exp_id in completed:
                raise RuntimeError("Iteration did not produce a new experiment ID")
            path = experiment_dir(qpath, exp_id)
            completion = get_record(qpath, "completion", exp_id)
            if not path.is_dir() or (
                not completion
                and read_yaml(path / "experiment.yaml", {}).get("status") != "draft"
            ):
                raise RuntimeError(
                    "Runner returned no finalized experiment or valid draft"
                )
        except Exception as exc:
            # The owning lifecycle records failure. A competing launcher must
            # never overwrite the state/lock of a run it does not own.
            result.update({"failed": True, "error": str(exc)})
            return result
        completed.append(exp_id)
        if completion:
            result["completed_iterations"] += 1
        else:
            result["prepared_iterations"] += 1
    return result
