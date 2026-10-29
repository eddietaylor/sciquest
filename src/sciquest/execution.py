"""Bounded process execution and filesystem isolation for local research workers."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Any
from collections.abc import Callable

from .io import content_hash, finite_number, utc_now


def isolation_prefix(
    workspace: Path, hidden_root: Path, *, network: bool = False
) -> list[str]:
    binary = shutil.which("bwrap")
    if not binary:
        raise RuntimeError(
            "Isolated execution requires bubblewrap; use explicit trusted-local exploration only if appropriate"
        )
    # Hide the complete quest, not just its database. The worker receives only
    # staged inputs. /proc is from its own PID namespace; /tmp is private.
    args = [binary, "--die-with-parent", "--new-session", "--unshare-all"]
    if network:
        args += ["--share-net"]
    args += ["--ro-bind", "/", "/", "--tmpfs", "/tmp"]
    for prefix in dict.fromkeys(
        [Path(sys.prefix).absolute(), Path(sys.base_prefix).absolute()]
    ):
        if prefix.is_relative_to(Path("/tmp")):
            if prefix == hidden_root.resolve() or prefix.is_relative_to(
                hidden_root.resolve()
            ):
                raise ValueError(
                    "Install the worker runtime outside the protected quest directory"
                )
            args += ["--ro-bind", str(prefix), str(prefix)]
    args += [
        "--tmpfs",
        str(hidden_root.resolve()),
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--dir",
        "/tmp/sciquest-workspace",
        "--bind",
        str(workspace.resolve()),
        "/tmp/sciquest-workspace",
        "--chdir",
        "/tmp/sciquest-workspace",
    ]
    return args


def run_command(
    argv: list[str],
    workspace: Path,
    log: Path,
    *,
    hidden_root: Path,
    timeout: float = 600,
    isolated: bool = True,
    network: bool = False,
    max_file_bytes: int = 64 * 1024 * 1024,
    on_tick: Callable[[], None] | None = None,
    blind: bool = False,
) -> dict[str, Any]:
    timeout = finite_number(timeout, "timeout")
    if timeout <= 0 or not argv:
        raise ValueError("A positive wall-time budget and a command are required")
    log.parent.mkdir(parents=True, exist_ok=True)
    started, monotonic = utc_now(), time.monotonic()
    from . import blind as sealed

    if blind and (not isolated or network):
        raise ValueError("Scientist code requires isolation with no network")
    prefix = (
        sealed.prefix(workspace)
        if blind
        else (
            isolation_prefix(workspace, hidden_root, network=network)
            if isolated
            else []
        )
    )
    # The helper sets resource limits in a fresh interpreter, avoiding unsafe
    # preexec_fn hooks in a potentially multithreaded supervisor.
    command = [
        *prefix,
        sys.executable,
        "-m",
        "sciquest.worker",
        "command",
        str(max_file_bytes),
        "--",
        *argv,
    ]
    if blind:
        command = [
            *prefix,
            sys.executable,
            sealed.WORKER,
            "command",
            str(max_file_bytes),
            "--",
            *sealed.worker_argv(argv),
        ]
    env = (
        dict(os.environ)
        if network
        else {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "PATH",
                "LANG",
                "LC_ALL",
                "TZ",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            }
        }
    )
    for name in ("SCIQUEST_OWNER_TOKEN",):
        env.pop(name, None)
    if blind:
        env = sealed.environment()
    env["JUPYTER_RUNTIME_DIR"] = (
        "/tmp/sciquest-jupyter" if isolated else str(workspace / ".jupyter")
    )
    env["IPYTHONDIR"] = (
        "/tmp/sciquest-ipython" if isolated else str(workspace / ".ipython")
    )
    status, returncode = "failed", None
    with log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                remaining = timeout - (time.monotonic() - monotonic)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    returncode = process.wait(timeout=min(remaining, 2))
                    break
                except subprocess.TimeoutExpired:
                    if on_tick:
                        on_tick()
            status = "succeeded" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timed_out"
        finally:
            # Terminate the entire group, including orphaned agent/tool/kernel
            # children even when their immediate parent returned successfully.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return {
        "status": status,
        "returncode": returncode,
        "started_at": started,
        "finished_at": utc_now(),
        "elapsed_seconds": time.monotonic() - monotonic,
        "timeout_seconds": timeout,
        "isolated": isolated,
        "network": network,
        "blind_filesystem": blind,
        "executable": argv[0],
        "command_hash": content_hash(argv),
        "log_bytes": log.stat().st_size,
    }
