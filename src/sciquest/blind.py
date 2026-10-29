"""Allowlisted runtime for untrusted scientist code; no host root or SciQuest import.

The supervisor and model client remain trusted. This is a data boundary, not a
defence against a compromised kernel or model service. Every invocation gets new
PID, IPC, mount, /tmp and user state; scientific continuity is explicitly staged.
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import sysconfig


# Deliberately exclude research/supervisor packages and editable import hooks.
# Only these distributions (and their declared dependency closure) are mounted.
LIBRARIES = (
    "nbformat",
    "nbclient",
    "ipykernel",
    "PyYAML",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
)
WORKER = "/opt/sciquest-worker.py"


def runtime_paths() -> list[Path]:
    paths = {Path(sys.executable), Path(sys.executable).resolve()}
    paths.update((Path(sys.base_prefix) / "lib").glob("libpython*.so*"))
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if cfg.exists():
        paths.add(cfg)
    stdlib = Path(sysconfig.get_path("stdlib"))
    paths.add(stdlib)
    pending, seen = list(LIBRARIES), set()
    from packaging.requirements import Requirement

    while pending:
        name = pending.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        if key == "sciquest":
            raise ValueError("Scientist runtime must not depend on SciQuest")
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            if name in LIBRARIES:
                continue  # optional numerical libraries are not installed
            raise
        for requirement in dist.requires or []:
            req = Requirement(requirement)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                pending.append(req.name)
        for file in dist.files or []:
            first = file.parts[0]
            if first in (".", "..", "__pycache__") or first.endswith(
                (".pth", ".egg-link")
            ):
                continue
            path = Path(dist.locate_file(first))
            if path.exists():
                paths.add(path)
    return sorted(paths, key=lambda p: (len(p.parts), str(p)))


def environment(*, client=False) -> dict[str, str]:
    """No arbitrary inherited secrets, preload hooks, proxies or chat context."""
    env = dict(
        PATH="/usr/bin:/bin",
        LANG="C.UTF-8",
        TZ="UTC",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        JUPYTER_RUNTIME_DIR="/tmp/jupyter",
        IPYTHONDIR="/tmp/ipython",
        MPLCONFIGDIR="/tmp/matplotlib",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    # Preserve the caller's home path, but it denotes an EMPTY directory inside
    # the mount namespace. Never mount the caller's home directory.
    env["HOME"] = str(Path.home())
    if client and "CODEX_HOME" in os.environ:
        env["CODEX_HOME"] = os.environ["CODEX_HOME"]
    return env


def prefix(workspace: Path, *, client_binary: Path | None = None) -> list[str]:
    binary = shutil.which("bwrap")
    if not binary:
        raise RuntimeError(
            "Blind discovery requires bubblewrap; no unisolated fallback"
        )
    args = [
        binary,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--cap-drop",
        "ALL",
        "--tmpfs",
        "/tmp",
    ]
    if client_binary is not None:
        args += ["--share-net"]  # only the tool-free authenticated client
    # Start with an empty root. Mount dependencies, never /, /home, /opt or a
    # complete virtual environment (which could contain editable source hooks).
    arch = sysconfig.get_config_var("MULTIARCH")
    for directory in (f"/usr/lib/{arch}", f"/lib/{arch}"):
        if Path(directory).exists():
            args += ["--ro-bind", directory, directory]
    for directory in (Path("/lib"), Path("/lib64")):
        for loader in directory.glob("ld-linux*.so.*"):
            args += ["--ro-bind", str(loader), str(loader)]
    stdlib = Path(sysconfig.get_path("stdlib"))
    paths = runtime_paths()
    for path in paths:
        if path != stdlib and path.is_relative_to(stdlib):
            continue
        args += ["--ro-bind", str(path), str(path)]
    # A standalone Python installation may store site-packages under stdlib.
    # Mask that broad directory, then remount only allowlisted distributions.
    for name in ("site-packages", "dist-packages"):
        site = stdlib / name
        if site.exists():
            args += ["--tmpfs", str(site)]
            for path in paths:
                if path.is_relative_to(site) and path != site:
                    args += ["--ro-bind", str(path), str(path)]
    args += ["--ro-bind", str(Path(__file__).with_name("worker.py")), WORKER]
    if client_binary is not None:
        args += ["--ro-bind", str(client_binary), str(client_binary)]
        for path in (
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/nsswitch.conf",
            "/etc/ssl/certs",
        ):
            if Path(path).exists():
                args += ["--ro-bind", path, path]
    args += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--dir",
        str(Path.home()),
        "--bind",
        str(workspace.resolve()),
        "/tmp/sciquest-workspace",
        "--chdir",
        "/tmp/sciquest-workspace",
    ]
    return args


def worker_argv(argv: list[str]) -> list[str]:
    if len(argv) >= 3 and argv[1:3] == ["-m", "sciquest.worker"]:
        return [argv[0], WORKER, *argv[3:]]
    return argv
