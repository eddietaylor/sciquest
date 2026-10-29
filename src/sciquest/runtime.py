"""Identify the actual interpreter, installed dependencies and SciQuest source."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys

from . import __version__
from .io import content_hash, file_hash


def runtime_manifest() -> dict:
    package = Path(__file__).resolve().parent
    files = {
        str(path.relative_to(package)): file_hash(path)
        for path in sorted(package.rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".yaml"}
        and "__pycache__" not in path.parts
    }
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    identity = {
        "sciquest_version": __version__,
        "package_hash": content_hash(files),
        "python_version": sys.version.split()[0],
        "packages": packages,
    }
    return {
        **identity,
        "fingerprint": content_hash(identity),
        "interpreter": sys.executable,
    }
