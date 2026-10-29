"""Byte-level generator replay. Reproducibility is distinct from model validity."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from .execution import run_command
from .io import contained_path, file_hash


def verify_generator(qpath: Path, dataset: dict[str, Any], log: Path) -> dict[str, Any]:
    generator = contained_path(qpath, dataset.get("generator"))
    output = dataset.get("generated_output")
    args = dataset.get("generator_args")
    seed = dataset.get("seed")
    if not generator.is_file() or generator.suffix != ".py":
        raise ValueError("Synthetic data requires an executable Python generator")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Synthetic data requires an explicit integer seed")
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("generator_args must be an explicit string list")
    expected = file_hash(contained_path(qpath, dataset["path"]))
    with tempfile.TemporaryDirectory(prefix="sciquest-generator-") as temp:
        workspace = Path(temp)
        shutil.copyfile(generator, workspace / "generator.py")
        for relative in dataset.get("generator_inputs", []):
            source, destination = (
                contained_path(qpath, relative),
                contained_path(workspace, relative),
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        target = contained_path(workspace, output)
        execution = run_command(
            [sys.executable, "generator.py", *args],
            workspace,
            log,
            hidden_root=qpath,
            timeout=60,
            isolated=True,
            network=False,
        )
        if (
            execution["status"] != "succeeded"
            or not target.is_file()
            or target.is_symlink()
            or file_hash(target) != expected
        ):
            raise ValueError(
                "Synthetic generator replay did not reproduce the declared dataset bytes"
            )
        return {
            "generator_sha256": file_hash(generator),
            "dataset_sha256": expected,
            "seed": seed,
            "execution": execution,
            "scope": "Exact byte replay under declared arguments; not validation of scientific realism",
        }
