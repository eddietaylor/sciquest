from __future__ import annotations

from pathlib import Path
import sys

import nbformat


def create_stub_notebook(path: Path, experiment_id: str) -> None:
    nb = nbformat.v4.new_notebook()
    code = '''from pathlib import Path
import yaml

EXP = Path.cwd()
ARTIFACTS = EXP / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
print("Dataset preview: agent must load or create data before substantive execution")
print("Dataset size before transforms: unknown")
print("Dataset size after transforms: unknown")
TARGET = "agent_defined_target"
INPUTS = []
FEATURES = []

def describe_preprocessing():
    """Document preprocessing and feature engineering for this experiment."""
    return "Agent stub; replace with concrete preprocessing."

metrics = {}
with open(EXP / "validation_results.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump({"metrics": metrics, "notes": "Agent stub produced no metrics."}, f)
'''
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            f"# {experiment_id}\n\nSciQuest experiment notebook. Dataset preview, target, inputs, preprocessing, features, plots, and metrics are documented here."
        ),
        nbformat.v4.new_code_cell(code),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)


def execute_notebook(
    notebook_path: Path, output_path: Path, cwd: Path, timeout: int = 600
) -> None:
    """Explicit trusted-local convenience execution, with a wall-time bound and logs.

    Managed research runs use the isolated supervisor path instead. Neither path
    infers scientific support from successful notebook execution.
    """
    from .execution import run_command

    source = notebook_path.resolve()
    destination = output_path.resolve()
    result = run_command(
        [
            sys.executable,
            "-m",
            "sciquest.worker",
            "notebook",
            str(source),
            str(destination),
            str(timeout),
        ],
        cwd.resolve(),
        cwd / "logs/notebook_execution.log",
        hidden_root=cwd,
        timeout=timeout,
        isolated=False,
    )
    if result["status"] != "succeeded":
        raise RuntimeError(
            f"Notebook execution {result['status']}; inspect logs/notebook_execution.log"
        )
