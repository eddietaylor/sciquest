from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


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


def execute_notebook(notebook_path: Path, output_path: Path, cwd: Path, timeout: int = 600) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(cwd)}})
    client.execute()
    nbformat.write(nb, output_path)
