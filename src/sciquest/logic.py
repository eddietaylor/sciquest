from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_yaml


def run_logic_check(quest_path: Path, experiment_id: str | None = None) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    hyp = read_yaml(quest_path / "hypotheses.yaml", {})
    val = read_yaml(quest_path / "validation.yaml", {})
    data = read_yaml(quest_path / "data_manifest.yaml", {})

    if not hyp.get("hypotheses"):
        issues.append("No hypothesis recorded; hypothesis must be testable in notebook.")
    if val.get("status") == "agent_required" or not val.get("metrics"):
        issues.append("Validation suite is missing or agent_required; metrics cannot yet align with problem.")
    if data.get("status") == "missing_user_data":
        warnings.append("Core data missing; agent must document discovered/generated data provenance and limitations.")

    if experiment_id:
        exp = quest_path / "experiments" / experiment_id
        nb = exp / "notebook.ipynb"
        report = exp / "experiment_report.md"
        results = exp / "validation_results.yaml"
        if not nb.exists():
            issues.append("Experiment notebook is missing; hypothesis is not testable in notebook.")
        else:
            text = nb.read_text(encoding="utf-8", errors="ignore").lower()
            for term in ["preview", "dataset", "target", "feature", "validation_results"]:
                if term not in text:
                    warnings.append(f"Notebook may not document required element: {term}.")
        if not results.exists():
            issues.append("validation_results.yaml missing; no silent-failure check possible.")
        if report.exists():
            rtext = report.read_text(encoding="utf-8", errors="ignore").lower()
            if "speculation" not in rtext or "evidence" not in rtext:
                warnings.append("Report should clearly separate speculation vs evidence.")
        else:
            warnings.append("Experiment report missing; conclusions cannot be checked against results.")

    return {"passed": not issues, "issues": issues, "warnings": warnings}
