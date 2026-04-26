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
        experiment_meta = read_yaml(exp / "experiment.yaml", {})
        diagrams_dir = exp / "artifacts" / "diagrams"
        plots_dir = exp / "artifacts" / "plots"

        if not (exp / "experiment.yaml").exists():
            issues.append("experiment.yaml missing; experiment metadata cannot be audited.")
        for dirname, path in [("artifacts/plots", plots_dir), ("artifacts/diagrams", diagrams_dir), ("logs", exp / "logs")]:
            if not path.exists() or not path.is_dir():
                issues.append(f"Required experiment directory missing: {dirname}.")

        for field in ["task_type", "model_architecture", "input_features", "target_features", "validation_technique"]:
            if not experiment_meta.get(field):
                warnings.append(f"experiment.yaml should document dashboard field: {field}.")

        declared_diagrams = experiment_meta.get("technical_diagrams") or []
        if not declared_diagrams and not list(diagrams_dir.glob("*.svg")):
            issues.append("Technical diagrams are missing; add model and validation architecture diagrams under artifacts/diagrams/.")
        for rel in declared_diagrams:
            if not (exp / rel).exists():
                issues.append(f"Declared technical diagram is missing: {rel}.")

        generated_datasets = [d for d in data.get("datasets", []) or [] if "synthetic" in (d.get("provenance", "") + d.get("name", "")).lower()]
        if generated_datasets and not any((exp / "artifacts").glob("*generation*.py")) and not any((exp / "artifacts").glob("*generator*.py")):
            issues.append("Synthetic/generated data was used but no data generation script artifact was preserved.")

        if not nb.exists():
            issues.append("Experiment notebook is missing; hypothesis is not testable in notebook.")
        else:
            text = nb.read_text(encoding="utf-8", errors="ignore").lower()
            for term in ["preview", "dataset", "target", "feature", "validation_results"]:
                if term not in text:
                    warnings.append(f"Notebook may not document required element: {term}.")
            section_markers = ["setup", "data", "preprocess", "feature", "model", "validation", "artifact"]
            if sum(1 for marker in section_markers if marker in text) < 4:
                warnings.append("Notebook should be split into readable sections: setup, data, preprocessing/features, model, validation, artifacts.")
        if not results.exists():
            issues.append("validation_results.yaml missing; no silent-failure check possible.")
        if report.exists():
            rtext = report.read_text(encoding="utf-8", errors="ignore").lower()
            if "speculation" not in rtext or "evidence" not in rtext:
                warnings.append("Report should clearly separate speculation vs evidence.")
        else:
            warnings.append("Experiment report missing; conclusions cannot be checked against results.")

    return {"passed": not issues, "issues": issues, "warnings": warnings}
