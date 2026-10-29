from __future__ import annotations

from pathlib import Path
from typing import Any

import nbformat

from .io import contained_path, content_hash, experiment_dir, file_hash, read_yaml
from .methods.ledger import classify_report_claims, parse_report_claims
from .state import authority_dir, get_record
from .validation import compute_aggregate_score


def run_logic_check(
    quest_path: Path, experiment_id: str | None = None
) -> dict[str, Any]:
    """Check artifact structure and recorded integrity, with explicit evidence scope."""
    issues: list[str] = []
    warnings: list[str] = []
    verified = False
    try:
        hypotheses = read_yaml(quest_path / "hypotheses.yaml", {})
        if not isinstance(hypotheses, dict) or not hypotheses.get("hypotheses"):
            issues.append("No hypothesis registry recorded.")
        suite = read_yaml(quest_path / "validation.yaml", {})
        if experiment_id:
            exp = experiment_dir(quest_path, experiment_id)
            meta = read_yaml(exp / "experiment.yaml", {})
            if not isinstance(meta, dict) or not meta:
                issues.append("experiment.yaml missing or invalid.")
                meta = {}
            if meta.get("status") in {"draft", "agent_stub", "agent_required"}:
                issues.append("Experiment is a draft, not a completed execution.")
            if not (exp / "logs").is_dir():
                issues.append("Execution logs directory missing.")
            for relative in meta.get("technical_diagrams", []) or []:
                if not contained_path(exp, relative).is_file():
                    issues.append(f"Declared technical diagram is missing: {relative}.")
            nb_path, executed_path = (
                exp / "notebook.ipynb",
                exp / "executed_notebook.ipynb",
            )
            if not nb_path.is_file() or not executed_path.is_file():
                issues.append(
                    "Source and executed notebooks are required to inspect execution."
                )
            else:
                source = nbformat.read(nb_path, as_version=4)
                executed = nbformat.read(executed_path, as_version=4)
                nbformat.validate(source)
                nbformat.validate(executed)
                sources = lambda nb: [(c.cell_type, c.source) for c in nb.cells]
                if sources(source) != sources(executed):
                    issues.append(
                        "Executed notebook source differs from the proposed notebook."
                    )
                cells = [
                    c
                    for c in executed.cells
                    if c.cell_type == "code" and c.source.strip()
                ]
                if not cells or any(c.get("execution_count") is None for c in cells):
                    issues.append(
                        "Notebook contains unexecuted code or no executable cells."
                    )
                if any(
                    o.output_type == "error"
                    for c in cells
                    for o in c.get("outputs", [])
                ):
                    issues.append("Executed notebook contains an error.")
            registration = get_record(quest_path, "registration", experiment_id)
            evaluation = get_record(quest_path, "evaluation", experiment_id)
            execution = get_record(quest_path, "execution", experiment_id)
            completion = get_record(quest_path, "completion", experiment_id)
            if registration:
                frozen = {
                    key: value
                    for key, value in registration.items()
                    if key != "registration_hash"
                }
                if content_hash(frozen) != registration["registration_hash"]:
                    issues.append("Supervisor registration hash is invalid.")
                snapshot = authority_dir(quest_path) / "snapshots" / experiment_id
                for relative, expected in registration["file_hashes"].items():
                    if file_hash(contained_path(snapshot, relative)) != expected:
                        issues.append(f"Registered input changed: {relative}.")
                suite = registration["validation_suite"]
                if not execution or execution.get("status") != "succeeded":
                    issues.append("Successful supervisor execution is not recorded.")
                if not evaluation or not completion:
                    issues.append("Supervisor evaluation/finalization is incomplete.")
                elif (
                    completion.get("evaluation_hash") != content_hash(evaluation)
                    or evaluation.get("registration_hash")
                    != registration["registration_hash"]
                ):
                    issues.append(
                        "Evaluation is not bound to the registered experiment."
                    )
                else:
                    verified = True
            else:
                warnings.append(
                    "No authoritative registration exists; legacy artifacts cannot establish preregistration or independent evaluation."
                )
            results = read_yaml(exp / "validation_results.yaml")
            computed = compute_aggregate_score(suite, results)
            if results.get("aggregate_score") != computed["aggregate_score"]:
                issues.append(
                    "Stored aggregate score does not match the complete metric suite."
                )
            if (
                evaluation
                and evaluation.get("validation_status") != computed["validation_status"]
            ):
                issues.append(
                    "Stored validation status does not match metric roles and constraints."
                )
            if (
                evaluation
                and "constraint_outcomes" in evaluation
                and evaluation["constraint_outcomes"] != computed["constraint_outcomes"]
            ):
                issues.append(
                    "Stored constraint outcomes do not match registered requirements."
                )
            if execution and execution.get("status") == "succeeded":
                if file_hash(executed_path) != execution.get("output_notebook_hash"):
                    issues.append(
                        "Published executed notebook differs from the execution record."
                    )
                for name, expected in execution.get(
                    "prediction_artifact_hashes", {}
                ).items():
                    if (
                        file_hash(contained_path(exp / "worker_artifacts", name))
                        != expected
                    ):
                        issues.append(
                            f"Prediction artifact changed after execution: {name}."
                        )
                for name, expected in (
                    execution.get("model_export_verification", {})
                    .get("model_artifact_hashes", {})
                    .items()
                ):
                    if (
                        file_hash(contained_path(exp / "worker_artifacts", name))
                        != expected
                    ):
                        issues.append(f"Exported model artifact changed: {name}.")
            if evaluation and results != evaluation:
                issues.append(
                    "Published results differ from the supervisor evaluation."
                )
            report = exp / "experiment_report.md"
            if report.exists():
                classes = classify_report_claims(
                    registration or {},
                    parse_report_claims(report.read_text()),
                    evaluation,
                )
                if classes.get("method_violation"):
                    issues.append(
                        "Method violation: report contains an unlabeled post-hoc change."
                    )
                if classes.get("unverified_claim"):
                    warnings.append(
                        "Report includes confirmatory claims without linked authoritative evidence."
                    )
            else:
                warnings.append("No interpretation report exists.")
        else:
            from .validation import validate_suite

            validate_suite(suite)
    except (ValueError, TypeError, KeyError, OSError, nbformat.ValidationError) as exc:
        issues.append(str(exc))
    return {
        "passed": not issues,
        "scope": "artifact_schema_and_recorded_integrity",
        "recorded_integrity_verified": verified and not issues,
        "scientific_verified": False,
        "interpretation": "A valid record does not by itself establish causal identification, novelty, or statistical error control.",
        "issues": issues,
        "warnings": warnings,
    }
