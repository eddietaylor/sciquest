from __future__ import annotations

from pathlib import Path
from typing import Any

from sciquest.io import read_yaml, utc_now, write_yaml

from .prompt_builder import build_method_journal_template
from .registry import MethodRegistry
from .resolver import resolve_method_stack
from .schema import MethodStack


def load_quest_method_stack(qpath: Path) -> MethodStack:
    return resolve_method_stack(read_yaml(qpath / "method_stack.yaml", {"primary_method": "standard_empirical"}))


def preregister_experiment(qpath: Path, exp_id: str, exp_path: Path, predicted_observations: list[str] | None = None) -> dict[str, Any]:
    stack = load_quest_method_stack(qpath)
    registry = MethodRegistry.default()
    profiles = [registry.get(m) for m in stack.methods_used]
    profile_versions = {profile.id: profile.profile_version for profile in profiles}
    evaluation_rules: list[str] = []
    success_criteria: list[str] = []
    allowed_post_hoc: list[str] = [
        "Exploratory analyses are allowed only when explicitly labeled exploratory_observation.",
        "Post hoc reinterpretations must be labeled post_hoc_reinterpretation and must not replace the pre-registered confirmatory evaluation.",
        "Switching method lenses after seeing results is a method_violation unless recorded as a new future experiment pre-registration.",
    ]
    for profile in profiles:
        evaluation_rules.extend(profile.evidence_evaluation_rules)
        success_criteria.extend(profile.validation_rules)
    ledger = {
        "experiment_id": exp_id,
        "created_at": utc_now(),
        "pre_registration": {
            "method_stack": stack.to_dict(),
            "primary_method": stack.primary_method,
            "phase_methods": stack.phases,
            "profile_versions": profile_versions,
            "confirmatory_status": "confirmatory",
            "predicted_observations": predicted_observations or ["Agent must specify predicted observations before notebook execution."],
            "falsification_or_success_criteria": success_criteria or ["Agent must specify success criteria before notebook execution."],
            "evaluation_rules": evaluation_rules,
            "allowed_post_hoc_analysis_rules": allowed_post_hoc,
        },
        "post_experiment_classification": {
            "confirmatory_result": [],
            "exploratory_observation": [],
            "post_hoc_reinterpretation": [],
            "method_violation": [],
            "unresolved_anomaly": [],
        },
    }
    write_yaml(exp_path / "method_ledger.yaml", ledger)
    return ledger


def classify_result_claim(pre_registration: dict[str, Any], claim: dict[str, Any]) -> str:
    claim_type = claim.get("claim_type", "confirmatory")
    if claim_type == "exploratory":
        return "exploratory_observation"
    if claim_type == "post_hoc":
        return "post_hoc_reinterpretation" if claim.get("labeled_post_hoc") else "method_violation"
    if claim.get("method_changed_after_results"):
        return "method_violation"
    if claim_type == "confirmatory":
        return "confirmatory_result" if claim.get("matched_prediction") else "unresolved_anomaly"
    return "unresolved_anomaly"


def _coerce_claim_value(value: str) -> Any:
    lower = value.strip().lower()
    if lower in {"true", "yes"}:
        return True
    if lower in {"false", "no"}:
        return False
    return value.strip()


def parse_report_claims(report_text: str) -> list[dict[str, Any]]:
    """Parse structured claim sections from an experiment report.

    Supported headings are confirmatory result, exploratory observation, post hoc
    reinterpretation, method violation, and unresolved anomaly. Bullets may use
    `key: value`; free-text bullets become `claim`.
    """
    heading_to_type = {
        "confirmatory result": "confirmatory",
        "confirmatory results": "confirmatory",
        "exploratory observation": "exploratory",
        "exploratory observations": "exploratory",
        "post hoc reinterpretation": "post_hoc",
        "post-hoc reinterpretation": "post_hoc",
        "method violation": "method_violation",
        "unresolved anomaly": "unresolved_anomaly",
    }
    claims: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if line.startswith("##"):
            title = line.lstrip("#").strip().lower()
            claim_type = heading_to_type.get(title)
            current = {"claim_type": claim_type, "claim": ""} if claim_type else None
            if current is not None:
                claims.append(current)
            continue
        if current is None or not line:
            continue
        if line.startswith(("-", "*")):
            item = line[1:].strip()
            if ":" in item:
                key, value = item.split(":", 1)
                current[key.strip()] = _coerce_claim_value(value)
            else:
                current["claim"] = (current.get("claim", "") + " " + item).strip()
        elif not line.startswith("#"):
            current["claim"] = (current.get("claim", "") + " " + line).strip()
    return claims


def classify_report_claims(pre_registration: dict[str, Any], claims: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "confirmatory_result": [],
        "exploratory_observation": [],
        "post_hoc_reinterpretation": [],
        "method_violation": [],
        "unresolved_anomaly": [],
    }
    for claim in claims:
        if claim.get("claim_type") == "method_violation":
            bucket = "method_violation"
        elif claim.get("claim_type") == "unresolved_anomaly":
            bucket = "unresolved_anomaly"
        else:
            bucket = classify_result_claim(pre_registration, claim)
        buckets.setdefault(bucket, []).append(claim)
    return buckets


def append_method_sections_to_report(report_path: Path, stack: MethodStack | None = None) -> None:
    if stack is None:
        stack = resolve_method_stack({"primary_method": "standard_empirical"})
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if "## Method Ledger" not in existing:
        report_path.write_text(existing.rstrip() + "\n\n" + build_method_journal_template(stack), encoding="utf-8")
