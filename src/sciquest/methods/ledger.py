from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sciquest.io import (
    contained_path,
    content_hash,
    experiment_dir,
    file_hash,
    finite_number,
    read_yaml,
    utc_now,
    write_yaml,
)
from sciquest.state import authority_dir, evidence_was_used, get_record, record
from sciquest.validation import validate_suite

from .prompt_builder import build_method_journal_template
from .registry import MethodRegistry
from .resolver import resolve_method_stack
from .schema import MethodStack


def load_quest_method_stack(qpath: Path) -> MethodStack:
    return resolve_method_stack(
        read_yaml(qpath / "method_stack.yaml", {"primary_method": "standard_empirical"})
    )


def create_draft_ledger(qpath: Path, exp_id: str, exp_path: Path) -> dict[str, Any]:
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
        "schema_version": 2,
        "registration_status": "draft",
        "experiment_id": exp_id,
        "created_at": utc_now(),
        "pre_registration": {
            "method_stack": stack.to_dict(),
            "primary_method": stack.primary_method,
            "phase_methods": stack.phases,
            "profile_versions": profile_versions,
            "confirmatory_status": "exploratory",
            "predicted_observations": [],
            "falsification_or_success_criteria": success_criteria,
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
    write_yaml(exp_path / "method_ledger.yaml", ledger, exclusive=True)
    return ledger


def validate_plan(plan: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("experiment_plan.yaml must be a mapping")
    for field in [
        "question",
        "hypothesis",
        "scope",
        "assumptions",
        "data_access",
        "stopping_rule",
    ]:
        value = plan.get(field)
        if (
            not isinstance(value, str)
            or not value.strip()
            or value.strip().lower() in {"pending", "agent_required", "todo"}
        ):
            raise ValueError(f"Plan requires a concrete {field}")
    actions = {
        "hypothesis_test",
        "rival_test",
        "replication",
        "measurement",
        "data_acquisition",
    }
    if not isinstance(plan.get("action"), str) or plan["action"] not in actions:
        raise ValueError("Plan action must be a string: " + ", ".join(sorted(actions)))
    if not isinstance(plan.get("confirmatory_status"), str) or plan[
        "confirmatory_status"
    ] not in {"confirmatory", "exploratory"}:
        raise ValueError("Plan must declare confirmatory or exploratory status")
    predictions = plan.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Plan requires measurable predictions")
    names = {m["name"] for m in suite["metrics"]}
    ids = set()
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("Predictions must be mappings")
        pid = prediction.get("id")
        if not isinstance(pid, str) or not pid.strip() or pid in ids:
            raise ValueError("Prediction IDs must be nonempty and unique")
        ids.add(pid)
        if (
            not isinstance(prediction.get("metric"), str)
            or not isinstance(prediction.get("operator"), str)
            or prediction.get("metric") not in names
            or prediction.get("operator")
            not in {
                "<",
                "<=",
                ">",
                ">=",
                "==",
            }
        ):
            raise ValueError(
                "Prediction must reference a metric and a supported operator"
            )
        finite_number(prediction.get("threshold"), f"{pid}.threshold")
    export = plan.get("model_export")
    if export is not None:
        if (
            not isinstance(export, dict)
            or not isinstance(export.get("artifacts"), list)
            or not export["artifacts"]
            or not all(isinstance(p, str) and p for p in export["artifacts"])
        ):
            raise ValueError("Model export requires explicit artifact paths")
        if (
            not isinstance(export.get("reload_notebook"), str)
            or not export["reload_notebook"]
        ):
            raise ValueError("Model export requires a standalone reload notebook")
        for name in [*export["artifacts"], export["reload_notebook"]]:
            path = Path(name)
            if not path.parts or path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "Model export paths must be contained relative file paths"
                )
        prediction_files = {
            m.get("computation", {}).get("artifact", "predictions.csv")
            for m in suite["metrics"]
        }
        if (
            prediction_files.intersection(export["artifacts"])
            or export["reload_notebook"] in export["artifacts"]
            or any(Path(name).parts[0] == "data" for name in export["artifacts"])
        ):
            raise ValueError(
                "Exported model artifacts must be separate from staged data, predictions, and the reload notebook"
            )
        if finite_number(export.get("absolute_tolerance", 0), "reload tolerance") < 0:
            raise ValueError("Reload tolerance must be nonnegative")
    return plan


def preregister_experiment(
    qpath: Path,
    exp_id: str,
    exp_path: Path | None = None,
    predicted_observations: list[str] | None = None,
    *,
    owner: str | None = None,
) -> dict[str, Any]:
    """Freeze concrete inputs once, before execution, in supervisor storage."""
    exp = experiment_dir(qpath, exp_id)
    if exp_path is not None and exp_path.resolve() != exp.resolve():
        raise ValueError("Experiment path does not match its ID")
    if predicted_observations is not None:
        raise ValueError("Use structured predictions in experiment_plan.yaml")
    if get_record(qpath, "registration", exp_id):
        raise FileExistsError(
            "Registration already exists; create a new experiment for amendments"
        )
    if (
        (exp / "execution.json").exists()
        or (exp / "validation_results.yaml").exists()
        or (exp / "executed_notebook.ipynb").exists()
    ):
        raise ValueError("Results already exist; registration cannot be retrospective")
    suite = validate_suite(read_yaml(qpath / "validation.yaml", {}))
    plan = validate_plan(read_yaml(exp / "experiment_plan.yaml"), suite)
    manifest = read_yaml(qpath / "data_manifest.yaml", {})
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "ready"
        or not manifest.get("datasets")
    ):
        raise ValueError(
            "Concrete ready data provenance is required before registration"
        )
    files = {"notebook.ipynb": exp / "notebook.ipynb"}
    generation_records = {}
    for index, dataset in enumerate(manifest["datasets"]):
        if not isinstance(dataset, dict) or not dataset.get("provenance"):
            raise ValueError("Each dataset requires provenance")
        path = contained_path(qpath, dataset.get("path"))
        if not path.is_file():
            raise ValueError("Declared dataset is missing")
        files[f"data/{index}/{path.name}"] = path
        if any(
            word in str(dataset.get("provenance", "")).lower()
            for word in ["synthetic", "generated", "simulated"]
        ):
            generator = contained_path(qpath, dataset.get("generator"))
            if not generator.is_file():
                raise ValueError(
                    "Synthetic data requires a concrete generator artifact"
                )
            files[f"generators/{index}/{generator.name}"] = generator
            from sciquest.provenance import verify_generator

            generation_records[str(index)] = verify_generator(
                qpath, dataset, exp / f"logs/generator_{index}.log"
            )
            for relative in dataset.get("generator_inputs", []):
                files[f"generator_inputs/{index}/{relative}"] = contained_path(
                    qpath, relative
                )
    if not files["notebook.ipynb"].is_file():
        raise ValueError("Notebook is missing")
    stack = load_quest_method_stack(qpath)
    registry = MethodRegistry.default()
    profiles = {method: asdict(registry.get(method)) for method in stack.methods_used}
    if plan["confirmatory_status"] == "confirmatory":
        for previous in (qpath / "experiments").glob("exp_*"):
            if (
                previous.name != exp_id
                and (previous / "validation_results.yaml").is_file()
                and not get_record(qpath, "registration", previous.name)
            ):
                raise ValueError(
                    "Legacy evaluation feedback cannot establish fresh confirmation; continue as exploratory or start a separate fresh confirmation campaign"
                )
        evaluation_hashes = {
            digest
            for name, digest in (
                (name, file_hash(path)) for name, path in files.items()
            )
            if name.startswith("data/")
            and manifest["datasets"][int(name.split("/")[1])].get("role")
            == "evaluation"
        }
        if evidence_was_used(qpath, evaluation_hashes):
            raise ValueError(
                "Evaluation evidence has already provided adaptive feedback; use exploratory status or fresh confirmation data"
            )
    from .policies import validate_policy_plan, validate_prior_chain

    policy_contract = validate_policy_plan(stack.to_dict(), plan)
    hashes = {name: file_hash(path) for name, path in files.items()}
    prior_chain = validate_prior_chain(
        qpath, exp_id, plan, policy_contract, hashes, manifest
    )
    payload = {
        "schema_version": 2,
        "registration_status": "registered",
        "experiment_id": exp_id,
        "created_at": utc_now(),
        "plan": plan,
        "validation_suite": suite,
        "data_manifest": manifest,
        "generation_replays": generation_records,
        "method_stack": stack.to_dict(),
        "profiles": profiles,
        "policy_contract": policy_contract,
        "file_hashes": hashes,
    }
    if prior_chain is not None:
        payload["prior_chain"] = prior_chain
    from sciquest.runtime import runtime_manifest

    payload["runtime"] = runtime_manifest()
    payload["registration_hash"] = content_hash(payload)
    snapshot = authority_dir(qpath) / "snapshots" / exp_id
    snapshot.mkdir(parents=True, exist_ok=False)
    try:
        for name, path in files.items():
            target = contained_path(snapshot, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if file_hash(target) != payload["file_hashes"][name]:
                raise RuntimeError(
                    "Input changed while registration was being captured"
                )
            target.chmod(0o400)
        write_yaml(snapshot / "registration.yaml", payload, exclusive=True)
        record(qpath, "registration", exp_id, payload, owner=owner)
    except BaseException:
        # Preserve an interrupted snapshot for diagnosis; never reuse its ID.
        raise
    ledger = {
        **payload,
        "pre_registration": {
            **plan,
            "primary_method": stack.primary_method,
            "profile_versions": {
                key: value["profile_version"] for key, value in profiles.items()
            },
        },
    }
    write_yaml(exp / "method_ledger.yaml", ledger)
    return payload


def classify_result_claim(
    pre_registration: dict[str, Any],
    claim: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
) -> str:
    claim_type = claim.get("claim_type", "confirmatory")
    if claim.get("method_changed_after_results"):
        return "method_violation"
    if claim_type == "exploratory":
        return "exploratory_observation"
    if claim_type == "post_hoc":
        return (
            "post_hoc_reinterpretation"
            if claim.get("labeled_post_hoc")
            else "method_violation"
        )
    if claim_type == "confirmatory":
        plan = pre_registration.get("plan", pre_registration)
        if (
            pre_registration.get("registration_status") != "registered"
            or plan.get("confirmatory_status") != "confirmatory"
        ):
            return "unverified_claim"
        registered = {p["id"] for p in plan.get("predictions", [])}
        prediction_id = claim.get("prediction_id")
        if not evaluation or evaluation.get(
            "registration_hash"
        ) != pre_registration.get("registration_hash"):
            return "unverified_claim"
        if (
            evaluation.get("evidence_source") != "supervisor_computed"
            or not evaluation.get("isolated_execution")
            or evaluation.get("confirmatory_status") != "confirmatory"
        ):
            return "unverified_claim"
        outcome = evaluation.get("prediction_outcomes", {}).get(prediction_id)
        return (
            "confirmatory_result"
            if prediction_id in registered
            and outcome in {"supported", "contradicted", "inconclusive"}
            else "unverified_claim"
        )
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


def classify_report_claims(
    pre_registration: dict[str, Any],
    claims: list[dict[str, Any]],
    evaluation: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "confirmatory_result": [],
        "exploratory_observation": [],
        "post_hoc_reinterpretation": [],
        "method_violation": [],
        "unresolved_anomaly": [],
        "unverified_claim": [],
    }
    for claim in claims:
        if claim.get("claim_type") == "method_violation":
            bucket = "method_violation"
        elif claim.get("claim_type") == "unresolved_anomaly":
            bucket = "unresolved_anomaly"
        else:
            bucket = classify_result_claim(pre_registration, claim, evaluation)
        buckets.setdefault(bucket, []).append(claim)
    return buckets


def append_method_sections_to_report(
    report_path: Path, stack: MethodStack | None = None
) -> None:
    if stack is None:
        stack = resolve_method_stack({"primary_method": "standard_empirical"})
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if "## Method Ledger" not in existing:
        report_path.write_text(
            existing.rstrip() + "\n\n" + build_method_journal_template(stack),
            encoding="utf-8",
        )
