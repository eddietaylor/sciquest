from __future__ import annotations

import math
import operator
from copy import deepcopy
from pathlib import Path
from typing import Any

from .decisions import DECISION_KINDS, validate_decision_config
from .io import (
    content_hash,
    experiment_dir,
    finite_number,
    read_yaml,
    utc_now,
    write_yaml,
)

COMPARATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}


def _as_float(value: Any) -> float:
    return finite_number(value, "Metric value")


def validate_suite(suite: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(suite, dict) or suite.get("status", "ready") != "ready":
        raise ValueError("Validation suite must be a ready mapping")
    if not isinstance(suite.get("metrics"), list) or not suite["metrics"]:
        raise ValueError("Validation suite must contain metrics")
    clean = deepcopy(suite)
    clean["schema_version"] = 1
    names: set[str] = set()
    for metric in clean["metrics"]:
        if not isinstance(metric, dict):
            raise ValueError("Every metric must be a mapping")
        name = metric.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Metric names must be nonempty and unique")
        names.add(name)
        role = metric.get("role", "primary")
        if not isinstance(role, str) or role not in {
            "primary",
            "diagnostic",
            "constraint",
        }:
            raise ValueError(f"Invalid metric role for {name}")
        cfg = metric.get("computation", {})
        kind = cfg.get("kind") if isinstance(cfg, dict) else None
        if kind is not None and not isinstance(kind, str):
            raise ValueError(f"{name}: computation kind must be a string")
        default_direction = (
            cfg.get("objective", "maximize")
            if kind == "policy_value"
            else "minimize"
            if kind in DECISION_KINDS
            else "maximize"
        )
        direction = metric.setdefault("direction", default_direction)
        if not isinstance(direction, str) or direction not in {
            "maximize",
            "minimize",
            "target",
        }:
            raise ValueError(f"Invalid direction for {name}: {direction!r}")
        if kind in DECISION_KINDS:
            validate_decision_config(cfg, direction)
        metric["weight"] = finite_number(
            metric.get("weight", 1 if role == "primary" else 0), f"{name}.weight"
        )
        if role == "primary" and metric["weight"] <= 0:
            raise ValueError(f"{name}.weight must be positive")
        if role != "primary" and metric["weight"] != 0:
            raise ValueError(f"{name}: diagnostic and constraint weights must be zero")
        if role == "constraint":
            requirement = metric.get("requirement")
            if (
                not isinstance(requirement, dict)
                or not isinstance(requirement.get("operator"), str)
                or requirement["operator"] not in COMPARATORS
            ):
                raise ValueError(
                    f"{name}: constraint requires an operator and threshold"
                )
            finite_number(requirement.get("threshold"), f"{name}.threshold")
        elif "requirement" in metric:
            raise ValueError(
                f"{name}: requirement is only valid for constraint metrics"
            )
        norm = metric.get("normalization", {"type": "identity"})
        norm = {"type": norm} if isinstance(norm, str) else norm
        if (
            not isinstance(norm, dict)
            or not isinstance(norm.get("type", "identity"), str)
            or norm.get("type", "identity") not in {"identity", "minmax", "target"}
        ):
            raise ValueError(f"Invalid normalization for {name}")
        norm.setdefault("type", "identity")
        if norm["type"] == "minmax":
            for key in ("min", "max"):
                norm[key] = finite_number(norm.get(key), f"{name}.{key}")
            if norm["max"] <= norm["min"]:
                raise ValueError(f"{name}: max must exceed min")
            if not math.isfinite(norm["max"] - norm["min"]):
                raise ValueError(f"{name}: normalization interval overflowed")
        if norm["type"] == "target" or direction == "target":
            if norm["type"] != "target" or direction != "target":
                raise ValueError(
                    f"{name}: target direction requires target normalization"
                )
            norm["target"] = finite_number(
                norm.get("target", metric.get("target")), f"{name}.target"
            )
            norm["tolerance"] = finite_number(
                norm.get("tolerance"), f"{name}.tolerance"
            )
            if norm["tolerance"] <= 0:
                raise ValueError(f"{name}.tolerance must be positive")
        metric["normalization"] = norm
        bounds = metric.get("bounds")
        if bounds is not None:
            if not isinstance(bounds, dict) or set(bounds) != {"min", "max"}:
                raise ValueError(f"{name}.bounds requires min and max")
            bounds["min"] = finite_number(bounds["min"])
            bounds["max"] = finite_number(bounds["max"])
            if bounds["min"] > bounds["max"]:
                raise ValueError(f"{name}: invalid bounds")
    try:
        total_weight = math.fsum(m["weight"] for m in clean["metrics"])
    except OverflowError as exc:
        raise ValueError("Total metric weight must be finite") from exc
    if not math.isfinite(total_weight):
        raise ValueError("Total metric weight must be finite")
    return clean


def normalize_value(value: float, metric: dict[str, Any]) -> float:
    metric = validate_suite({"metrics": [metric]})["metrics"][0]
    value = finite_number(value, metric["name"])
    bounds = metric.get("bounds")
    if bounds and not bounds["min"] <= value <= bounds["max"]:
        raise ValueError(f"{metric['name']} is outside its physical bounds")
    norm = metric["normalization"]
    if norm["type"] == "target":
        score = 1.0 - abs(value - norm["target"]) / norm["tolerance"]
    else:
        score = (
            (value - norm["min"]) / (norm["max"] - norm["min"])
            if norm["type"] == "minmax"
            else value
        )
        if metric["direction"] == "minimize":
            score = 1.0 - score
    if not math.isfinite(score):
        raise ValueError(f"{metric['name']} normalization overflowed")
    return max(0.0, min(1.0, score))


def compute_aggregate_score(
    suite: dict[str, Any], results: dict[str, Any]
) -> dict[str, Any]:
    suite = validate_suite(suite)
    if not isinstance(results, dict):
        raise ValueError("Validation results must be a mapping")
    raw_metrics = results.get("metrics", results)
    if not isinstance(raw_metrics, dict):
        raise ValueError("Results metrics must be a mapping")
    undefined = results.get("undefined_metrics", {}) if "metrics" in results else {}
    names = {m["name"] for m in suite["metrics"]}
    if (
        not isinstance(undefined, dict)
        or set(undefined) - names
        or any(
            not isinstance(reason, str) or not reason.strip()
            for reason in undefined.values()
        )
        or set(undefined) & raw_metrics.keys()
    ):
        raise ValueError("Invalid or conflicting undefined metric evidence")
    missing = names - raw_metrics.keys() - undefined.keys()
    if missing:
        raise ValueError(f"Missing required metrics: {', '.join(sorted(missing))}")
    for name, raw in raw_metrics.items():
        finite_number(raw, str(name))
    scores: dict[str, dict[str, float]] = {}
    constraints = {}
    for metric in suite["metrics"]:
        name = metric["name"]
        role = metric.get("role", "primary")
        if name in undefined:
            if role == "constraint":
                constraints[name] = "inconclusive"
            continue
        raw = _as_float(raw_metrics[name])
        bounds = metric.get("bounds")
        if bounds and not bounds["min"] <= raw <= bounds["max"]:
            raise ValueError(f"{name} is outside its physical bounds")
        if role == "constraint":
            rule = metric["requirement"]
            constraints[name] = (
                "passed"
                if COMPARATORS[rule["operator"]](raw, rule["threshold"])
                else "failed"
            )
        if role != "primary":
            continue
        normalized = normalize_value(raw, metric)
        weight = _as_float(metric.get("weight", 1.0))
        scores[name] = {"raw": raw, "normalized": normalized, "weight": weight}
    total_weight = math.fsum(s["weight"] for s in scores.values())
    aggregate = (
        math.fsum(
            s["normalized"] * (s["weight"] / total_weight) for s in scores.values()
        )
        if scores
        else None
    )
    status = "scored" if scores else "evaluated"
    if any(
        m["name"] in undefined and m.get("role", "primary") != "diagnostic"
        for m in suite["metrics"]
    ):
        aggregate, status = None, "inconclusive"
    if "failed" in constraints.values():
        aggregate, status = None, "constraint_failed"
    return {
        "aggregate_score": aggregate,
        "scores": scores,
        "suite_hash": content_hash(suite),
        "validation_status": status,
        "metric_roles": {m["name"]: m.get("role", "primary") for m in suite["metrics"]},
        "constraint_outcomes": constraints,
    }


def validate_experiment(quest_path: Path, experiment_id: str) -> dict[str, Any]:
    from .state import get_record

    exp = experiment_dir(quest_path, experiment_id)
    if not exp.is_dir():
        raise FileNotFoundError(f"Experiment does not exist: {experiment_id}")
    registration = get_record(quest_path, "registration", experiment_id)
    suite = (
        registration["validation_suite"]
        if registration
        else read_yaml(quest_path / "validation.yaml", {})
    )
    results_path = exp / "validation_results.yaml"
    if not results_path.is_file():
        raise FileNotFoundError(
            "validation_results.yaml is missing; no evidence to score"
        )
    results = read_yaml(results_path)
    scored = compute_aggregate_score(suite, results)
    merged = {
        **results,
        **scored,
        "scored_at": utc_now(),
        "evidence_source": "agent_reported",
        "scientific_status": "unverified",
    }
    if get_record(quest_path, "evaluation", experiment_id):
        raise RuntimeError(
            "This experiment already has an immutable supervisor evaluation"
        )
    write_yaml(results_path, merged)
    return merged
