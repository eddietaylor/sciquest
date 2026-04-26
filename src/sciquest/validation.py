from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_yaml, write_yaml


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric value {value!r} is not numeric") from exc


def normalize_value(value: float, metric: dict[str, Any]) -> float:
    direction = metric.get("direction", "maximize")
    norm = metric.get("normalization") or {"type": "identity"}
    if isinstance(norm, str):
        norm = {"type": norm}
    typ = norm.get("type", "identity")

    if typ == "minmax":
        lo = _as_float(norm.get("min", 0))
        hi = _as_float(norm.get("max", 1))
        score = 0.0 if hi == lo else (value - lo) / (hi - lo)
    elif typ == "target":
        target = _as_float(norm.get("target", metric.get("target", 0)))
        tolerance = abs(_as_float(norm.get("tolerance", 1))) or 1.0
        score = 1.0 - min(abs(value - target) / tolerance, 1.0)
        return max(0.0, min(1.0, score))
    else:
        score = value

    if direction == "minimize":
        score = 1.0 - score
    elif direction == "target":
        target = _as_float(norm.get("target", metric.get("target", 0)))
        tolerance = abs(_as_float(norm.get("tolerance", 1))) or 1.0
        score = 1.0 - min(abs(value - target) / tolerance, 1.0)
    return max(0.0, min(1.0, score))


def compute_aggregate_score(suite: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    raw_metrics = results.get("metrics", results)
    scores: dict[str, dict[str, float]] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for metric in suite.get("metrics", []) or []:
        name = metric["name"]
        if name not in raw_metrics:
            continue
        raw = _as_float(raw_metrics[name])
        normalized = normalize_value(raw, metric)
        weight = _as_float(metric.get("weight", 1.0))
        scores[name] = {"raw": raw, "normalized": normalized, "weight": weight}
        weighted_sum += normalized * weight
        total_weight += weight
    aggregate = weighted_sum / total_weight if total_weight else 0.0
    return {"aggregate_score": aggregate, "scores": scores}


def validate_experiment(quest_path: Path, experiment_id: str) -> dict[str, Any]:
    suite = read_yaml(quest_path / "validation.yaml", {})
    exp = quest_path / "experiments" / experiment_id
    results_path = exp / "validation_results.yaml"
    results = read_yaml(results_path, {"metrics": {}})
    scored = compute_aggregate_score(suite, results)
    merged = dict(results)
    merged.update(scored)
    write_yaml(results_path, merged)
    return merged
