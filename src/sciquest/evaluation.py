"""Compute metrics from registered truth and prediction artifacts, never claimed numbers."""

from __future__ import annotations

import csv
import math
import operator
from collections import defaultdict
from pathlib import Path
from typing import Any

from .decisions import DECISION_KINDS, evaluate_decisions, validate_decision_config
from .io import contained_path, content_hash, file_hash, finite_number, utc_now
from .state import authority_dir, get_record, record
from .validation import compute_aggregate_score

COMPARATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}


class UndefinedMetric(ValueError):
    """Valid evidence cannot define this statistic; do not invent a numeric score."""


def _number(value: str, label: str) -> float:
    try:
        return finite_number(float(value), label)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric value in {label}") from exc


def _table(path: Path, key: str) -> dict[str, dict[str, str]]:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Tabular evidence exceeds the 64 MiB per-artifact budget")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if (
            not reader.fieldnames
            or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or key not in reader.fieldnames
        ):
            raise ValueError(f"Missing or ambiguous key column {key} in {path.name}")
        rows = {}
        for row in reader:
            if len(rows) >= 1000000:
                raise ValueError("Tabular evidence exceeds the row budget")
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row in {path.name}")
            identity = row[key]
            if not identity or identity in rows:
                raise ValueError(f"Empty or duplicate row ID in {path.name}")
            rows[identity] = row
    if not rows:
        raise ValueError(f"Empty evidence artifact: {path.name}")
    return rows


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for index in order[i:j]:
            ranks[index] = (i + j - 1) / 2
        i = j
    return ranks


def _correlation(a: list[float], b: list[float]) -> float:
    a, b = _rank(a), _rank(b)
    ma, mb = math.fsum(a) / len(a), math.fsum(b) / len(b)
    va, vb = math.fsum((v - ma) ** 2 for v in a), math.fsum((v - mb) ** 2 for v in b)
    if va == 0 or vb == 0:
        raise UndefinedMetric("Rank correlation is undefined for constant rankings")
    return math.fsum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def compute_artifact_metric(
    snapshot: Path,
    outputs: Path,
    metric: dict[str, Any],
    registration: dict[str, Any],
    *,
    decision_evidence: dict | None = None,
) -> float:
    cfg = metric.get("computation")
    if not isinstance(cfg, dict):
        raise ValueError(f"{metric['name']} requires a trusted metric computation")
    kind = cfg.get("kind")
    if (
        kind
        not in {"mae", "rmse", "wape", "rank_correlation", "monotonicity"}
        | DECISION_KINDS
    ):
        raise ValueError(f"Unsupported metric computation: {kind!r}")
    truth_name, prediction_name = (
        cfg.get("truth"),
        cfg.get("artifact", "predictions.csv"),
    )
    if truth_name not in registration["file_hashes"] or not str(truth_name).startswith(
        "data/"
    ):
        raise ValueError("Metric truth must name a registered dataset snapshot")
    key = cfg.get("key", "id")
    truth = _table(contained_path(snapshot, truth_name), key)
    predictions = _table(contained_path(outputs, prediction_name), key)
    if truth.keys() != predictions.keys():
        raise ValueError(
            "Prediction IDs must match the entire registered evaluation dataset"
        )
    if kind in DECISION_KINDS:
        validate_decision_config(
            cfg,
            metric.get(
                "direction", cfg["objective"] if kind == "policy_value" else "minimize"
            ),
        )
        value, detail = evaluate_decisions(truth, predictions, cfg)
        if decision_evidence is not None:
            decision_evidence[metric["name"]] = detail
        return value
    target, pred = cfg.get("target", "actual"), cfg.get("prediction", "prediction")
    actual, predicted = [], []
    groups: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    for identity, row in truth.items():
        if target not in row or pred not in predictions[identity]:
            raise ValueError("Metric columns are missing")
        a = _number(row[target], target)
        p = _number(predictions[identity][pred], pred)
        minimum = cfg.get("prediction_min")
        maximum_column = cfg.get("prediction_max_column")
        if minimum is not None and p < finite_number(minimum):
            raise ValueError("Prediction violates its registered lower bound")
        if maximum_column is not None and (
            maximum_column not in row
            or p > _number(row[maximum_column], maximum_column)
        ):
            raise ValueError("Prediction violates its registered capacity/upper bound")
        actual.append(a)
        predicted.append(p)
        if kind in {"rank_correlation", "monotonicity"}:
            columns = cfg.get("group_by", [])
            if not isinstance(columns, list) or any(col not in row for col in columns):
                raise ValueError("Invalid grouping columns")
            price = cfg.get("order_by")
            if price not in row:
                raise ValueError("Rank/monotonicity checks require an order_by column")
            groups[tuple(row[col] for col in columns)].append(
                (_number(row[price], price), a, p)
            )
    if kind == "mae":
        result = math.fsum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)
    elif kind == "rmse":
        result = math.sqrt(
            math.fsum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
        )
    elif kind == "wape":
        denominator = math.fsum(abs(a) for a in actual)
        if denominator == 0:
            raise UndefinedMetric(
                "WAPE is undefined when total actual magnitude is zero"
            )
        result = math.fsum(abs(a - p) for a, p in zip(actual, predicted)) / denominator
    else:
        scores = []
        for rows in groups.values():
            if len(rows) < 2:
                raise ValueError("Each action group needs at least two observations")
            rows.sort()
            if len({r[0] for r in rows}) != len(rows):
                raise ValueError("Duplicate intervention levels within a group")
            if kind == "rank_correlation":
                scores.append(_correlation([r[1] for r in rows], [r[2] for r in rows]))
            else:
                scores.append(float(all(a[2] >= b[2] for a, b in zip(rows, rows[1:]))))
        result = math.fsum(scores) / len(scores)
    return finite_number(result, metric["name"])


def evaluate_registered(
    qpath: Path,
    exp_id: str,
    outputs: Path,
    execution: dict[str, Any],
    *,
    owner: str | None = None,
) -> dict[str, Any]:
    if get_record(qpath, "evaluation", exp_id):
        raise FileExistsError("Evaluation already finalized")
    registration = get_record(qpath, "registration", exp_id)
    if not registration:
        raise ValueError("No supervisor registration exists")
    if (
        execution.get("status") != "succeeded"
        or execution.get("registration_hash") != registration["registration_hash"]
    ):
        raise ValueError("Successful registered execution is required")
    if get_record(qpath, "execution", exp_id) != execution:
        raise ValueError("Execution evidence must come from the supervisor record")
    snapshot = authority_dir(qpath) / "snapshots" / exp_id
    for name, expected in registration["file_hashes"].items():
        if file_hash(contained_path(snapshot, name)) != expected:
            raise ValueError("Registered input integrity check failed")
    metrics, undefined, decisions = {}, {}, {}
    for metric in registration["validation_suite"]["metrics"]:
        try:
            artifact = metric.get("computation", {}).get("artifact", "predictions.csv")
            expected = execution.get("prediction_artifact_hashes", {}).get(artifact)
            if not expected or file_hash(contained_path(outputs, artifact)) != expected:
                raise ValueError(
                    "Prediction artifacts are not bound to the execution record"
                )
            metrics[metric["name"]] = compute_artifact_metric(
                snapshot, outputs, metric, registration, decision_evidence=decisions
            )
        except UndefinedMetric as exc:
            undefined[metric["name"]] = str(exc)
        except (ArithmeticError, KeyError) as exc:
            raise ValueError(
                f"Invalid numerical evidence for {metric['name']}"
            ) from exc
    scored = compute_aggregate_score(
        registration["validation_suite"],
        {"metrics": metrics, "undefined_metrics": undefined},
    )
    plan = registration["plan"]
    outcomes = {
        p["id"]: "inconclusive"
        if p["metric"] in undefined
        else "supported"
        if COMPARATORS[p["operator"]](metrics[p["metric"]], p["threshold"])
        else "contradicted"
        for p in plan["predictions"]
    }
    from .methods.policies import policy_evidence

    comparison_group = content_hash(
        {
            "suite": scored["suite_hash"],
            "data": {
                key: value
                for key, value in registration["file_hashes"].items()
                if key.startswith("data/")
            },
            "scope": plan["scope"],
        }
    )
    result = {
        **scored,
        "experiment_id": exp_id,
        "metrics": metrics,
        "undefined_metrics": undefined,
        "decision_evidence": decisions,
        "prediction_outcomes": outcomes,
        "registration_hash": registration["registration_hash"],
        "created_at": utc_now(),
        "policy_evidence": policy_evidence(registration, outcomes),
        "comparison_group": comparison_group,
        "evidence_source": "supervisor_computed",
        "isolated_execution": execution.get("isolated", False),
        "confirmatory_status": plan["confirmatory_status"]
        if execution.get("isolated")
        else "exploratory",
        "scientific_status": "registered_rule_evaluated",
        "evaluation_scope": "Registered decision rules; not automatic causal identification or statistical error control",
    }
    record(qpath, "evaluation", exp_id, result, owner=owner)
    return result
