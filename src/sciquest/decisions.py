"""Evaluate finite action-grid decisions against owner-supplied protected truth."""

from __future__ import annotations

import math
from typing import Any

from .io import finite_number

DECISION_KINDS = {"policy_value", "regret", "constraint_violation_rate"}


def validate_decision_config(cfg: dict, direction: str) -> None:
    kind = cfg["kind"]
    for field in ("target", "prediction", "action"):
        if not isinstance(cfg.get(field), str) or not cfg[field].strip():
            raise ValueError(f"Decision metric requires an explicit {field} column")
    groups = cfg.get("group_by")
    if (
        not isinstance(groups, list)
        or not groups
        or any(not isinstance(v, str) or not v.strip() for v in groups)
        or len(set(groups)) != len(groups)
    ):
        raise ValueError("Decision metrics require unique nonempty group_by columns")
    if cfg.get("objective") not in ("maximize", "minimize"):
        raise ValueError("Decision metrics require objective: maximize or minimize")
    if cfg["action"] in groups:
        raise ValueError("Decision action cannot also be a grouping column")
    if cfg.get("tie_break") not in ("lowest_action", "highest_action"):
        raise ValueError(
            "Decision metrics require an explicit numeric action tie_break"
        )
    expected_direction = cfg["objective"] if kind == "policy_value" else "minimize"
    if direction != expected_direction:
        raise ValueError(f"{kind} requires direction: {expected_direction}")
    if (
        kind != "constraint_violation_rate"
        and cfg.get("value_semantics") != "expected_interventional"
    ):
        raise ValueError(
            "Policy value and regret require value_semantics: expected_interventional"
        )


def _number(value: str, label: str) -> float:
    try:
        return finite_number(float(value), label)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric decision evidence in {label}") from exc


def evaluate_decisions(
    truth: dict[str, dict[str, str]], predictions: dict[str, dict[str, str]], cfg: dict
) -> tuple[float, dict[str, Any]]:
    """Equal weight per state, exact prediction ties, oracle over listed actions.

    The truth owner must supply identified expected intervention values for value
    and regret. Binary violation labels describe whether each action would violate
    the registered constraint. No implicit feasibility filtering is performed.
    """
    if truth.keys() != predictions.keys():
        raise ValueError(
            "Prediction IDs must match the entire registered evaluation dataset"
        )
    grouped: dict[tuple, list] = {}
    for identity, row in truth.items():
        required = [*cfg["group_by"], cfg["target"], cfg["action"]]
        if any(col not in row or row[col] == "" for col in required):
            raise ValueError("Missing decision grouping, action or truth columns")
        if cfg["prediction"] not in predictions[identity]:
            raise ValueError("Decision prediction column is missing")
        target = _number(row[cfg["target"]], cfg["target"])
        prediction = _number(
            predictions[identity][cfg["prediction"]], cfg["prediction"]
        )
        action = _number(row[cfg["action"]], cfg["action"])
        minimum, maximum = cfg.get("prediction_min"), cfg.get("prediction_max_column")
        if minimum is not None and prediction < finite_number(minimum):
            raise ValueError("Decision prediction violates its registered lower bound")
        if maximum is not None and (
            maximum not in row or prediction > _number(row[maximum], maximum)
        ):
            raise ValueError("Decision prediction violates its registered upper bound")
        if cfg["kind"] == "constraint_violation_rate" and target not in (0, 1):
            raise ValueError("Constraint violation truth must be binary 0 or 1")
        group = tuple(row[col] for col in cfg["group_by"])
        grouped.setdefault(group, []).append((action, target, prediction, identity))
    if not grouped:
        raise ValueError("Decision evidence is empty")
    values, selected_values, oracle_values, trace = [], [], [], []
    ties = 0
    for group in sorted(grouped):
        rows = grouped[group]
        if len({r[0] for r in rows}) != len(rows):
            raise ValueError("Duplicate actions within a decision group")
        best = max if cfg["objective"] == "maximize" else min
        predicted_best = best(r[2] for r in rows)
        candidates = [r for r in rows if r[2] == predicted_best]
        ties += len(candidates) > 1
        choose = min if cfg["tie_break"] == "lowest_action" else max
        chosen = choose(candidates, key=lambda r: r[0])
        value = chosen[1]
        detail = {
            "group": list(group),
            "selected_id": chosen[3],
            "action": chosen[0],
            "truth": chosen[1],
        }
        if cfg["kind"] != "constraint_violation_rate":
            oracle = best(r[1] for r in rows)
            gap = finite_number(
                oracle - value if cfg["objective"] == "maximize" else value - oracle,
                "regret",
            )
            selected_values.append(value)
            oracle_values.append(oracle)
            detail.update(oracle_value=oracle, regret=gap)
            if cfg["kind"] == "regret":
                value = gap
        values.append(finite_number(value, cfg["kind"]))
        if len(trace) < 100:
            trace.append(detail)
    result = finite_number(math.fsum(v / len(values) for v in values), cfg["kind"])
    evidence = {
        "objective": cfg["objective"],
        "tie_break": cfg["tie_break"],
        "group_by": cfg["group_by"],
        "action": cfg["action"],
        "group_count": len(grouped),
        "weighting": "equal_per_group",
        "selection_tie_groups": ties,
        "trace_sample": trace,
        "trace_sample_limit": 100,
        "trace_truncated": len(grouped) > len(trace),
        "scope": "Registered finite action grid; truth semantics and causal identification are owner assumptions. No sequential-policy guarantee or implicit feasibility filtering.",
    }
    if selected_values:
        evidence.update(
            selected_value=math.fsum(v / len(values) for v in selected_values),
            oracle_value=math.fsum(v / len(values) for v in oracle_values),
            value_semantics=cfg["value_semantics"],
        )
    return result, evidence
