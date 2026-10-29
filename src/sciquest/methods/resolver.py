from __future__ import annotations

from pathlib import Path
from typing import Any

from sciquest.io import content_hash, finite_number, read_yaml
from sciquest.state import get_record, list_records

from .registry import MethodRegistry
from .schema import METHOD_PHASES, MethodStack


def resolve_method_stack(
    selection: str | dict[str, Any] | None = None,
    registry: MethodRegistry | None = None,
) -> MethodStack:
    registry = registry or MethodRegistry.default()
    if selection is None:
        selection = {"primary_method": "standard_empirical", "mode": "simple"}
    if isinstance(selection, str):
        selection = {"primary_method": selection, "mode": "simple"}
    if not isinstance(selection, dict):
        raise ValueError("Method stack must be a name or mapping")
    if set(selection) - {"primary_method", "phases", "mode", "rationale"}:
        raise ValueError("Unknown method stack fields")

    primary = selection.get("primary_method", "standard_empirical")
    if not isinstance(primary, str) or not primary.strip():
        raise ValueError("primary_method must be a nonempty method ID")
    registry.get(primary)
    mode = selection.get("mode", ("advanced" if selection.get("phases") else "simple"))
    provided = selection.get("phases", {})
    if not isinstance(provided, dict) or set(provided) - set(METHOD_PHASES):
        raise ValueError(
            "Unknown method phase; use the supported research-cycle phases"
        )
    if not isinstance(mode, str) or mode not in {"simple", "advanced", "auto"}:
        raise ValueError("Method mode must be simple, advanced, or auto")
    if not isinstance(selection.get("rationale", ""), str):
        raise ValueError("Method rationale must be text")
    phases: dict[str, str] = {}
    for phase in METHOD_PHASES:
        method_id = provided.get(phase, primary)
        if not isinstance(method_id, str) or not method_id.strip():
            raise ValueError("Phase assignments must be nonempty method IDs")
        registry.get(method_id)
        phases[phase] = method_id
    return MethodStack(
        primary_method=primary,
        phases=phases,
        mode=mode,
        rationale=selection.get("rationale", ""),
    )


def _history_signals(quest_path: Path | None) -> dict[str, Any]:
    if not quest_path or not (quest_path / ".sciquest/state.sqlite3").is_file():
        return {"anomaly_count": 0, "declining_scores": False, "text": ""}
    groups: dict[str, list[float]] = {}
    # Complete records are newest first; compare only within a registered group.
    for expid, completion in reversed(
        list_records(quest_path, "completion", limit=100)
    ):
        evaluation = get_record(quest_path, "evaluation", expid)
        registration = get_record(quest_path, "registration", expid)
        if (
            not evaluation
            or not registration
            or completion.get("evaluation_hash") != content_hash(evaluation)
            or evaluation.get("registration_hash")
            != registration.get("registration_hash")
            or evaluation.get("evidence_source") != "supervisor_computed"
            or evaluation.get("validation_status") != "scored"
        ):
            continue
        score = evaluation.get("aggregate_score")
        if score is not None:
            groups.setdefault(evaluation["comparison_group"], []).append(
                finite_number(score)
            )
    declining = any(
        len(scores) >= 3 and scores[-3] > scores[-2] > scores[-1]
        for scores in groups.values()
    )
    return {
        "anomaly_count": 0,
        "declining_scores": declining,
        "text": "",
    }


def auto_recommend_method_stack(
    quest_metadata: dict[str, Any], registry: MethodRegistry | None = None
) -> MethodStack:
    quest_path_value = quest_metadata.get("quest_path")
    quest_path = Path(quest_path_value) if quest_path_value else None
    if quest_path and quest_path.exists():
        quest_file_metadata = read_yaml(quest_path / "quest.yaml", {})
        explicit_metadata = {
            k: v
            for k, v in quest_metadata.items()
            if k == "quest_path" or v not in (None, "")
        }
        quest_metadata = {**quest_file_metadata, **explicit_metadata}
    history = _history_signals(quest_path)
    text = " ".join(
        str(quest_metadata.get(k, ""))
        for k in (
            "hero_statement",
            "problem_statement",
            "initial_hypothesis",
            "description",
        )
    ).lower()
    if any(
        word in text
        for word in [
            "intervention",
            "causal",
            "cause",
            "effect",
            "treatment",
            "counterfactual",
            "confound",
        ]
    ):
        phases = {
            "problem_framing": "causal_interventionist",
            "experiment_design": "causal_interventionist",
            "evidence_evaluation": "causal_interventionist",
            "stress_testing": "popperian_falsificationist",
        }
        if (
            history["anomaly_count"] >= 2
            or history["declining_scores"]
            or "anomal" in history["text"]
        ):
            phases["stress_testing"] = "kuhnian"
            phases["long_term_tracking"] = "lakatosian"
        return resolve_method_stack(
            {
                "mode": "auto",
                "primary_method": "causal_interventionist",
                "phases": phases,
                "rationale": "Quest language suggests treatment/effect/counterfactual reasoning; accumulated history adjusts stress testing and long-term tracking.",
            },
            registry,
        )
    if history["anomaly_count"] >= 2 or history["declining_scores"]:
        return resolve_method_stack(
            {
                "mode": "auto",
                "primary_method": "lakatosian",
                "phases": {
                    "stress_testing": "kuhnian",
                    "long_term_tracking": "lakatosian",
                },
                "rationale": "Comparable supervisor-evaluated scores declined over the last three tests. Heuristic suggestion to review assumptions; this does not establish a paradigm crisis or a degenerating research programme.",
            },
            registry,
        )
    if any(
        word in text
        for word in ["mechanism", "mechanistic", "process", "pathway", "component"]
    ):
        return resolve_method_stack(
            {
                "mode": "auto",
                "primary_method": "mechanistic",
                "rationale": "Quest language asks for mechanism or process explanation.",
            },
            registry,
        )
    if any(
        word in text
        for word in [
            "rival",
            "multiple explanation",
            "competing",
            "different explanations",
        ]
    ):
        return resolve_method_stack(
            {
                "mode": "auto",
                "primary_method": "strong_inference",
                "rationale": "Quest has multiple plausible explanations to discriminate.",
            },
            registry,
        )
    if any(
        word in text
        for word in [
            "noisy",
            "uncertain",
            "confidence",
            "partial",
            "probability",
            "probabilistic",
        ]
    ):
        return resolve_method_stack(
            {
                "mode": "auto",
                "primary_method": "bayesian",
                "rationale": "Quest emphasizes uncertainty and confidence updating.",
            },
            registry,
        )
    if any(
        word in text for word in ["stale", "patch", "anomaly", "paradigm", "crisis"]
    ):
        return resolve_method_stack(
            {
                "mode": "auto",
                "primary_method": "lakatosian",
                "phases": {"stress_testing": "kuhnian"},
                "rationale": "Quest mentions anomalies or research-program drift.",
            },
            registry,
        )
    if any(
        word in text
        for word in ["ambiguous", "early", "explore", "pattern", "vague", "unknown"]
    ):
        return resolve_method_stack(
            {
                "mode": "auto",
                "primary_method": "exploratory_inductive",
                "phases": {
                    "hypothesis_generation": "abductive",
                    "evidence_evaluation": "exploratory_inductive",
                    "stress_testing": "standard_empirical",
                },
                "rationale": "Early ambiguous topic: explore patterns, then infer best explanations.",
            },
            registry,
        )
    return resolve_method_stack(
        {
            "mode": "auto",
            "primary_method": "standard_empirical",
            "rationale": "Default general-purpose empirical loop.",
        },
        registry,
    )
