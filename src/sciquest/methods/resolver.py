from __future__ import annotations

from pathlib import Path
from typing import Any

from sciquest.io import read_yaml

from .registry import MethodRegistry
from .schema import METHOD_PHASES, MethodStack


def resolve_method_stack(selection: str | dict[str, Any] | None = None, registry: MethodRegistry | None = None) -> MethodStack:
    registry = registry or MethodRegistry.default()
    if selection is None:
        selection = {"primary_method": "standard_empirical", "mode": "simple"}
    if isinstance(selection, str):
        selection = {"primary_method": selection, "mode": "simple"}

    primary = selection.get("primary_method") or "standard_empirical"
    registry.get(primary)
    mode = selection.get("mode") or ("advanced" if selection.get("phases") else "simple")
    provided = selection.get("phases") or {}
    phases: dict[str, str] = {}
    for phase in METHOD_PHASES:
        method_id = provided.get(phase, primary)
        registry.get(method_id)
        phases[phase] = method_id
    return MethodStack(primary_method=primary, phases=phases, mode=mode, rationale=selection.get("rationale", ""))


def _history_signals(quest_path: Path | None) -> dict[str, Any]:
    if not quest_path:
        return {"anomaly_count": 0, "declining_scores": False, "text": ""}
    state = read_yaml(quest_path / "state.yaml", {})
    text_parts = [str(item) for item in state.get("failures", []) or []]
    anomaly_count = sum("anomal" in part.lower() for part in text_parts)
    scores: list[float] = []
    for exp in sorted((quest_path / "experiments").glob("exp_*")) if (quest_path / "experiments").exists() else []:
        validation = read_yaml(exp / "validation_results.yaml", {})
        score = validation.get("aggregate_score")
        try:
            if score is not None:
                scores.append(float(score))
        except (TypeError, ValueError):
            pass
        ledger = read_yaml(exp / "method_ledger.yaml", {})
        classifications = ledger.get("post_experiment_classification", {}) or {}
        unresolved = classifications.get("unresolved_anomaly") or []
        anomaly_count += len(unresolved)
        text_parts.append(str(classifications))
    declining = len(scores) >= 3 and all(a > b for a, b in zip(scores, scores[1:]))
    return {"anomaly_count": anomaly_count, "declining_scores": declining, "text": " ".join(text_parts).lower()}


def auto_recommend_method_stack(quest_metadata: dict[str, Any], registry: MethodRegistry | None = None) -> MethodStack:
    quest_path_value = quest_metadata.get("quest_path")
    quest_path = Path(quest_path_value) if quest_path_value else None
    if quest_path and quest_path.exists():
        quest_file_metadata = read_yaml(quest_path / "quest.yaml", {})
        explicit_metadata = {k: v for k, v in quest_metadata.items() if k == "quest_path" or v not in (None, "")}
        quest_metadata = {**quest_file_metadata, **explicit_metadata}
    history = _history_signals(quest_path)
    text = " ".join(str(quest_metadata.get(k, "")) for k in ("hero_statement", "problem_statement", "initial_hypothesis", "description")).lower()
    if any(word in text for word in ["intervention", "causal", "cause", "effect", "treatment", "counterfactual", "confound"]):
        phases = {"problem_framing": "causal_interventionist", "experiment_design": "causal_interventionist", "evidence_evaluation": "causal_interventionist", "stress_testing": "popperian_falsificationist"}
        if history["anomaly_count"] >= 2 or history["declining_scores"] or "anomal" in history["text"]:
            phases["stress_testing"] = "kuhnian"
            phases["long_term_tracking"] = "lakatosian"
        return resolve_method_stack({
            "mode": "auto",
            "primary_method": "causal_interventionist",
            "phases": phases,
            "rationale": "Quest language suggests treatment/effect/counterfactual reasoning; accumulated history adjusts stress testing and long-term tracking.",
        }, registry)
    if history["anomaly_count"] >= 2 or history["declining_scores"]:
        return resolve_method_stack({
            "mode": "auto",
            "primary_method": "lakatosian",
            "phases": {"stress_testing": "kuhnian", "long_term_tracking": "lakatosian"},
            "rationale": "Quest history shows repeated anomalies or a declining validation trajectory; inspect research-program drift and paradigm-level assumptions.",
        }, registry)
    if any(word in text for word in ["mechanism", "mechanistic", "process", "pathway", "component"]):
        return resolve_method_stack({"mode": "auto", "primary_method": "mechanistic", "rationale": "Quest language asks for mechanism or process explanation."}, registry)
    if any(word in text for word in ["rival", "multiple explanation", "competing", "different explanations"]):
        return resolve_method_stack({"mode": "auto", "primary_method": "strong_inference", "rationale": "Quest has multiple plausible explanations to discriminate."}, registry)
    if any(word in text for word in ["noisy", "uncertain", "confidence", "partial", "probability", "probabilistic"]):
        return resolve_method_stack({"mode": "auto", "primary_method": "bayesian", "rationale": "Quest emphasizes uncertainty and confidence updating."}, registry)
    if any(word in text for word in ["stale", "patch", "anomaly", "paradigm", "crisis"]):
        return resolve_method_stack({"mode": "auto", "primary_method": "lakatosian", "phases": {"stress_testing": "kuhnian"}, "rationale": "Quest mentions anomalies or research-program drift."}, registry)
    if any(word in text for word in ["ambiguous", "early", "explore", "pattern", "vague", "unknown"]):
        return resolve_method_stack({
            "mode": "auto",
            "primary_method": "exploratory_inductive",
            "phases": {"hypothesis_generation": "abductive", "evidence_evaluation": "exploratory_inductive", "stress_testing": "standard_empirical"},
            "rationale": "Early ambiguous topic: explore patterns, then infer best explanations.",
        }, registry)
    return resolve_method_stack({"mode": "auto", "primary_method": "standard_empirical", "rationale": "Default general-purpose empirical loop."}, registry)
