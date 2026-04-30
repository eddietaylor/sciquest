from __future__ import annotations

from typing import Any

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


def auto_recommend_method_stack(quest_metadata: dict[str, Any], registry: MethodRegistry | None = None) -> MethodStack:
    text = " ".join(str(quest_metadata.get(k, "")) for k in ("hero_statement", "problem_statement", "initial_hypothesis", "description")).lower()
    if any(word in text for word in ["intervention", "causal", "cause", "effect", "treatment", "counterfactual", "confound"]):
        return resolve_method_stack({
            "mode": "auto",
            "primary_method": "causal_interventionist",
            "phases": {"problem_framing": "causal_interventionist", "experiment_design": "causal_interventionist", "evidence_evaluation": "causal_interventionist", "stress_testing": "popperian_falsificationist"},
            "rationale": "Quest language suggests treatment/effect/counterfactual reasoning.",
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
