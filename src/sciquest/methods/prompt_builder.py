from __future__ import annotations

from .registry import MethodRegistry
from .schema import METHOD_PHASES, MethodStack


def build_method_prompt_addendum(stack: MethodStack, registry: MethodRegistry | None = None) -> str:
    registry = registry or MethodRegistry.default()
    lines = [
        "## Method-aware reasoning",
        "Do not blend all methods into one vague super-method. Apply each method only to its assigned research-cycle phase.",
        f"Method Stack mode: {stack.mode}",
        f"Primary method: {registry.get(stack.primary_method).display_name} ({stack.primary_method})",
        "",
        "Phase governance:",
    ]
    for phase in METHOD_PHASES:
        profile = registry.get(stack.phases[phase])
        lines.append(f"- {phase}: {profile.display_name} — {profile.plain_language_label}")
    lines.append("\nMethod instructions:")
    for method_id in stack.methods_used:
        profile = registry.get(method_id)
        lines.extend([
            f"\n### {profile.display_name} ({method_id})",
            profile.description,
            f"Prompt addendum: {profile.prompt_addendum}",
            "Hypothesis rules: " + "; ".join(profile.hypothesis_generation_rules),
            "Experiment rules: " + "; ".join(profile.experiment_design_rules),
            "Evidence rules: " + "; ".join(profile.evidence_evaluation_rules),
            "Validation rules: " + "; ".join(profile.validation_rules),
            "Next-step policy: " + "; ".join(profile.next_step_policy),
        ])
    lines.extend([
        "",
        "Integrity guardrail:",
        "Before running the experiment, read or create method_ledger.yaml and treat it as a pre-registration. After results are known, label claims as confirmatory_result, exploratory_observation, post_hoc_reinterpretation, method_violation, or unresolved_anomaly.",
    ])
    return "\n".join(lines).strip() + "\n"


def build_method_journal_template(stack: MethodStack, registry: MethodRegistry | None = None) -> str:
    registry = registry or MethodRegistry.default()
    sections: list[str] = []
    for method_id in stack.methods_used:
        for section in registry.get(method_id).journal_sections:
            if section not in sections:
                sections.append(section)
    lines = ["## Method Ledger", "", f"Primary method: {stack.primary_method}", "", "Phase methods:"]
    for phase, method_id in stack.phases.items():
        lines.append(f"- {phase}: {method_id}")
    lines.append("\n## Method-specific Journal Sections")
    for section in sections:
        lines.append(f"\n### {section}\nPending agent entry.")
    return "\n".join(lines).strip() + "\n"
