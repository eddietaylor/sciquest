from __future__ import annotations

from .policies import policy_requirements
from .registry import MethodRegistry
from .schema import METHOD_PHASES, MethodStack


def build_method_prompt_addendum(
    stack: MethodStack, registry: MethodRegistry | None = None
) -> str:
    registry = registry or MethodRegistry.default()
    lines = [
        "## Method-aware reasoning",
        "Do not blend all methods into one vague super-method. Apply each method only to its assigned research-cycle phase.",
        f"Method Stack mode: {stack.mode}",
        f"Default primary method (assigned phases govern): {registry.get(stack.primary_method).display_name} ({stack.primary_method})",
        "",
        "Phase governance:",
    ]
    for phase in METHOD_PHASES:
        profile = registry.get(stack.phases[phase])
        lines.append(
            f"- {phase}: {profile.display_name} — {profile.plain_language_label}"
        )
    lines.append("\nPhase-scoped method instructions:")
    fields = {
        "problem_framing": (),
        "hypothesis_generation": ("hypothesis_generation_rules",),
        "experiment_design": ("experiment_design_rules",),
        "evidence_evaluation": ("evidence_evaluation_rules", "validation_rules"),
        "stress_testing": ("experiment_design_rules", "validation_rules"),
        "long_term_tracking": ("next_step_policy",),
    }
    for phase in METHOD_PHASES:
        profile = registry.get(stack.phases[phase])
        lines.extend(
            [
                f"\n### {phase}: {profile.display_name} ({profile.id})",
                profile.description,
            ]
        )
        for field in fields[phase]:
            lines.append(
                field.replace("_", " ") + ": " + "; ".join(getattr(profile, field))
            )
    capabilities = policy_requirements(stack.to_dict())
    lines.append(
        "Required structured policies in this assignment: "
        + (", ".join(capabilities["required_plan_extensions"]) or "none")
    )
    lines.append(
        "\nStructured contracts apply only as declared in policy_contract.yaml; other assigned phases provide guidance. No automatic information-gain optimization, causal identification or philosophical fidelity is certified."
    )
    lines.extend(
        [
            "",
            "Integrity guardrail:",
            "Write experiment_plan.yaml before execution. The supervisor alone freezes registration and owns method_ledger.yaml, execution, evaluation and history. A draft ledger is not registration. Stop after preparation; interpret later recorded evidence with its registered scope.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_method_journal_template(
    stack: MethodStack, registry: MethodRegistry | None = None
) -> str:
    registry = registry or MethodRegistry.default()
    sections: list[str] = []
    for method_id in stack.methods_used:
        for section in registry.get(method_id).journal_sections:
            if section not in sections:
                sections.append(section)
    lines = [
        "## Method Ledger",
        "",
        f"Primary method: {stack.primary_method}",
        "",
        "Phase methods:",
    ]
    for phase, method_id in stack.phases.items():
        lines.append(f"- {phase}: {method_id}")
    lines.append("\n## Method-specific Journal Sections")
    for section in sections:
        lines.append(f"\n### {section}\nPending agent entry.")
    return "\n".join(lines).strip() + "\n"
