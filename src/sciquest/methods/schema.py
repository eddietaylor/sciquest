from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import Any


METHOD_PHASES = (
    "problem_framing",
    "hypothesis_generation",
    "experiment_design",
    "evidence_evaluation",
    "stress_testing",
    "long_term_tracking",
)


@dataclass(frozen=True)
class MethodProfile:
    id: str
    display_name: str
    plain_language_label: str
    description: str
    best_for: list[str] = field(default_factory=list)
    bad_for: list[str] = field(default_factory=list)
    phase_affinities: list[str] = field(default_factory=list)
    hypothesis_generation_rules: list[str] = field(default_factory=list)
    experiment_design_rules: list[str] = field(default_factory=list)
    evidence_evaluation_rules: list[str] = field(default_factory=list)
    validation_rules: list[str] = field(default_factory=list)
    journal_sections: list[str] = field(default_factory=list)
    scoring_transform: str = "none"
    next_step_policy: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    incompatible_or_tense_with: list[str] = field(default_factory=list)
    recommended_pairings: list[str] = field(default_factory=list)
    prompt_addendum: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MethodProfile":
        values: dict[str, Any] = {}
        for dataclass_field in cls.__dataclass_fields__.values():
            if dataclass_field.name in data:
                values[dataclass_field.name] = data[dataclass_field.name]
            elif dataclass_field.default_factory is not MISSING:  # type: ignore[attr-defined]
                values[dataclass_field.name] = dataclass_field.default_factory()  # type: ignore[misc]
            elif dataclass_field.default is not MISSING:
                values[dataclass_field.name] = dataclass_field.default
            else:
                raise ValueError(f"Method profile missing required field: {dataclass_field.name}")
        return cls(**values)


@dataclass(frozen=True)
class MethodStack:
    primary_method: str
    phases: dict[str, str]
    mode: str = "simple"
    rationale: str = ""

    @property
    def methods_used(self) -> list[str]:
        methods = [self.primary_method, *self.phases.values()]
        return list(dict.fromkeys(methods))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "primary_method": self.primary_method,
            "phases": dict(self.phases),
            "rationale": self.rationale,
        }
