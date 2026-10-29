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
    profile_version: str = "1.0.0"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MethodProfile":
        if not isinstance(data, dict) or set(data) - cls.__dataclass_fields__.keys():
            raise ValueError("Method profile must be a mapping with known field names")
        values: dict[str, Any] = {}
        for dataclass_field in cls.__dataclass_fields__.values():
            if dataclass_field.name in data:
                values[dataclass_field.name] = data[dataclass_field.name]
            elif dataclass_field.default_factory is not MISSING:  # type: ignore[attr-defined]
                values[dataclass_field.name] = dataclass_field.default_factory()  # type: ignore[misc]
            elif dataclass_field.default is not MISSING:
                values[dataclass_field.name] = dataclass_field.default
            else:
                raise ValueError(
                    f"Method profile missing required field: {dataclass_field.name}"
                )
        for name, value in values.items():
            if name in {
                "id",
                "display_name",
                "plain_language_label",
                "description",
                "profile_version",
                "scoring_transform",
            }:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Method profile {name} requires nonempty text")
            elif name == "prompt_addendum":
                if not isinstance(value, str):
                    raise ValueError("Method prompt_addendum must be text")
            elif not isinstance(value, list) or any(
                not isinstance(v, str) or not v.strip() for v in value
            ):
                raise ValueError(
                    f"Method profile {name} requires a list of nonempty strings"
                )
        if set(values["phase_affinities"]) - set(METHOD_PHASES):
            raise ValueError("Method profile has unknown phase affinities")
        if values["scoring_transform"] not in {"none", "method_lens_metadata_only_v1"}:
            raise ValueError(
                "Method profiles cannot silently transform evaluation scores"
            )
        return cls(**values)


@dataclass(frozen=True)
class MethodStack:
    primary_method: str
    phases: dict[str, str]
    mode: str = "simple"
    rationale: str = ""

    @property
    def methods_used(self) -> list[str]:
        methods = (
            [self.primary_method]
            if not self.phases or self.primary_method in self.phases.values()
            else []
        ) + list(self.phases.values())
        return list(dict.fromkeys(methods))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "primary_method": self.primary_method,
            "phases": dict(self.phases),
            "rationale": self.rationale,
        }
