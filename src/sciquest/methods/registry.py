from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from .schema import MethodProfile


class MethodRegistry:
    """Data-driven registry for philosophy-of-science method profiles."""

    def __init__(self, profiles: dict[str, MethodProfile]):
        self._profiles = profiles

    @classmethod
    def default(cls) -> "MethodRegistry":
        package = resources.files("sciquest.methods.profiles")
        profiles: dict[str, MethodProfile] = {}
        for resource in package.iterdir():
            if resource.name.endswith((".yaml", ".yml")):
                data = yaml.safe_load(resource.read_text(encoding="utf-8"))
                profile = MethodProfile.from_dict(data)
                profiles[profile.id] = profile
        return cls(profiles)

    @classmethod
    def from_directory(cls, directory: Path) -> "MethodRegistry":
        profiles: dict[str, MethodProfile] = {}
        for path in sorted(directory.glob("*.y*ml")):
            profile = MethodProfile.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
            profiles[profile.id] = profile
        return cls(profiles)

    def get(self, method_id: str) -> MethodProfile:
        try:
            return self._profiles[method_id]
        except KeyError as exc:
            choices = ", ".join(sorted(self._profiles))
            raise KeyError(f"Unknown SciQuest method '{method_id}'. Available: {choices}") from exc

    def list_profiles(self) -> list[MethodProfile]:
        return [self._profiles[k] for k in sorted(self._profiles)]

    def ids(self) -> list[str]:
        return sorted(self._profiles)
