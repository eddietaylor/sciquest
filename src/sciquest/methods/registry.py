from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from sciquest.io import _UniqueLoader, read_yaml

from .schema import MethodProfile


class MethodRegistry:
    """Data-driven registry for philosophy-of-science method profiles."""

    def __init__(self, profiles: dict[str, MethodProfile]):
        for profile in profiles.values():
            if (
                set(profile.recommended_pairings)
                | set(profile.incompatible_or_tense_with)
            ) - profiles.keys():
                raise ValueError(f"Unknown method reference in {profile.id}")
        self._profiles = profiles

    @classmethod
    def default(cls) -> "MethodRegistry":
        package = resources.files("sciquest.methods.profiles")
        profiles: dict[str, MethodProfile] = {}
        for resource in package.iterdir():
            if resource.name.endswith((".yaml", ".yml")):
                data = yaml.load(
                    resource.read_text(encoding="utf-8"), Loader=_UniqueLoader
                )
                profile = MethodProfile.from_dict(data)
                if profile.id in profiles:
                    raise ValueError(f"Duplicate method profile ID: {profile.id}")
                profiles[profile.id] = profile
        return cls(profiles)

    @classmethod
    def from_directory(cls, directory: Path) -> "MethodRegistry":
        profiles: dict[str, MethodProfile] = {}
        for path in sorted(directory.glob("*.y*ml")):
            profile = MethodProfile.from_dict(read_yaml(path))
            if profile.id in profiles:
                raise ValueError(f"Duplicate method profile ID: {profile.id}")
            profiles[profile.id] = profile
        return cls(profiles)

    def get(self, method_id: str) -> MethodProfile:
        try:
            return self._profiles[method_id]
        except KeyError as exc:
            choices = ", ".join(sorted(self._profiles))
            raise KeyError(
                f"Unknown SciQuest method '{method_id}'. Available: {choices}"
            ) from exc

    def list_profiles(self) -> list[MethodProfile]:
        return [self._profiles[k] for k in sorted(self._profiles)]

    def ids(self) -> list[str]:
        return sorted(self._profiles)
