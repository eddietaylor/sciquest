from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re
import uuid
import json
import math
import os
import tempfile
import hashlib
import fcntl

import yaml


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: yaml.SafeLoader, node: Any, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("YAML mapping keys must be strings")
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or f"quest-{uuid.uuid4().hex[:8]}"


def quests_dir(root: Path) -> Path:
    return root / "quests"


def quest_dir(root: Path, slug: str) -> Path:
    return contained_path(quests_dir(root), slug, component=True)


def contained_path(root: Path, relative: str, *, component: bool = False) -> Path:
    """Reject absolute paths, traversal and symlinks escaping an artifact root."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Expected a nonempty relative path")
    parts = Path(relative).parts
    if any(part in {".", ".."} for part in parts) or (component and len(parts) != 1):
        raise ValueError(f"Unsafe artifact path: {relative!r}")
    base = root.resolve()
    candidate = root / relative
    if not candidate.resolve().is_relative_to(base):
        raise ValueError(f"Artifact escapes its root: {relative!r}")
    return candidate


def experiment_dir(quest_path: Path, experiment_id: str) -> Path:
    if not re.fullmatch(r"exp_[0-9]{3,}", experiment_id):
        raise ValueError(f"Invalid experiment ID: {experiment_id!r}")
    return contained_path(quest_path / "experiments", experiment_id, component=True)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any, label: str = "value") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def atomic_text(path: Path, text: str, *, exclusive: bool = False) -> None:
    """Publish a whole file atomically; exclusive publication never replaces history."""
    if path.is_symlink():
        raise ValueError(f"Refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_yaml(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_UniqueLoader)
    return default if data is None else data


def write_yaml(path: Path, data: Any, *, exclusive: bool = False) -> None:
    atomic_text(
        path,
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        exclusive=exclusive,
    )


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
