"""Bounded opaque bytes explicitly produced by one campaign's offline analyses.

The supervisor never imports or deserializes these files. Publication follows
termination of the sandbox. Authorization comes from frozen event manifests and
a campaign nonce, never from a content hash or a discovered filename.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

VERSION = "campaign-artifacts-v1.0"


class ArtifactIntegrityError(ValueError):
    """Authoritative storage corruption/orphans require an owner audit."""
_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_ANALYSIS = re.compile(r"a[0-9]{4,8}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ArtifactLimits:
    max_file_bytes: int = 1024 * 1024
    max_analysis_bytes: int = 2 * 1024 * 1024
    max_campaign_bytes: int = 8 * 1024 * 1024
    max_analysis_files: int = 16
    max_campaign_files: int = 64

    def validate(self):
        if any(type(x) is not int or x <= 0 for x in asdict(self).values()):
            raise ValueError("Artifact limits must be positive integers")
        if not (
            self.max_file_bytes <= self.max_analysis_bytes <= self.max_campaign_bytes
            and self.max_analysis_files <= self.max_campaign_files
        ):
            raise ValueError("Artifact limits must be nested")
        return self


def encoded(value):
    """Exact manifest JSON UTF-8 bytes, not content_hash(value)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative_name(name):
    if (
        not isinstance(name, str)
        or len(name.encode()) > 512
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError("Artifact path must be a bounded relative POSIX name")
    parts = name.split("/")
    if not parts or any(not _PART.fullmatch(x) or x in (".", "..") for x in parts):
        raise ValueError("Artifact path has an unsafe component")
    if str(PurePosixPath(name)) != name:
        raise ValueError("Artifact path must be canonical")
    return name


def _read_regular(root, name, limit):
    """Open every component without symlinks; bound reads, not just stat."""
    relative_name(name)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = name.split("/")
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("Artifact must be a regular non-hardlinked file")
            if before.st_size > limit:
                raise ValueError("Artifact exceeds its byte limit")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(limit + 1)
            after = os.fstat(fd)
            if (
                len(data) > limit
                or len(data) != before.st_size
                or (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            ):
                raise ValueError("Artifact changed during collection")
            return data
        finally:
            os.close(fd)
    finally:
        os.close(directory)


def _identity(campaign_id, analysis_id):
    if not isinstance(campaign_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{16,128}", campaign_id):
        raise ValueError("Artifact campaign binding must be an explicit nonce")
    if not isinstance(analysis_id, str) or not _ANALYSIS.fullmatch(analysis_id):
        raise ValueError("Invalid analysis version")


def _collect_locked(
    store, workspace, outputs, *, campaign_id, analysis_id, source_revision,
    available_revision, code_sha256, context_sha256, execution, previous=(),
    limits=None,
):
    """Publish all declared outputs atomically after successful execution.

    Previous manifests must be verified and event-authorized by this campaign.
    Failed analyses publish nothing. Limits bound retained bytes, not arbitrary
    temporary writes while the sandbox is executing.
    """
    limits = (limits or ArtifactLimits()).validate()
    _identity(campaign_id, analysis_id)
    if (
        execution.get("status") != "succeeded"
        or execution.get("returncode") != 0
        or not execution.get("blind_filesystem")
        or not execution.get("isolated")
        or execution.get("network")
    ):
        raise ValueError("Only a terminated successful blind analysis may publish artifacts")
    if (
        type(source_revision) is not int or type(available_revision) is not int
        or source_revision < 1 or available_revision <= source_revision
        or not isinstance(code_sha256, str) or not _SHA.fullmatch(code_sha256)
        or not isinstance(context_sha256, str) or not _SHA.fullmatch(context_sha256)
    ):
        raise ValueError("Invalid artifact source/availability commitment")
    if (
        not isinstance(outputs, list) or len(outputs) > limits.max_analysis_files
        or any(not isinstance(x, str) for x in outputs)
        or len(set(outputs)) != len(outputs)
    ):
        raise ValueError("Declare a unique bounded list of output files")
    old_files, old_bytes = 0, 0
    for manifest in previous:
        if manifest["campaign_id"] != campaign_id:
            raise ValueError("Cross-campaign artifact accounting is forbidden")
        old_files += len(manifest["objects"])
        old_bytes += sum(x["bytes"] for x in manifest["objects"])
    objects, payloads, total = [], [], 0
    for name in outputs:
        relative_name(name)
        if not name.startswith("outputs/"):
            raise ValueError("Publish only explicitly declared files under outputs/")
        logical = relative_name(name[len("outputs/"):])
        try:
            data = _read_regular(workspace, name, limits.max_file_bytes)
        except OSError as exc:
            raise ValueError("Declared artifact missing or unsafe: " + name) from exc
        total += len(data)
        if total > limits.max_analysis_bytes:
            raise ValueError("Declared analysis artifacts exceed aggregate byte limit")
        objects.append(dict(
            id=analysis_id + "/" + logical, name=logical, bytes=len(data),
            sha256=sha(data), validation="opaque_bytes_only",
        ))
        payloads.append(data)
    if old_files + len(objects) > limits.max_campaign_files or old_bytes + total > limits.max_campaign_bytes:
        raise ValueError("Campaign artifact retention allowance exhausted")
    manifest = dict(
        version=VERSION, campaign_id=campaign_id, analysis_id=analysis_id,
        source_revision=source_revision, available_revision=available_revision,
        code_sha256=code_sha256, context_sha256=context_sha256, objects=objects,
    )
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    if store.is_symlink():
        raise ValueError("Artifact store cannot be a symlink")
    destination = store / analysis_id
    if destination.exists() or destination.is_symlink():
        raise ValueError("An immutable artifact version cannot be replaced")
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=store))
    try:
        for obj, data in zip(objects, payloads):
            path = pending / "objects" / obj["name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        raw = encoded(manifest)
        with (pending / "manifest.json").open("xb") as stream:
            stream.write(raw)
        # Existing complete versions are nonempty, so POSIX rename cannot
        # replace them even if two legitimate publishers race.
        pending.rename(destination)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    return dict(manifest=manifest, manifest_sha256=sha(raw))


def collect(store, workspace, outputs, **kwargs):
    """Serialize publication and require the complete preceding authorized catalog.

    An orphaned committed directory or interrupted pending publication is an
    explicit audit condition. It is never silently imported into campaign memory.
    The caller must also hold its session lock through event authorization.
    """
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    if store.is_symlink():
        raise ValueError("Artifact store cannot be a symlink")
    fd = os.open(store / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        previous = list(kwargs.get("previous", ()))
        kwargs["previous"] = previous
        expected = {m["analysis_id"] for m in previous}
        actual = set()
        for path in store.iterdir():
            if path.name == ".lock":
                continue
            if path.is_symlink() or not path.is_dir() or not _ANALYSIS.fullmatch(path.name):
                raise ArtifactIntegrityError("Uncommitted or unexpected artifact storage requires an audit")
            actual.add(path.name)
        if len(expected) != len(previous) or actual != expected:
            raise ArtifactIntegrityError("Artifact accounting needs the complete preceding authorized catalog")
        for manifest in previous:
            try:
                verify(store, dict(manifest=manifest, manifest_sha256=sha(encoded(manifest))),
                       kwargs["campaign_id"], limits=kwargs.get("limits"))
            except (ValueError, OSError, KeyError) as exc:
                raise ArtifactIntegrityError("Prior authorized artifact failed integrity verification") from exc
        return _collect_locked(store, workspace, outputs, **kwargs)
    finally:
        os.close(fd)


def verify(store, record, campaign_id, *, limits=None):
    """Verify the exact event-authorized manifest and every opaque object."""
    limits = (limits or ArtifactLimits()).validate()
    expected = record["manifest"]
    _identity(campaign_id, expected["analysis_id"])
    if expected["campaign_id"] != campaign_id or expected["version"] != VERSION:
        raise ValueError("Artifact belongs to another campaign or version")
    directory = Path(store) / expected["analysis_id"]
    raw = _read_regular(directory, "manifest.json", 128 * 1024)
    if sha(raw) != record["manifest_sha256"] or raw != encoded(expected):
        raise ValueError("Published artifact manifest changed")
    names, total = set(), 0
    for obj in expected["objects"]:
        name = relative_name(obj["name"])
        if name in names or obj["id"] != expected["analysis_id"] + "/" + name:
            raise ValueError("Duplicate or inconsistent artifact identity")
        names.add(name)
        data = _read_regular(directory, "objects/" + name, limits.max_file_bytes)
        if len(data) != obj["bytes"] or sha(data) != obj["sha256"]:
            raise ValueError("Published artifact bytes changed")
        total += len(data)
    if len(names) > limits.max_analysis_files or total > limits.max_analysis_bytes:
        raise ValueError("Published artifact collection exceeds frozen limits")
    return expected


def stage(store, workspace, records, requested, *, campaign_id, at_revision, limits=None):
    """Copy explicit IDs authorized by this campaign's preceding events."""
    limits = (limits or ArtifactLimits()).validate()
    if type(at_revision) is not int or at_revision < 1:
        raise ValueError("Artifact retrieval needs an explicit evidence cutoff")
    if (
        not isinstance(requested, list) or any(not isinstance(x, str) for x in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("Artifact inputs must be a unique explicit list")
    available, total, count = {}, 0, 0
    for record in records:
        manifest = verify(store, record, campaign_id, limits=limits)
        if manifest["available_revision"] > at_revision:
            raise ValueError("Artifact was not yet available at this evidence cutoff")
        for obj in manifest["objects"]:
            if obj["id"] in available:
                raise ValueError("Repeated artifact version in history")
            available[obj["id"]] = obj
            total += obj["bytes"]
            count += 1
    if total > limits.max_campaign_bytes or count > limits.max_campaign_files:
        raise ValueError("Authorized artifact history exceeds frozen campaign limits")
    if any(x not in available for x in requested):
        raise ValueError("Artifact input is not available in this campaign at this cutoff")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if workspace.is_symlink():
        raise ValueError("Artifact staging workspace cannot be a symlink")
    # A caller cannot substitute a symlink or a populated tree for this namespace.
    # No scientist process may execute concurrently with staging.
    artifact_root = workspace / "artifacts"
    try:
        artifact_root.mkdir()
    except FileExistsError as exc:
        raise ValueError("Artifact staging needs a fresh exclusive destination namespace") from exc
    copied = []
    for identity in requested:
        obj = available[identity]
        analysis_id, name = identity.split("/", 1)
        data = _read_regular(Path(store) / analysis_id, "objects/" + name, limits.max_file_bytes)
        if sha(data) != obj["sha256"]:
            raise ValueError("Artifact changed before staging")
        target_name = "artifacts/" + identity
        target = workspace / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        # The caller supplies a fresh workspace which has never executed code.
        with target.open("xb") as stream:
            stream.write(data)
        copied.append(dict(**obj, path=target_name))
    return copied
