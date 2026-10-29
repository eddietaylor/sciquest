"""Explicit, nonmutating adoption of a frozen v2 bank by new comparison code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from sciquest.io import content_hash, file_hash
from sciquest.runtime import runtime_manifest
from .admission import admission_report
from .collection_v2 import signature
from .worldbank import dump, verify_call

# Exact source equality for every transitive numerical/normalization dependency.
PHYSICS_FILES = (
    "demand/world.py",
    "demand/designed.py",
    "demand/designed_v2.py",
    "demand/admission.py",
    "io.py",
)
VERSION = "v2-comparison-adoption-v1"


def _read_bank(bank):
    bank = Path(bank).resolve()
    generation = json.loads((bank / "generation_protocol.json").read_text())
    manifest = json.loads((bank / "manifest.json").read_text())
    private = json.loads((bank / "private/worlds.json").read_text())
    if (
        content_hash(generation)
        != (bank / "generation_protocol.sha256").read_text().strip()
        or content_hash(manifest) != (bank / "manifest.sha256").read_text().strip()
        or manifest["generation_protocol_hash"] != content_hash(generation)
        or manifest["private_commitment"] != content_hash(private)
        or manifest["worlds"]
        != [dict(id=w["id"], commitment=content_hash(w)) for w in private["worlds"]]
    ):
        raise ValueError("Frozen collection commitments changed")
    return generation, manifest, private


def adopt_collection(path, bank):
    """Replay original verifier under archived source, then match unchanged physics.

    Requires the original interpreter/dependency environment. No runtime guard is
    bypassed, no old session/protocol is migrated, no model call is made.
    """
    bank = Path(bank).resolve()
    generation, manifest, private = _read_bank(bank)
    capsule = bank / "environment/source.tar.gz"
    package = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="sciquest-v2-adoption-") as temp:
        dest = Path(temp)
        with tarfile.open(capsule) as archive:
            members = archive.getmembers()
            for member in members:
                if (
                    Path(member.name).is_absolute()
                    or ".." in Path(member.name).parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("Unsafe frozen source capsule")
            archive.extractall(dest, members=members)
        archived = dest / "src/sciquest"
        physics = {name: file_hash(archived / name) for name in PHYSICS_FILES}
        if physics != {name: file_hash(package / name) for name in PHYSICS_FILES}:
            raise ValueError(
                "Physics changed; explicit new-world validation is required"
            )
        script = "from sciquest.demand.collection_v2 import verify_collection; import json,sys; print(json.dumps(verify_collection(sys.argv[1])))"
        env = {
            **os.environ,
            "PYTHONPATH": str(dest / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [sys.executable, "-c", script, str(bank)],
            cwd=dest,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
            check=True,
        )
        original = json.loads(result.stdout)
    if not original["verified"] or original["manifest_hash"] != content_hash(manifest):
        raise ValueError("Original frozen verifier did not verify this collection")
    for w in private["worlds"]:
        if w["validation"] != admission_report(w["spec"], w["seed"]) or w[
            "signature"
        ] != signature(w["spec"]):
            raise ValueError("Current admission or response signature differs")
    record = dict(
        version=VERSION,
        bank=str(bank),
        manifest_hash=content_hash(manifest),
        original_runtime=generation["runtime"],
        adopted_runtime=runtime_manifest(),
        original_verification=original,
        source_capsule_sha256=file_hash(capsule),
        physics_files=physics,
        scope="Same fixed specs,seeds,physics,actions and original accepted/rejected provenance; new scoring/guidance runtime only. Development data remain development. No old results or sessions upgraded.",
    )
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    dump(path / "adoption.json", record)
    (path / "adoption.sha256").write_text(content_hash(record) + "\n")
    return record


def load_adopted(path):
    path = Path(path).resolve()
    record = json.loads((path / "adoption.json").read_text())
    if (
        record["version"] != VERSION
        or content_hash(record) != (path / "adoption.sha256").read_text().strip()
    ):
        raise ValueError("Adoption commitment changed")
    if record["adopted_runtime"]["fingerprint"] != runtime_manifest()["fingerprint"]:
        raise ValueError(
            "Adopted comparison runtime changed; preserve this adoption and create another explicit adoption"
        )
    package = Path(__file__).resolve().parents[1]
    if record["physics_files"] != {
        name: file_hash(package / name) for name in PHYSICS_FILES
    }:
        raise ValueError("Adopted physics changed")
    generation, manifest, private = _read_bank(record["bank"])
    if (
        record["manifest_hash"] != content_hash(manifest)
        or record["original_runtime"] != generation["runtime"]
        or file_hash(Path(record["bank"]) / "environment/source.tar.gz")
        != record["source_capsule_sha256"]
    ):
        raise ValueError("Original collection or capsule changed")
    for attempt in private["attempts"]:
        directory = Path(record["bank"]) / attempt["call"]
        verify_call(directory)
        if file_hash(directory / "receipt.json") != attempt["receipt_sha256"]:
            raise ValueError("Original designer provenance changed")
    return record, manifest, private
