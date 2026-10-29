"""Declared fitted-state structure and finite public component prediction parity.

These checks do not certify arbitrary scientific intent, inference, calibration
or code semantics. The earlier live reference is excluded from the clean checker
workspace. Array payloads and scientist Python are read/executed only in sandboxes.
"""

from __future__ import annotations

import json
import csv
import io
import math
import sys
import tempfile
from pathlib import Path

from sciquest.execution import run_command
from .artifacts_v1 import _read_regular, encoded, relative_name, sha

VERSION = "fitted-state-check-v1.0"
ABS_TOLERANCE = 1e-8
REL_TOLERANCE = 1e-8
MAX_BYTES = 1024 * 1024
RESERVED_CHECKER_PATHS = (
    "fidelity-internal-manifest.json", "fidelity-internal-structure.py", "fidelity-internal-limits.py",
    "fidelity-internal-structure.json", "fidelity-internal-reloaded.json",
)
LIMITED_RUNNER = """import os,resource,sys
resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.execv(sys.executable,[sys.executable,*sys.argv[1:]])
"""


def _json(data):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(data, parse_constant=invalid)


def declaration(root, name, *, source_revision):
    """Validate bounded JSON declarations and file commitments; no array loading."""
    if type(source_revision) is not int or source_revision < 1:
        raise ValueError("Fitted check requires an integer source revision")
    manifest = _json(_read_regular(root, relative_name(name), MAX_BYTES))
    required = {
        "version", "instance_id", "source_revision", "training_cutoff",
        "families", "arrays", "model_files", "probe_file", "live_reference",
        "reload_program", "measurement_assumptions",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("Fitted-state declaration must have exactly these fields, with no unknown fields: " + ",".join(sorted(required)))
    if manifest["version"] != VERSION:
        raise ValueError("Fitted-state version must equal " + VERSION)
    if not isinstance(manifest["instance_id"], str) or not manifest["instance_id"].strip() or len(manifest["instance_id"]) > 128:
        raise ValueError("Fitted-state instance_id must be a nonblank string of at most128 characters")
    if type(manifest["source_revision"]) is not int or manifest["source_revision"] != source_revision:
        raise ValueError(f"Fitted-state source_revision must be the exact JSON integer {source_revision} from context.json; bool, float and string are invalid")
    if type(manifest["training_cutoff"]) is not int or not 1 <= manifest["training_cutoff"] <= source_revision:
        raise ValueError(f"Fitted-state training_cutoff must be an exact JSON integer in1..{source_revision}; use an evidence revision, not an observation ID, object or prose")
    if not isinstance(manifest["measurement_assumptions"], str) or not manifest["measurement_assumptions"].strip():
        raise ValueError("Fitted-state measurement_assumptions must be a nonblank string; list and object are invalid")
    families = manifest["families"]
    if not isinstance(families, list) or not 1 <= len(families) <= 16:
        raise ValueError("Declare1..16 fitted components")
    family_map = {}
    for family in families:
        if not isinstance(family, dict) or set(family) != {"id", "basis"}:
            raise ValueError("Each family needs id and named basis order")
        identity, basis = family["id"], family["basis"]
        if (
            not isinstance(identity, str) or not identity.strip() or identity in family_map
            or not isinstance(basis, list) or len(basis) > 128
            or any(not isinstance(x, str) or not x.strip() for x in basis)
            or len(set(basis)) != len(basis)
        ):
            raise ValueError("Fitted component identities/basis names must be unique")
        family_map[identity] = basis
    files = manifest["model_files"]
    if not isinstance(files, list) or not files or len(files) > 16 or len(set(files)) != len(files):
        raise ValueError("Declare a bounded unique model file list")
    for field in ("probe_file", "live_reference", "reload_program"):
        relative_name(manifest[field])
    for filename in files:
        relative_name(filename)
    if len(set([*files, manifest["probe_file"], manifest["live_reference"], manifest["reload_program"]])) != len(files) + 3:
        raise ValueError("Model, probe, reference and reload program must be distinct files")
    specs = manifest["arrays"]
    if not isinstance(specs, list) or len(specs) > 32:
        raise ValueError("At most32 array declarations are allowed")
    seen = set()
    dimensions = {}
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {"file", "key", "family_id", "shape", "axes", "basis_axis", "role"}:
            raise ValueError("Incomplete fitted-array declaration")
        shape, axes = spec["shape"], spec["axes"]
        family_id = spec["family_id"]
        if (
            spec["file"] not in files or not spec["file"].endswith(".npz")
            or not isinstance(spec["key"], str) or not spec["key"]
            or family_id not in family_map
            or not isinstance(shape, list) or not 1 <= len(shape) <= 4
            or any(type(n) is not int or n < 1 for n in shape)
            or math.prod(shape) > 1000000
            or not isinstance(axes, list) or len(axes) != len(shape)
            or any(not isinstance(x, str) or not x for x in axes)
            or len(set(axes)) != len(axes)
            or spec["role"] not in ("parameters", "particle_weights", "other")
        ):
            raise ValueError("Invalid fitted array axes, dimensions, family or file")
        identity = (spec["file"], spec["key"])
        if identity in seen:
            raise ValueError("An array cannot silently serve two declared component identities")
        seen.add(identity)
        axis = spec["basis_axis"]
        if type(axis) is not int or axis < -1 or axis >= len(shape):
            raise ValueError("Invalid basis axis")
        if axis >= 0 and shape[axis] != len(family_map[family_id]):
            raise ValueError("Array basis dimension does not match declared named basis")
        if spec["role"] == "particle_weights" and (len(shape) != 1 or axes != ["particle"]):
            raise ValueError("Particle weights need one particle axis")
        for label, width in zip(axes, shape):
            key = (family_id, label)
            if key in dimensions and dimensions[key] != width:
                raise ValueError("Component parameter and weight axes disagree")
            dimensions[key] = width
    names = [*files, manifest["probe_file"], manifest["live_reference"], manifest["reload_program"]]
    if name in names:
        raise ValueError("Declaration cannot overwrite one of its declared assets")
    commitments = {n: sha(_read_regular(root, n, MAX_BYTES)) for n in names}
    return manifest, commitments


def validate_probe_result(value, manifest, probe_sha256):
    required = {"instance_id", "source_revision", "training_cutoff", "probe_sha256",
                "families", "component_weights", "probe_ids", "targets", "predictions"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Component probe result must have exactly these fields, with no unknown fields: " + ",".join(sorted(required)))
    for key in ("instance_id", "source_revision", "training_cutoff", "families"):
        if type(value[key]) is not type(manifest[key]) or value[key] != manifest[key]:
            raise ValueError("Component probe " + key + " must exactly match the declaration's type, value and order")
    if value["probe_sha256"] != probe_sha256:
        raise ValueError("Component probe feature bytes changed")
    ids, targets, weights = value["probe_ids"], value["targets"], value["component_weights"]
    if not isinstance(ids, list) or not 1 <= len(ids) <= 128 or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Component probe probe_ids must be1..128 unique nonempty strings in actual probe CSV order")
    if not isinstance(targets, list) or not 1 <= len(targets) <= 16 or any(not isinstance(x, str) or not x for x in targets) or len(set(targets)) != len(targets):
        raise ValueError("Component probe targets must be1..16 unique nonempty strings naming the compared quantities")
    if (
        not isinstance(weights, list) or len(weights) != len(manifest["families"])
        or any(type(w) not in (float, int) or not math.isfinite(w) or w < 0 for w in weights)
        or not math.isclose(sum(weights), 1, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE)
    ):
        raise ValueError("Component probe component_weights must be one finite nonnegative JSON number per family, summing to1 within the host tolerance; bool is invalid")
    predictions = value["predictions"]
    if not isinstance(predictions, list) or len(predictions) != len(weights):
        raise ValueError("One prediction matrix per declared component is required")
    for component in predictions:
        if not isinstance(component, list) or len(component) != len(ids):
            raise ValueError("Component probe row count differs")
        for row in component:
            if (
                not isinstance(row, list) or len(row) != len(targets)
                or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e12 for x in row)
            ):
                raise ValueError("Component probe predictions must be finite bounded numeric matrices")
    return value


def compare(reference, reloaded, manifest, probe_sha256):
    for value in (reference, reloaded):
        validate_probe_result(value, manifest, probe_sha256)
    for key in ("probe_ids", "targets"):
        if reference[key] != reloaded[key]:
            raise ValueError("Reload reordered component probe rows or targets")
    for a, b in zip(reference["component_weights"], reloaded["component_weights"]):
        if not math.isclose(a, b, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE):
            raise ValueError("Reload changed declared component weights")
    maximum = 0.0
    for left, right in zip(reference["predictions"], reloaded["predictions"]):
        for lrow, rrow in zip(left, right):
            for a, b in zip(lrow, rrow):
                maximum = max(maximum, abs(a-b))
                if not math.isclose(a, b, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE):
                    raise ValueError("Reload changes a live component prediction")
    return maximum


def check_bundle(root, name, directory, *, source_revision, expected_probe_sha256=None, timeout=30):
    """Retain a one-shot bounded structural/parity check, including failures."""
    root, directory = Path(root), Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(version=VERSION, status="failed", error=None, checks=[], verified_quantities=[], output_sha256={},
                  tolerance=dict(absolute=ABS_TOLERANCE, relative=REL_TOLERANCE),
                  scope="Declared structure and finite live-to-reload component parity only; not arbitrary intent, inference, scientific validity or calibration.")
    try:
        manifest, commitments = declaration(root, name, source_revision=source_revision)
        record.update(declaration=manifest, input_sha256=commitments,
                      declaration_sha256=sha(_read_regular(root, name, MAX_BYTES)))
        commitments[name] = record["declaration_sha256"]
        reference = _json(_read_regular(root, manifest["live_reference"], MAX_BYTES))
        validate_probe_result(reference, manifest, commitments[manifest["probe_file"]])
        record["requested_quantities"] = reference["targets"]
        record["quantity_scope"] = (
            "Only the listed finite probe quantities are compared. Means alone "
            "do not establish uncertainty or distribution fidelity. Unlisted "
            "moments, dependence, draws and transforms remain unverified."
        )
        if expected_probe_sha256 is not None and commitments[manifest["probe_file"]] != expected_probe_sha256:
            raise ValueError("Fitted check changed the prospectively supplied public probes")
        probes = list(csv.DictReader(io.StringIO(_read_regular(root, manifest["probe_file"], MAX_BYTES).decode("utf-8"))))
        if [row.get("probe_id") for row in probes] != reference["probe_ids"]:
            raise ValueError("Reference probe IDs differ from actual probe feature rows")
        reference_bytes = encoded(reference)
        (directory / "live_reference.json").write_bytes(reference_bytes)
        record["output_sha256"]["live_reference.json"] = sha(reference_bytes)
        with tempfile.TemporaryDirectory(prefix="sciquest-fitted-check-") as temp:
            work = Path(temp)
            # The earlier reference is deliberately not provided to the reload.
            staged = [name, *manifest["model_files"], manifest["probe_file"], manifest["reload_program"]]
            reserved = set(RESERVED_CHECKER_PATHS)
            if any(n in reserved or n.split("/")[0] in reserved for n in staged):
                raise ValueError("Fitted files collide with reserved checker paths")
            for filename in staged:
                target = work / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_read_regular(root, filename, MAX_BYTES))
            (work / "fidelity-internal-manifest.json").write_bytes(encoded(manifest))
            (work / "fidelity-internal-structure.py").write_bytes(Path(__file__).with_name("fidelity_worker_v1.py").read_bytes())
            (work / "fidelity-internal-limits.py").write_text(LIMITED_RUNNER)
            commands = [
                ("structure", ["-I", "fidelity-internal-structure.py", "fidelity-internal-manifest.json", "fidelity-internal-structure.json"]),
                ("reload", [manifest["reload_program"], manifest["probe_file"], "fidelity-internal-reloaded.json"]),
            ]
            for label, arguments in commands:
                log = directory / (label + ".log")
                execution = run_command(
                    [sys.executable, "-I", "fidelity-internal-limits.py", *arguments], work, log,
                    hidden_root=directory, blind=True, timeout=timeout, max_file_bytes=MAX_BYTES,
                )
                record["checks"].append(dict(kind=label, execution=execution, log_sha256=sha(log.read_bytes())))
                if execution["status"] != "succeeded":
                    raise ValueError("Fitted-state " + label + " check " + execution["status"])
                output = "fidelity-internal-structure.json" if label == "structure" else "fidelity-internal-reloaded.json"
                data = _read_regular(work, output, MAX_BYTES)
                (directory / (label + ".json")).write_bytes(data)
                record["output_sha256"][label + ".json"] = sha(data)
                for filename in staged:
                    if sha(_read_regular(work, filename, MAX_BYTES)) != commitments[filename]:
                        raise ValueError("Checker mutated a committed fitted-state input")
            reloaded = _json((directory / "reload.json").read_bytes())
            record["max_absolute_prediction_difference"] = compare(
                reference, reloaded, manifest, commitments[manifest["probe_file"]]
            )
        record["verified_quantities"] = list(reference["targets"])
        record["status"] = "succeeded"
    except Exception as exc:
        # Ordinary parser/runtime errors are terminal evidence too. Interruptions
        # (BaseException) remain visible incomplete attempts.
        record["error"] = type(exc).__name__ + ": " + str(exc)
    (directory / "fidelity.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def verify_check(directory, record, *, expected_manifest, expected_input_sha256,
                 expected_reference, expected_probe_sha256, source_revision, timeout):
    """Read-only verification of retained finite checks, never rerun a subject.

    Expected inputs/reference must be reconstructed from the authorized source
    analysis, not from a later predictor's declarations.
    """
    from sciquest.io import content_hash

    directory = Path(directory)
    actual = _json(_read_regular(directory, "fidelity.json", MAX_BYTES))
    declaration_names = set(expected_input_sha256) - {
        *expected_manifest["model_files"], expected_manifest["probe_file"],
        expected_manifest["live_reference"], expected_manifest["reload_program"],
    }
    if (
        actual != record or record["version"] != VERSION
        or record.get("declaration") != expected_manifest
        or record.get("input_sha256") != expected_input_sha256
        or expected_manifest["source_revision"] != source_revision
        or expected_input_sha256[expected_manifest["probe_file"]] != expected_probe_sha256
        or record["tolerance"] != dict(absolute=ABS_TOLERANCE, relative=REL_TOLERANCE)
        or len(declaration_names) != 1
        or record.get("declaration_sha256") != expected_input_sha256[next(iter(declaration_names), "")]
    ):
        raise ValueError("Fitted check source/input/tolerance binding changed")
    checks = record["checks"]
    if len(checks) > 2 or [c["kind"] for c in checks] != ["structure", "reload"][:len(checks)]:
        raise ValueError("Unexpected fitted-check execution sequence")
    arguments = [
        ["-I", "fidelity-internal-structure.py", "fidelity-internal-manifest.json", "fidelity-internal-structure.json"],
        [expected_manifest["reload_program"], expected_manifest["probe_file"], "fidelity-internal-reloaded.json"],
    ]
    for index, check in enumerate(checks):
        execution = check["execution"]
        if (
            not execution.get("blind_filesystem") or not execution.get("isolated") or execution.get("network")
            or execution["timeout_seconds"] != timeout
            or execution["command_hash"] != content_hash([sys.executable, "-I", "fidelity-internal-limits.py", *arguments[index]])
            or sha(_read_regular(directory, check["kind"]+".log", MAX_BYTES)) != check["log_sha256"]
        ):
            raise ValueError("Fitted check execution/log provenance changed")
    for name, digest in record["output_sha256"].items():
        if name not in {"live_reference.json","structure.json","reload.json"} or sha(_read_regular(directory, name, MAX_BYTES)) != digest:
            raise ValueError("Retained fitted check output bytes changed")
    if record["status"] == "succeeded":
        if set(record["output_sha256"]) != {"live_reference.json","structure.json","reload.json"}:
            raise ValueError("Successful fitted check lacks raw output commitments")
        if len(checks) != 2 or any(c["execution"]["status"] != "succeeded" for c in checks):
            raise ValueError("Successful fitted check lacks completed isolated executions")
        reference = _json(_read_regular(directory, "live_reference.json", MAX_BYTES))
        if reference != expected_reference:
            raise ValueError("Fitted live reference changed from its source analysis")
        reloaded = _json(_read_regular(directory, "reload.json", MAX_BYTES))
        difference = compare(reference, reloaded, expected_manifest, expected_probe_sha256)
        if difference != record["max_absolute_prediction_difference"] or record["verified_quantities"] != reference["targets"]:
            raise ValueError("Fitted finite parity result changed")
        report = _json(_read_regular(directory, "structure.json", MAX_BYTES))
        if report["status"] != "succeeded" or len(report["arrays"]) != len(expected_manifest["arrays"]):
            raise ValueError("Fitted structural report is incomplete")
        for item, spec in zip(report["arrays"], expected_manifest["arrays"]):
            if any(item[k] != spec[k] for k in ("family_id", "file", "key", "shape")):
                raise ValueError("Fitted structural report refers to a different array")
    elif record["status"] != "failed" or not record.get("error") or record["verified_quantities"]:
        raise ValueError("Invalid fitted failure report")
    return record
