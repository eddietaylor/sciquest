"""Prospective delivery operations and byte-exact, bounded campaign transport.

This module supplies contracts, not a runner. Legacy attempts keep their original
schemas, reserves and archives. A format allowance is campaign-global; successful
analysis does not replenish it. A runtime/publication failure is retained work,
not a format correction and never an implicit permission for more calls.
"""

from __future__ import annotations

import base64
import binascii
import csv
import io
import json
import math
from dataclasses import asdict, dataclass

from .artifacts_v1 import relative_name, sha

VERSION = "scientist-delivery-v4.0"
ARCHIVE_VERSION = "campaign-byte-archive-v1.0"
CHUNK_CHARS = 16000
ARCHIVE_FIELDS = (
    "id", "version", "campaign_id", "cutoff", "path", "bytes", "file_sha256",
    "part", "parts", "encoding", "payload", "chunk_sha256",
)


@dataclass(frozen=True)
class DeliveryPolicy:
    format_corrections: int = 2
    finalization_input_tokens: int = 60000
    finalization_output_tokens: int = 18000
    backend_overhead_tokens: int = 16000
    max_notes_bytes: int = 8000
    max_program_bytes: int = 262144
    max_notebook_bytes: int = 1048576
    max_plan_bytes: int = 65536
    max_proposal_bytes: int = 32768
    analysis_timeout_seconds: int = 120
    preflight_timeout_seconds: int = 120
    fidelity_timeout_seconds: int = 30

    def validate(self, limits):
        values = asdict(self)
        if (
            any(type(v) is not int for v in values.values())
            or not 0 <= self.format_corrections <= 10
            or any(v <= 0 for k, v in values.items() if k != "format_corrections")
            or limits.max_calls <= 2 + self.format_corrections
            or self.finalization_input_tokens >= limits.input_token_stop
            or self.finalization_output_tokens >= limits.output_token_stop
            or self.max_notes_bytes > limits.max_notes_chars
        ):
            raise ValueError("Invalid prospective delivery allowance/reserves")
        return self


def controller(phase, *, calls_used, analyses_used, queries_used, format_errors,
               limits, policy, input_tokens=0, output_tokens=0, observation_remaining=True):
    """Legal next operations from the authorized prefix, before a model call.

    C is the number of additional corrections still allowed after a first failed
    preparation, not a promise that every remaining submission can be retried.
    Discovery preserves stop + prediction + C opportunities. Post-stop preparation
    is legal only if spending this call leaves prediction + C opportunities.
    Hard token/prompt/usage bounds remain separately enforced before dispatch.
    """
    policy.validate(limits)
    counters = (calls_used, analyses_used, queries_used, format_errors, input_tokens, output_tokens)
    if any(type(v) is not int or v < 0 for v in counters):
        raise ValueError("Usage counters must be nonnegative integers")
    if calls_used > limits.max_calls or analyses_used > limits.max_analyses or queries_used > limits.max_queries:
        raise ValueError("Usage exceeds frozen opportunities")
    if phase not in ("collecting", "prediction_required", "completed", "failed"):
        raise ValueError("Unknown scientist phase")
    remaining = limits.max_calls - calls_used
    corrections = max(0, policy.format_corrections - format_errors)
    terminal = phase in ("completed", "failed") or remaining == 0 or format_errors > policy.format_corrections
    token_pressure = (
        input_tokens + policy.finalization_input_tokens + policy.backend_overhead_tokens >= limits.input_token_stop
        or output_tokens + policy.finalization_output_tokens >= limits.output_token_stop
    )
    allowed = []
    if not terminal:
        if phase == "collecting":
            allowed = ["stop"]
            # After a discovery call: still stop, prediction, C corrections.
            if remaining - 1 >= 2 + corrections and not token_pressure:
                if queries_used < limits.max_queries and observation_remaining:
                    allowed.insert(0, "query")
                if analyses_used < limits.max_analyses:
                    allowed.insert(0, "analysis")
        else:
            allowed = ["predict"]
            if remaining - 1 >= 1 + corrections and analyses_used < limits.max_analyses and not token_pressure:
                allowed.insert(0, "prepare")
    return dict(
        version=VERSION, calls_remaining=remaining,
        analyses_remaining=limits.max_analyses - analyses_used,
        queries_remaining=limits.max_queries - queries_used,
        format_corrections_remaining=corrections, allowed_operations=allowed,
        stop_required=allowed == ["stop"], token_reserve_pressure=token_pressure,
        terminal=terminal,
    )


def _variant(kind, fields):
    properties = {"kind": {"type": "string", "enum": [kind]}, **fields}
    return dict(type="object", properties=properties, required=list(properties),
                additionalProperties=False)


_STRING = {"type": "string"}
_IDS = {"type": "array", "items": _STRING}
_ANALYSIS = dict(program=_STRING, inputs=_IDS, outputs=_IDS, fitted_bundle=_STRING)
VARIANTS = {
    "query": _variant("query", dict(query_json=_STRING)),
    "stop": _variant("stop", dict(decision_json=_STRING)),
    "analysis": _variant("analysis", _ANALYSIS),
    "prepare": _variant("prepare", _ANALYSIS),
    "predict": _variant("predict", dict(
        notebook_json=_STRING, plan_yaml=_STRING, artifact_ids=_IDS, fitted_bundle=_STRING,
    )),
}


def response_schema(allowed):
    if not allowed or len(set(allowed)) != len(allowed) or any(k not in VARIANTS for k in allowed):
        raise ValueError("An open call needs an explicit unique legal operation set")
    return dict(
        type="object",
        properties=dict(operation={"anyOf": [VARIANTS[k] for k in allowed]}, campaign_notes=_STRING),
        required=["operation", "campaign_notes"], additionalProperties=False,
    )


def _bounded(value, maximum, label, *, empty=False):
    if (
        not isinstance(value, str) or len(value.encode("utf-8")) > maximum
        or (not empty and not value.strip())
    ):
        raise ValueError(label + " must be bounded UTF-8 text")
    return value


def _ids(value, *, maximum=64, outputs=False):
    if not isinstance(value, list) or len(value) > maximum or any(not isinstance(x, str) for x in value):
        raise ValueError("Artifact IDs must be a bounded list of strings")
    if len(set(value)) != len(value):
        raise ValueError("Duplicate artifact IDs")
    for name in value:
        relative_name(name)
        if outputs and not name.startswith("outputs/"):
            raise ValueError("Analysis outputs must be explicitly under outputs/")
    return value


def validate_response(response, allowed, policy):
    """Reject mixed or extraneous payloads; never normalize a legacy submission."""
    if not isinstance(response, dict) or set(response) != {"operation", "campaign_notes"}:
        raise ValueError("Response needs exactly operation and campaign_notes")
    _bounded(response["campaign_notes"], policy.max_notes_bytes, "campaign_notes", empty=True)
    op = response["operation"]
    if not isinstance(op, dict) or op.get("kind") not in allowed:
        raise ValueError("Operation is not legal at this campaign cutoff")
    kind = op["kind"]
    if kind not in VARIANTS or set(op) != set(VARIANTS[kind]["properties"]):
        raise ValueError("Invalid or mixed operation payload")
    if kind in ("query", "stop"):
        field = "query_json" if kind == "query" else "decision_json"
        _bounded(op[field], policy.max_proposal_bytes, field)
        json_payload(op[field])
    elif kind in ("analysis", "prepare"):
        _bounded(op["program"], policy.max_program_bytes, "program")
        _ids(op["inputs"])
        _ids(op["outputs"], maximum=16, outputs=True)
        _bounded(op["fitted_bundle"], 512, "fitted_bundle", empty=True)
        if op["fitted_bundle"] and op["fitted_bundle"] not in op["outputs"]:
            raise ValueError("A fitted declaration must be one of this analysis's published outputs")
    else:
        _bounded(op["notebook_json"], policy.max_notebook_bytes, "notebook_json")
        json_payload(op["notebook_json"])
        _bounded(op["plan_yaml"], policy.max_plan_bytes, "plan_yaml")
        _ids(op["artifact_ids"])
        _bounded(op["fitted_bundle"], 512, "fitted_bundle", empty=True)
        if op["fitted_bundle"] and op["fitted_bundle"] not in op["artifact_ids"]:
            raise ValueError("Prediction must explicitly select its fitted declaration")
    return op


def validate_data(value, *, depth=0, remaining=None):
    """Bound JSON-like metadata and reject escaped invalid Unicode/cycles."""
    remaining = [100000] if remaining is None else remaining
    remaining[0] -= 1
    if depth > 64 or remaining[0] < 0:
        raise ValueError("Structured payload exceeds nesting/element bound")
    if isinstance(value, str):
        value.encode("utf-8")
    elif value is None or type(value) in (int, bool):
        pass
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Nonfinite structured value")
    elif isinstance(value, list):
        for item in value:
            validate_data(item, depth=depth+1, remaining=remaining)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Structured keys must be UTF-8 strings")
            key.encode("utf-8")
            validate_data(item, depth=depth+1, remaining=remaining)
    else:
        raise ValueError("Structured values must be JSON-like")
    return value


def json_payload(text):
    try:
        return validate_data(json.loads(text))
    except RecursionError as exc:
        raise ValueError("Structured payload exceeds nesting bound") from exc


@dataclass(frozen=True)
class ArchiveLimits:
    max_file_bytes: int = 1048576
    max_total_bytes: int = 25165824
    max_files: int = 256
    max_rows: int = 8192
    max_encoded_bytes: int = 41943040

    def validate(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("Archive limits must be positive integers")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("Archive file allowance exceeds aggregate allowance")
        return self


def _archive_identity(campaign_id, cutoff):
    import re
    if not isinstance(campaign_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", campaign_id):
        raise ValueError("Invalid campaign archive identity")
    if type(cutoff) is not int or cutoff < 1:
        raise ValueError("Archive needs an explicit evidence cutoff")


def encode_archive(files, *, campaign_id, cutoff, limits=None):
    """Canonical CSV of opaque bytes. Hash labels always refer to raw bytes."""
    limits = (limits or ArchiveLimits()).validate()
    _archive_identity(campaign_id, cutoff)
    if not isinstance(files, dict) or not files or len(files) > limits.max_files:
        raise ValueError("Archive needs a bounded explicit file map")
    total = 0
    rows = []
    for name in sorted(files):
        relative_name(name)
        data = files[name]
        if not isinstance(data, bytes) or len(data) > limits.max_file_bytes:
            raise ValueError("Archive file exceeds opaque byte bound")
        total += len(data)
        if total > limits.max_total_bytes:
            raise ValueError("Archive decoded aggregate allowance exceeded")
        payload = base64.b64encode(data).decode("ascii")
        chunks = [payload[i:i+CHUNK_CHARS] for i in range(0, len(payload), CHUNK_CHARS)] or [""]
        for part, chunk in enumerate(chunks, 1):
            rows.append(dict(
                id=f"instrument-{campaign_id}-{len(rows):06d}", version=ARCHIVE_VERSION,
                campaign_id=campaign_id, cutoff=str(cutoff), path=name, bytes=str(len(data)),
                file_sha256=sha(data), part=str(part), parts=str(len(chunks)), encoding="base64",
                payload=chunk, chunk_sha256=sha(chunk.encode("ascii")),
            ))
        if len(rows) > limits.max_rows:
            raise ValueError("Archive row allowance exceeded")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ARCHIVE_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    result = stream.getvalue().encode("utf-8")
    if len(result) > limits.max_encoded_bytes:
        raise ValueError("Archive encoded byte allowance exceeded")
    return result


def decode_archive(raw, *, campaign_id, cutoff, expected_sha256, limits=None):
    """Read only a frozen, canonical archive with exact order and both quotas.

    Campaign identity/cutoff must come from authorized history, not the CSV.
    This utility does not authorize a subject-selected path/hash on its own.
    """
    limits = (limits or ArchiveLimits()).validate()
    _archive_identity(campaign_id, cutoff)
    if not isinstance(raw, bytes) or len(raw) > limits.max_encoded_bytes or sha(raw) != expected_sha256:
        raise ValueError("Archive encoded bytes changed or exceed bound")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    try:
        fields = tuple(reader.fieldnames or ())
    except csv.Error as exc:
        raise ValueError("Malformed or oversized archive CSV field") from exc
    if fields != ARCHIVE_FIELDS:
        raise ValueError("Unexpected archive CSV columns")
    groups = {}
    total_rows = 0
    for row in _csv_rows(reader):
        if total_rows >= limits.max_rows:
            raise ValueError("Archive row allowance exceeded")
        if set(row) != set(ARCHIVE_FIELDS) or any(value is None for value in row.values()):
            raise ValueError("Malformed archive row")
        if (
            row["id"] != f"instrument-{campaign_id}-{total_rows:06d}"
            or row["version"] != ARCHIVE_VERSION or row["campaign_id"] != campaign_id
            or row["cutoff"] != str(cutoff) or row["encoding"] != "base64"
        ):
            raise ValueError("Archive row identity/cutoff/encoding changed")
        name = relative_name(row["path"])
        if len(row["payload"]) > CHUNK_CHARS or sha(row["payload"].encode("ascii")) != row["chunk_sha256"]:
            raise ValueError("Archive payload chunk changed or exceeds bound")
        group = groups.setdefault(name, [])
        # Groups are contiguous and strictly increasing by logical path.
        if list(groups)[-1] != name or row["part"] != str(len(group) + 1):
            raise ValueError("Reordered or duplicate archive chunks")
        group.append(row)
        if len(groups) > limits.max_files:
            raise ValueError("Archive file allowance exceeded")
        total_rows += 1
    files, total = {}, 0
    if not groups or list(groups) != sorted(groups):
        raise ValueError("Archive needs ordered complete file groups")
    for name, group in groups.items():
        first = group[0]
        if any(
            row["parts"] != str(len(group))
            or any(row[k] != first[k] for k in ("bytes", "file_sha256"))
            for row in group
        ):
            raise ValueError("Incomplete or inconsistent archive chunks")
        try:
            size = int(first["bytes"])
            if str(size) != first["bytes"] or not 0 <= size <= limits.max_file_bytes:
                raise ValueError("Archive decoded file allowance exceeded")
            total += size
            if total > limits.max_total_bytes:
                raise ValueError("Archive decoded aggregate allowance exceeded")
            encoded_length = 4 * ((size + 2) // 3)
            if (
                sum(len(row["payload"]) for row in group) != encoded_length
                or len(group) != max(1, (encoded_length + CHUNK_CHARS - 1) // CHUNK_CHARS)
                or any(len(row["payload"]) != CHUNK_CHARS for row in group[:-1])
            ):
                raise ValueError("Archive encoded payload disagrees with bounded decoded size")
            data = base64.b64decode("".join(row["payload"] for row in group), validate=True)
        except (binascii.Error, TypeError) as exc:
            raise ValueError("Invalid archive base64") from exc
        if len(data) != size or sha(data) != first["file_sha256"]:
            raise ValueError("Archive decoded bytes changed")
        files[name] = data
    # Reject alternate chunking/whitespace or noncanonical CSV representations.
    if encode_archive(files, campaign_id=campaign_id, cutoff=cutoff, limits=limits) != raw:
        raise ValueError("Archive is not the frozen canonical representation")
    return files


def _csv_rows(reader):
    try:
        yield from reader
    except csv.Error as exc:
        raise ValueError("Malformed or oversized archive CSV field") from exc
