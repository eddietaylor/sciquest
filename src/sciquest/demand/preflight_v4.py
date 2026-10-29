"""Prospective public preflight with retained exact native export bytes."""

from __future__ import annotations

import json
from pathlib import Path

from sciquest.io import content_hash, file_hash
from . import preflight_v3 as base
from .artifacts_v1 import _read_regular, relative_name, sha
from .export_v1 import verify_export

VERSION = "public-predictor-preflight-v2.0"


def run(session, notebook, submitted, context, directory, timeout):
    captures = {}

    def observe(work, export, evidence, stage):
        verify_export(work, export)
        records = {}
        for name in [*export["artifacts"], export["reload_notebook"]]:
            relative_name(name)
            source = "notebook.ipynb" if stage == "reload" and name == export["reload_notebook"] else name
            data = _read_regular(work, source, 1048576)
            dest = evidence / "exports" / stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            records[name] = dict(bytes=len(data), sha256=sha(data))
        captures[stage] = records

    record = base.run(session, notebook, submitted, context, directory, timeout,
                      export_observer=observe)
    # Keep the base record intact; the wrapper binds it and its retained bytes.
    result = dict(
        version=VERSION, status=record["status"], error=record["error"],
        base_record_sha256=file_hash(Path(directory) / "preflight.json"),
        base_record_hash=content_hash(record), captures=captures,
        checked_fitted_export=(submitted.get("model_export") or {}).get("checked_fitted_export"),
        scope="Public execution and byte/finite forecast parity only; no truth, scoring, inference or calibration validation.",
    )
    (Path(directory) / "instrument_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_record(path, expected_notebook, expected_plan_hash, revision, *,
                  context, expected_archive, expected_export):
    path = Path(path)
    result = json.loads((path / "instrument_preflight.json").read_text())
    record = base.verify_record(
        path, expected_notebook, expected_plan_hash, revision,
        context=context, expected_archive=expected_archive,
    )
    if (
        result["version"] != VERSION or result["base_record_sha256"] != file_hash(path / "preflight.json")
        or result["base_record_hash"] != content_hash(record)
        or result["status"] != record["status"] or result["error"] != record["error"]
        or result["checked_fitted_export"] != (expected_export or {}).get("checked_fitted_export")
    ):
        raise ValueError("Instrument preflight binding changed")
    for stage, records in result["captures"].items():
        if stage not in ("original", "reload") or not expected_export:
            raise ValueError("Unexpected native export capture")
        if set(records) != set([*expected_export["artifacts"], expected_export["reload_notebook"]]):
            raise ValueError("Incomplete native export capture")
        directory = path / "exports" / stage
        for name, item in records.items():
            data = _read_regular(directory, relative_name(name), 1048576)
            if len(data) != item["bytes"] or sha(data) != item["sha256"]:
                raise ValueError("Retained native export changed")
        verify_export(directory, expected_export)
    if result["status"] == "succeeded" and expected_export and set(result["captures"]) != {"original", "reload"}:
        raise ValueError("Successful preflight lacks both native export captures")
    return result
