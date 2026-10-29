"""Campaign-local memory and auditable opportunity budgets for matched guidance."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from sciquest.io import content_hash, file_hash
from .policies_v2 import COMMON, GUIDANCE, ResourceLimits, validate_assignment


def response_schema(base):
    schema = copy.deepcopy(base)
    schema["properties"]["campaign_notes"] = {"type": "string"}
    schema["required"].append("campaign_notes")
    return schema


def instructions(protocol, model_instructions, assignment):
    validate_assignment(assignment)
    # Override the old repeated-profile/mandatory-extension instructions explicitly.
    return (
        protocol
        + "\n"
        + model_instructions
        + "\nCOMPARISON RULES (replace policy-specific requirements above):\n"
        + COMMON
        + "\nFIXED INQUIRY GUIDANCE:\n"
        + GUIDANCE[assignment["policy"]]
    )


def events(session):
    with session.transaction() as (c, _):
        return [e["payload"] for e in session._events(c)]


def usage_ledger(session):
    calls = [e for e in events(session) if e["type"] == "model_preparation"]
    records = []
    for e in calls:
        directory = session.path / e["call"]
        receipt = json.loads((directory / "receipt.json").read_text())
        if file_hash(directory / "receipt.json") != e["receipt_sha256"]:
            raise ValueError("Campaign usage receipt changed")
        records.append(receipt)
    known = [r["usage"] for r in records if r.get("usage") is not None]
    return dict(
        calls=len(calls),
        missing_usage_calls=len(records) - len(known),
        input_tokens=sum(u["input_tokens"] for u in known),
        output_tokens=sum(u["output_tokens"] for u in known),
        cached_input_tokens=sum(u.get("cached_input_tokens", 0) for u in known),
        elapsed_call_seconds=sum(r.get("elapsed_seconds", 0) for r in records),
    )


def compact_memory(session, assignment):
    limits = ResourceLimits(**assignment["limits"])
    history = events(session)
    analyses = [e["analysis"] for e in history if e["type"] == "scientist_analysis"]
    notes = [e["text"] for e in history if e["type"] == "campaign_notes"]
    index = []
    for i, analysis in enumerate(analyses, 1):
        entry = dict(
            id=f"a{i:04d}",
            status=analysis["status"],
            code_hash=content_hash(analysis["code"]),
            output_hash=analysis.get(
                "full_output_hash", content_hash(analysis["output"])
            ),
            code_file=f"analysis/a{i:04d}.py",
            output_file=f"analysis/a{i:04d}.txt",
        )
        if i == len(analyses):
            entry["recent_output"] = analysis["output"][: limits.recent_analysis_chars]
            entry["output_excerpted"] = len(
                analysis["output"]
            ) > limits.recent_analysis_chars or analysis.get(
                "full_output_excerpted", False
            )
        index.append(entry)
    return dict(
        campaign_notes=notes[-1] if notes else "",
        analysis_history=index,
        memory_rules=dict(
            max_notes_chars=limits.max_notes_chars,
            raw_analysis_retrieval="Offline analysis can read the indexed same-campaign files; no hidden files or other campaign history are mounted.",
        ),
    )


def stage_analyses(session, workspace):
    analyses = [
        e["analysis"] for e in events(session) if e["type"] == "scientist_analysis"
    ]
    dest = Path(workspace) / "analysis"
    dest.mkdir()
    for i, a in enumerate(analyses, 1):
        (dest / f"a{i:04d}.py").write_text(a["code"])
        if a.get("full_output_file"):
            source = session.path / a["full_output_file"]
            if file_hash(source) != a["full_output_hash"]:
                raise ValueError("Archived own-campaign analysis changed")
            text = source.read_text(errors="replace")
        else:
            text = a["output"]
        (dest / f"a{i:04d}.txt").write_text(text)


def resource_stop(ledger, limits, prompt=None):
    if ledger["missing_usage_calls"]:
        return "missing_model_usage"
    if ledger["input_tokens"] >= limits.input_token_stop:
        return "input_token_stop"
    if ledger["output_tokens"] >= limits.output_token_stop:
        return "output_token_stop"
    if prompt is not None and len(prompt.encode()) > limits.max_prompt_bytes:
        return "prompt_byte_limit"
    return None
