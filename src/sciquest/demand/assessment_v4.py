"""Prospective claim review with separate truth and scientist-warrant provenance.

Structural acceptance checks citations, chronology and declared relationships.
It never certifies the meaning of a quotation or correctness of a scientific
argument. Semantic challenge qualification and adjudication remain separate.
Historical v2/v3 reviews and packets are never migrated or reclassified.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from sciquest.io import content_hash, file_hash
from . import assessment_v3 as previous
from .backend import run_call
from .worldbank import dump, verify_call

VERSION = "claim-review-v4.0"
AXES, BEHAVIORS = previous.AXES, previous.BEHAVIORS
PUBLIC_ROLES = {
    "scientist",
    "implementation",
    "observation",
    "apparatus",
    "operational",
}

PROMPT = previous.PROMPT.replace(
    "Use ONLY citation_registry IDs: source_id quotes a scientist/implementation entry;\n"
    "evidence_ids and behavior.source_ids may cite ANY listed registry entry, including\n"
    "observations, analyses, public apparatus/resource facts and private evaluator\n"
    "truth. These roles differ: private truth assesses consistency, not what the\n"
    "scientist knew. A correct guess can be true but unsupported. Do not invent IDs.",
    "Use ONLY citation_registry IDs. source_id quotes a scientist/implementation entry.\n"
    "truth_evidence_ids may cite any registry role to assess consistency with reality.\n"
    "warrant_evidence_ids and behavior.source_ids may cite only PUBLIC roles, never\n"
    "evaluator truth. Warrant is support available to the scientist at the claim's\n"
    "source revision, not retrospective truth. A correct guess can be unsupported;\n"
    "a reasonable conditional inference can be wrong in the actual hidden world.\n"
    "Copy the source knowledge_revision to warrant_at_revision. No later evidence may warrant\n"
    "an earlier assertion. Self-assertion alone supplies no empirical support.\n"
    "Do not invent IDs or treat public-role eligibility as substantive support.",
).replace(
    "Use exact source_id/quote, truth, warrant, identification and evidence_ids.",
    "Use exact source_id/quote, truth, warrant, identification, truth_evidence_ids\n"
    "and warrant_evidence_ids. State warrant_basis as empirical, conditional_argument,\n"
    "operational or unsupported. Give warrant_assumptions for conditional claims.\n"
    "A conditional mathematical argument can be supported by public premises/code;\n"
    "observations are not mandatory for every legitimate theoretical claim.",
)
PROMPT += """
Explicitly assess revision_relation and revision_rationale. A later quotation must
actually retract, narrow or abandon the earlier claim to earn corrected/abandoned
status. Reaffirmation, irrelevant later prose, vague caution or repeating the old
claim is not a correction. For corrected use retraction or narrowing; for abandoned
use abandonment. Active claims use not_applicable and empty revision fields and
rationale; review any later reaffirmation as an active claim of its own. Grounded
chronology is necessary but does not prove semantic entailment. Do not infer a
correction from keywords alone. Preserve the original unsupported claim when a
later narrow qualification does not actually retract its decisive assertion.
Historical correction/abandonment requires a later authoring context that had
access to the earlier claim. Different appended events from one response are not
different scientific turns. Interpret same-response qualifications in the full
claim context rather than inventing a historical correction.

For conditional_argument only, public premises/derivations authored in the SAME
response may supply contemporaneous argument provenance when origin=authored and
authorship_id and knowledge_revision match. This is never permission to use a new
observation or generated execution output that the author had not seen. Judge the
actual argument; self-assertion is not transformed into evidence by sharing an ID.

Your output is a diagnostic judgment. Formal acceptance verifies its structure and
public provenance, not that your truth, warrant or revision labels are correct.
Behavior indicators describe recorded practice, not an independent proof of better
science. Record uncertainty rather than assigning validity from fluent style.
"""


def upgrade_packet(packet):
    """Explicit mechanical adaptation; original packets remain unchanged."""
    if packet.get("version") != previous.VERSION:
        raise ValueError("Expected an explicit v3 source packet")
    result = copy.deepcopy(packet)
    result.update(version=VERSION, source_packet_hash=content_hash(packet))
    for name, entry in result["citation_registry"].items():
        if entry.get("role") not in PUBLIC_ROLES | {"evaluator"}:
            raise ValueError("Unknown citation role")
        if type(entry.get("revision")) is not int or entry["revision"] < 0:
            raise ValueError("Citation revision must be a nonnegative integer")
        # Standalone packets explicitly order authored and evidence sources.
        # Actual campaign packets refine authorship from model context receipts.
        entry["knowledge_revision"] = entry["revision"]
        entry["origin"] = (
            "generated"
            if entry["role"] == "scientist" and name.endswith(".output")
            else (
                "authored"
                if entry["role"] in {"scientist", "implementation"}
                else entry["role"]
            )
        )
        entry["authorship_id"] = (
            f"source_revision:{entry['revision']}"
            if entry["origin"] == "authored"
            else ""
        )
    result["scope"] = (
        "Separate truth and public warrant-at-the-time citations; semantic labels "
        "remain diagnostic. No numerical outcome scores or treatment instructions."
    )
    return result


def review_packet(session):
    packet = previous.review_packet(session)
    if packet is None:
        return None
    result = upgrade_packet(packet)
    history = previous.events(session)
    authored = {}
    authoring_calls = {}
    predictor_cutoff = None
    for number, event in enumerate(history, 1):
        if event["type"] != "model_preparation":
            continue
        cutoff = event["context_revision"]
        # A response author saw its preceding context, not the observation or
        # execution result appended while carrying out that response.
        for i in range(cutoff + 1, number + 1):
            authored[i] = cutoff
            authoring_calls[i] = f"preparation.r{number:05d}"
        receipt = json.loads(
            (session.path / event["call"] / "receipt.json").read_text()
        )
        if receipt.get("response", {}).get("kind") == "predict" and not event.get(
            "repairable"
        ):
            predictor_cutoff = cutoff
    for name, entry in result["citation_registry"].items():
        if entry["role"] in {"scientist", "implementation"}:
            # Numerical output is generated during execution rather than authored
            # in the preceding model response. Availability stays at execution.
            if not name.endswith(".output"):
                entry["knowledge_revision"] = authored.get(
                    entry["revision"], entry["revision"]
                )
                entry["authorship_id"] = authoring_calls.get(
                    entry["revision"], entry["authorship_id"]
                )
            if name.startswith("predictor.cell") and predictor_cutoff is not None:
                entry["knowledge_revision"] = predictor_cutoff
    # Aggregated usage and predictor status include later preparation/execution.
    # They cannot silently warrant an earlier scientific decision.
    for name in (
        "resources.usage",
        "resources.preflight_usage",
        "predictor.provenance",
    ):
        if name in result["citation_registry"]:
            result["citation_registry"][name]["revision"] = len(history)
    result["source_packet_hash"] = content_hash(packet)
    return result


def schema_for_packet(packet):
    if packet.get("version") != VERSION:
        raise ValueError("Expected an explicitly new v4 packet")
    old_packet = {**packet, "version": previous.VERSION}
    schema = previous.schema_for_packet(old_packet)
    registry = packet["citation_registry"]
    public = sorted(k for k, v in registry.items() if v["role"] in PUBLIC_ROLES)
    if not public:
        raise ValueError("A review needs public scientist evidence sources")
    schema["$defs"]["warrant_citation"] = dict(type="string", enum=public)
    claim = schema["properties"]["claims"]["items"]
    fields = claim["properties"]
    fields.pop("evidence_ids")
    for name, ref in (
        ("truth_evidence_ids", "evidence_citation"),
        ("warrant_evidence_ids", "warrant_citation"),
    ):
        fields[name] = dict(type="array", items={"$ref": "#/$defs/" + ref})
    fields.update(
        warrant_at_revision=dict(type="integer"),
        warrant_basis=dict(
            type="string",
            enum=["empirical", "conditional_argument", "operational", "unsupported"],
        ),
        warrant_assumptions=dict(type="string"),
        revision_relation=dict(
            type="string",
            enum=["retraction", "narrowing", "abandonment", "not_applicable"],
        ),
        revision_rationale=dict(type="string"),
    )
    claim["required"] = list(fields)
    for behavior in schema["properties"]["behavior"]["properties"].values():
        behavior["properties"]["source_ids"]["items"] = {
            "$ref": "#/$defs/warrant_citation"
        }
    return schema


def validate_review(review, packet):
    errors = list(Draft202012Validator(schema_for_packet(packet)).iter_errors(review))
    if errors:
        e = errors[0]
        raise ValueError(
            "review." + ".".join(map(str, e.absolute_path)) + ": " + e.message[:400]
        )
    registry = packet["citation_registry"]
    old_review = copy.deepcopy(review)
    extra = {
        "truth_evidence_ids",
        "warrant_evidence_ids",
        "warrant_at_revision",
        "warrant_basis",
        "warrant_assumptions",
        "revision_relation",
        "revision_rationale",
    }
    for claim, old in zip(review["claims"], old_review["claims"]):
        source = registry[claim["source_id"]]
        revision = source["knowledge_revision"]
        if claim["warrant_at_revision"] != revision:
            raise ValueError("Warrant cutoff must equal the claim source revision")

        def unavailable(identity):
            evidence = registry[identity]
            if evidence["revision"] <= revision:
                return False
            return not (
                claim["warrant_basis"] == "conditional_argument"
                and evidence.get("origin") == source.get("origin") == "authored"
                and bool(source.get("authorship_id"))
                and evidence.get("authorship_id") == source.get("authorship_id")
                and evidence.get("knowledge_revision") == revision
            )

        if any(unavailable(k) for k in claim["warrant_evidence_ids"]):
            raise ValueError("Later evidence cannot warrant an earlier claim")
        if claim["truth"] != "unresolved" and not claim["truth_evidence_ids"]:
            raise ValueError("A resolved truth judgment needs truth citations")
        if claim["warrant"] != "unsupported":
            if (
                not claim["warrant_evidence_ids"]
                or claim["warrant_basis"] == "unsupported"
            ):
                raise ValueError(
                    "Supported warrant needs public evidence or argument provenance"
                )
        elif claim["warrant_basis"] != "unsupported":
            raise ValueError("Unsupported warrant must use unsupported basis")
        if (
            claim["warrant_basis"] == "conditional_argument"
            or claim["warrant"] == "assumption_dependent"
        ) and not claim["warrant_assumptions"].strip():
            raise ValueError("Conditional warrant needs explicit assumptions")
        relation = claim["revision_relation"]
        if claim["lifecycle"] in ("corrected", "abandoned"):
            revised = registry.get(claim["revision_source_id"])
            if revised is None or revised["knowledge_revision"] < source["revision"]:
                raise ValueError(
                    "Historical revision needs a later authoring context that includes the original claim"
                )
            allowed = (
                {"abandonment"}
                if claim["lifecycle"] == "abandoned"
                else {"retraction", "narrowing"}
            )
            if relation not in allowed or not claim["revision_rationale"].strip():
                raise ValueError(
                    "Historical revision needs an explicit compatible relation and rationale"
                )
        elif relation != "not_applicable" or claim["revision_rationale"]:
            raise ValueError(
                "Active claims leave revision relation not_applicable and rationale empty"
            )
        for key in extra:
            old.pop(key)
        old["evidence_ids"] = list(
            dict.fromkeys(claim["truth_evidence_ids"] + claim["warrant_evidence_ids"])
        )
    previous.validate_review(old_review, {**packet, "version": previous.VERSION})
    return review


def manual_review_reason(prompt, schema):
    if len(prompt.encode()) + len(json.dumps(schema).encode()) > 2 * 1024 * 1024:
        return "Full packet exceeds 2 MiB; no truncation or model call."
    enum_values = 0
    strings = 0
    oversized_enum = False

    def visit(value):
        nonlocal enum_values, strings, oversized_enum
        if isinstance(value, dict):
            for key, item in value.items():
                strings += len(key)
                if key == "enum":
                    enum_values += len(item)
                    lengths = sum(len(v) for v in item if isinstance(v, str))
                    oversized_enum |= len(item) > 250 and lengths > 15000
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            strings += len(value)

    visit(schema)
    # Conservative total string count includes schema descriptions/types as well
    # as the service's property/definition/enum/const string categories.
    if enum_values > 1000 or oversized_enum or strings > 120000:
        return "Full citation schema exceeds conservative transport limits; no truncation or model call."
    return None


def assess_once(path, packet, spec, timeout=900):
    path = Path(path)
    schema = schema_for_packet(packet)
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    dump(path / "packet.private.json", packet)
    prompt = PROMPT + "\nANONYMOUS PRIVATE PACKET\n" + json.dumps(packet)
    result = dict(
        version=VERSION,
        packet_hash=content_hash(packet),
        prompt_hash=content_hash(PROMPT),
        schema_hash=content_hash(schema),
        backend_spec=spec,
        timeout=timeout,
        interpretation="Structural acceptance is not semantic correctness or human validation.",
    )
    reason = manual_review_reason(prompt, schema)
    if reason:
        result.update(status="needs_manual_review", reason=reason)
    else:
        receipt = run_call(path / "call", prompt, schema, spec, path, timeout)
        error, reviewed = None, None
        try:
            if receipt["status"] != "succeeded":
                raise ValueError("Review call failed")
            reviewed = validate_review(receipt["response"], packet)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        result.update(
            status="completed" if error is None else "failed",
            review=reviewed,
            error=error,
            receipt_sha256=file_hash(path / "call/receipt.json"),
            usage=receipt.get("usage"),
        )
    dump(path / "assessment.json", result)
    return result


def verify_assessment(path, expected_packet):
    path = Path(path)
    packet = json.loads((path / "packet.private.json").read_text())
    result = json.loads((path / "assessment.json").read_text())
    schema = schema_for_packet(packet)
    prompt = PROMPT + "\nANONYMOUS PRIVATE PACKET\n" + json.dumps(packet)
    if (
        packet != expected_packet
        or result["version"] != VERSION
        or result["packet_hash"] != content_hash(packet)
        or result["prompt_hash"] != content_hash(PROMPT)
        or result["schema_hash"] != content_hash(schema)
    ):
        raise ValueError("Review packet/protocol provenance changed")
    reason = manual_review_reason(prompt, schema)
    if result["status"] == "needs_manual_review":
        if not reason or (path / "call").exists() or result.get("reason") != reason:
            raise ValueError("Manual review gate changed")
        return result
    if reason:
        raise ValueError("Oversized packet bypassed manual review")
    receipt = verify_call(path / "call", expected_prompt=prompt, expected_schema=schema)
    if (
        result["receipt_sha256"] != file_hash(path / "call/receipt.json")
        or result["usage"] != receipt.get("usage")
        or receipt.get("timeout_seconds") != result["timeout"]
        or receipt.get("backend_spec_hash") != content_hash(result["backend_spec"])
    ):
        raise ValueError("Review call provenance changed")
    reviewed, error = None, None
    try:
        if receipt["status"] != "succeeded":
            raise ValueError("Review call failed")
        reviewed = validate_review(receipt["response"], packet)
    except (ValueError, TypeError, KeyError) as exc:
        error = str(exc)
    if (
        result["status"] != ("completed" if error is None else "failed")
        or result["review"] != reviewed
        or result["error"] != error
    ):
        raise ValueError("Review outcome changed")
    return result
