"""New, one-shot diagnostic review. Never changes v2 packets or acceptances."""

from __future__ import annotations

import copy
import json
import hashlib
from pathlib import Path

from jsonschema import Draft202012Validator

from sciquest.io import content_hash, file_hash
from . import assessment_v2
from .backend import run_call
from .campaign_v2 import events, usage_ledger
from .worldbank import dump, verify_call

VERSION = "claim-review-v3.3"
AXES = assessment_v2.AXES
BEHAVIORS = assessment_v2.BEHAVIORS


def obj(fields):
    return dict(
        type="object",
        properties={k: copy.deepcopy(v) for k, v in fields.items()},
        required=list(fields),
        additionalProperties=False,
    )


TEXT = dict(type="string", pattern=r"\S")
IDS = dict(type="array", items=dict(type="string"))
CLAIM = {
    **copy.deepcopy(assessment_v2.CLAIM_FIELDS),
    "lifecycle": dict(
        type="string",
        enum=["conjecture", "retained", "unresolved", "corrected", "abandoned"],
    ),
    "decisive": dict(type="boolean"),
    "mechanistic": dict(type="boolean"),
    "revision_source_id": dict(type="string"),
    "revision_quote": dict(type="string"),
}
CLAIM["evidence_ids"] = IDS
for key in ("source_id", "quote", "rationale"):
    CLAIM[key] = TEXT
SCHEMA = obj(
    {
        **{a: dict(type="integer", enum=[0, 1, 2, 3, 4]) for a in AXES},
        "claims": dict(type="array", minItems=1, items=obj(CLAIM)),
        "code_checks": dict(
            type="array",
            items=obj(
                dict(
                    source_id=TEXT,
                    quote=TEXT,
                    validity=dict(
                        type="string", enum=["valid", "invalid", "unverified"]
                    ),
                    rationale=TEXT,
                )
            ),
        ),
        "critical_errors": dict(type="array", items=TEXT),
        "positive_mechanistic_content": dict(type="boolean"),
        "behavior": obj(
            {
                b: obj(
                    dict(
                        observed=dict(
                            type="string",
                            enum=["yes", "partial", "no", "not_applicable"],
                        ),
                        source_ids=IDS,
                        rationale=TEXT,
                    )
                )
                for b in BEHAVIORS
            }
        ),
        "rationale": TEXT,
    }
)

PROMPT = """Review this anonymous scientific trace against the supplied PRIVATE evaluator
truth. You are never the scientist. Numerical outcome scores, policy labels and
other reviews are withheld; scientific prose may reveal style. Do not reward it.
Use ONLY citation_registry IDs: source_id quotes a scientist/implementation entry;
evidence_ids and behavior.source_ids may cite ANY listed registry entry, including
observations, analyses, public apparatus/resource facts and private evaluator
truth. These roles differ: private truth assesses consistency, not what the
scientist knew. A correct guess can be true but unsupported. Do not invent IDs.

Review EVERY declared claim quote and material decisive assertions elsewhere.
Use exact source_id/quote, truth, warrant, identification and evidence_ids.
Lifecycle is conjecture, retained, unresolved, corrected or abandoned. For corrected
or abandoned claims give a later scientist revision_source_id and exact
revision_quote showing actual correction/abandonment. Otherwise both revision
strings are empty. Final declared claims cannot be marked corrected/abandoned to
avoid assessment. decisive means material to the scientific conclusion, rather
than a minor label or ancillary historical diagnostic. mechanistic means a positive
account of response/measurement structure, not generic caution or an operational
resource fact. Preserve earlier errors and assess their actual follow-through.

Rate these independent axes 0..4:
mechanism_content: 0 central mechanism contradicted/no content; 1 small correct
fragment or unknown; 2 partly faithful with important gaps; 3 substantive correct
account at the claimed resolution; 4 faithful relevant mechanism at that resolution.
evidence_warrant: 0 assertion; 1 weak support; 2 plausible conditional inference with
important gaps; 3 defensible integration; 4 systematic integration of evidence,
dependence, contradictions and limitations.
identification_discipline: 0 major overclaim; 1 repeated/central scope error;
2 mixed/generic caution; 3 specific appropriate limits; 4 defensible claim-by-claim
identification reasoning. Distinguish population identification under assumptions,
finite precision, transfer and full-law recovery. Do not demand hidden model-class
knowledge. Observationally equivalent kernels remain indistinguishable with more
samples; finite-grid interventions do not identify arbitrary off-grid laws.
test_revision_quality: 0 no meaningful test; 1 weak sampling; 2 plausible test;
3 discriminating/justified measurement with evidence-based response; 4 strong
diagnostic tests with warranted follow-through. Changing a belief alone is not
revision quality. Forecast calibration has separate numeric scores; never infer it
from cautious prose or these ratings.

Apply EXACT caps: any decisive ACTIVE claim marked overclaimed caps identification
at 1; any decisive ACTIVE mechanistic claim contradicted caps mechanism at 1.
ACTIVE means conjecture/retained/unresolved, excluding grounded corrected/abandoned
claims. No positive mechanistic content caps mechanism at 1. If all active
mechanistic claims are unsupported, positive mechanistic content cannot earn
evidence_warrant above 1. Do not penalize a final conclusion as though a grounded
earlier correction never happened; do not erase the earlier mistake either.

Check EVERY required_code_source_id once in code_checks with an exact code quote.
Read semantics and actual output: dimensions, log densities versus densities,
weight normalization, held-out versus reused data, evidence dependence and whether
printed diagnostics support the prose. Mark invalid for a demonstrable defect and
explain it; mark unverified when validity cannot be established. Execution success
or arithmetic alone does not validate a likelihood/model/diagnostic. Do not assume
unexecuted rejected preparations were delivered or that a registered notebook
proves a scientific conclusion. predictor.provenance distinguishes these cases.
Public-input preflight is implementation preparation, not sealed evaluation or new
scientific evidence. Rejected preparation code may be referenced by the registered
predictor; its separate sources are included so that reference can be checked.

Audit all four behavior indicators with source IDs: substantive rivals;
discriminating design under error assumptions; conditional prior/likelihood/
dependence updates used later; anticipated diagnostic failure and actual warranted
response checking auxiliaries. Mark partial/no/NA where appropriate.
Resource facts and implementation claims are assessable only within packet scope.
Return concise grounded rationales, preserving disagreements. No composite score.
These are diagnostic AI judgments requiring calibration and eventual human checks.
"""


def _entry(text, role, revision=0):
    return dict(
        text=text if isinstance(text, str) else json.dumps(text, sort_keys=True),
        role=role,
        revision=revision,
    )


def review_packet(session):
    base = assessment_v2.review_packet(session)
    if base is None:
        return None
    history = events(session)
    with session.transaction() as (_, state):
        state = copy.deepcopy(state)
    stop_revision = next(i for i, e in enumerate(history, 1) if e["type"] == "stop")
    registry = {
        k: _entry(v, "scientist", stop_revision)
        for k, v in assessment_v2.citation_sources(base).items()
    }
    analysis_no = 0
    code_ids = []
    for revision, event in enumerate(history, 1):
        if event["type"] == "query":
            identity = event["observation"]["id"]
            registry[identity]["role"] = "observation"
            for key in registry:
                if key == identity or key.startswith(identity + ".plan."):
                    registry[key]["revision"] = revision
            registry[identity + ".plan.predictions"] = _entry(
                event["plan"]["predictions"], "scientist", revision
            )
        elif event["type"] == "scientist_analysis":
            analysis_no += 1
            prefix = f"analysis.a{analysis_no:04d}"
            for suffix in ("code", "output"):
                registry[prefix + "." + suffix]["revision"] = revision
            registry[prefix + ".code"]["role"] = "implementation"
            registry[prefix + ".provenance"] = _entry(
                dict(
                    status=event["analysis"]["status"],
                    code_hash=content_hash(event["analysis"]["code"]),
                    output_hash=event["analysis"]["full_output_hash"],
                ),
                "operational",
                revision,
            )
            code_ids.append(prefix + ".code")
        elif event["type"] == "campaign_notes":
            # Retain earlier notes too: an actual later revision needs chronology.
            identity = f"notes.r{revision:05d}"
            registry[identity] = _entry(event["text"], "scientist", revision)
            if registry.get("campaign_notes", {}).get("text") == event["text"]:
                registry["campaign_notes"]["revision"] = revision
    registry["evaluator.access"] = _entry(base["access"], "evaluator")
    for name in (
        "private_world",
        "truth_contract",
        "private_identification_limits",
        "private_intervention_truth",
    ):
        registry["evaluator." + name] = _entry(base[name], "evaluator")
    registry["apparatus.public_contract"] = _entry(state["public"], "apparatus")
    limits = state["comparison"]["limits"]
    registry["resources.limits"] = _entry(
        dict(**limits, observation_budget=state["public"]["budget"]), "operational"
    )
    registry["resources.usage"] = _entry(
        dict(
            **usage_ledger(session),
            queries=len(base["evidence"]),
            observation_cost=state["spent"],
            analyses=analysis_no,
        ),
        "operational",
    )
    delivery_contracts = [e for e in history if e["type"] == "delivery_contract"]
    if delivery_contracts:
        delivery = delivery_contracts[0]
        registry["resources.delivery_policy"] = _entry(
            dict(
                version=delivery["version"],
                policy=delivery["policy"],
                repair_limit=delivery["repair_limit"],
                offline_analysis_seconds=delivery["timeout"],
                native_predictor_seconds=120,
            ),
            "operational",
        )
    registry["resources.preflight_usage"] = _entry(
        [
            dict(
                status=e["record"]["status"],
                error=e["record"].get("error"),
                execution_seconds=(e["record"].get("execution") or {}).get(
                    "elapsed_seconds"
                ),
                reload_seconds=(e["record"].get("reload_execution") or {}).get(
                    "elapsed_seconds"
                ),
            )
            for e in history
            if e["type"] == "predictor_preflight"
        ],
        "operational",
    )
    provenance = dict(
        status="not_registered",
        native_execution="unavailable",
        rejected_preparations=[],
    )
    if state["quest"]:
        from sciquest.state import get_record

        quest = Path(state["quest"])
        registration = get_record(quest, "registration", state["experiment"])
        if registration:
            notebook = quest / "experiments" / state["experiment"] / "notebook.ipynb"
            native = json.loads(notebook.read_text())
            provenance.update(
                status="registered",
                notebook_sha256=file_hash(notebook),
                native_execution=(
                    state["phase"]
                    if state["phase"] in ("completed", "failed")
                    else "pending"
                ),
                registered_plan={
                    k: v
                    for k, v in registration["plan"].items()
                    if k
                    in (
                        "question",
                        "hypothesis",
                        "scope",
                        "assumptions",
                        "data_access",
                        "stopping_rule",
                        "action",
                        "confirmatory_status",
                        "predictions",
                        "model_export",
                    )
                },
            )
            for i, cell in enumerate(native["cells"], 1):
                if cell["cell_type"] == "code" and cell["source"]:
                    identity = f"predictor.cell{i:04d}"
                    source = cell["source"]
                    registry[identity] = _entry(
                        "".join(source) if isinstance(source, list) else source,
                        "implementation",
                        len(history),
                    )
                    code_ids.append(identity)
    preparation_number = 0
    for revision, event in enumerate(history, 1):
        if (
            event["type"] == "model_preparation"
            and event.get("repairable")
            and event.get("error")
        ):
            receipt = json.loads(
                (session.path / event["call"] / "receipt.json").read_text()
            )
            if (
                file_hash(session.path / event["call"] / "receipt.json")
                != event["receipt_sha256"]
            ):
                raise ValueError("Rejected preparation receipt changed")
            if receipt.get("response", {}).get("kind") == "predict":
                preparation_number += 1
                prefix = f"prepared_predictor.p{preparation_number:04d}"
                raw = receipt["response"]["notebook_json"]
                registry[prefix + ".raw"] = _entry(raw, "scientist", revision)
                code_sources = []
                try:
                    prepared = json.loads(raw)
                    cells = (
                        prepared.get("cells", []) if isinstance(prepared, dict) else []
                    )
                    for i, cell in enumerate(cells, 1):
                        if (
                            isinstance(cell, dict)
                            and cell.get("cell_type") == "code"
                            and cell.get("source")
                        ):
                            text = cell["source"]
                            text = "".join(text) if isinstance(text, list) else text
                            if not isinstance(text, str):
                                continue
                            identity = prefix + f".cell{i:04d}"
                            registry[identity] = _entry(
                                text, "implementation", revision
                            )
                            code_sources.append(identity)
                            code_ids.append(identity)
                except (ValueError, TypeError):
                    pass  # malformed preparation is retained verbatim, never repaired
                checks = [
                    e["record"]["status"]
                    for e in history
                    if e["type"] == "predictor_preflight"
                    and e["path"] == event["call"] + "/preflight"
                ]
                provenance["rejected_preparations"].append(
                    dict(
                        status="rejected_before_native_registration",
                        public_preflight_status=checks[0] if checks else "not_run",
                        code_source_ids=code_sources,
                        error=event["error"],
                        notebook_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                        plan_yaml=receipt["response"]["plan_yaml"],
                    )
                )
    registry["predictor.provenance"] = _entry(provenance, "operational")
    return dict(
        version=VERSION,
        decision=base["decision"],
        citation_registry=registry,
        required_code_source_ids=code_ids,
        scope="Full available scientific trace, public apparatus/resource ledger, actual registered predictor source and preparation provenance. No outcome scores or policy instructions; partial prose masking only.",
    )


def schema_for_packet(packet):
    if packet.get("version") != VERSION:
        raise ValueError("Use an explicitly new v3 review packet")
    registry = packet["citation_registry"]
    source_ids = sorted(
        k for k, v in registry.items() if v["role"] in ("scientist", "implementation")
    )
    all_ids = sorted(registry)
    schema = copy.deepcopy(SCHEMA)
    # Define each registry once. Repeating it in every behavior/citation field
    # exceeds the service's total enum-value limit on realistic long traces.
    # uniqueItems is deliberately absent: the strict transport rejects it.
    schema["$defs"] = dict(
        scientist_citation=dict(type="string", enum=source_ids),
        evidence_citation=dict(type="string", enum=all_ids),
    )
    claim = schema["properties"]["claims"]["items"]["properties"]
    claim["source_id"] = {"$ref": "#/$defs/scientist_citation"}
    claim["evidence_ids"]["items"] = {"$ref": "#/$defs/evidence_citation"}
    claim["revision_source_id"] = dict(
        anyOf=[dict(type="string", enum=[""]), {"$ref": "#/$defs/scientist_citation"}]
    )
    check = schema["properties"]["code_checks"]["items"]["properties"]
    if packet["required_code_source_ids"]:
        check["source_id"]["enum"] = packet["required_code_source_ids"]
    for behavior in schema["properties"]["behavior"]["properties"].values():
        behavior["properties"]["source_ids"]["items"] = {
            "$ref": "#/$defs/evidence_citation"
        }
    return schema


def validate_review(review, packet):
    errors = list(Draft202012Validator(schema_for_packet(packet)).iter_errors(review))
    if errors:
        error = errors[0]
        raise ValueError(
            "review."
            + ".".join(map(str, error.absolute_path))
            + ": "
            + error.message[:400]
        )
    registry = packet["citation_registry"]
    rationales = [
        review["rationale"],
        *[c["rationale"] for c in review["claims"]],
        *[c["rationale"] for c in review["code_checks"]],
        *[b["rationale"] for b in review["behavior"].values()],
    ]
    if any(not rationale.strip() for rationale in rationales):
        raise ValueError("Review rationales must contain substantive text")
    declared = {c["quote"] for c in packet["decision"].get("claims", [])}
    quoted = set()
    for claim in review["claims"]:
        source = registry[claim["source_id"]]
        if not claim["quote"].strip() or claim["quote"] not in source["text"]:
            raise ValueError("Ungrounded review quotation")
        quoted.add(claim["quote"])
        if claim["lifecycle"] in ("corrected", "abandoned"):
            revision = registry.get(claim["revision_source_id"])
            if (
                claim["quote"] in declared
                or not revision
                or revision["revision"] <= source["revision"]
                or not claim["revision_quote"].strip()
                or claim["revision_quote"] not in revision["text"]
            ):
                raise ValueError(
                    "Correction/abandonment requires a later grounded revision; final declared claims remain assessable"
                )
        elif claim["revision_source_id"] or claim["revision_quote"]:
            raise ValueError("Active claims must leave revision fields empty")
    if not declared <= quoted:
        raise ValueError("Review omitted declared claims")
    active = [
        c for c in review["claims"] if c["lifecycle"] not in ("corrected", "abandoned")
    ]
    if (
        any(c["decisive"] and c["identification"] == "overclaimed" for c in active)
        and review["identification_discipline"] > 1
    ):
        raise ValueError("Decisive active overclaim caps identification at 1")
    if (
        any(
            c["decisive"] and c["mechanistic"] and c["truth"] == "contradicted"
            for c in active
        )
        and review["mechanism_content"] > 1
    ):
        raise ValueError("Decisive active mechanism contradiction caps mechanism at 1")
    if not review["positive_mechanistic_content"] and review["mechanism_content"] > 1:
        raise ValueError("Generic caution cannot earn mechanism credit")
    mechanisms = [c for c in active if c["mechanistic"]]
    if (
        review["positive_mechanistic_content"]
        and (not mechanisms or all(c["warrant"] == "unsupported" for c in mechanisms))
        and review["evidence_warrant"] > 1
    ):
        raise ValueError("Wholly unsupported positive account caps warrant at 1")
    checked = []
    for check in review["code_checks"]:
        checked.append(check["source_id"])
        if (
            not check["quote"].strip()
            or check["quote"] not in registry[check["source_id"]]["text"]
        ):
            raise ValueError("Ungrounded code check")
    if sorted(checked) != sorted(packet["required_code_source_ids"]):
        raise ValueError(
            "Review must check every available analysis/predictor code source exactly once"
        )
    for behavior in review["behavior"].values():
        if behavior["observed"] == "yes" and not behavior["source_ids"]:
            raise ValueError("Observed behavior needs evidence")
    return review


def assess_once(path, packet, spec, timeout=600):
    """Fresh diagnostic call; no retry or reinterpretation of historical ratings."""
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
    )
    if len(prompt.encode()) + len(json.dumps(schema).encode()) > 2 * 1024 * 1024:
        result.update(
            status="needs_manual_review",
            reason="Full packet exceeds 2 MiB; no truncation or model call.",
        )
    else:
        receipt = run_call(path / "call", prompt, schema, spec, path, timeout)
        error, review = None, None
        try:
            if receipt["status"] != "succeeded":
                raise ValueError("Review call failed")
            review = validate_review(receipt["response"], packet)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        result.update(
            status="completed" if error is None else "failed",
            review=review,
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
    too_large = (
        len(prompt.encode()) + len(json.dumps(schema).encode()) > 2 * 1024 * 1024
    )
    if result["status"] == "needs_manual_review":
        if (
            not too_large
            or (path / "call").exists()
            or result.get("reason")
            != "Full packet exceeds 2 MiB; no truncation or model call."
        ):
            raise ValueError("Manual review gate changed")
        return result
    if too_large:
        raise ValueError("Oversized packet bypassed manual review")
    receipt = verify_call(path / "call", expected_prompt=prompt, expected_schema=schema)
    if (
        result["receipt_sha256"] != file_hash(path / "call/receipt.json")
        or result["usage"] != receipt.get("usage")
        or receipt.get("timeout_seconds", result["timeout"]) != result["timeout"]
        or receipt.get("backend_spec_hash") != content_hash(result["backend_spec"])
    ):
        raise ValueError("Review call provenance changed")
    review, error = None, None
    try:
        if receipt["status"] != "succeeded":
            raise ValueError("Review call failed")
        review = validate_review(receipt["response"], packet)
    except (ValueError, TypeError, KeyError) as exc:
        error = str(exc)
    if (
        result["status"] != ("completed" if error is None else "failed")
        or result["review"] != review
        or result["error"] != error
    ):
        raise ValueError("Review outcome changed")
    return result
