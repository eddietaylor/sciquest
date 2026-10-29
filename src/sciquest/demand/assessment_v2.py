"""Claim-grounded explanation and behavior review, masked to treatment identity."""

from __future__ import annotations

import json
import copy
from pathlib import Path

from sciquest.io import content_hash, file_hash
from .grading import packet as legacy_packet
from .worldbank import dump, verify_call
from .backend import run_call

VERSION = "claim-review-v2.3"
AXES = (
    "mechanism_content",
    "evidence_warrant",
    "identification_discipline",
    "test_revision_quality",
)
CLAIM_FIELDS = {
    "source_id": dict(type="string"),
    "quote": dict(type="string"),
    "truth": dict(type="string", enum=["consistent", "contradicted", "unresolved"]),
    "warrant": dict(
        type="string", enum=["supported", "assumption_dependent", "unsupported"]
    ),
    "identification": dict(
        type="string", enum=["appropriately_scoped", "overclaimed", "not_applicable"]
    ),
    "evidence_ids": dict(type="array", items=dict(type="string")),
    "rationale": dict(type="string"),
}
BEHAVIORS = (
    "substantive_rivals",
    "discriminating_design",
    "conditional_belief_update",
    "diagnostic_failure_response",
)
SCHEMA = dict(
    type="object",
    properties={
        **{a: dict(type="integer", enum=[0, 1, 2, 3, 4]) for a in AXES},
        "claims": dict(
            type="array",
            items=dict(
                type="object",
                properties=CLAIM_FIELDS,
                required=list(CLAIM_FIELDS),
                additionalProperties=False,
            ),
        ),
        "critical_errors": dict(type="array", items=dict(type="string")),
        "positive_mechanistic_content": dict(type="boolean"),
        "behavior": dict(
            type="object",
            properties={
                b: dict(
                    type="object",
                    properties=dict(
                        observed=dict(
                            type="string",
                            enum=["yes", "partial", "no", "not_applicable"],
                        ),
                        source_ids=dict(type="array", items=dict(type="string")),
                        rationale=dict(type="string"),
                    ),
                    required=["observed", "source_ids", "rationale"],
                    additionalProperties=False,
                )
                for b in BEHAVIORS
            },
            required=list(BEHAVIORS),
            additionalProperties=False,
        ),
        "rationale": dict(type="string"),
    },
    required=[
        *AXES,
        "claims",
        "critical_errors",
        "positive_mechanistic_content",
        "behavior",
        "rationale",
    ],
    additionalProperties=False,
)
PROMPT = """Assess this anonymous scientific trace using the supplied PRIVATE truth and
access/measurement contract. You are an evaluator, never the scientist. Policy
identity, instructions, numerical forecast scores and other reviews are withheld.
Style can reveal a treatment: do not reward rhetoric or a philosophical label.

First review the decisive scientific claims, including EVERY declared claim quote
and any material assertion elsewhere in the decision or plans. For each give its
exact source_id and exact quote (declared claim IDs are valid source IDs too); truth (consistent/contradicted/unresolved),
warrant (supported/assumption_dependent/unsupported), identification
(appropriately_scoped/overclaimed/not_applicable), actual evidence_ids and rationale.
A correct guess can be consistent and unsupported. An unknown law is not recovered
by generic caution. Distinguish causal identification in population under stated
assumptions, finite-budget precision, transfer and exact-law recovery. More data
from an observationally equivalent access kernel cannot identify its causal law.
Interventions alone do not identify arbitrary off-grid laws. Do not demand that a
scientist guess an undisclosed model class, noise distribution or true parameter.
Judge the scope actually claimed. Unsupported blanket 'nothing can be learned'
can also be wrong when local controlled comparisons are available.

Rate four independent axes 0..4:
mechanism_content: 0 contradicted central mechanism or no substantive content;
1 small correct fragment; 2 partly faithful explanation with important gaps;
3 substantive correct account of relevant responses/capacity/measurement;
4 faithful mechanism at the claimed resolution. Unknown alone earns at most 1.
evidence_warrant: 0 assertion/no support; 1 weak or cherry-picked support;
2 plausible conditional inference with important gaps; 3 defensible integration;
4 systematic integration of relevant evidence, dependence, contradictions and limits.
identification_discipline: 0 major overclaim; 1 repeated or central scope error;
2 mixed, vague or generic caution; 3 specific appropriate limitations;
4 explicit, defensible claim-by-claim identification reasoning, including access
and assumptions and why plausible alternatives are or are not distinguished.
Cautious wording alone cannot earn 4. Numerical interval calibration is scored
by a separate proper score; NEVER infer numerical calibration from this rating.
test_revision_quality: 0 no meaningful test; 1 weak sampling; 2 plausible test;
3 discriminating or justified measurement/replication with evidence-based response;
4 strong diagnostic tests with actual warranted follow-through. Belief changes
alone and declared failure_action fields alone do not establish revision.

Set positive_mechanistic_content honestly; list critical_errors with explanations.
An overclaimed decisive claim caps identification_discipline at 1. A contradicted
decisive mechanism caps mechanism_content at 1. A wholly unsupported positive
account caps evidence_warrant at 1. Unsupported caution also needs assessment.

Audit ALL four behavior indicators regardless of the unknown assigned treatment:
substantive rivals (not renamed synonyms), discriminating design (predictions and
measurement can distinguish rivals under stated error assumptions), conditional
belief update (prior/likelihood/dependence and posterior used in later reasoning),
diagnostic failure response (anticipated alternative-sensitive failure plus actual
follow-through checking auxiliaries). Cite source_ids. Mark partial/no/NA rather
than inventing evidence. A valid numerical calculation does not certify a credible
likelihood or test power. This assesses observable behavior, not sincere allegiance.

Return concise rationales. Contradictory claims are not repaired by appending
'provisional'. Honest scoped abstention can rate highly on identification while
remaining low on mechanism_content. No composite scientific-success score.
These are diagnostic model judgments; shared model biases and partial blinding
remain. They require calibration and eventual human adjudication for paper claims.
"""


def review_packet(session):
    base = legacy_packet(session)
    if base is None:
        return None
    sources = {
        "decision." + k: v for k, v in base["decision"].items() if isinstance(v, str)
    }
    observations = {}
    for i, q in enumerate(base["evidence"], 1):
        identity = q["observation"]["id"]
        observations[identity] = q["observation"]
        for k, v in q["plan"].items():
            if isinstance(v, str):
                sources[f"{identity}.plan.{k}"] = v
    # Include methodological records uniformly, without enforced labels/identity.
    # Their prose can still reveal style: masking is explicitly partial.
    with session.transaction() as (c, state):
        all_events = [e["payload"] for e in session._events(c)]
        query_events = [
            e["payload"] for e in session._events(c) if e["payload"]["type"] == "query"
        ]
    analysis_metadata = []
    for event in all_events:
        if event["type"] == "scientist_analysis":
            a = event["analysis"]
            identity = f"analysis.a{len(analysis_metadata) + 1:04d}"
            output = a["output"]
            if a.get("full_output_file"):
                raw = session.path / a["full_output_file"]
                if file_hash(raw) != a["full_output_hash"]:
                    raise ValueError("Review analysis archive changed")
                output = raw.read_text(errors="replace")
            sources[identity + ".code"] = a["code"]
            sources[identity + ".output"] = output
            analysis_metadata.append(dict(id=identity, status=a["status"]))
        elif event["type"] == "campaign_notes":
            sources["campaign_notes"] = event["text"]
    records = []
    for q in query_events:
        identity = q["observation"]["id"]
        extensions = {
            k: q["plan"][k]
            for k in (
                "rivals",
                "discriminating_prediction_id",
                "belief_model",
                "severe_test",
            )
            if k in q["plan"]
        }
        for k, v in extensions.items():
            sources[f"{identity}.plan.{k}"] = json.dumps(v, sort_keys=True)
        records.append(dict(observation_id=identity, records=extensions))
    return dict(
        version=VERSION,
        decision=base["decision"],
        evidence=base["evidence"],
        sources=sources,
        observations=observations,
        methodological_records=records,
        analysis_metadata=analysis_metadata,
        private_world=base["private_world"],
        truth_contract=base["truth_contract"],
        private_intervention_truth=base["private_intervention_truth"],
        access=dict(
            interventions_available=state["public"]["interventions_available"],
            query_prices=state["public"]["prices"],
            query_contexts=state["public"]["contexts"],
        ),
        private_identification_limits="Observational-only worlds have a distinct causal/decision twin with identical available kernels. Query-setting means under stable control and calibration assumptions are population-identifiable with interventions; finite budgets do not guarantee accurate estimates. Finite query prices/contexts cannot nonparametrically identify the entire off-grid response. No hidden filtered-grammar membership was supplied to the scientist.",
    )


def citation_sources(packet):
    """Resolve declared claim IDs only to their exact, already-grounded quotations."""
    sources = dict(packet["sources"])
    for claim in packet["decision"].get("claims", []):
        identity, quote = claim["id"], claim["quote"]
        if not any(quote in text for text in packet["sources"].values()):
            raise ValueError("Declared claim has no prose source")
        if identity in sources and sources[identity] != quote:
            raise ValueError("Ambiguous claim source ID")
        sources[identity] = quote
    for identity, observation in packet["observations"].items():
        if identity in sources:
            raise ValueError("Ambiguous observation citation ID")
        sources[identity] = json.dumps(observation, sort_keys=True)
    return sources


def schema_for_packet(packet):
    schema = copy.deepcopy(SCHEMA)
    identities = sorted(citation_sources(packet))
    schema["properties"]["claims"]["items"]["properties"]["source_id"]["enum"] = (
        identities
    )
    for behavior in schema["properties"]["behavior"]["properties"].values():
        behavior["properties"]["source_ids"]["items"]["enum"] = identities
    return schema


def validate_review(review, packet):
    sources = citation_sources(packet)
    if not isinstance(review, dict) or set(review) != set(SCHEMA["properties"]):
        raise ValueError("Invalid review fields")
    if any(type(review[a]) is not int or not 0 <= review[a] <= 4 for a in AXES):
        raise ValueError("Invalid review rating")
    if (
        not isinstance(review["rationale"], str)
        or not review["rationale"].strip()
        or type(review["positive_mechanistic_content"]) is not bool
    ):
        raise ValueError("Review needs rationale and positive-content judgment")
    if not isinstance(review["claims"], list) or not review["claims"]:
        raise ValueError("Review needs grounded claim assessments")
    quoted = set()
    for c in review["claims"]:
        if (
            set(c) != set(CLAIM_FIELDS)
            or c["source_id"] not in sources
            or not isinstance(c["quote"], str)
            or not c["quote"].strip()
            or c["quote"] not in sources[c["source_id"]]
        ):
            raise ValueError("Ungrounded review quotation")
        if (
            c["truth"] not in ("consistent", "contradicted", "unresolved")
            or c["warrant"] not in ("supported", "assumption_dependent", "unsupported")
            or c["identification"]
            not in ("appropriately_scoped", "overclaimed", "not_applicable")
        ):
            raise ValueError("Invalid claim judgment")
        if (
            not isinstance(c["evidence_ids"], list)
            or any(
                not isinstance(x, str) or x not in packet["observations"]
                for x in c["evidence_ids"]
            )
            or not isinstance(c["rationale"], str)
            or not c["rationale"].strip()
        ):
            raise ValueError("Invalid review evidence reference")
        quoted.add(c["quote"])
    if not {c["quote"] for c in packet["decision"].get("claims", [])} <= quoted:
        raise ValueError("Review omitted declared claims")
    if not isinstance(review["critical_errors"], list) or any(
        not isinstance(x, str) or not x.strip() for x in review["critical_errors"]
    ):
        raise ValueError("Invalid critical errors")
    if not review["positive_mechanistic_content"] and review["mechanism_content"] > 1:
        raise ValueError("Generic caution cannot earn mechanism credit")
    if (
        any(c["identification"] == "overclaimed" for c in review["claims"])
        and review["identification_discipline"] > 1
    ):
        raise ValueError("Identification rating conflicts with overclaim")
    if (
        all(c["warrant"] == "unsupported" for c in review["claims"])
        and review["evidence_warrant"] > 1
    ):
        raise ValueError("Unsupported claims cannot earn high warrant")
    behavior = review["behavior"]
    if not isinstance(behavior, dict) or set(behavior) != set(BEHAVIORS):
        raise ValueError("Missing behavior assessments")
    for b in behavior.values():
        if (
            set(b) != {"observed", "source_ids", "rationale"}
            or b["observed"] not in ("yes", "partial", "no", "not_applicable")
            or not isinstance(b["source_ids"], list)
            or any(x not in sources for x in b["source_ids"])
            or not isinstance(b["rationale"], str)
            or not b["rationale"].strip()
            or (b["observed"] == "yes" and not b["source_ids"])
        ):
            raise ValueError("Ungrounded behavior judgment")
    return review


def assess_once(path, packet, spec, timeout=360):
    """One masked review with retained raw response/failure; never auto-retry."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    dump(path / "packet.private.json", packet)
    prompt = PROMPT + "\nANONYMOUS PRIVATE PACKET\n" + json.dumps(packet)
    schema = schema_for_packet(packet)
    if len(prompt.encode()) + len(json.dumps(schema).encode()) > 2 * 1024 * 1024:
        result = dict(
            version=VERSION,
            packet_hash=content_hash(packet),
            status="needs_manual_review",
            reason="Full scientific packet exceeds 2 MiB; no evidence truncated and no model called.",
        )
        dump(path / "assessment.json", result)
        return result
    receipt = run_call(path / "call", prompt, schema, spec, path, timeout)
    error, review = None, None
    try:
        if receipt["status"] != "succeeded":
            raise ValueError("Review call failed")
        review = validate_review(receipt["response"], packet)
    except (ValueError, TypeError, KeyError) as exc:
        error = str(exc)
    result = dict(
        version=VERSION,
        packet_hash=content_hash(packet),
        prompt_hash=content_hash(PROMPT),
        schema_hash=content_hash(schema),
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
    if result["status"] == "needs_manual_review":
        prompt = PROMPT + "\nANONYMOUS PRIVATE PACKET\n" + json.dumps(packet)
        expected = dict(
            version=VERSION,
            packet_hash=content_hash(packet),
            status="needs_manual_review",
            reason="Full scientific packet exceeds 2 MiB; no evidence truncated and no model called.",
        )
        if (
            result != expected
            or packet != expected_packet
            or len(prompt.encode())
            + len(json.dumps(schema_for_packet(packet)).encode())
            <= 2 * 1024 * 1024
            or (path / "call").exists()
        ):
            raise ValueError("Manual-review gate changed")
        return result
    receipt = verify_call(
        path / "call",
        expected_prompt=PROMPT + "\nANONYMOUS PRIVATE PACKET\n" + json.dumps(packet),
        expected_schema=schema_for_packet(packet),
    )
    if (
        result["version"] != VERSION
        or packet != expected_packet
        or result["packet_hash"] != content_hash(packet)
        or result["prompt_hash"] != content_hash(PROMPT)
        or result["schema_hash"] != content_hash(schema_for_packet(packet))
        or result["receipt_sha256"] != file_hash(path / "call/receipt.json")
        or result["usage"] != receipt.get("usage")
    ):
        raise ValueError("Review provenance changed")
    error, review = None, None
    try:
        if receipt["status"] != "succeeded":
            raise ValueError("Review call failed")
        review = validate_review(receipt["response"], packet)
    except (ValueError, TypeError, KeyError) as exc:
        error = str(exc)
    if (
        result["status"] != ("completed" if error is None else "failed")
        or result["error"] != error
        or result["review"] != review
    ):
        raise ValueError("Review outcome changed")
    return result
