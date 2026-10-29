"""Target meaning, contemporaneous support, later critique and truth are separate.

Formal validation establishes bounded location/provenance/stage constraints. It
does not establish entailment, scientific correctness, expertise or philosophical
validity. Historical reviewers/packets remain unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator
from .delivery_v4 import validate_data

VERSION = "claim-review-v5.0"


def raw_sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    validate_data(value)
    return json.dumps(value, sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def H(value):
    return raw_sha(encoded(value))


def loads(raw):
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError("Duplicate assessment JSON key")
            result[key] = value
        return result
    return validate_data(json.loads(raw, object_pairs_hook=unique))


_DATA = loads(Path(__file__).with_name("assessment_v5_schema.yaml").read_bytes())
KINDS = {name:value["properties"]["document_type"]["const"]
         for name,value in _DATA["$defs"].items() if "document_type" in value.get("properties",{})}
AUTHORED = {"authored_prose","authored_code"}
PURPOSES = {
    "measurement":{"observation"},
    "execution_result":{"execution_output","asset_inspection"},
    "authored_premise":AUTHORED,
    "static_program":{"authored_code"},
    "provided_contract":{"public_contract"},
    "prior_operational_fact":{"operational_record","artifact_manifest"},
}


def schema(name):
    result = copy.deepcopy(_DATA["$defs"][name])
    result["$defs"] = copy.deepcopy(_DATA["$defs"])
    result["$schema"] = _DATA["$schema"]
    return result


def _shape(name, value):
    validate_data(value)
    errors = sorted(Draft202012Validator(schema(name)).iter_errors(value),
                    key=lambda e:str(list(e.absolute_path)))
    if errors:
        raise ValueError(f"{name}: {list(errors[0].absolute_path)}: {errors[0].message}")


def _unique(values, label):
    if len(set(values)) != len(values):
        raise ValueError("Duplicate "+label)
    return set(values)


def span(source, location):
    text = source["text"]
    start,end = location["start"],location["end"]
    if not 0 <= start < end <= len(text) or text[start:end] != location["quote"]:
        raise ValueError("Source span is not an exact Unicode-codepoint location")


def validate_public_packet(packet, *, expected_sources=None):
    _shape("public_packet",packet)
    sources = {s["id"]:s for s in packet["sources"]}
    _unique([s["id"] for s in packet["sources"]],"source IDs")
    if expected_sources is not None and sources != expected_sources:
        raise ValueError("Packet sources differ from authoritative custody")
    for source in sources.values():
        if source["text_utf8_sha256"] != raw_sha(source["text"].encode("utf-8")):
            raise ValueError("Source text raw UTF-8 hash changed")
        if source["visibility"] == "private_truth" or source["kind"] == "private_truth":
            raise ValueError("Private truth cannot enter a public packet")
        if source["kind"] in AUTHORED:
            if (
                source["origin"] != "authored" or not source["authoring_id"]
                or type(source["actor_cutoff"]) is not int
                or type(source["event_revision"]) is not int
                or source["event_revision"] <= source["actor_cutoff"]
            ):
                raise ValueError("Authored source needs its actual author/cutoff/append provenance")
        elif source["origin"] == "authored" or source["authoring_id"] is not None or source["actor_cutoff"] is not None:
            raise ValueError("Generated/provided records cannot acquire source authorship")
        if source["kind"] in {"observation","execution_output","artifact_manifest","asset_inspection"} and source["origin"] != "generated":
            raise ValueError("Execution/observation/assets must retain generated origin")
        if source["kind"] == "public_contract" and source["origin"] != "provided":
            raise ValueError("Public contract has provided origin")
        if source["visibility"] == "actor_public":
            if (
                type(source["event_revision"]) is not int or type(source["available_revision"]) is not int
                or source["event_revision"] > source["available_revision"]
                or source["available_revision"] > packet["review_asof_revision"]
            ):
                raise ValueError("Actor-public record has invalid actual availability")
        elif (
            source["kind"] in AUTHORED or source["authoring_id"] is not None
            or any(source[k] is not None for k in ("actor_cutoff","event_revision","available_revision"))
        ):
            raise ValueError("Reviewer-public records have no inferred actor availability")
        asset = source["asset_ref"]
        if asset:
            producer = sources.get(asset["producer_source_id"])
            if (
                asset["campaign_id"] != packet["campaign_id"] or not producer
                or producer["kind"] != "authored_code" or producer["operation_kind"] not in {"analysis","prepare"}
                or asset["source_revision"] != producer["actor_cutoff"]
                or asset["source_revision"] >= asset["available_revision"]
                or asset["available_revision"] > packet["review_asof_revision"]
                or (source["visibility"] == "actor_public" and source["available_revision"] < asset["available_revision"])
                or (source["visibility"] == "reviewer_public" and source["kind"] != "asset_inspection")
            ):
                raise ValueError("Asset source/producer/campaign/availability binding is invalid")
    targets = {t["id"]:t for t in packet["targets"]}
    _unique([t["id"] for t in packet["targets"]],"target IDs")
    required = _unique(packet["required_target_ids"],"required targets")
    decisions = _unique(packet["scientific_decision_target_ids"],"scientific decision targets")
    required_code = _unique(packet["required_code_source_ids"],"required code IDs")
    if not required <= set(targets) or not decisions <= set(targets):
        raise ValueError("Required/decision target does not resolve")
    if not targets and not required_code:
        raise ValueError("Assessment packet has neither targets nor code checks")
    if any(identity not in sources or sources[identity]["kind"] != "authored_code" for identity in required_code):
        raise ValueError("Required code source is missing or has wrong origin")
    for target in targets.values():
        source = sources.get(target["source_id"])
        if not source:
            raise ValueError("Target source is absent")
        span(source,target["span"])
        context = target["context_span"]
        if not 0 <= context["start"] <= target["span"]["start"] < target["span"]["end"] <= context["end"] <= len(source["text"]):
            raise ValueError("Target context must contain its exact source span")
        if source["kind"] in AUTHORED:
            if target["actor_cutoff"] != source["actor_cutoff"] or target["endorsement_source_id"] is not None:
                raise ValueError("Authored target cannot change its source cutoff")
        elif target["endorsement_source_id"] is None:
            if target["actor_cutoff"] is not None:
                raise ValueError("Unendorsed generated target has no actor cutoff")
        else:
            endorsement = sources.get(target["endorsement_source_id"])
            if (
                not endorsement or endorsement["kind"] not in AUTHORED
                or target["actor_cutoff"] != endorsement["actor_cutoff"]
                or source["visibility"] != "actor_public"
                or source["available_revision"] > endorsement["actor_cutoff"]
            ):
                raise ValueError("Generated target endorsement lacked actual prior access")
    stop = packet["scientific_stop_revision"]
    if stop is not None and stop > packet["review_asof_revision"]:
        raise ValueError("Frozen scientific stop is after packet cutoff")
    for identity in decisions:
        source = sources[targets[identity]["source_id"]]
        if source["operation_kind"] != "stop" or source["event_revision"] != stop:
            raise ValueError("Scientific decision target is not bound to the actual frozen stop")
    if packet["presentation_mode"] == "actor_cutoff":
        cutoff = packet["frame_cutoff"]
        if type(cutoff) is not int or any(t["actor_cutoff"] != cutoff for t in targets.values()):
            raise ValueError("Actor presentation requires one exact actor cutoff")
        author_ids = {sources[t["endorsement_source_id"] or t["source_id"]]["authoring_id"] for t in targets.values()}
        author_ids.update(s["authoring_id"] for s in sources.values()
                          if s["kind"] in AUTHORED and s["actor_cutoff"] == cutoff)
        if len(author_ids) > 1:
            raise ValueError("A strict actor frame cannot mix authoring responses")
        if (
            any(not re.fullmatch(r"s[0-9]{4,8}", identity) for identity in sources)
            or any(not re.fullmatch(r"t[0-9]{4,8}", identity) for identity in targets)
        ):
            raise ValueError("Actor presentation requires neutral numeric source/target identities")
        for source in sources.values():
            same = (source["kind"] in AUTHORED and source["actor_cutoff"] == cutoff and source["authoring_id"] in author_ids)
            if source["visibility"] != "actor_public" or (source["available_revision"] > cutoff and not same):
                raise ValueError("Actor view includes later/generated/reviewer evidence")
    elif packet["frame_cutoff"] is not None:
        raise ValueError("Retrospective packet has no single actor presentation cutoff")
    return packet


def actor_presentation(packet):
    """Do not disclose custody metadata or later outcome-dependent identities.

    Packet creation/target selection must itself use an outcome-independent rule.
    This projection cannot retroactively prove an owner's selection was blinded.
    """
    validate_public_packet(packet)
    if packet["presentation_mode"] != "actor_cutoff":
        raise ValueError("Actor presentation requires an explicit single-cutoff packet")
    return dict(
        version=VERSION, mode="single_actor_frame",
        sources=[dict(id=s["id"],kind=s["kind"],text=s["text"],origin=s["origin"],
                      exposure="current_authorship" if s["kind"] in AUTHORED and s["actor_cutoff"] == packet["frame_cutoff"] else "prior_public")
                 for s in packet["sources"]],
        targets=[{k:t[k] for k in ("id","source_id","span","context_span","kind","force","proposition","endpoint")}
                 for t in packet["targets"]],
        required_target_ids=packet["required_target_ids"],
        required_code_source_ids=packet["required_code_source_ids"],
        unreviewed_scope=["Only this authorship frame and the listed available sources are presented; other frames and later outcomes are withheld."],
        limits=(
            "Only the listed scientific-evidence projection is judged, not every actual prompt field or retrievable file. "
            "Supplied assignment/guidance may be masked; exact authored disclosures can still identify it. "
            "Common scientific and operational definitions must be retained. Policy compliance is outside this common-warrant review. "
            "If meaning or support depends on omitted guidance or undisplayed indexed files, retain an unresolved/adjudication-needed judgment; "
            "absence from this display does not establish absence from the campaign. "
            "Owner custody, append/availability metadata, registration status and later totals are withheld; literal prior revisions in supplied evidence may remain. "
            "Only the chosen frame is judged. Final endpoint materiality and later revision/fulfillment are deferred."
        ),
    )


def response_schema(name,fields):
    """Only requested annotations and their reachable schema definitions."""
    result = schema(name)
    result["properties"] = {k:v for k,v in result["properties"].items() if k in fields}
    result["required"] = list(result["properties"])
    definitions = result.pop("$defs")
    needed = set()
    def visit(value):
        if isinstance(value,dict):
            ref = value.get("$ref")
            if ref:
                identity = ref.removeprefix("#/$defs/")
                if identity not in needed:
                    needed.add(identity)
                    visit(definitions[identity])
            for child in value.values():
                visit(child)
        elif isinstance(value,list):
            for child in value:
                visit(child)
    visit(result)
    result["$defs"] = {k:v for k,v in definitions.items() if k in needed}
    return result


def public_response_schema():
    """Semantic annotations only; hidden custody hashes never enter transport."""
    return response_schema("public_review",{"annotations","revision_relations","code_checks","unreviewed_scope"})


def assemble_public_review(response,packet,*,reviewer,rubric_hash):
    """Lossless annotation copy plus fixed owner custody metadata; no repairs."""
    Draft202012Validator(public_response_schema()).validate(response)
    presented = actor_presentation(packet) if packet["presentation_mode"] == "actor_cutoff" else packet
    review = dict(
        document_type=KINDS["public_review"],public_packet_hash=H(packet),
        target_manifest_hash=H(packet["targets"]),rubric_hash=rubric_hash,
        reviewer=copy.deepcopy(reviewer),presentation_sha256=H(presented),
        **copy.deepcopy(response),
    )
    return validate_public_review(review,packet,rubric_hash=rubric_hash,reviewer=reviewer)


def _author(target,sources):
    return sources[target["endorsement_source_id"] or target["source_id"]]


def _actor_support(support,target,sources):
    cutoff = target["actor_cutoff"]
    refs = support["actor_basis_refs"]
    _unique([(r["source_id"],r["purpose"]) for r in refs],"actor support references")
    _unique(support["basis_modes"],"basis modes")
    if cutoff is None:
        if refs or support["verdict"] != "not_applicable":
            raise ValueError("Unendorsed generated record cannot have actor warrant")
        return
    author = _author(target,sources)
    for ref in refs:
        source = sources.get(ref["source_id"])
        if not source or source["visibility"] != "actor_public" or source["kind"] not in PURPOSES[ref["purpose"]]:
            raise ValueError("Actor support has wrong visibility/source purpose")
        same = (
            ref["purpose"] in {"authored_premise","static_program"}
            and source["kind"] in AUTHORED and source["origin"] == "authored"
            and source["authoring_id"] == author["authoring_id"] and source["actor_cutoff"] == cutoff
        )
        if source["available_revision"] > cutoff and not same:
            raise ValueError("Later observation/execution cannot support an earlier assertion")
    if support["verdict"] in {"supported","assumption_dependent"} and (not refs or not support["basis_modes"]):
        raise ValueError("Resolved positive support needs eligible provenance and basis mode")
    if (support["verdict"] == "assumption_dependent" or "conditional" in support["basis_modes"]) and not support["assumptions"]:
        raise ValueError("Conditional support needs explicit assumptions")


def _critique_refs(refs,sources):
    _unique([(r["source_id"],r["purpose"],H(r["span"])) for r in refs],"critique references")
    for ref in refs:
        source = sources.get(ref["source_id"])
        if not source or source["visibility"] == "private_truth":
            raise ValueError("Public critique cannot cite private truth")
        if ref["span"] is not None:
            span(source,ref["span"])


def _code_checks(checks,packet):
    sources = {s["id"]:s for s in packet["sources"]}
    ids = [c["source_id"] for c in checks]
    _unique(ids,"code checks")
    if set(ids) != set(packet["required_code_source_ids"]):
        raise ValueError("Required code checks must appear exactly once")
    for check in checks:
        source = sources[check["source_id"]]
        span(source,check["span"])
        _critique_refs(check["critique_refs"],sources)
        # A generated stdout source alone does not establish which code/inputs
        # were executed, or that a finite probe was actually performed. Until
        # typed execution-receipt binding is implemented, these scopes are
        # deliberately unavailable, even if a reviewer claims to have used them.
        if check["scope"] not in {"static","unverified"}:
            raise ValueError("Execution/probe certification requires an implemented receipt binding")
        if check["scope"] == "unverified" and check["judgment"] != "unverified":
            raise ValueError("Unverified code scope cannot certify validity")


def _materiality(material,target,packet):
    if material["endpoint"] != target["endpoint"]:
        raise ValueError("Materiality cannot silently change the named endpoint")
    sources = {s["id"]:s for s in packet["sources"]}
    targets = {t["id"]:t for t in packet["targets"]}
    ids = _unique(material["dependent_target_ids"],"materiality targets")
    if not ids <= set(targets):
        raise ValueError("Materiality dependency target is absent")
    dependencies = material["dependency_refs"]
    for ref in dependencies:
        source = sources.get(ref["source_id"])
        if not source or source["visibility"] == "private_truth" or ref["endpoint"] != material["endpoint"]:
            raise ValueError("Materiality dependency has wrong public endpoint/source")
        span(source,ref["span"])
    if material["judgment"] == "material" and not ids and not dependencies:
        raise ValueError("Materiality needs an explicit endpoint dependency")
    if material["endpoint"] == "scientific_conclusion":
        if not ids <= set(packet["scientific_decision_target_ids"]):
            raise ValueError("Scientific materiality depends on an unrelated target")
        for ref in dependencies:
            source = sources[ref["source_id"]]
            if source["operation_kind"] != "stop" or source["event_revision"] != packet["scientific_stop_revision"]:
                raise ValueError("Scientific dependency is not the frozen scientific decision")


def _revisions(relations,packet,*, original_targets=None):
    sources = {s["id"]:s for s in packet["sources"]}
    targets = original_targets or {t["id"]:t for t in packet["targets"]}
    _unique([(r["original_target_id"],r["later_source_id"],r["later_span"]["start"],r["later_span"]["end"]) for r in relations],"revision relations")
    for relation in relations:
        target = targets.get(relation["original_target_id"])
        later = sources.get(relation["later_source_id"])
        if not target or not later:
            raise ValueError("Revision source/target is absent")
        original = sources[target["source_id"]]
        span(later,relation["later_span"])
        if (
            original["kind"] not in AUTHORED or later["kind"] not in AUTHORED
            or later["authoring_id"] == original["authoring_id"]
            or later["actor_cutoff"] < original["available_revision"]
            or later["event_revision"] <= original["event_revision"]
        ):
            raise ValueError("Historical revision needs a distinct later author with prior access")


def validate_public_review(review,packet,*, rubric_hash,reviewer=None,presentation=None):
    validate_public_packet(packet)
    _shape("public_review",review)
    if (
        review["public_packet_hash"] != H(packet) or review["target_manifest_hash"] != H(packet["targets"])
        or review["rubric_hash"] != rubric_hash or reviewer is not None and review["reviewer"] != reviewer
    ):
        raise ValueError("Public review packet/target/rubric/reviewer binding changed")
    expected_presentation = actor_presentation(packet) if packet["presentation_mode"] == "actor_cutoff" else packet
    if presentation is not None and presentation != expected_presentation:
        raise ValueError("Public presentation differs from its declared projection")
    if review["presentation_sha256"] != H(expected_presentation):
        raise ValueError("Reviewer presentation bytes/content binding changed")
    sources = {s["id"]:s for s in packet["sources"]}
    targets = {t["id"]:t for t in packet["targets"]}
    annotation_ids = [a["target_id"] for a in review["annotations"]]
    if _unique(annotation_ids,"review targets") != set(targets):
        raise ValueError("Every frozen target must be assessed exactly once")
    for annotation in review["annotations"]:
        target = targets[annotation["target_id"]]
        _actor_support(annotation["actor_support"],target,sources)
        _critique_refs(annotation["critique_refs"],sources)
        _materiality(annotation["materiality"],target,packet)
        if packet["presentation_mode"] == "actor_cutoff":
            material = annotation["materiality"]
            if material["judgment"] != "not_applicable" or material["dependent_target_ids"] or material["dependency_refs"]:
                raise ValueError("Final endpoint materiality is deferred from actor-only review")
    if packet["presentation_mode"] == "actor_cutoff" and review["revision_relations"]:
        raise ValueError("Later revision judgments are deferred from actor-only review")
    _revisions(review["revision_relations"],packet)
    _code_checks(review["code_checks"],packet)
    return review


def validate_truth_packet(packet, *, public_packet=None):
    _shape("truth_packet",packet)
    validate_public_packet(packet["public_packet"])
    if packet["public_packet_hash"] != H(packet["public_packet"]) or public_packet is not None and packet["public_packet"] != public_packet:
        raise ValueError("Truth packet changed its frozen public record")
    public_ids = {s["id"] for s in packet["public_packet"]["sources"]}
    _unique([s["id"] for s in packet["private_sources"]],"private source IDs")
    for source in packet["private_sources"]:
        if (
            source["id"] in public_ids or source["kind"] != "private_truth"
            or source["visibility"] != "private_truth" or source["origin"] != "provided"
            or any(source[k] is not None for k in ("authoring_id","actor_cutoff","event_revision","available_revision","asset_ref"))
            or source["text_utf8_sha256"] != raw_sha(source["text"].encode())
        ):
            raise ValueError("Private truth source has invalid origin/identity")
    return packet


def validate_truth_review(review,truth,freeze,release,*, public_review,reviewer=None):
    _shape("truth_review",review)
    validate_truth_packet(truth)
    _shape("freeze",freeze);_shape("release",release)
    validate_public_review(public_review,truth["public_packet"],rubric_hash=freeze["rubric_hash"],reviewer=reviewer)
    if (
        release["public_freeze_hash"] != H(freeze) or release["truth_packet_hash"] != H(truth)
        or release["reviewer_id"] != freeze["reviewer_id"]
        or release["ledger_sequence"] <= freeze["ledger_sequence"]
        or review["reviewer"]["id"] != freeze["reviewer_id"]
        or reviewer is not None and review["reviewer"] != reviewer
        or review["public_freeze_hash"] != H(freeze) or review["truth_release_hash"] != H(release)
        or review["truth_packet_hash"] != H(truth)
        or freeze["public_packet_hash"] != truth["public_packet_hash"]
        or freeze["target_manifest_hash"] != H(truth["public_packet"]["targets"])
        or H(public_review) != freeze["public_review_hash"]
        or freeze["presentation_sha256"] != public_review["presentation_sha256"]
        or freeze["reviewer_id"] != public_review["reviewer"]["id"]
        or review["reviewer"] != public_review["reviewer"]
    ):
        raise ValueError("Truth review is not bound to its actual frozen public stage/release")
    targets = {t["id"] for t in truth["public_packet"]["targets"]}
    if _unique([a["target_id"] for a in review["annotations"]],"truth targets") != targets:
        raise ValueError("Truth review targets changed")
    ids = {s["id"] for s in truth["public_packet"]["sources"]+truth["private_sources"]}
    interpretations = {a["target_id"]:a for a in public_review["annotations"]}
    for annotation in review["annotations"]:
        refs = annotation["truth_evidence_ids"]
        if not _unique(refs,"truth evidence IDs") <= ids:
            raise ValueError("Truth evidence source does not resolve")
        if annotation["judgment"] in {"consistent","contradicted"} and not refs:
            raise ValueError("Resolved truth judgment needs evidence provenance")
        original = interpretations[annotation["target_id"]]
        # A disputed proposed target may have a precise alternative frozen by
        # the reviewer. Truth then concerns that alternative conditionally; this
        # does not resolve the recorded interpretation dispute or adjudicate it.
        if (
            annotation["judgment"] in {"consistent","contradicted"}
            and (original["interpretation_status"] == "adjudication_needed"
                 or original["interpretation"]["kind"] == "ambiguous"
                 or original["interpretation"]["force"] == "ambiguous")
        ):
            raise ValueError("Unresolved public meaning cannot become resolved truth by reinterpretation")
    return review


def validate_later_packet(actor_packet,later_packet):
    validate_public_packet(actor_packet);validate_public_packet(later_packet)
    if later_packet["presentation_mode"] != "retrospective_public":
        raise ValueError("Later critique needs an explicit retrospective public record")
    if (
        actor_packet["targets"] != later_packet["targets"]
        or actor_packet["required_target_ids"] != later_packet["required_target_ids"]
        or actor_packet["required_code_source_ids"] != later_packet["required_code_source_ids"]
        or actor_packet["campaign_id"] != later_packet["campaign_id"]
        or actor_packet["instrument_version"] != later_packet["instrument_version"]
        or later_packet["review_asof_revision"] < actor_packet["review_asof_revision"]
        or (actor_packet["scientific_stop_revision"] is not None
            and actor_packet["scientific_stop_revision"] != later_packet["scientific_stop_revision"])
    ):
        raise ValueError("Later packet changed frozen actor coverage/campaign")
    later_sources = {s["id"]:s for s in later_packet["sources"]}
    if any(later_sources.get(s["id"]) != s for s in actor_packet["sources"]):
        raise ValueError("Later critique rewrote the frozen public sources")
    return later_packet


def validate_later_critique(critique,actor_packet,later_packet,freeze,*, release=None,reviewer=None):
    _shape("later_critique",critique)
    _shape("freeze",freeze)
    if release is not None:
        _shape("release",release)
    validate_later_packet(actor_packet,later_packet)
    presented = actor_presentation(actor_packet) if actor_packet["presentation_mode"] == "actor_cutoff" else actor_packet
    if (
        critique["public_freeze_hash"] != H(freeze) or freeze["public_packet_hash"] != H(actor_packet)
        or critique["later_public_packet_hash"] != H(later_packet)
        or critique["target_manifest_hash"] != H(actor_packet["targets"])
        or freeze["target_manifest_hash"] != H(actor_packet["targets"])
        or freeze["presentation_sha256"] != H(presented)
        or critique["reviewer"]["id"] != freeze["reviewer_id"]
        or reviewer is not None and critique["reviewer"] != reviewer
    ):
        raise ValueError("Later critique changed the frozen actor targets/reviewer/presentation")
    if critique["exposure"] == "private_informed":
        if (
            not release or critique["truth_release_hash"] != H(release)
            or release["public_freeze_hash"] != H(freeze)
            or release["reviewer_id"] != freeze["reviewer_id"]
            or release["ledger_sequence"] <= freeze["ledger_sequence"]
        ):
            raise ValueError("Private-informed critique lacks its actual truth release")
    elif critique["truth_release_hash"] is not None or critique["exposure"] == "public_only" and release is not None:
        raise ValueError("Critique exposure record contradicts private release")
    targets = {t["id"]:t for t in actor_packet["targets"]}
    if _unique([a["target_id"] for a in critique["annotations"]],"later critique targets") != set(targets):
        raise ValueError("Later critique must preserve every original target")
    sources = {s["id"]:s for s in later_packet["sources"]}
    for annotation in critique["annotations"]:
        _critique_refs(annotation["critique_refs"],sources)
        _materiality(annotation["materiality"],targets[annotation["target_id"]],later_packet)
    _revisions(critique["revision_relations"],later_packet)
    _code_checks(critique["code_checks"],later_packet)
    return critique
