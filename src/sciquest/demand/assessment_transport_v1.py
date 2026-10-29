"""Opt-in annotation transport: model judgments, deterministic host bookkeeping.

No scientific interpretation, support, source choice, truth, revision relation or
endpoint reliance is inferred here. Exact quote locations are resolved only in
the source chosen by the reviewer. Legacy returned lists are never deduplicated.
"""

from __future__ import annotations

import copy
from pathlib import Path

from jsonschema import Draft202012Validator

from . import assessment_v5 as a, assessment_source_locations_v1 as source_locations

VERSION = "selective-host-assembly-v1.2"
ASSEMBLY = "claim-review-annotation-assembly-v1.0"
RULES = {
    "source_locations": "Source locations describe original JSON paths and representation, not scientist-provided evidence, endorsement, operation admission or truth. Interpret a JSON value in its original parent context; identical text can have different roles.",
    "actor_deferred": "HOST_DEFERRED: final materiality and revision are absent from actor transport; not model judgments.",
    "code_unit": "HOST_WHOLE_ASSIGNED_CODE_UNIT: assigned source locator, not an exhaustive audit or selected witness.",
    "quotes": "Model selects source/purpose/EXACT quote and optional exact context. Unique literal occurrence only; no repair, normalization or first-match selection.",
    "truth_domain": "truth_packet.public_packet.sources UNION truth_packet.private_sources; excludes later-only public IDs.",
    "later_domain": "Public sources in later_public_packet only; private sources cannot be public critique/dependency evidence.",
    "conditional_assumptions": "If actor_support.verdict is assumption_dependent OR basis_modes contains conditional, actor_support.assumptions must be a nonempty list of explicit model-owned premises. Premises stated only in rationale or elsewhere do not fill this field.",
}


def mode(value):
    if value not in (None, VERSION):
        raise ValueError("Unknown annotation_transport; explicit supported version required")
    return value


def identity():
    root = Path(__file__).parent
    names = ("assessment_transport_v1.py", "assessment_v5.py", "assessment_v5_schema.yaml",
             "assessment_custody_v5.py", "assessment_runner_v5.py", "assessment_source_locations_v1.py")
    return dict(version=VERSION, implementation={n: a.raw_sha((root / n).read_bytes()) for n in names})


def _ids(values):
    return sorted(set(values))


def _public(packet):
    if packet["presentation_mode"] != "actor_cutoff":
        raise ValueError("New annotation transport supports actor_cutoff only; retrospective materiality cannot be deferred")
    a.actor_presentation(packet)
    sources = {s["id"]: s for s in packet["sources"]}
    for sid in packet["required_code_source_ids"]:
        if not sources[sid]["text"]:
            raise ValueError("Assigned whole code unit must be nonempty")
    return sources


def public_contract(packet):
    sources = _public(packet)
    basis = {}
    for target in packet["targets"]:
        author = sources[target["endorsement_source_id"] or target["source_id"]]
        per_purpose = {}
        for purpose, kinds in a.PURPOSES.items():
            eligible = []
            for source in sources.values():
                same = (purpose in {"authored_premise", "static_program"}
                        and source["kind"] in a.AUTHORED and source["origin"] == "authored"
                        and source["authoring_id"] == author["authoring_id"]
                        and source["actor_cutoff"] == target["actor_cutoff"])
                if (source["visibility"] == "actor_public" and source["kind"] in kinds
                    and target["actor_cutoff"] is not None
                    and (source["available_revision"] <= target["actor_cutoff"] or same)):
                    eligible.append(source["id"])
            per_purpose[purpose] = _ids(eligible)
        basis[target["id"]] = per_purpose
    return dict(identity=identity(), stage="public", rules=copy.deepcopy(RULES),
                **source_locations.display_binding(packet),
                targets=[target["id"] for target in packet["targets"]],
                code_units=list(packet["required_code_source_ids"]),
                source_domains=dict(actor_basis_by_target_and_purpose=basis,
                                    critique_refs=_ids(sources)))


def stage2_contract(truth, later):
    a.validate_truth_packet(truth)
    public = truth["public_packet"]
    _public(public)
    a.validate_later_packet(public, later)
    targets = {t["id"]: t for t in public["targets"]}
    original = {s["id"]: s for s in public["sources"]}
    revisions = {}
    for tid, target in targets.items():
        source = original[target["source_id"]]
        revisions[tid] = _ids(s["id"] for s in later["sources"]
                             if source["kind"] in a.AUTHORED and s["kind"] in a.AUTHORED
                             and s["authoring_id"] != source["authoring_id"]
                             and s["actor_cutoff"] >= source["available_revision"]
                             and s["event_revision"] > source["event_revision"])
    return dict(identity=identity(), stage="private", rules=copy.deepcopy(RULES),
                truth_packet_hash=a.H(truth), later_public_packet_hash=a.H(later),
                targets=[target["id"] for target in public["targets"]], code_units=list(public["required_code_source_ids"]),
                source_domains=dict(truth_evidence_ids=_ids(s["id"] for s in public["sources"] + truth["private_sources"]),
                                    critique_and_dependency_refs=_ids(s["id"] for s in later["sources"]),
                                    scientific_stop_dependency_sources=_ids(s["id"] for s in later["sources"]
                                        if s["operation_kind"] == "stop" and s["event_revision"] == later["scientific_stop_revision"]),
                                    revision_sources_by_target=revisions))


def _prune(schema):
    definitions = schema.pop("$defs")
    needed = set()
    def visit(value):
        if isinstance(value, dict):
            ref = value.get("$ref")
            if ref:
                key = ref.removeprefix("#/$defs/")
                if key not in needed:
                    needed.add(key);visit(definitions[key])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(schema)
    schema["$defs"] = {k: definitions[k] for k in sorted(needed)}
    return schema


def _common(schema, source_ids, code_ids, target_ids):
    defs = schema["$defs"]
    defs["span"] = dict(type="object", properties=dict(
        quote=dict(type="string", minLength=1), context=dict(type="string", minLength=1)),
        required=["quote"], additionalProperties=False)
    if "support" in defs:
        defs["support"]["properties"]["assumptions"]["description"] = RULES["conditional_assumptions"]
    defs["critique_ref"]["properties"]["source_id"] = dict(type="string", enum=source_ids)
    value = copy.deepcopy(defs["code_check"])
    for field in ("source_id", "span"):
        value["properties"].pop(field);value["required"].remove(field)
    defs["transport_code_check"] = value
    keyed = dict(type="object", properties={sid: {"$ref": "#/$defs/transport_code_check"} for sid in code_ids},
                 required=list(code_ids), additionalProperties=False)
    for definition in ("public_annotation", "critique_annotation", "truth_annotation"):
        if definition in defs:
            defs[definition]["properties"]["target_id"] = dict(type="string", enum=target_ids)
    if "revision" in defs:
        defs["revision"]["properties"]["original_target_id"] = dict(type="string", enum=target_ids)
        defs["revision"]["properties"]["later_source_id"] = dict(type="string", enum=source_ids)
    if "materiality" in defs:
        refs = defs["materiality"]["properties"]["dependency_refs"]["items"]
        refs["properties"]["source_id"] = dict(type="string", enum=source_ids)
        dependent = defs["materiality"]["properties"]["dependent_target_ids"]
        if target_ids:
            dependent["items"] = dict(type="string", enum=target_ids)
        else:
            _empty_array(dependent)
    return keyed


def _empty_array(schema):
    # Boolean items:false is legal JSON Schema but rejected by the strict backend.
    # The typed item is unreachable: maxItems:0 preserves the sole legal value [].
    # No fake domain ID, semantic record, or reviewer judgment is introduced.
    schema.update(items=dict(type="string"), maxItems=0)


def public_schema(packet):
    contract = public_contract(packet)
    schema = a.public_response_schema()
    keyed = _common(schema, contract["source_domains"]["critique_refs"], contract["code_units"], contract["targets"])
    annotation = schema["$defs"]["public_annotation"]
    annotation["properties"].pop("materiality");annotation["required"].remove("materiality")
    schema["properties"].pop("revision_relations");schema["required"].remove("revision_relations")
    schema["properties"]["code_checks"] = keyed
    count = len(contract["targets"])
    schema["properties"]["annotations"].update(minItems=count, maxItems=count)
    if not count:
        _empty_array(schema["properties"]["annotations"])
    choices = []
    by_target = contract["source_domains"]["actor_basis_by_target_and_purpose"]
    for purpose in a.PURPOSES:
        ids = _ids(sid for table in by_target.values() for sid in table[purpose])
        if ids:
            choices.append(dict(type="object", properties=dict(
                source_id=dict(type="string", enum=ids), purpose=dict(type="string", enum=[purpose])),
                required=["source_id", "purpose"], additionalProperties=False))
    if choices:
        schema["$defs"]["actor_ref"] = dict(anyOf=choices)
    else:
        schema["$defs"]["support"]["properties"]["actor_basis_refs"]["maxItems"] = 0
    return _prune(schema)


def private_schema(truth, later):
    contract = stage2_contract(truth, later)
    tr = a.response_schema("truth_review", {"annotations", "unreviewed_scope"})
    lr = a.response_schema("later_critique", {"annotations", "revision_relations", "code_checks", "unreviewed_scope"})
    definitions = {**tr.pop("$defs"), **lr.pop("$defs")}
    schema = dict(type="object", properties=dict(truth_review=tr, later_critique=lr),
                  required=["truth_review", "later_critique"], additionalProperties=False,
                  **{"$defs": definitions, "$schema": a._DATA["$schema"]})
    keyed = _common(schema, contract["source_domains"]["critique_and_dependency_refs"], contract["code_units"], contract["targets"])
    schema["properties"]["later_critique"]["properties"]["code_checks"] = keyed
    definitions["truth_annotation"]["properties"]["truth_evidence_ids"].update(uniqueItems=True)
    definitions["truth_annotation"]["properties"]["truth_evidence_ids"]["items"] = dict(
        type="string", enum=contract["source_domains"]["truth_evidence_ids"])
    revision_ids = _ids(sid for ids in contract["source_domains"]["revision_sources_by_target"].values() for sid in ids)
    if revision_ids:
        definitions["revision"]["properties"]["later_source_id"] = dict(type="string", enum=revision_ids)
    else:
        _empty_array(schema["properties"]["later_critique"]["properties"]["revision_relations"])
    for key in ("truth_review", "later_critique"):
        schema["properties"][key]["properties"]["annotations"].update(
            minItems=len(contract["targets"]), maxItems=len(contract["targets"]))
        if not contract["targets"]:
            _empty_array(schema["properties"][key]["properties"]["annotations"])
    return _prune(schema)


def _occurrences(text, quote):
    found = []
    start = 0
    while True:
        start = text.find(quote, start)
        if start < 0:
            return found
        found.append(start);start += 1


def locate(text, locator):
    """Unique exact source/optional-context occurrence; Unicode codepoints."""
    if not isinstance(locator, dict) or set(locator) - {"quote", "context"}:
        raise ValueError("Quote locator has missing/extra fields")
    quote = locator.get("quote")
    if not isinstance(quote, str) or not quote:
        raise ValueError("Quote locator needs nonempty exact text")
    context = locator.get("context")
    offset, search = 0, text
    if "context" in locator:
        if not isinstance(context, str) or not context:
            raise ValueError("Quote context must be nonempty exact text")
        contexts = _occurrences(text, context)
        if len(contexts) != 1:
            raise ValueError("Quote context is missing or ambiguous")
        offset, search = contexts[0], context
    hits = _occurrences(search, quote)
    if len(hits) != 1:
        raise ValueError("Quote is missing, repeated or ambiguous in selected source/context")
    start = offset + hits[0]
    return dict(start=start, end=start + len(quote), quote=quote)


def _resolve(value, sources, resolved, path=""):
    if isinstance(value, list):
        return [_resolve(v, sources, resolved, path + "/" + str(i)) for i, v in enumerate(value)]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result = {}
    sid = value.get("source_id", value.get("later_source_id"))
    for key, child in value.items():
        if key in {"span", "later_span"} and sid is not None and child is not None:
            if sid not in sources:
                raise ValueError("Quote locator source does not resolve in its stage")
            location = locate(sources[sid]["text"], child)
            resolved.append(dict(pointer=path + "/" + key, source_id=sid,
                                 model_locator=copy.deepcopy(child), derived_span=location))
            result[key] = location
        else:
            result[key] = _resolve(child, sources, resolved, path + "/" + key)
    return result


def _code_units(keyed, packet, resolved, path):
    required = packet["required_code_source_ids"]
    if set(keyed) != set(required):
        raise ValueError("Required code slots must match exactly; no extras or missing units")
    sources = {s["id"]: s for s in packet["sources"]}
    return [dict(source_id=sid, span=dict(start=0, end=len(sources[sid]["text"]), quote=sources[sid]["text"]),
                 **_resolve(keyed[sid], sources, resolved, path + "/" + sid)) for sid in required]


def _decode(raw, schema, contract):
    response = a.loads(raw)  # Strict duplicate keys, nonfinite and Unicode checks.
    Draft202012Validator(schema).validate(response)
    origin = dict(document_type=ASSEMBLY, contract=contract, schema_sha256=a.H(schema),
                  raw_response_sha256=a.raw_sha(raw), decoded_response_sha256=a.H(response),
                  resolved_model_witnesses=[], host_derived_fields=[])
    return response, origin


def seal_assembly(origin, derived):
    return dict(**origin, derived_sha256=a.H(derived))


def assemble_public(raw, packet, *, reviewer, rubric_hash):
    contract = public_contract(packet)
    response, origin = _decode(raw, public_schema(packet), contract)
    sources = {s["id"]: s for s in packet["sources"]}
    target = {t["id"]: t for t in packet["targets"]}
    resolved = origin["resolved_model_witnesses"]
    derived = _resolve({k: v for k, v in response.items() if k != "code_checks"}, sources, resolved)
    for i, annotation in enumerate(derived["annotations"]):
        endpoint = target[annotation["target_id"]]["endpoint"]
        annotation["materiality"] = dict(judgment="not_applicable", endpoint=endpoint,
            dependent_target_ids=[], dependency_refs=[], rationale="HOST_DEFERRED actor stage; no model materiality judgment.")
        origin["host_derived_fields"].append(dict(pointer=f"/annotations/{i}/materiality", origin="HOST_DEFERRED", endpoint=endpoint))
    derived["revision_relations"] = []
    origin["host_derived_fields"].append(dict(pointer="/revision_relations", origin="HOST_DEFERRED"))
    derived["code_checks"] = _code_units(response["code_checks"], packet, resolved, "/code_checks")
    for i, sid in enumerate(packet["required_code_source_ids"]):
        origin["host_derived_fields"].append(dict(pointer=f"/code_checks/{i}", source_id=sid,
            derived_fields=["source_id", "span"], origin="HOST_WHOLE_ASSIGNED_CODE_UNIT",
            limitation="Assigned unit only; model scope/judgment/rationale/critique remain unchanged and need not be exhaustive."))
    review = a.assemble_public_review(derived, packet, reviewer=reviewer, rubric_hash=rubric_hash)
    return review, seal_assembly(origin, review)


def assemble_private_payload(raw, truth, later):
    contract = stage2_contract(truth, later)
    response, origin = _decode(raw, private_schema(truth, later), contract)
    sources = {s["id"]: s for s in later["sources"]}
    derived = copy.deepcopy(response)
    derived["later_critique"] = _resolve({k: v for k, v in response["later_critique"].items() if k != "code_checks"},
                                        sources, origin["resolved_model_witnesses"], "/later_critique")
    derived["later_critique"]["code_checks"] = _code_units(
        response["later_critique"]["code_checks"], later, origin["resolved_model_witnesses"], "/later_critique/code_checks")
    for i, sid in enumerate(later["required_code_source_ids"]):
        origin["host_derived_fields"].append(dict(pointer=f"/later_critique/code_checks/{i}", source_id=sid,
            derived_fields=["source_id", "span"], origin="HOST_WHOLE_ASSIGNED_CODE_UNIT",
            limitation="Assigned unit only; not an exhaustive code audit."))
    return derived, origin
