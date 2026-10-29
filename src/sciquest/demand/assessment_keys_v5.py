"""Finite precommitted review assertions, never a free-prose semantic scorer.

Freeze this matcher before authoring qualification instances/keys. Keys bind exact
actor/later/truth packets and named categorical/provenance assertions. Accepted
narratives can still contain errors these finite assertions do not detect.
"""

from __future__ import annotations

from . import assessment_v5 as a

VERSION="finite-review-key-v1.0"
VALUE_PATHS={
    ("public_review","annotations"):{
        ("interpretation_status",),("interpretation","kind"),("interpretation","force"),
        ("actor_support","verdict")},
    ("truth_review","annotations"):{("judgment",)},
    ("later_critique","annotations"):{("materiality","judgment")},
    ("public_review","code_checks"):{("scope",),("judgment",)},
    ("later_critique","code_checks"):{("scope",),("judgment",)},
}
REFERENCE_PATHS={
    ("public_review","actor_basis_refs"):("actor_support","actor_basis_refs"),
    ("truth_review","truth_evidence_ids"):("truth_evidence_ids",),
    ("later_critique","critique_refs"):("critique_refs",),
    ("later_critique","dependency_refs"):("materiality","dependency_refs"),
}


def _get(value,path):
    for part in path:
        value=value[part]
    return value


def _match(value,pattern):
    return isinstance(value,dict) and all(k in value and value[k]==v for k,v in pattern.items())


def _labels(record,collection,path):
    definition = "code_check" if collection=="code_checks" else {
        "public_review":"public_annotation", "truth_review":"truth_annotation",
        "later_critique":"critique_annotation"}[record]
    node=a.schema(definition)
    for part in path:
        while "$ref" in node:
            node=a.schema(node["$ref"].rsplit("/",1)[1])
        node=node["properties"][part]
    return node["enum"]


def _span(source,value):
    if (not isinstance(value,dict) or set(value)!={"start","end","quote"}
        or type(value["start"]) is not int or type(value["end"]) is not int
        or not isinstance(value["quote"],str)):
        raise ValueError("Key witness must be an exact typed span")
    a.span(source,value)


def _within(location,anchor,context):
    return context["start"]<=location["start"]<=anchor["start"] and anchor["end"]<=location["end"]<=context["end"]


def validate_key(key,bundle):
    a.validate_data(key)
    if not isinstance(key,dict) or set(key)!={"version","public_packet_sha256","later_packet_sha256","truth_packet_sha256","assertions"} or key["version"]!=VERSION:
        raise ValueError("Invalid finite key version/shape")
    for name in ("public_packet","later_packet","truth_packet"):
        if key[name+"_sha256"]!=a.H(bundle[name]):
            raise ValueError("Key differs from its exact packet instance")
    assertions=key["assertions"]
    if not isinstance(assertions,list) or not assertions:
        raise ValueError("Finite key has no precommitted assertions")
    if any(not isinstance(c,dict) or not isinstance(c.get("id"),str) or not c["id"] for c in assertions):
        raise ValueError("Invalid assertion identity/shape")
    identities=[c["id"] for c in assertions]
    if len(set(identities))!=len(identities):
        raise ValueError("Duplicate/empty assertion identities")
    targets={t["id"]:t for t in bundle["public_packet"]["targets"]}
    public={s["id"]:s for s in bundle["public_packet"]["sources"]}
    later={s["id"]:s for s in bundle["later_packet"]["sources"]}
    truth={**public,**{s["id"]:s for s in bundle["truth_packet"]["private_sources"]}}
    code=set(bundle["public_packet"]["required_code_source_ids"])
    for check in assertions:
        kind=check.get("kind")
        if kind=="value":
            if set(check)!={"id","kind","record","collection","identity","path","allowed"}:
                raise ValueError("Invalid categorical assertion")
            pair=(check["record"],check["collection"])
            if not isinstance(check["path"],list) or any(not isinstance(v,str) for v in check["path"]) or tuple(check["path"]) not in VALUE_PATHS.get(pair,set()):
                raise ValueError("Matcher does not score that field/free rationale")
            identity=check["identity"]
            expected="target_id" if check["collection"]=="annotations" else "source_id"
            if not isinstance(identity,dict) or set(identity)!={expected} or identity[expected] not in (targets if expected=="target_id" else code):
                raise ValueError("Key annotation/code source does not resolve")
            if not isinstance(check["allowed"],list) or not check["allowed"] or any(not isinstance(v,str) for v in check["allowed"]):
                raise ValueError("Categorical alternatives must be explicit nonempty labels")
            if not set(check["allowed"])<=set(_labels(*pair,check["path"])):
                raise ValueError("Key categorical label is not in the response schema")
            if check["collection"]=="code_checks" and check["path"]==["scope"] and not set(check["allowed"])<={"static","unverified"}:
                raise ValueError("Key asks for a code certification not currently enabled")
        elif kind=="reference":
            if set(check)!={"id","kind","record","target_id","field","match","min","max"}:
                raise ValueError("Invalid reference assertion")
            if (check["record"],check["field"]) not in REFERENCE_PATHS or check["target_id"] not in targets:
                raise ValueError("Unknown reference dimension/target")
            if type(check["min"]) is not int or check["min"]<0 or (
                check["max"] is not None and (type(check["max"]) is not int or check["max"]<check["min"])):
                raise ValueError("Invalid finite reference count bounds")
            match=check["match"]
            fields={"actor_basis_refs":{"source_id","purpose"},"truth_evidence_ids":{"source_id"},
                    "critique_refs":{"source_id","purpose","span"},"dependency_refs":{"source_id","span","endpoint"}}[check["field"]]
            sources={"public_review":public,"truth_review":truth,"later_critique":later}[check["record"]]
            if (not isinstance(match,dict) or not {"source_id"}<=set(match)
                or set(match)-fields or match["source_id"] not in sources):
                raise ValueError("Reference key source/pattern does not resolve")
            if "purpose" in match and match["purpose"] not in (a.PURPOSES if check["field"]=="actor_basis_refs" else {"context","negative_warrant","counterexample","fulfillment","code_check"}):
                raise ValueError("Reference key purpose does not resolve")
            if check["field"]=="dependency_refs" and "span" in match and match["span"] is None:
                raise ValueError("Dependency witness span cannot be null")
            if check["field"]=="actor_basis_refs" and check["min"]>0:
                # A positive key must ask for a citation the formal actor stage
                # can actually produce at this target's authorship.
                purposes=[match["purpose"]] if "purpose" in match else list(a.PURPOSES)
                reachable=False
                for purpose in purposes:
                    try:
                        a._actor_support(dict(actor_basis_refs=[dict(source_id=match["source_id"],purpose=purpose)],
                            basis_modes=[],verdict="unsupported",assumptions=[]),targets[check["target_id"]],public)
                        reachable=True
                    except ValueError:
                        pass
                if not reachable:
                    raise ValueError("Positive actor reference key is unreachable at the target cutoff")
            if match.get("span") is not None:
                _span(sources[match["source_id"]],match["span"])
            if "endpoint" in match and match["endpoint"]!=targets[check["target_id"]]["endpoint"]:
                raise ValueError("Reference key changed endpoint")
            if check["field"]=="dependency_refs" and check["min"]>0:
                source=sources[match["source_id"]]
                witness=match.get("span") or dict(start=0,end=len(source["text"]),quote=source["text"])
                a._materiality(dict(judgment="not_applicable",endpoint=targets[check["target_id"]]["endpoint"],
                    dependent_target_ids=[],dependency_refs=[dict(source_id=source["id"],span=witness,
                        endpoint=targets[check["target_id"]]["endpoint"])]),targets[check["target_id"]],bundle["later_packet"])
        elif kind=="revision_count":
            if set(check)!={"id","kind","target_id","later_source_id","later_span","context_span","relations","allowed_counts"}:
                raise ValueError("Invalid revision assertion")
            if check["target_id"] not in targets or check["later_source_id"] not in later:
                raise ValueError("Revision key source/target does not resolve")
            _span(later[check["later_source_id"]],check["later_span"])
            _span(later[check["later_source_id"]],check["context_span"])
            if not _within(check["later_span"],check["later_span"],check["context_span"]):
                raise ValueError("Revision witness anchor lies outside its context")
            if not isinstance(check["relations"],list) or not check["relations"] or any(not isinstance(v,str) for v in check["relations"]) or not set(check["relations"])<={"retraction","narrowing","abandonment","reaffirmation","unclear"}:
                raise ValueError("Invalid finite revision alternatives")
            if not isinstance(check["allowed_counts"],list) or not check["allowed_counts"] or any(type(v) is not int or v<0 for v in check["allowed_counts"]):
                raise ValueError("Invalid finite revision counts")
            if any(v>0 for v in check["allowed_counts"]):
                a._revisions([dict(original_target_id=check["target_id"],later_source_id=check["later_source_id"],
                    later_span=check["later_span"])],bundle["later_packet"],original_targets=targets)
        else:
            raise ValueError("Unknown finite key assertion kind")
    return key


def _evaluate(key,records):
    results=[]
    for check in key["assertions"]:
        kind=check["kind"]
        record=records.get(check.get("record","later_critique"))
        observed=None
        passed=False
        if record is not None:
            if kind=="value":
                selected=[v for v in record[check["collection"]] if _match(v,check["identity"])]
                if len(selected)==1:  # no favorable selection among duplicates
                    observed=_get(selected[0],check["path"])
                    passed=observed in check["allowed"]
            elif kind=="reference":
                selected=[v for v in record["annotations"] if v["target_id"]==check["target_id"]]
                if len(selected)==1:
                    refs=_get(selected[0],REFERENCE_PATHS[(check["record"],check["field"])])
                    refs=[dict(source_id=v) if isinstance(v,str) else v for v in refs]
                    # Repeated copies of one citation are one finite witness,
                    # never extra evidence toward a minimum count.
                    unique={a.H(v):v for v in refs}
                    observed=sum(_match(v,check["match"]) for v in unique.values())
                    passed=observed>=check["min"] and (check["max"] is None or observed<=check["max"])
            else:
                eligible=[v for v in record["revision_relations"] if
                    v["original_target_id"]==check["target_id"] and v["later_source_id"]==check["later_source_id"]
                    and _within(v["later_span"],check["later_span"],check["context_span"])]
                observed=sum(v["relation"] in check["relations"] for v in eligible)
                passed=observed in check["allowed_counts"] and len({v["relation"] for v in eligible})<=1
        results.append(dict(id=check["id"],passed=passed,observed=observed,
                            record_available=record is not None))
    return dict(version=VERSION,assertion_count=len(results),passed_count=sum(r["passed"] for r in results),
                all_passed=all(r["passed"] for r in results),assertions=results,
                scope="Only these precommitted categorical/reference/revision predicates; free-text correctness and expertise are not certified.")


def evaluate_ledger(key,bundle,ledger):
    validate_key(key,bundle)
    state=ledger.verify()
    if state["prepared"]["public_packet"]!=bundle["public_packet"]:
        raise ValueError("Key packet differs from actual reviewed custody")
    if "truth_packet" in state and (state["truth_packet"]!=bundle["truth_packet"] or state["later_packet"]!=bundle["later_packet"]):
        raise ValueError("Key truth/later evidence differs from actual release")
    return _evaluate(key,{name:state.get(name) for name in ("public_review","truth_review","later_critique")})
