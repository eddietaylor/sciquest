"""Exact first-attempt stop science, independent of operation admission.

V5 remains available with its original quote-linked coverage. V6 preserves whole
prose fields and original JSON metadata records, including rejected claims. It
never repairs a quotation, guesses scientific meaning or changes a frozen stop.
"""

from __future__ import annotations

import copy

from . import assessment_packets_v5 as legacy, assessment_v5 as a, assessment_source_locations_v1 as source_locations

VERSION = "instrument-review-packets-v6.0"
EXTRACTION = "first_nested_operation_complete_stop_v1"
COVERAGE = "complete-stop-source-coverage-v1.0"
PROSE = ("reason", "explanation", "limitations")
MAX_CLAIM_RECORDS = 20
MAX_DECISION_FIELDS = 32
MAX_QUOTE_LOCATIONS = 2048


class PacketUnavailable(legacy.PacketUnavailable):
    """No complete packet; structured source coverage remains diagnostic."""

    def __init__(self, reason, coverage):
        self.coverage = copy.deepcopy(coverage)
        super().__init__(reason)


_value_spans = source_locations._value_spans


def _pointer(path):
    return "/operation/decision_json/" + "/".join(
        str(key).replace("~", "~0").replace("/", "~1") for key in path)


def _stop_targets(registry, frame, fields, parsed, code):
    if frame["operation"].get("kind") != "stop":
        targets = legacy._targets(registry, frame, fields, parsed, code)
        registry.stop_coverage = dict(version=COVERAGE, operation=frame["operation"].get("kind"),
                                      mode="unchanged_nonstop_fixed_fields", required_target_ids=[t["id"] for t in targets])
        return targets
    coverage = dict(version=COVERAGE, operation="stop", mode="complete_original_stop_sources",
                    prose_fields=[], metadata=[], claim_quote_mappings=[], required_target_ids=[],
                    limits=dict(claim_records=MAX_CLAIM_RECORDS, decision_fields=MAX_DECISION_FIELDS, quote_locations=MAX_QUOTE_LOCATIONS),
                    complete_source_coverage=False, semantic_coverage="not_certified")
    registry.stop_coverage = coverage
    document = parsed.get("decision_json")
    decision = document.get("decision") if isinstance(document, dict) else None
    if not isinstance(decision, dict):
        raise PacketUnavailable("Original decision is not an unambiguous strict JSON object", coverage)
    for key in PROSE:
        value = decision.get(key)
        status = ("missing" if key not in decision else "nonstring" if not isinstance(value, str)
                  else "empty" if not value.strip() else "literal_prose")
        coverage["prose_fields"].append(dict(field=key, status=status))
    if any(row["status"] != "literal_prose" for row in coverage["prose_fields"]):
        statuses = ", ".join(row["field"] + "=" + row["status"] for row in coverage["prose_fields"])
        raise PacketUnavailable("Incomplete original stop prose: " + statuses, coverage)
    claims = decision.get("claims")
    coverage["claims_state"] = ("missing" if "claims" not in decision else
                                "list_nonempty" if isinstance(claims, list) and claims else
                                "list_empty" if isinstance(claims, list) else "nonlist")
    coverage["decision_field_count"] = len(decision)
    coverage["claim_record_count"] = len(claims) if isinstance(claims, list) else None
    if len(decision) > MAX_DECISION_FIELDS or isinstance(claims, list) and len(claims) > MAX_CLAIM_RECORDS:
        raise PacketUnavailable("Original stop expansion exceeds fixed field/claim bounds; no truncation", coverage)
    raw_document = frame["operation"]["decision_json"]
    locations = _value_spans(raw_document)
    targets = []
    quote_location_count = 0

    def target(source, rule):
        text = source["text"]
        item = dict(id=f"t{len(targets)+1:04d}", source_id=source["id"],
                    span=dict(start=0, end=len(text), quote=text), context_span=dict(start=0, end=len(text)),
                    actor_cutoff=frame["cutoff"], endorsement_source_id=None, kind="ambiguous", force="ambiguous",
                    proposition=text, endpoint="scientific_conclusion")
        targets.append(item)
        registry.mapping[source["id"]].setdefault("selected_targets", []).append(
            dict(target_id=item["id"], rule=rule, start=0, end=len(text)))
        return item["id"]

    for row in coverage["prose_fields"]:
        key = row["field"]
        source = fields["/operation/decision_json/decision/" + key]
        source_locations.attach(source, fields["/operation/decision_json"], ["decision", key], "literal_decoded_string")
        registry.mapping[source["id"]]["decision_value_path"] = [key]
        row.update(source_id=source["id"], target_id=target(source, "complete_literal_prose/" + key))

    def metadata(path, role):
        start, end = locations[path]
        text = raw_document[start:end]
        revision = frame["stop_revision"] or frame["append_revision"]
        source = registry.add(text, "authored_prose", frame=frame, pointer=_pointer(path),
            event=revision, available=revision, author=frame["call"], cutoff=frame["cutoff"], operation="stop",
            provenance=frame["call"] + "/trace.jsonl#selected_final_message",
            extraction="strict_embedded_json_exact_value_slice",
            exposure="current_authorship_or_retrospective_authored_record")
        source_locations.attach(source, fields["/operation/decision_json"], path, "original_json_value")
        registry.mapping[source["id"]].update(
            embedded_json_path=list(path), embedded_json_span=dict(start=start, end=end, quote=text),
            decision_value_path=list(path[1:]))
        item = dict(path=list(path), role=role, source_id=source["id"],
                    target_id=target(source, "complete_original_json_value/" + role))
        coverage["metadata"].append(item)
        return item

    if "status" in decision:
        metadata(("decision", "status"), "status")
    if isinstance(claims, list) and claims:
        for index, record in enumerate(claims):
            item = metadata(("decision", "claims", index), "claim_record")
            quote = record.get("quote") if isinstance(record, dict) else None
            matches = []
            if isinstance(quote, str) and quote:
                for key in PROSE:
                    source = fields["/operation/decision_json/decision/" + key]
                    start = 0
                    while True:
                        start = source["text"].find(quote, start)
                        if start < 0:
                            break
                        quote_location_count += 1
                        if quote_location_count > MAX_QUOTE_LOCATIONS:
                            coverage.update(quote_locations_observed_lower_bound=quote_location_count,
                                            quote_mapping_status="fixed_limit_exceeded_no_truncation")
                            raise PacketUnavailable("Original quote-location inventory exceeds fixed bound; no truncation", coverage)
                        matches.append(dict(field=key, source_id=source["id"], start=start,
                                            end=start+len(quote)))
                        start += 1
            status = ("malformed_record" if not isinstance(record, dict) else
                      "missing_quote" if "quote" not in record else
                      "nonstring_quote" if not isinstance(quote, str) else "empty_quote" if not quote else
                      "unlocated" if not matches else "exact_once" if len(matches) == 1 else "exact_multiple")
            coverage["claim_quote_mappings"].append(dict(index=index, record_source_id=item["source_id"],
                record_target_id=item["target_id"], status=status, locations=matches))
    elif "claims" in decision:
        metadata(("decision", "claims"), "empty_or_nonlist_claims_field")
    for key in sorted(set(decision) - {*PROSE, "status", "claims"}):
        metadata(("decision", key), "other_decision_field")
    notes = fields.get("/campaign_notes")
    coverage["campaign_notes"] = dict(status="nonempty" if notes and notes["text"] else "absent_or_empty")
    if notes and notes["text"]:
        source_locations.attach(notes, fields["/"], ["campaign_notes"], "literal_decoded_string")
        coverage["campaign_notes"].update(source_id=notes["id"],
            target_id=target(notes, "complete_literal_campaign_notes"))
    coverage.update(complete_source_coverage=True, required_target_ids=[t["id"] for t in targets],
        literal_quote_mapping_complete=bool(coverage["claim_quote_mappings"]) and
            all(row["locations"] for row in coverage["claim_quote_mappings"]),
        unreviewed_scope=["Whole-field/record targets are not exhaustive semantic atomization or all-trace coverage.",
                          "Outer transport/wrapper bookkeeping is retained context, not a scientific target."])
    return targets


def _decision_targets(registry, frame, events, parsed, targets):
    if frame["operation"].get("kind") != "stop" or frame["stop_revision"] is None:
        return []
    revision = frame["stop_revision"]
    if not 1 <= revision <= len(events) or events[revision-1].get("type") != "stop":
        raise ValueError("Selected stop has no authenticated frozen-stop event")
    actual = events[revision-1].get("decision")
    original = parsed["decision_json"]["decision"]
    if a.encoded(actual) != a.encoded(original):
        raise ValueError("Selected fields do not equal the authenticated frozen-stop decision")
    ids = []
    for target in targets:
        mapping = registry.mapping[target["source_id"]]
        path = mapping.get("decision_value_path")
        if path is None:  # notes and raw wrappers are not frozen decision fields
            continue
        expected, stored = original, actual
        for key in path:
            expected, stored = expected[key], stored[key]
        if a.encoded(expected) != a.encoded(stored):
            raise ValueError("Selected field differs from authenticated frozen-stop value")
        ids.append(target["id"])
    return ids


def _actor_packet(registry, selected, events, frames):
    return legacy._actor_packet(registry, selected, events, frames,
        target_builder=_stop_targets, extraction=EXTRACTION, decision_targets=_decision_targets)


def _owner_extension(registry, selected, packet):
    return dict(source_coverage=dict(**copy.deepcopy(registry.stop_coverage),
        selected_call=selected["call"], actor_cutoff=selected["cutoff"],
        receipt_append_revision=selected["append_revision"], admitted_selected_stop_revision=selected["stop_revision"],
        scientific_decision_target_ids=packet["scientific_decision_target_ids"],
        scope="Source/operation provenance only; no meaning, warrant, truth, correction or calibration inferred."))


def build_packets(session, *, first_operation):
    """New extraction only; never rewrite an old packet or change its qualification."""
    try:
        return legacy._build_packets(session, first_operation=first_operation, actor_builder=_actor_packet,
            version=VERSION, extraction=EXTRACTION, owner_extension=_owner_extension)
    except PacketUnavailable:
        raise
    except legacy.PacketUnavailable as exc:
        raise PacketUnavailable(str(exc), dict(version=COVERAGE, operation=first_operation,
            mode="first_frame_unavailable", complete_source_coverage=False,
            semantic_coverage="not_certified", required_target_ids=[], reason=str(exc))) from exc


def verify_packets(bundle, session, *, first_operation):
    expected = build_packets(session, first_operation=first_operation)
    if bundle != expected:
        raise ValueError("V6 packets/source coverage differ from exact original campaign evidence")
    return dict(verified=True, version=VERSION,
                scope="Exact first-attempt source coverage only; not scientific assessment validity.")
