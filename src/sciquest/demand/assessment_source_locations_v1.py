"""Neutral original-field locations for prospective review displays.

These facts describe representation and authorship, never scientific support,
operation admission, truth or outcome. Legacy review projections are unchanged.
"""
from __future__ import annotations

import copy
import json
from . import assessment_v5 as a

PREFIX = "authored-json-location-v1:"
EXTRACTION = "first_nested_operation_complete_stop_v1;stop"

def _value_spans(text):
    """Exact codepoint spans for strict JSON values, with no reserialization."""
    a.loads(text)  # duplicate keys, Unicode, nesting/elements, nonfinite values
    decoder = json.JSONDecoder()
    spans = {}

    def whitespace(pos):
        while pos < len(text) and text[pos] in " \t\r\n":
            pos += 1
        return pos

    def visit(pos, path):
        start = whitespace(pos)
        value, end = decoder.raw_decode(text, start)
        spans[path] = (start, end)
        if isinstance(value, (dict, list)):
            cursor = start + 1
            for index, key in enumerate(value if isinstance(value, dict) else range(len(value))):
                cursor = whitespace(cursor)
                if index:
                    if text[cursor] != ",":
                        raise ValueError("JSON source locator lost an item separator")
                    cursor = whitespace(cursor + 1)
                if isinstance(value, dict):
                    actual_key, cursor = decoder.raw_decode(text, cursor)
                    cursor = whitespace(cursor)
                    if actual_key != key or text[cursor] != ":":
                        raise ValueError("JSON source locator lost a field")
                    cursor += 1
                cursor = visit(cursor, path + (key,))
            cursor = whitespace(cursor)
            closing = "}" if isinstance(value, dict) else "]"
            if cursor >= len(text) or text[cursor] != closing or cursor + 1 != end:
                raise ValueError("JSON source locator lost its container boundary")
        return end

    end = visit(0, ())
    if whitespace(end) != len(text):
        raise ValueError("JSON source locator left unconsumed text")
    return spans



def attach(source, parent, path, representation):
    start, end = _value_spans(parent["text"])[tuple(path)]
    location = dict(parent_source_id=parent["id"], path=list(path), representation=representation,
                    original_value_span=dict(start=start, end=end, quote=parent["text"][start:end]))
    source["provenance_ref"] = PREFIX + a.encoded(location).decode("utf-8")


def public_locations(packet):
    """Reconstruct every supplied locator against its already-public parent."""
    sources = {s["id"]: s for s in packet["sources"]}
    result = {}
    for source in sources.values():
        provenance = source.get("provenance_ref")
        if not isinstance(provenance, str) or not provenance.startswith(PREFIX):
            continue
        loc = a.loads(provenance[len(PREFIX):])
        if not isinstance(loc, dict) or set(loc) != {"parent_source_id", "path", "representation", "original_value_span"}:
            raise ValueError("Invalid original-field locator fields")
        parent = sources.get(loc["parent_source_id"])
        path = loc["path"]
        if (parent is None or parent["id"] == source["id"] or not isinstance(path, list) or not path
            or any(type(key) not in (str, int) for key in path)
            or any(s["kind"] not in a.AUTHORED or s["origin"] != "authored" or s["visibility"] != "actor_public" for s in (source, parent))
            or source["authoring_id"] != parent["authoring_id"] or source["actor_cutoff"] != parent["actor_cutoff"]):
            raise ValueError("Original-field locator needs the same public authored parent")
        try:
            start, end = _value_spans(parent["text"])[tuple(path)]
            value = a.loads(parent["text"])
            for key in path:
                if isinstance(value, list):
                    if type(key) is not int or key < 0:
                        raise ValueError("Invalid original array path")
                elif not isinstance(value, dict) or not isinstance(key, str):
                    raise ValueError("Invalid original object path")
                value = value[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Original-field path does not resolve") from exc
        exact = dict(start=start, end=end, quote=parent["text"][start:end])
        if loc["original_value_span"] != exact:
            raise ValueError("Original-field span differs from its exact JSON path")
        representation = loc["representation"]
        if representation == "literal_decoded_string":
            valid = isinstance(value, str) and source["text"] == value
        elif representation == "original_json_value":
            valid = source["text"] == exact["quote"]
        else:
            valid = False
        if not valid:
            raise ValueError("Original-field text differs from its declared representation")
        result[source["id"]] = copy.deepcopy(loc)
    if packet["selection_rule"] == EXTRACTION and not {t["source_id"] for t in packet["targets"]} <= set(result):
        raise ValueError("Complete-stop review requires public locations for every target source")
    return result


def display_binding(packet):
    presentation = a.actor_presentation(packet)
    locations = public_locations(packet)
    return dict(source_locations=locations, actor_presentation_sha256=a.H(presentation),
                source_display_sha256=a.H(dict(presentation=presentation, source_locations=locations)))
