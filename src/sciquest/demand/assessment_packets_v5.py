"""Read-only, original-response review packets for the new campaign instrument.

The raw prompt is retained in owner custody. Reviewer evidence is an explicit
scientific-evidence projection, not the entire prompt or every retrievable file.
Only supplied assignment/guidance is masked; authored disclosures are unchanged.
No source is inferred from the final registered notebook or a legacy top-level
response kind. Missing/invalid selected frames are never replaced.
"""

from __future__ import annotations

import copy
import inspect
import json
import re

import yaml

from sciquest.io import content_hash
from sciquest.readonly import database_reads_only
from . import assessment_v5 as a, instrument_v4 as instrument
from .artifacts_v1 import _read_regular
from .backend import parse_events
from .session import public_query
from .worldbank import verify_call

VERSION = "instrument-review-packets-v5.2"
EXTRACTION = "first_nested_operation_fixed_fields_v3"
GUIDANCE_MARKER = "FIXED INQUIRY GUIDANCE:\n"
MAX_FILE = 16 * 1024 * 1024
CONTEXT_FIELDS = {
    "instrument_version","information_mode","campaign_id","phase","revision","contract",
    "comparison","controller","resource_usage","delivery_policy","remaining","history",
    "decision","campaign_notes","archive_index","artifact_index","analysis_index",
    "validation_feedback","apparatus_action_schemas","collecting_plan_schema",
    "final_plan_schema","fidelity_probes","fitted_state_contract","native_model_export_contract","recent_analysis_output",
    "evaluation_features","validation_suite","available_libraries","prediction_files",
    "public_task",
}


class PacketUnavailable(ValueError):
    """The fixed frame cannot be admitted; retain its slot without replacement."""


def _text(value):
    return value if isinstance(value,str) else a.encoded(value).decode()


def _yaml(raw):
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader,node,deep=False):
        result={}
        for key_node,value_node in node.value:
            key=loader.construct_object(key_node,deep=deep)
            if not isinstance(key,str) or key in result:
                raise ValueError("Ambiguous/non-string plan YAML key")
            result[key]=loader.construct_object(value_node,deep=deep)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
    return a.validate_data(yaml.load(raw,Loader=UniqueLoader))


def _frames(session,wrapped,state):
    """Every successful service response, including rejected operations."""
    events=[e["payload"] for e in wrapped]
    frozen=instrument.contract(events)
    frames=[]
    for index,event in enumerate(events):
        if event["type"]!="instrument_preparation":
            continue
        call=event["call"]
        if not re.fullmatch(r"instrument_calls/c[0-9]{4,8}",call):
            raise ValueError("Noncanonical call source")
        cutoff=event["author_cutoff"]
        ctx=instrument.context(session,events=events[:cutoff],state=state)
        prompt=instrument.instructions(ctx["comparison"])+instrument._MARKER+json.dumps(
            ctx,ensure_ascii=False,separators=(",",":"))
        schema=instrument.delivery.response_schema(ctx["controller"]["allowed_operations"])
        if event.get("dispatch_exception") and not event.get("receipt_sha256"):
            continue
        directory=session.path/call
        receipt=verify_call(directory,expected_prompt=prompt,expected_schema=schema)
        if receipt.get("backend_spec_hash")!=content_hash(frozen["backend_spec"]) or receipt.get("timeout_seconds")!=frozen["timeout"]:
            raise ValueError("Selected response backend/timeout differs from frozen instrument")
        if receipt["status"]!="succeeded":
            continue
        rawfiles={name:_read_regular(session.path,call+"/"+name,MAX_FILE) for name in (
            "prompt.txt","schema.json","trace.jsonl","receipt.json","response.json")}
        if rawfiles["prompt.txt"]!=prompt.encode() or a.raw_sha(rawfiles["receipt.json"])!=event["receipt_sha256"]:
            raise ValueError("Original raw prompt/receipt source changed")
        rawresponse=parse_events(rawfiles["trace.jsonl"].decode())["final_message"]
        try:
            response=a.loads(rawresponse)
        except ValueError:
            # The original raw trace still exists. This cannot enter the strict
            # assessment contract and is never normalized into a passing frame.
            response=None
        if response is not None and response!=receipt["response"]:
            raise ValueError("Strict original response differs from verified receipt")
        # Selection uses the verified service JSON even if strict assessment
        # admission fails. Otherwise an invalid FIRST predictor could disappear
        # and be silently replaced by a later corrected predictor.
        supplied=receipt["response"]
        op=supplied.get("operation") if isinstance(supplied,dict) else None
        op=op if isinstance(op,dict) else {}
        effects=events[cutoff:index]
        stop=next((cutoff+n+1 for n,e in enumerate(effects) if e["type"]=="stop"),None)
        frames.append(dict(
            call=call,cutoff=cutoff,append_revision=index+1,stop_revision=stop,
            response=response,raw_response=rawresponse,operation=op,context=ctx,
            common_instructions=prompt.split(instrument._MARKER,1)[0].split(GUIDANCE_MARKER,1)[0],
            raw_sha256={name:a.raw_sha(raw) for name,raw in rawfiles.items()},
            response_utf8_sha256=a.raw_sha(rawresponse.encode()),
            event_hash=wrapped[index]["hash"],dispatch_hash=wrapped[cutoff-1]["hash"],
        ))
    return frames


class _Registry:
    def __init__(self):
        self.sources=[]
        self.mapping={}

    def add(self,value,kind,*,frame=None,pointer=None,event=None,available=None,
            provenance,visibility="actor_public",operation="system",author=None,cutoff=None,
            extraction="exact_decoded_value",exposure="literal_current_prompt",
            parent_document="selected_final_message"):
        text=_text(value)
        identity=f"s{len(self.sources)+1:04d}"
        source=dict(
            id=identity,kind=kind,visibility=visibility,text=text,
            text_utf8_sha256=a.raw_sha(text.encode()),
            origin="authored" if kind in a.AUTHORED else ("provided" if kind in {"public_contract","private_truth"} else "generated"),
            event_revision=event,available_revision=available,authoring_id=author,
            actor_cutoff=cutoff,provenance_ref=provenance,operation_kind=operation,asset_ref=None,
        )
        self.sources.append(source)
        parent_hash=None
        if frame:
            parent_hash=(
                frame["response_utf8_sha256"] if parent_document=="selected_final_message"
                else frame["raw_sha256"]["schema.json"] if parent_document=="transport_schema"
                else frame["raw_sha256"]["prompt.txt"]
            )
        self.mapping[identity]=dict(
            call=frame["call"] if frame else None,pointer=pointer,extraction=extraction,
            exposure=exposure,source_utf8_sha256=source["text_utf8_sha256"],
            parent_document=parent_document if frame else None,parent_raw_sha256=parent_hash,
            associated_response_utf8_sha256=frame["response_utf8_sha256"] if frame else None,
            raw_prompt_sha256=frame["raw_sha256"]["prompt.txt"] if frame else None,
        )
        return source


def _authored(registry,frame):
    """Lossless decoded fields, with raw wrapper retained as independent source."""
    response,op=frame["response"],frame["operation"]
    admitted=isinstance(response,dict)
    response=response if admitted else {}
    op=op if admitted else {}
    kind=op.get("kind")
    if not isinstance(kind,str) or kind not in {"query","analysis","stop","prepare","predict"}:
        kind="system"
    base=dict(
        frame=frame,event=frame["append_revision"],available=frame["append_revision"],
        author=frame["call"],cutoff=frame["cutoff"],operation=kind,
        provenance=frame["call"]+"/trace.jsonl#selected_final_message",
        exposure="current_authorship_or_retrospective_authored_record",
    )
    fields={}
    fields["/"]=registry.add(frame["raw_response"],"authored_prose",pointer="",extraction="exact_selected_final_message",**base)
    def add(pointer,value,kind="authored_prose",**changes):
        fields[pointer]=registry.add(value,kind,pointer=pointer,**{**base,**changes})
    if isinstance(response.get("campaign_notes"),str):
        add("/campaign_notes",response["campaign_notes"])
    for key in ("query_json","decision_json","plan_yaml","notebook_json","program"):
        if isinstance(op.get(key),str):
            add("/operation/"+key,op[key],"authored_code" if key=="program" else "authored_prose")
    parsed={}
    for key in ("query_json","decision_json"):
        if key in op and isinstance(op[key],str):
            try:
                parsed[key]=a.loads(op[key])
            except (ValueError,RecursionError):
                pass
    if isinstance(parsed.get("decision_json"),dict):
        decision=parsed["decision_json"].get("decision")
        if isinstance(decision,dict):
            for key in ("reason","explanation","limitations"):
                if isinstance(decision.get(key),str):
                    # Accepted stop text is bound to its actual frozen decision
                    # event. Rejected stop text remains authored at receipt append.
                    revision=frame["stop_revision"] or frame["append_revision"]
                    add("/operation/decision_json/decision/"+key,decision[key],
                        event=revision,available=revision)
    if isinstance(op.get("plan_yaml"),str):
        try:
            parsed["plan_yaml"]=_yaml(op["plan_yaml"])
        except (ValueError,yaml.YAMLError,RecursionError):
            pass
        if isinstance(parsed.get("plan_yaml"),dict):
            for key,value in parsed["plan_yaml"].items():
                if isinstance(value,str):
                    add("/operation/plan_yaml/"+key,value,extraction="strict_yaml_exact_string_value")
    code=[]
    if isinstance(op.get("notebook_json"),str):
        try:
            notebook=a.loads(op["notebook_json"])
            cells=notebook["cells"]
            if not isinstance(cells,list):
                raise ValueError("Notebook cells are not a list")
            for n,cell in enumerate(cells):
                if not isinstance(cell,dict) or cell.get("cell_type") not in {"code","markdown","raw"}:
                    raise ValueError("Invalid original notebook cell")
                value=cell.get("source")
                if isinstance(value,list) and all(isinstance(s,str) for s in value):
                    value="".join(value)
                if not isinstance(value,str):
                    raise ValueError("Invalid original notebook source")
                pointer=f"/operation/notebook_json/cells/{n}/source"
                add(pointer,value,"authored_code" if cell["cell_type"]=="code" else "authored_prose",
                    extraction="strict_json_source_list_concatenation")
                if cell["cell_type"]=="code":
                    # Empty cells are legal and retained exactly. They contain
                    # no nonempty program span for the code-check schema.
                    if value:
                        code.append(fields[pointer]["id"])
            parsed["notebook_json"]=notebook
        except (ValueError,KeyError,TypeError,RecursionError):
            # Preserve exact original raw text and any typed projections already
            # decoded; callers requiring notebook coverage mark the frame missing.
            parsed.pop("notebook_json",None)
    return fields,parsed,code


def _prior_context(registry,frame,events,frames):
    ctx=frame["context"]
    if set(ctx)-CONTEXT_FIELDS:
        raise ValueError("Unreviewed new context fields require an explicit projection version")
    base=dict(frame=frame,event=frame["cutoff"],available=frame["cutoff"],
              provenance=frame["call"]+"/prompt.txt#context",operation="system",
              parent_document="prompt_context")
    registry.add(frame["common_instructions"],"public_contract",
                 pointer="instruction_prefix/common",extraction="exact_common_prefix_before_fixed_guidance",
                 **{**base,"parent_document":"prompt_prefix"})
    schema=instrument.delivery.response_schema(ctx["controller"]["allowed_operations"])
    registry.add(schema,"public_contract",pointer="transport_schema",
                 extraction="verified_actual_transport_schema",**{**base,"parent_document":"transport_schema"})
    projected=copy.deepcopy(ctx)
    projected.pop("campaign_id")
    projected["comparison"]={k:v for k,v in ctx["comparison"].items() if k not in {"policy","guidance_hash"}}
    registry.add(projected,"operational_record",pointer="context.json",
                 extraction="canonical_json_context_without_campaign_identity_and_supplied_assignment",**base)
    registry.add(ctx["contract"],"public_contract",pointer="/contract",**base)
    if "public_task" in ctx:
        registry.add(ctx["public_task"],"public_contract",pointer="/public_task",
                     extraction="exact_registered_public_task_from_actual_prompt",**base)
    query_events=[(n,e) for n,e in enumerate(events[:frame["cutoff"]],1) if e["type"]=="query"]
    if len(query_events)!=len(ctx["history"]):
        raise ValueError("Actor query history projection changed")
    for n,((revision,event),query) in enumerate(zip(query_events,ctx["history"])):
        if query["observation"]!=public_query(event)["observation"]:
            raise ValueError("Actor observation projection changed")
        registry.add(query["observation"],"observation",pointer=f"/history/{n}/observation",
                     **{**base,"event":revision,"available":revision,"operation":"query"})
    if ctx["campaign_notes"]:
        note=next((e for e in reversed(events[:frame["cutoff"]]) if e["type"]=="instrument_notes"),None)
        if not note or note["text"]!=ctx["campaign_notes"]:
            raise ValueError("Actor notes do not resolve to actual authorship")
        revision=next(n for n,e in enumerate(events,1) if e is note)
        registry.add(ctx["campaign_notes"],"authored_prose",pointer="/campaign_notes",
                     **{**base,"event":revision,"available":revision,"author":note["source_call"],
                        "cutoff":note["author_cutoff"]})
    if "recent_analysis_output" in ctx:
        revision,event=next((n,e) for n,e in reversed(list(enumerate(events[:frame["cutoff"]],1)))
                            if e["type"]=="instrument_analysis")
        registry.add(ctx["recent_analysis_output"],"execution_output",pointer="/recent_analysis_output",
                     extraction="exact_actual_prompt_truncated_output",
                     **{**base,"event":revision,"available":revision,"operation":event["kind"]})
    if ctx["decision"] is not None:
        revision,event=next((n,e) for n,e in enumerate(events[:frame["cutoff"]],1) if e["type"]=="stop")
        author=next(f for f in frames if f["stop_revision"]==revision)
        registry.add(ctx["decision"],"authored_prose",pointer="/decision",
                     **{**base,"event":revision,"available":revision,"author":author["call"],
                        "cutoff":author["cutoff"],"operation":"stop"})


def _targets(registry,frame,fields,parsed,code):
    kind=frame["operation"].get("kind")
    targets=[]
    def add(source,start,end,endpoint,selection):
        text=source["text"]
        target=dict(
            id=f"t{len(targets)+1:04d}",source_id=source["id"],
            span=dict(start=start,end=end,quote=text[start:end]),
            context_span=dict(start=0,end=len(text)),actor_cutoff=frame["cutoff"],
            endorsement_source_id=None,kind="ambiguous",force="ambiguous",
            proposition=text[start:end],endpoint=endpoint,
        )
        targets.append(target)
        registry.mapping[source["id"]].setdefault("selected_targets",[]).append(
            dict(target_id=target["id"],rule=selection,start=start,end=end))
    if kind=="stop":
        document=parsed.get("decision_json")
        decision=document.get("decision",{}) if isinstance(document,dict) else {}
        decision=decision if isinstance(decision,dict) else {}
        claims=decision.get("claims")
        if not isinstance(claims,list) or not claims:
            raise PacketUnavailable("Fixed stop has no nonempty explicit claim array")
        for n,claim in enumerate(claims):
            quote=claim.get("quote") if isinstance(claim,dict) else None
            if not isinstance(quote,str) or not quote:
                raise PacketUnavailable("Fixed stop contains an unavailable claim quote")
            found=False
            for key in ("reason","explanation","limitations"):
                source=fields.get("/operation/decision_json/decision/"+key)
                if source:
                    start=0
                    while True:
                        start=source["text"].find(quote,start)
                        if start<0:
                            break
                        add(source,start,start+len(quote),"scientific_conclusion",
                            f"claims/{n}/quote;all_exact_occurrences;{key}")
                        start+=1
                        found=True
            if not found:
                raise PacketUnavailable("Fixed stop quote has no exact decision-text location")
    elif kind=="predict":
        if "notebook_json" not in parsed:
            raise PacketUnavailable("Fixed predictor notebook cannot supply original exact code coverage")
        plan=parsed.get("plan_yaml")
        if not isinstance(plan,dict):
            raise PacketUnavailable("Fixed predictor plan cannot be parsed without repair")
        for key in ("hypothesis","scope","assumptions"):
            source=fields.get("/operation/plan_yaml/"+key)
            if not source or not source["text"]:
                raise PacketUnavailable("Fixed predictor plan lacks required nonempty target field: "+key)
            add(source,0,len(source["text"]),"predictor_fidelity","complete_plan_field/"+key)
        notes=fields.get("/campaign_notes")
        if notes and notes["text"]:
            add(notes,0,len(notes["text"]),"predictor_fidelity","complete_nonempty_campaign_notes")
    else:
        # Additional frames are development/engineering checks, not the two fixed
        # qualification stop/predict frames. The raw response remains assessable.
        source=fields["/"]
        add(source,0,len(source["text"]),"operational_delivery","whole_original_response_development_only")
    return targets


def _physical_truth(world):
    """Parameter facts plus exact law/measurement definitions, never scores.

    Definition text may refer to a sampling seed; the actual seed and all
    sampled evaluation labels are omitted. Source code is private-stage evidence.
    """
    from . import world as w,designed,designed_v2
    physical=dict(model=copy.deepcopy(world["model"]))
    if "spec" in world:
        physical["spec"]={k:copy.deepcopy(v) for k,v in world["spec"].items()
                          if k not in {"description","scientific_challenge"}}
        module=designed_v2 if world["spec"]["version"]==designed_v2.VERSION else designed
        functions=[designed.raw,designed.latent,module.draw]
        if module is designed_v2:
            functions.insert(2,designed_v2.latent)
        physical["definitions"]=dict(
            nuisance_values=list(designed.Z_GRID),historical_randomized_prices=list(w.PRICES),
            population_scope="Uniform nuisance and fresh Gaussian instrument draws; a particular hidden PRNG seed is not supplied.",
            aliases={"original_latent":"sciquest.demand.designed.latent"} if module is designed_v2 else {},
            function_source={f.__module__+"."+f.__name__:inspect.getsource(f) for f in functions},
        )
    else:
        config=w.DemandConfig(**world["config"])
        contract=w.assisted_contract(config)
        physical["scenario"]=config.scenario
        physical["measurement"]={k:contract[k] for k in (
            "historical_assignment","historical_capacity","sensor_sigma","precise_sensor_sigma",
            "nuisance_model","observation_model")}
        physical["definitions"]=dict(
            nuisance_values=[-1,0,1],historical_randomized_prices=list(w.PRICES),
            function_source={f.__name__:inspect.getsource(f) for f in (w.effect,w.demand,w.components,w.draw)},
        )
    return physical


def _actor_packet(registry,selected,events,frames,*,target_builder=None,
                  extraction=EXTRACTION,decision_targets=None):
    """Pure projection after caller authenticates inputs; no owner truth input."""
    _prior_context(registry,selected,events,frames)
    fields,parsed,code=_authored(registry,selected)
    targets=(target_builder or _targets)(registry,selected,fields,parsed,code)
    stop=next((n for n,e in enumerate(events,1) if e["type"]=="stop"),None)
    known_stop=stop if stop is not None and stop<=selected["append_revision"] else None
    packet=dict(
        document_type=a.KINDS["public_packet"],
        campaign_id=instrument.contract(events)["campaign_id"],instrument_version=instrument.VERSION,
        review_asof_revision=selected["append_revision"],scientific_stop_revision=known_stop,
        sources=copy.deepcopy(registry.sources),targets=targets,
        required_target_ids=[t["id"] for t in targets],required_code_source_ids=code,
        scientific_decision_target_ids=(decision_targets(registry,selected,events,parsed,targets)
            if decision_targets else [t["id"] for t in targets]
            if selected["operation"].get("kind")=="stop" and selected["stop_revision"] is not None else []),
        presentation_mode="actor_cutoff",frame_cutoff=selected["cutoff"],
        selection_rule=extraction+";"+selected["operation"]["kind"],
        unreviewed_scope=[
            "Common-warrant scientific-evidence subset; supplied assignment/guidance masked, authored disclosures preserved.",
            "Indexed/retrievable files are not all displayed; missing essential context requires an unresolved judgment.",
            "Whole-field targets can be compound; no exhaustive semantic atomization or all-trace claim coverage.",
        ],
    )
    return a.validate_public_packet(packet,expected_sources={s["id"]:s for s in registry.sources})


def build_packets(session,*,first_operation):
    """First submitted operation of the specified kind, independent of success.

    This is a quiescent-owner API. The entire instrument is first verified.
    Incomplete/unverifiable campaigns and unparseable fixed frames are unavailable.
    Returns owner custody plus separate actor/later/truth packets; never writes.
    """
    return _build_packets(session,first_operation=first_operation,
                          actor_builder=_actor_packet,version=VERSION,extraction=EXTRACTION)


def _build_packets(session,*,first_operation,actor_builder,version,extraction,owner_extension=None):
    """Shared authenticated custody; named versioned entrypoints own projections."""
    if first_operation not in {"query","analysis","stop","prepare","predict"}:
        raise ValueError("Unknown fixed frame operation")
    with database_reads_only():
        proof=instrument.verify(session)
        with session.transaction() as (connection,state):
            wrapped=session._events(connection)
            state=copy.deepcopy(state)
        events=[e["payload"] for e in wrapped]
        frames=_frames(session,wrapped,state)
        selected=next((f for f in frames if f["operation"].get("kind")==first_operation),None)
        if selected is None:
            raise PacketUnavailable("First "+first_operation+" frame is missing; no replacement allowed")
        if not isinstance(selected["response"],dict):
            raise PacketUnavailable("First selected response cannot enter strict JSON/UTF-8 assessment; no replacement allowed")
        registry=_Registry()
        packet=actor_builder(registry,selected,events,frames)
        extension=owner_extension(registry,selected,packet) if owner_extension else {}
        stop=next((n for n,e in enumerate(events,1) if e["type"]=="stop"),None)
        actor_mapping=copy.deepcopy(registry.mapping)
        for frame in frames:
            if frame["call"]!=selected["call"]:
                _authored(registry,frame)
        for revision,event in enumerate(events,1):
            if event["type"]=="query":
                registry.add(public_query(event),"observation",event=revision,available=revision,
                             operation="query",provenance=f"event/{revision}",
                             exposure="retrospective_public_query;not_new_actor_warrant")
            elif event["type"]=="instrument_analysis":
                output=_read_regular(session.path,event["output_file"],MAX_FILE)
                if a.raw_sha(output)!=event["output_sha256"]:
                    raise ValueError("Retrospective public analysis output changed")
                source=registry.add(output.decode(errors="replace"),"execution_output",event=revision,available=revision,
                             operation=event["kind"],provenance=event["output_file"],
                             exposure="retrievable_public_output;not_necessarily_read",
                             extraction="utf8_replacement_decode;raw_bytes_separately_committed")
                registry.mapping[source["id"]]["original_bytes_sha256"]=a.raw_sha(output)
        registry.add(dict(phase=proof["phase"],completed_delivery=proof["completed_delivery"],
                          scope=proof["scope"]),"operational_record",visibility="reviewer_public",
                     provenance="read_only_instrument_verification",exposure="retrospective_public_delivery_status")
        later=copy.deepcopy(packet)
        later.update(sources=copy.deepcopy(registry.sources),presentation_mode="retrospective_public",
                     frame_cutoff=None,review_asof_revision=len(events),scientific_stop_revision=stop)
        a.validate_later_packet(packet,later)
        # Truth intentionally omits seed, utility/error scores, arm/guidance,
        # campaign labels and author-written scientific-challenge summaries.
        physical=_physical_truth(state["world"])
        private=[]
        values=[
            ("physical_world",physical),
            ("realized_units",[dict(observation_id=e["observation"]["id"],private=e["private"])
                               for e in events if e["type"]=="query"]),
        ]
        for n,(name,value) in enumerate(values,1):
            text=_text(value)
            private.append(dict(
                id=f"s{90000000+n}",kind="private_truth",visibility="private_truth",
                text=text,text_utf8_sha256=a.raw_sha(text.encode()),
                origin="provided",event_revision=None,available_revision=None,
                authoring_id=None,actor_cutoff=None,provenance_ref="private/"+name,
                operation_kind="system",asset_ref=None))
        truth=dict(document_type=a.KINDS["truth_packet"],public_packet_hash=a.H(packet),
                   public_packet=copy.deepcopy(packet),private_sources=private)
        a.validate_truth_packet(truth,public_packet=packet)
        owner=dict(
            version=version,extraction=extraction,selected_call=selected["call"],
            selected_operation=first_operation,raw_prompt_sha256=selected["raw_sha256"]["prompt.txt"],
            raw_response_utf8_sha256=selected["response_utf8_sha256"],
            campaign_head=wrapped[-1]["hash"],instrument_proof=proof,
            calls=[{k:f[k] for k in ("call","cutoff","append_revision","event_hash","dispatch_hash","raw_sha256","response_utf8_sha256")} for f in frames],
            actor_source_mapping=actor_mapping,later_source_mapping=registry.mapping,
            public_packet_sha256=a.H(packet),later_packet_sha256=a.H(later),truth_packet_sha256=a.H(truth),
            public_presentation_sha256=a.H(a.actor_presentation(packet)),
            masking="Supplied assignment and guidance only; exact authored disclosures may identify treatment.",
        )
        if set(extension)&set(owner):
            raise ValueError("Projection extension cannot replace authoritative custody")
        owner.update(extension)
        with session.transaction() as (connection,_):
            if session._events(connection)!=wrapped:
                raise ValueError("Campaign changed during packet construction")
        for frame in frames:
            for name,digest in frame["raw_sha256"].items():
                if a.raw_sha(_read_regular(session.path,frame["call"]+"/"+name,MAX_FILE))!=digest:
                    raise ValueError("Model-call bytes changed during packet construction")
        return dict(public_packet=packet,later_packet=later,truth_packet=truth,owner_custody=owner)


def verify_packets(bundle,session,*,first_operation):
    expected=build_packets(session,first_operation=first_operation)
    if bundle!=expected:
        raise ValueError("Review packets/source map differ from actual original campaign evidence")
    return dict(verified=True,version=VERSION,scope="Original response and deterministic scientific-evidence projection; not exhaustive semantics.")
