"""New prospective campaign instrument; legacy attempts use their frozen runner.

All subject files are explicit current-campaign inputs. Stop is immutable.
The caller must freeze a new validation allocation before using live model calls.
Development fixtures are never counted as scientific agent outcomes.
"""

from __future__ import annotations

import csv
import fcntl
import io
import json
import sys
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from sciquest.execution import run_command
from sciquest.io import content_hash, file_hash, read_yaml, write_yaml
from . import artifacts_v1 as artifacts, campaign_v2, delivery_v4 as delivery, fidelity_v1, public_contract_v1, public_tasks_v1
from .apparatus import SEMANTICS, action_schemas, plan_schema
from .policies_v2 import COMMON, GUIDANCE, ResourceLimits
from .session import observation_fields, public_query
from .world import EVAL_FEATURES, evaluation_rows

VERSION = "research-instrument-v4.2"
_MARKER = "\ncontext.json:\n"

PROTOCOL = """You are an experimental scientist investigating a hidden apparatus.
Return one operation object and your bounded campaign_notes. Only operations
listed in controller.allowed_operations are legal. You have no direct tools or
external memory. Use only the supplied context and your own campaign files.
If context.public_task is present, perform that registered public software
exercise through this same iterative loop. It supplies no scientific answer and
does not change legal operations, resource accounting or information boundaries.
If absent, the fitted-state route remains optional.

query: query_json contains {revision:CURRENT,action:ACTION,plan:PLAN}.
stop: decision_json contains {revision:CURRENT,decision:DECISION}. Stopping
permanently freezes scientific conclusions and closes further measurements.
analysis (before stop) or prepare (after stop): program is offline Python; inputs
explicitly names artifact IDs from artifact_index; outputs explicitly names files
under outputs/. Each consumes the SAME shared offline allowance, even if execution
or publication fails. Files are fresh for every run; selected inputs are copied
to artifacts/ID. Other indexed code/output/records are available at their listed
paths. Print concise diagnostics; explicitly save useful fitted state.
predict (after stop): submit complete notebook_json, plan_yaml, artifact_ids and
fitted_bundle (empty for a new/unverified predictor). Do not run the notebook.
No unused or mixed payload fields are accepted. Notes are your own bounded text;
empty retains the prior notes. Same-response notes cannot know later execution.

Analysis stdout and published outputs become available only after execution.
Artifacts preserve bytes, not scientific correctness. For fitted-state checking,
write a declaration using fitted_state_contract and include EVERY referenced
outputs/ file in outputs. Use the supervisor's public probes.csv unchanged.
fitted_bundle is the outputs/ declaration path. The check compares only explicitly
listed component quantities at finite probes; means alone do not validate
uncertainty, distributions, inference or calibration. Check feedback is public
software validation only. A failed check may retain bytes but gives no fidelity
endorsement. No likelihood, fitted model, physical law or rival is provided.
The contract supplies structural schemas AND exact runtime/cross-field rules;
follow both. In particular, training_cutoff is an integer evidence revision,
measurement_assumptions is text, and your Python reload_program receives the
probe path as sys.argv[1] and result JSON path as sys.argv[2] in a fresh root.
Its earlier live reference is withheld; do not read it in the reload program.

The final notebook receives ordinary training/features plus a campaign_archive
CSV at prediction_files.campaign_archive. It contains your own earlier code,
outputs, records and ONLY the artifact IDs selected by this predict response.
Group rows by path, require parts1..N, concatenate payload, base64-decode and
verify file_sha256 against the decoded raw bytes. chunk_sha256 hashes each
ASCII base64 chunk; the archive index labels raw file bytes explicitly.
A frozen archive hash binds the actual preflight and native inputs.

For a checked fitted handoff, select the earlier successful declaration and all
its model files and reload program. Native model_export.artifacts must use the
same artifacts/ID paths. Export their exact selected bytes; the supervisor binds
their hashes and the fixed1e-8 reload tolerance. No reconstruction from a code
search or a new fit qualifies as preservation of the earlier fitted state.
A predictor without this route is labeled new_or_unverified_fitted_state.
Even a checked route does not prove arbitrary predictor intent or scientific
validity. The standalone reload receives features and exported artifacts only.
Optional native model_export is specified by final_plan_schema and
native_model_export_contract. Its artifacts and reload_notebook are BOTH required
when opting in. The ORIGINAL notebook must CREATE the exported files and a valid
separate reload NOTEBOOK at that path. This notebook is distinct from the fitted
checker's Python reload_program. Omitted/null model_export keeps the basic route.
Emit predictions.csv for every evaluation ID with mean_demand, mean_sales,
lower_sales,upper_sales as specified in the public contract.

The two format/preparation corrections are campaign-global. Malformed requests
and failed public predictor preflights use them; offline runtime/publication
outcomes spend the shared analysis slot instead. Integrity/service failures are
terminal. Reserves preserve stopping, prediction and remaining correction
opportunities; hard resources can still prevent delivery. Once native handoff
begins, any error is terminal. No native scores or new measurements are available
for correcting a predictor. Never manufacture scores or declare confirmation.
"""


def instructions(assignment):
    # Replace legacy memory advice while preserving identical scientific rules.
    start, end = COMMON.index("Use campaign_notes"), COMMON.index("Freeze a final decision")
    return PROTOCOL + "\n" + COMMON[:start] + COMMON[end:] + "\nAPPARATUS SEMANTICS:\n" + SEMANTICS + "\nFIXED INQUIRY GUIDANCE:\n" + GUIDANCE[assignment["policy"]]


def history(session):
    return campaign_v2.events(session)


def contract(events):
    found = [e for e in events if e["type"] == "instrument_contract"]
    if len(found) != 1 or any(e["type"] == "delivery_contract" for e in events):
        raise ValueError("Require exactly one new instrument contract and no legacy delivery contract")
    frozen = found[0]
    if len(events) < 2 or events[1] != frozen or frozen["version"] != VERSION:
        raise ValueError("Instrument contract must precede every operation")
    if not {"public_task", "public_task_sha256"} <= set(frozen):
        raise ValueError("Instrument contract lacks its explicit public task binding")
    public_tasks_v1.verify_frozen(frozen["public_task"], frozen["public_task_sha256"])
    return frozen


@contextmanager
def attempt_lock(session):
    with (session.path / ".demand/instrument.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Instrument attempt is already running") from exc
        yield


def initialize(session, spec, *, policy=None, artifact_limits=None, archive_limits=None, timeout=360,
               allocation_sha256=None, public_task=None):
    """Fresh allocation only; never restarts an attempted campaign."""
    task = public_tasks_v1.get_task(public_task)
    policy = policy or delivery.DeliveryPolicy()
    artifact_limits = (artifact_limits or artifacts.ArtifactLimits()).validate()
    archive_limits = (archive_limits or delivery.ArchiveLimits()).validate()
    view = session.view(scientist=True)
    if not view.get("comparison") or view["information_mode"] != "black_box_v2":
        raise ValueError("New instrument requires a blind common-validation comparison")
    limits = ResourceLimits(**view["comparison"]["limits"])
    policy.validate(limits)
    if type(timeout) is not int or not 1 <= timeout <= 3600:
        raise ValueError("Invalid backend timeout")
    import re
    if allocation_sha256 is None:
        if spec.get("backend", {}).get("kind") != "test_fixture_not_a_live_model":
            raise ValueError("Freeze a fresh allocation before any live instrument attempt")
    elif not isinstance(allocation_sha256, str) or not re.fullmatch("[0-9a-f]{64}", allocation_sha256):
        raise ValueError("Allocation commitment must hash the frozen prospective protocol bytes")
    frozen = dict(
        type="instrument_contract", version=VERSION, campaign_id=uuid.uuid4().hex,
        policy=asdict(policy), artifact_limits=asdict(artifact_limits),
        archive_limits=asdict(archive_limits), backend_spec=spec, timeout=timeout,
        instructions_sha256=artifacts.sha(instructions(view["comparison"]).encode()),
        operation_schemas_hash=content_hash(delivery.VARIANTS),
        offline_failure_accounting="launched_runtime_or_publication_outcome_spends_analysis_not_format_correction",
        fidelity_scope="explicit_declared_structure_and_finite_component_quantities",
        public_software_contract_version=public_contract_v1.VERSION,
        public_task=task, public_task_sha256=content_hash(task) if task is not None else None,
        allocation_sha256=allocation_sha256,
    )
    with session.transaction() as (c, _):
        if len(session._events(c)) != 1:
            raise ValueError("Instrument requires a fresh campaign; never resume or salvage an old attempt")
        session._append(c, frozen)
    return frozen


def public_probes(public):
    rows = [
        dict(probe_id=f"probe{i:03d}", price=p, context=x, capacity=public["capacities"][-1])
        for i, (p, x) in enumerate(
            (p, x) for p in (public["prices"][0], public["prices"][len(public["prices"])//2], public["prices"][-1])
            for x in (public["contexts"][0], public["contexts"][len(public["contexts"])//2], public["contexts"][-1])
        )
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["probe_id", "price", "context", "capacity"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def authorized_artifacts(events):
    return [e["artifacts"] for e in events if e["type"] == "instrument_analysis" and e.get("artifacts")]


def audit_store(session, events):
    """Authoritative store integrity faults are terminal, never format feedback."""
    frozen = contract(events)
    records = authorized_artifacts(events)
    store = session.path / "artifacts"
    expected = {r["manifest"]["analysis_id"] for r in records}
    if store.is_symlink():
        raise RuntimeError("Artifact store integrity audit required")
    actual = {p.name for p in store.iterdir() if p.name != ".lock"} if store.exists() else set()
    if actual != expected:
        raise RuntimeError("Artifact store has missing/uncommitted versions; audit required")
    try:
        for record in records:
            artifacts.verify(store, record, frozen["campaign_id"],
                             limits=artifacts.ArtifactLimits(**frozen["artifact_limits"]))
    except (ValueError, OSError, KeyError) as exc:
        raise RuntimeError("Authorized artifact bytes failed integrity audit") from exc


def _receipt(root, prep):
    if prep["receipt_sha256"] is None:
        return dict(status="interrupted_missing_receipt", usage=None, elapsed_seconds=0)
    path = Path(root) / prep["call"] / "receipt.json"
    if file_hash(path) != prep["receipt_sha256"]:
        raise ValueError("Authorized model receipt changed")
    return json.loads(path.read_text())


def usage(events, root):
    preps = [e for e in events if e["type"] == "instrument_preparation"]
    receipts = [_receipt(root, p) for p in preps]
    known = [r["usage"] for r in receipts if r.get("usage") is not None]
    analyses = [e for e in events if e["type"] == "instrument_analysis"]
    return dict(
        calls=sum(e["type"] == "instrument_dispatch" for e in events),
        completed_calls=len(preps), missing_usage_calls=len(receipts)-len(known),
        pending_dispatch_calls=sum(e["type"] == "instrument_dispatch" for e in events)-len(preps),
        input_tokens=sum(u["input_tokens"] for u in known),
        output_tokens=sum(u["output_tokens"] for u in known),
        cached_input_tokens=sum(u.get("cached_input_tokens", 0) for u in known),
        elapsed_call_seconds=sum(r.get("elapsed_seconds", 0) for r in receipts),
        offline_launches=sum(e["type"] == "instrument_analysis_started" for e in events),
        offline_execution_seconds=sum((e.get("execution") or {}).get("elapsed_seconds", 0) for e in analyses),
        retained_artifact_bytes=sum(o["bytes"] for r in authorized_artifacts(events) for o in r["manifest"]["objects"]),
        retrieved_artifact_bytes=sum(e.get("retrieved_bytes", 0) for e in events if e["type"] == "instrument_analysis_started"),
        fidelity_execution_seconds=sum(
            c["execution"].get("elapsed_seconds", 0) for e in analyses
            for c in (e.get("fidelity") or {}).get("checks", [])
        ),
        predictor_preflights=sum(e["type"] == "instrument_preflight" for e in events),
    )


def archive_files(events, root, selected=()):
    """Only explicit public records and campaign-authorized selected model bytes."""
    frozen = contract(events)
    files = {
        "campaign/queries.json": artifacts.encoded([public_query(e) for e in events if e["type"] == "query"]),
        "campaign/notes.json": artifacts.encoded([
            dict(text=e["text"], source_call=e["source_call"], author_cutoff=e["author_cutoff"])
            for e in events if e["type"] == "instrument_notes"
        ]),
        "campaign/preparation_errors.json": artifacts.encoded([
            dict(error=e["error"], category=e["category"], source_call=e["call"])
            for e in events if e["type"] == "instrument_preparation" and e.get("error")
        ]),
    }
    for event in events:
        if event["type"] == "instrument_analysis":
            for suffix, field in ((".py", "code_file"), (".txt", "output_file")):
                source = Path(root) / event[field]
                key = "code_sha256" if suffix == ".py" else "output_sha256"
                if file_hash(source) != event[key]:
                    raise ValueError("Own analysis source/output changed")
                files["analysis/" + event["analysis_id"] + suffix] = source.read_bytes()
        elif event["type"] == "instrument_preparation":
            response = _receipt(root, event).get("response")
            response = response if isinstance(response, dict) else {}
            op = response.get("operation")
            op = op if isinstance(op, dict) else {}
            if op.get("kind") == "predict":
                prefix = "submissions/" + Path(event["call"]).name
                for field, suffix in (("notebook_json", ".ipynb"), ("plan_yaml", ".yaml")):
                    if not isinstance(op.get(field), str):
                        continue
                    try:
                        data = op[field].encode("utf-8")
                        if len(data) <= 1048576:
                            files[prefix+suffix] = data
                            continue
                        reason = "oversized_rejected_text"
                    except UnicodeEncodeError:
                        reason = "invalid_utf8_rejected_text"
                    files[prefix+suffix+".omitted.json"] = artifacts.encoded(dict(
                        reason=reason, field=field, source_call=event["call"],
                        raw_receipt_sha256=event["receipt_sha256"],
                        retention="Full rejected response retained in the immutable call receipt.",
                    ))
    catalog = {}
    for record in authorized_artifacts(events):
        manifest = artifacts.verify(Path(root) / "artifacts", record, frozen["campaign_id"],
                                    limits=artifacts.ArtifactLimits(**frozen["artifact_limits"]))
        for obj in manifest["objects"]:
            catalog[obj["id"]] = (manifest["analysis_id"], obj)
    if len(set(selected)) != len(selected) or any(x not in catalog for x in selected):
        raise ValueError("Prediction selected unavailable current-campaign artifact")
    for identity in selected:
        aid, obj = catalog[identity]
        data = artifacts._read_regular(Path(root) / "artifacts" / aid, "objects/" + obj["name"], 1048576)
        if artifacts.sha(data) != obj["sha256"]:
            raise ValueError("Selected artifact changed")
        files["artifacts/" + identity] = data
    return files


def context(session, *, events=None, state=None):
    """Deterministic allowlisted context from a historical authorized prefix."""
    if state is None:
        with session.transaction() as (c, state):
            events = [e["payload"] for e in session._events(c)] if events is None else events
    frozen = contract(events)
    assignment = state["comparison"]
    policy = delivery.DeliveryPolicy(**frozen["policy"])
    limits = ResourceLimits(**assignment["limits"])
    queries = [e for e in events if e["type"] == "query"]
    stops = [e["decision"] for e in events if e["type"] == "stop"]
    phase = "prediction_required" if stops else "collecting"
    ledger = usage(events, session.path)
    dispatched = [e for e in events if e["type"] == "instrument_dispatch"]
    completed = {e["call"] for e in events if e["type"] == "instrument_preparation"}
    pending = [e for e in dispatched if e["call"] not in completed]
    if pending and (len(pending) != 1 or events[-1] != pending[0]):
        raise RuntimeError("Incomplete dispatch prefix requires an owner audit")
    ctl = delivery.controller(
        phase, calls_used=ledger["calls"]-bool(pending), analyses_used=ledger["offline_launches"],
        queries_used=len(queries),
        format_errors=sum(e["category"] == "format_error" for e in events if e["type"] == "instrument_preparation"),
        limits=limits, policy=policy, input_tokens=ledger["input_tokens"], output_tokens=ledger["output_tokens"],
        observation_remaining=sum(q["cost"] for q in queries) < state["public"]["budget"],
    )
    notes = [e["text"] for e in events if e["type"] == "instrument_notes"]
    files = archive_files(events, session.path)
    analyses = [e for e in events if e["type"] == "instrument_analysis"]
    public = {**state["public"], "price_limit": float(state["public"]["price_limit"])}
    public["scoring"] = "Query/transfer forecasts, noisy sensor intervals, feasible pricing utility and scientific warrant are separate. No composite scientific-success grade."
    result = dict(
        instrument_version=VERSION, information_mode="black_box_v2",
        campaign_id=frozen["campaign_id"], phase=phase, revision=len(events),
        contract=public, comparison=assignment, controller=ctl,
        resource_usage=ledger, delivery_policy=frozen["policy"],
        remaining=public["budget"]-sum(q["cost"] for q in queries),
        history=[compact_query(q) for q in queries],
        decision=stops[-1] if stops else None, campaign_notes=notes[-1] if notes else "",
        archive_index={n: dict(raw_bytes=len(b), raw_sha256=artifacts.sha(b)) for n,b in sorted(files.items())},
        artifact_index=[dict(**obj, available_revision=r["manifest"]["available_revision"],
                             source_revision=r["manifest"]["source_revision"])
                        for r in authorized_artifacts(events) for obj in r["manifest"]["objects"]],
        analysis_index=[dict(
            id=e["analysis_id"], status=e["status"], code_file="analysis/"+e["analysis_id"]+".py",
            output_file="analysis/"+e["analysis_id"]+".txt", output_sha256=e["output_sha256"],
            publication_error=e.get("publication_error"),
            fidelity_status=(e.get("fidelity") or {}).get("status"),
            fidelity_error=(e.get("fidelity") or {}).get("error"),
            verified_quantities=(e.get("fidelity") or {}).get("verified_quantities", []),
        ) for e in analyses],
        validation_feedback=[e["error"] for e in events if e["type"] == "instrument_preparation" and e.get("error")][-2:],
        apparatus_action_schemas=action_schemas(public, [public_query(q)["observation"] for q in queries]),
        collecting_plan_schema=plan_schema(["observation"], collecting=True),
        final_plan_schema=public_contract_v1.final_plan_schema(plan_schema(["demand_rmse", "sales_rmse"])),
        native_model_export_contract=public_contract_v1.native_model_export_contract(),
        fidelity_probes=dict(path="probes.csv", raw_sha256=artifacts.sha(public_probes(public)),
                             csv=public_probes(public).decode()),
        fitted_state_contract=public_contract_v1.fitted_state_contract(
            source_revision=len(events), probe_sha256=artifacts.sha(public_probes(public)),
            artifact_limits=frozen["artifact_limits"], archive_limits=frozen["archive_limits"],
            policy=frozen["policy"]),
    )
    if frozen["public_task"] is not None:
        result["public_task"] = public_tasks_v1.get_task(frozen["public_task"]["id"])
    if analyses:
        recent = files["analysis/"+analyses[-1]["analysis_id"]+".txt"].decode(errors="replace")
        result["recent_analysis_output"] = recent[:limits.recent_analysis_chars]
    if stops:
        from .runner import available_libraries
        result.update(
            evaluation_features=[{k:r[k] for k in EVAL_FEATURES} for r in evaluation_rows(state["world"])],
            validation_suite=read_yaml(Path(state["quest"]) / "validation.yaml"),
            available_libraries=available_libraries(),
            prediction_files=dict(
                training="data/0/train.csv" if queries else None,
                evaluation_features=f"data/{int(bool(queries))}/features.csv",
                campaign_archive=f"data/{2 if queries else 1}/instrument_archive.csv",
                training_columns=list(observation_fields(public)) if queries else [],
                evaluation_columns=list(EVAL_FEATURES), campaign_archive_columns=list(delivery.ARCHIVE_FIELDS),
                output="predictions.csv",
            ),
        )
    return result


def compact_query(event):
    from sciquest.validation import COMPARATORS

    query = public_query(event)
    return dict(
        action=query["action"], design=query["design"], cost=query["cost"],
        observation=query["observation"],
        plan_reference=dict(path="campaign/queries.json", observation_id=query["observation"]["id"]),
        predictions=[dict(**p, outcome="supported" if COMPARATORS[p["operator"]](
            query["observation"]["value"], p["threshold"]) else "contradicted")
            for p in query["plan"]["predictions"]],
    )


def execute_analysis(session, op, ctx, call_dir):
    """One offline launch under attempt_lock; publication and event authorization."""
    events = history(session)
    frozen = contract(events)
    audit_store(session, events)
    policy = delivery.DeliveryPolicy(**frozen["policy"])
    records = authorized_artifacts(events)
    aid = f"a{sum(e['type']=='instrument_analysis_started' for e in events)+1:04d}"
    call_dir = Path(call_dir)
    code = op["program"].encode()
    raw_context = artifacts.encoded(ctx)
    (call_dir / "analysis.py").write_bytes(code)
    (call_dir / "analysis_context.json").write_bytes(raw_context)
    with tempfile.TemporaryDirectory(prefix="sciquest-instrument-analysis-") as temp:
        work = Path(temp)
        (work / "analysis.py").write_bytes(code)
        (work / "context.json").write_bytes(raw_context)
        (work / "probes.csv").write_bytes(public_probes(ctx["contract"]))
        for name, data in archive_files(events, session.path).items():
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        staged = artifacts.stage(
            session.path / "artifacts", work, records, op["inputs"],
            campaign_id=frozen["campaign_id"], at_revision=ctx["revision"],
            limits=artifacts.ArtifactLimits(**frozen["artifact_limits"]),
        )
        started = dict(
            type="instrument_analysis_started", analysis_id=aid, kind=op["kind"],
            source_call=str(call_dir.relative_to(session.path)), author_cutoff=ctx["revision"],
            code_sha256=artifacts.sha(code), context_sha256=artifacts.sha(raw_context),
            input_ids=op["inputs"], outputs=op["outputs"],
            retrieved_bytes=sum(o["bytes"] for o in staged),
        )
        with session.transaction() as (c, _):
            if len(session._events(c)) != ctx["revision"]:
                raise RuntimeError("Analysis cutoff changed before launch")
            session._append(c, started)
        log = call_dir / "analysis.log"
        execution = run_command(
            [sys.executable, "analysis.py"], work, log, hidden_root=session.path, blind=True,
            timeout=policy.analysis_timeout_seconds, max_file_bytes=1048576,
        )
        event = dict(
            type="instrument_analysis", analysis_id=aid, kind=op["kind"],
            source_call=started["source_call"], author_cutoff=ctx["revision"],
            code_file=str((call_dir/"analysis.py").relative_to(session.path)),
            code_sha256=artifacts.sha(code), context_sha256=artifacts.sha(raw_context),
            output_file=str(log.relative_to(session.path)), output_sha256=file_hash(log),
            execution=execution, status=execution["status"], artifacts=None,
            publication_error=None, fidelity=None, fitted_bundle=op["fitted_bundle"],
        )
        if execution["status"] == "succeeded":
            if op["fitted_bundle"]:
                try:
                    manifest, _ = fidelity_v1.declaration(work, op["fitted_bundle"], source_revision=ctx["revision"])
                    required = [op["fitted_bundle"], *manifest["model_files"], manifest["live_reference"], manifest["reload_program"]]
                    if manifest["probe_file"] != "probes.csv" or any(n not in op["outputs"] for n in required):
                        raise ValueError("Fitted declaration must use public probes.csv and publish every declared outputs/ asset")
                    event["fidelity"] = fidelity_v1.check_bundle(
                        work, op["fitted_bundle"], call_dir/"fidelity", source_revision=ctx["revision"],
                        expected_probe_sha256=ctx["fidelity_probes"]["raw_sha256"],
                        timeout=policy.fidelity_timeout_seconds,
                    )
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    event["publication_error"] = type(exc).__name__ + ": " + str(exc)
        with session.transaction() as (c, _):
            current = [e["payload"] for e in session._events(c)]
            if len(current) != ctx["revision"]+1 or current[-1] != started:
                raise RuntimeError("Campaign changed during offline execution; audit required")
            audit_store(session, current)
            if execution["status"] == "succeeded" and not event["publication_error"]:
                try:
                    event["artifacts"] = artifacts.collect(
                        session.path/"artifacts", work, op["outputs"], campaign_id=frozen["campaign_id"],
                        analysis_id=aid, source_revision=ctx["revision"],
                        available_revision=len(current)+1,
                        code_sha256=event["code_sha256"], context_sha256=event["context_sha256"],
                        execution=execution, previous=[r["manifest"] for r in records],
                        limits=artifacts.ArtifactLimits(**frozen["artifact_limits"]),
                    )
                except artifacts.ArtifactIntegrityError as exc:
                    raise RuntimeError("Artifact publication integrity audit required") from exc
                except (ValueError, OSError) as exc:
                    event["publication_error"] = type(exc).__name__ + ": " + str(exc)
            if event["publication_error"]:
                event["status"] = "publication_failed"
            session._append(c, event)
    return event


def bind_fitted_export(session, events, op, submitted, ctx):
    """Bind A's successful check to A's exact selected/exported bytes.

    Does not claim arbitrary submitted Python uses these files correctly.
    """
    from .export_v1 import VERSION as EXPORT_VERSION

    submitted = json.loads(json.dumps(submitted))
    export = submitted.get("model_export")
    if export:
        if "checked_fitted_export" in export:
            raise ValueError("Checked fitted export metadata is supervisor-owned")
        export["absolute_tolerance"] = fidelity_v1.ABS_TOLERANCE
    if not op["fitted_bundle"]:
        submitted["instrument_fitted_state"] = dict(status="new_or_unverified_fitted_state")
        return submitted
    if not export:
        raise ValueError("Checked fitted-state handoff requires native model_export")
    frozen = contract(events)
    identity = op["fitted_bundle"]
    try:
        aid, name = identity.split("/", 1)
    except ValueError as exc:
        raise ValueError("Unknown fitted declaration ID") from exc
    matches = [e for e in events if e["type"] == "instrument_analysis" and e["analysis_id"] == aid]
    if len(matches) != 1:
        raise ValueError("Fitted declaration has no authorized source analysis")
    source = matches[0]
    if source["fitted_bundle"] != "outputs/" + name or not source["artifacts"]:
        raise ValueError("Selected fitted declaration differs from checked source")
    record = source.get("fidelity") or {}
    if record.get("status") != "succeeded":
        raise ValueError("Selected fitted state has no successful finite check")
    published = artifacts.verify(session.path/"artifacts", source["artifacts"], frozen["campaign_id"],
                                 limits=artifacts.ArtifactLimits(**frozen["artifact_limits"]))
    available = {o["name"]:o for o in published["objects"]}
    original_names = [source["fitted_bundle"], *record["declaration"]["model_files"],
                      record["declaration"]["live_reference"], record["declaration"]["reload_program"]]
    raw = {}
    for path in original_names:
        if not path.startswith("outputs/") or path[8:] not in available:
            raise ValueError("Fitted source asset was not atomically published")
        raw[path] = artifacts._read_regular(
            session.path/"artifacts"/aid, "objects/"+path[8:], 1048576)
        if artifacts.sha(raw[path]) != available[path[8:]]["sha256"]:
            raise RuntimeError("Authorized fitted asset integrity failure")
    raw["probes.csv"] = public_probes(ctx["contract"])
    with tempfile.TemporaryDirectory(prefix="sciquest-fitted-source-verify-") as temp:
        work = Path(temp)
        for path, data in raw.items():
            target = work/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        try:
            manifest, expected_inputs = fidelity_v1.declaration(
                work, source["fitted_bundle"], source_revision=source["author_cutoff"])
            expected_inputs[source["fitted_bundle"]] = artifacts.sha(raw[source["fitted_bundle"]])
            fidelity_v1.verify_check(
                session.path/source["source_call"]/"fidelity", record,
                expected_manifest=manifest, expected_input_sha256=expected_inputs,
                expected_reference=fidelity_v1._json(raw[manifest["live_reference"]]),
                expected_probe_sha256=artifacts.sha(raw["probes.csv"]),
                source_revision=source["author_cutoff"],
                timeout=frozen["policy"]["fidelity_timeout_seconds"],
            )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise RuntimeError("Authorized fitted evidence failed integrity audit") from exc
    required = {
        aid+"/"+path[8:]: artifacts.sha(raw[path])
        for path in [*manifest["model_files"], manifest["reload_program"]]
    }
    if any(identity not in op["artifact_ids"] for identity in required):
        raise ValueError("Predict must select every checked model file and reload program")
    expected_export = {"artifacts/"+identity:digest for identity,digest in required.items()}
    if len(set(export["artifacts"])) != len(export["artifacts"]) or set(export["artifacts"]) != set(expected_export):
        raise ValueError("Native model_export must name exactly the selected checked model files and reload program")
    export["checked_fitted_export"] = dict(
        version=EXPORT_VERSION, campaign_id=frozen["campaign_id"], analysis_id=aid,
        check_sha256=file_hash(session.path/source["source_call"]/"fidelity/fidelity.json"),
        artifact_sha256=expected_export, scope="exact_declared_assets_and_finite_probes",
    )
    submitted["instrument_fitted_state"] = dict(
        status="declared_assets_and_finite_component_probes_checked",
        declaration_id=identity, verified_quantities=record["verified_quantities"],
        scope="Exact selected/exported bytes and listed finite component quantities only; arbitrary predictor intent, full distribution, inference and calibration unverified.",
    )
    return submitted


def freeze_native_archive(session, events, op, ctx, call_dir):
    frozen = contract(events)
    raw = delivery.encode_archive(
        archive_files(events, session.path, op["artifact_ids"]),
        campaign_id=frozen["campaign_id"], cutoff=ctx["revision"],
        limits=delivery.ArchiveLimits(**frozen["archive_limits"]),
    )
    (call_dir/"instrument_archive.csv").write_bytes(raw)
    with session.transaction() as (_, state):
        quest = Path(state["quest"])
    manifest = read_yaml(quest/"data_manifest.yaml")
    # Preserve evaluation dataset index; add/update this public training archive.
    dataset = dict(
        path="data/raw/instrument_archive.csv", role="training",
        provenance="Supervisor byte transport of this campaign's own public records, scientist-authored code, retained analysis output and explicitly selected artifacts; not new observations.",
    )
    existing = [i for i,d in enumerate(manifest["datasets"]) if d["path"] == dataset["path"]]
    if existing:
        if existing != [len(manifest["datasets"])-1] or manifest["datasets"][existing[0]] != dataset:
            raise RuntimeError("Instrument archive manifest integrity fault")
    else:
        manifest["datasets"].append(dataset)
    (quest/dataset["path"]).write_bytes(raw)
    write_yaml(quest/"data_manifest.yaml", manifest)
    expected_index = int(ctx["prediction_files"]["campaign_archive"].split("/")[1])
    if manifest["datasets"][expected_index] != dataset:
        raise RuntimeError("Instrument archive changed the public dataset layout")
    event = dict(
        type="instrument_archive", source_call=str(call_dir.relative_to(session.path)),
        author_cutoff=ctx["revision"], artifact_ids=op["artifact_ids"],
        raw_sha256=artifacts.sha(raw), bytes=len(raw),
        path=str((call_dir/"instrument_archive.csv").relative_to(session.path)),
        native_path=dataset["path"], data_manifest_hash=content_hash(manifest),
    )
    with session.transaction() as (c, _):
        session._append(c, event)
    return event


def execute_predict(session, op, ctx, call_dir):
    """Preflight then a single terminal native barrier; called under attempt_lock."""
    import nbformat
    import yaml
    from sciquest.methods.ledger import validate_plan
    from sciquest.methods.policies import validate_policy_plan
    from . import preflight_v3, preflight_v4

    events = history(session)
    audit_store(session, events)
    preflight_v3.validate_notebook_json(op["notebook_json"])
    notebook = nbformat.reads(op["notebook_json"], as_version=4)
    nbformat.validate(notebook)
    submitted = validate_plan(delivery.validate_data(yaml.safe_load(op["plan_yaml"])), ctx["validation_suite"])
    with session.transaction() as (_, state):
        validate_policy_plan(state["method_stack"], submitted)
    submitted = bind_fitted_export(session, events, op, submitted, ctx)
    archive = freeze_native_archive(session, events, op, ctx, call_dir)
    source = call_dir/"submitted.ipynb"
    source.write_text(op["notebook_json"])
    record = preflight_v4.run(
        session, source, submitted, ctx, call_dir/"preflight",
        contract(events)["policy"]["preflight_timeout_seconds"],
    )
    with session.transaction() as (c, _):
        session._append(c, dict(
            type="instrument_preflight", source_call=str(call_dir.relative_to(session.path)),
            author_cutoff=ctx["revision"], record=record, record_hash=content_hash(record),
        ))
    if record["status"] != "succeeded":
        raise ValueError("Public predictor preflight failed: "+str(record["error"]))
    native_plan = {
        **submitted, "demand_scientific_decision":ctx["decision"],
        "scientist_information_mode":"black_box_v2", "comparison_assignment":ctx["comparison"],
    }
    # Freeze all bytes before the terminal boundary, including no-score checks.
    with session.transaction() as (c, state):
        quest = Path(state["quest"])
        if (
            file_hash(quest/archive["native_path"]) != archive["raw_sha256"]
            or content_hash(read_yaml(quest/"data_manifest.yaml")) != archive["data_manifest_hash"]
        ):
            raise RuntimeError("Native public inputs changed after preflight")
        session._append(c, dict(
            type="instrument_native_barrier", source_call=str(call_dir.relative_to(session.path)),
            author_cutoff=ctx["revision"], notebook_sha256=file_hash(source),
            plan_hash=content_hash(native_plan), archive_sha256=archive["raw_sha256"],
            preflight_hash=content_hash(record),
        ))
    # Any subsequent exception is terminal, regardless of format allowance.
    session.finish(source, submitted)
    verify_native_predictions(session, ctx, call_dir, expected_export=submitted.get("model_export"))
    return dict(archive=archive, preflight=record)


def verify_native_predictions(session, ctx, call_dir, *, expected_export=None):
    """Anchor all four public forecast fields directly to frozen preflight.

    A declared export requires the actual canonical native clean output; basic
    no-export delivery does not invent a mandatory reload.
    """
    from .preflight_v3 import public_predictions, FIELDS
    from sciquest.state import get_record
    import math

    with session.transaction() as (_, state):
        quest, experiment = Path(state["quest"]), state["experiment"]
        exp = quest/"experiments"/experiment
    original = public_predictions(Path(call_dir)/"preflight", ctx)
    native = public_predictions(exp/"worker_artifacts", ctx)
    candidates = [native]
    clean = exp/"reload/worker_artifacts"
    if expected_export is not None and not (clean/"predictions.csv").is_file():
        raise RuntimeError("Native declared export lacks canonical clean-reload predictions")
    if expected_export is not None:
        execution = get_record(quest, "execution", experiment) or {}
        verification = execution.get("model_export_verification") or {}
        expected_hash = (verification.get("prediction_artifact_hashes") or {}).get("predictions.csv")
        raw = artifacts._read_regular(clean, "predictions.csv", 1048576)
        if expected_hash != artifacts.sha(raw):
            raise RuntimeError("Native clean-reload predictions differ from authoritative execution hash")
    if (clean/"predictions.csv").exists():
        candidates.append(public_predictions(clean, ctx))
    for values in candidates:
        if any(not math.isclose(original[i][key], values[i][key], rel_tol=0,
                                abs_tol=fidelity_v1.ABS_TOLERANCE)
               for i in original for key in FIELDS):
            raise RuntimeError("Native public forecasts differ from frozen preflight")
    return True


def run(session, *, backend_spec, policy=None, artifact_limits=None, archive_limits=None,
        timeout=360, allocation_sha256=None, on_progress=None, public_task=None):
    """One nonresumable prospective attempt; every backend attempt is retained."""
    from .backend import run_call
    import nbformat
    import yaml

    with attempt_lock(session):
        session.verify()
        frozen = initialize(
            session, backend_spec, policy=policy, artifact_limits=artifact_limits,
            archive_limits=archive_limits, timeout=timeout, allocation_sha256=allocation_sha256,
            public_task=public_task,
        )
        while True:
            events = history(session)
            view = session.view(scientist=True)
            if view["phase"] in ("completed", "failed"):
                return session.view()
            if any(e["type"] == "instrument_native_barrier" for e in events):
                raise RuntimeError("Native barrier is terminal; audit incomplete attempt")
            if view["phase"] == "prediction_required":
                session.prepare_quest()
            events = history(session)
            audit_store(session, events)
            ctx = context(session)
            limits = ResourceLimits(**ctx["comparison"]["limits"])
            ctl = ctx["controller"]
            prompt = instructions(ctx["comparison"]) + _MARKER + json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))
            reason = campaign_v2.resource_stop(ctx["resource_usage"], limits, prompt)
            if ctl["terminal"] or reason:
                reason = reason or "call_or_format_opportunities_exhausted"
                with session.transaction() as (c, _):
                    session._append(c, dict(type="instrument_resource_stop", reason=reason,
                                            author_cutoff=ctx["revision"], ledger=ctx["resource_usage"]))
                return {**session.view(), "runner_status":reason}
            schema = delivery.response_schema(ctl["allowed_operations"])
            call_dir = session.path/"instrument_calls"/f"c{ctx['resource_usage']['calls']+1:04d}"
            dispatch = dict(type="instrument_dispatch", call=str(call_dir.relative_to(session.path)),
                            prior_revision=len(events), backend_spec_hash=content_hash(backend_spec))
            # Compute the final handoff before committing a dispatch, so a hard
            # prompt stop does not invent an attempted backend request.
            projected = context(session, events=events+[dispatch])
            projected_prompt = instructions(projected["comparison"])+_MARKER+json.dumps(projected, ensure_ascii=False, separators=(",",":"))
            projected_reason = campaign_v2.resource_stop(projected["resource_usage"], limits, projected_prompt)
            if projected_reason:
                with session.transaction() as (c, _):
                    session._append(c, dict(type="instrument_resource_stop", reason=projected_reason,
                                            author_cutoff=ctx["revision"], ledger=ctx["resource_usage"],
                                            projected_dispatch=True))
                return {**session.view(), "runner_status":projected_reason}
            with session.transaction() as (c, _):
                if len(session._events(c)) != len(events):
                    raise RuntimeError("Campaign changed before dispatch")
                session._append(c, dispatch)
            ctx, prompt = projected, projected_prompt
            try:
                receipt = run_call(call_dir, prompt, schema, backend_spec, session.path, timeout)
            except Exception as exc:
                # Retain the authorized attempted dispatch even when the service
                # never supplies a complete receipt. No automatic retry.
                call_dir.mkdir(parents=True, exist_ok=True)
                failure = dict(type="instrument_preparation", call=str(call_dir.relative_to(session.path)),
                               author_cutoff=ctx["revision"], prompt_sha256=artifacts.sha(prompt.encode()),
                               schema_hash=content_hash(schema), response_hash=content_hash(None),
                               receipt_sha256=None, error=type(exc).__name__+": "+str(exc),
                               category="service_failure", dispatch_exception=True,
                               raw_receipt_sha256=file_hash(call_dir/"receipt.json") if (call_dir/"receipt.json").is_file() else None)
                if failure["raw_receipt_sha256"]:
                    from .worldbank import verify_call
                    try:
                        retained = verify_call(call_dir, expected_prompt=prompt, expected_schema=schema)
                        failure["receipt_sha256"] = failure["raw_receipt_sha256"]
                        failure["response_hash"] = content_hash(retained.get("response"))
                    except Exception:
                        pass  # Raw partial receipt remains committed, without guessed usage.
                (call_dir/"dispatch_failure.json").write_text(json.dumps(failure, indent=2)+"\n")
                with session.transaction() as (c, _):
                    session._append(c, failure)
                return {**session.view(), "agent_error":failure["error"], "runner_status":"service_failure"}
            error, category, op = None, "succeeded", None
            try:
                with session.transaction() as (c, _):
                    if len(session._events(c)) != ctx["revision"]:
                        raise RuntimeError("Campaign changed during model dispatch; audit required")
                if receipt["status"] != "succeeded":
                    category = "service_failure"
                    raise RuntimeError("Model preparation failed; inspect retained service receipt")
                op = delivery.validate_response(receipt.get("response"), ctl["allowed_operations"],
                                                delivery.DeliveryPolicy(**frozen["policy"]))
                if op["kind"] == "query":
                    session.query(json.loads(op["query_json"]))
                elif op["kind"] == "stop":
                    value = json.loads(op["decision_json"])
                    if not isinstance(value, dict) or set(value) != {"revision","decision"}:
                        raise ValueError("Stop request needs exactly revision and decision")
                    session.stop(value["decision"], value["revision"])
                elif op["kind"] in ("analysis", "prepare"):
                    outcome = execute_analysis(session, op, ctx, call_dir)
                    if outcome["status"] != "succeeded":
                        category = "offline_outcome"
                        error = outcome["publication_error"] or "Offline execution "+outcome["status"]
                else:
                    execute_predict(session, op, ctx, call_dir)
            except (ValueError, KeyError, TypeError, nbformat.ValidationError, yaml.YAMLError) as exc:
                error = type(exc).__name__ + ": " + str(exc)
                category = "format_error"
            except Exception as exc:
                error = type(exc).__name__ + ": " + str(exc)
                if category != "service_failure":
                    category = "integrity_failure"
            # The native barrier overrides every retry classification.
            after = history(session)
            if error and any(e["type"] == "instrument_native_barrier" for e in after):
                category = "native_terminal_failure"
            error = error.replace(str(session.path), "<campaign>") if error else None
            with session.transaction() as (c, _):
                if op is not None and category in ("succeeded", "offline_outcome") and receipt["response"]["campaign_notes"]:
                    session._append(c, dict(
                        type="instrument_notes", text=receipt["response"]["campaign_notes"],
                        source_call=str(call_dir.relative_to(session.path)), author_cutoff=ctx["revision"],
                    ))
                session._append(c, dict(
                    type="instrument_preparation", call=str(call_dir.relative_to(session.path)),
                    author_cutoff=ctx["revision"], prompt_sha256=artifacts.sha(prompt.encode()),
                    schema_hash=content_hash(schema), response_hash=content_hash(receipt.get("response")),
                    receipt_sha256=file_hash(call_dir/"receipt.json"), error=error, category=category,
                ))
            if on_progress:
                on_progress(dict(call=ctx["resource_usage"]["calls"],
                                 kind=op["kind"] if op else None, category=category,
                                 phase=session.view()["phase"], error=error))
            if category not in ("succeeded", "offline_outcome", "format_error"):
                return {**session.view(), "agent_error":error, "runner_status":category}
            if category == "format_error" and ctl["format_corrections_remaining"] == 0:
                return {**session.view(), "agent_error":error, "runner_status":"format_corrections_exhausted"}


def _validate_request(session, events, state, ctx, response):
    """Read-only deterministic validation, before any scientist execution."""
    import nbformat
    import yaml
    from .session import validate_decision
    from .policies_v2 import validate_claims
    from .world import normalize_action
    from sciquest.methods.ledger import validate_plan
    from sciquest.methods.policies import validate_policy_plan

    op = delivery.validate_response(response, ctx["controller"]["allowed_operations"],
                                    delivery.DeliveryPolicy(**contract(events)["policy"]))
    queries = [e for e in events if e["type"] == "query"]
    if op["kind"] == "query":
        value = json.loads(op["query_json"])
        if not isinstance(value, dict) or set(value) != {"revision","action","plan"} or type(value["revision"]) is not int or value["revision"] != len(events):
            raise ValueError("Invalid query request/revision")
        _, cost = normalize_action(value["action"], state["public"], queries)
        if cost > ctx["remaining"]:
            raise ValueError("Insufficient observation budget")
        session._query_plan(state["method_stack"], value["plan"], queries, comparison=True)
    elif op["kind"] == "stop":
        value = json.loads(op["decision_json"])
        if not isinstance(value, dict) or set(value) != {"revision","decision"} or type(value["revision"]) is not int or value["revision"] != len(events):
            raise ValueError("Invalid stop request/revision")
        decision = validate_decision(value["decision"], assisted=False, claims=True)
        validate_claims(decision, [q["observation"]["id"] for q in queries])
    elif op["kind"] in ("analysis", "prepare"):
        with tempfile.TemporaryDirectory(prefix="sciquest-retrieval-verify-") as temp:
            artifacts.stage(
                session.path/"artifacts", Path(temp), authorized_artifacts(events), op["inputs"],
                campaign_id=ctx["campaign_id"], at_revision=ctx["revision"],
                limits=artifacts.ArtifactLimits(**contract(events)["artifact_limits"]),
            )
    else:
        nbformat.validate(nbformat.reads(op["notebook_json"], as_version=4))
        submitted = validate_plan(delivery.validate_data(yaml.safe_load(op["plan_yaml"])), ctx["validation_suite"])
        validate_policy_plan(state["method_stack"], submitted)
        submitted = bind_fitted_export(session, events, op, submitted, ctx)
        delivery.encode_archive(
            archive_files(events, session.path, op["artifact_ids"]),
            campaign_id=ctx["campaign_id"], cutoff=ctx["revision"],
            limits=delivery.ArchiveLimits(**contract(events)["archive_limits"]),
        )
        return op, submitted
    return op, None


def verify(session):
    from sciquest.readonly import database_reads_only

    with database_reads_only():
        return _verify(session)


def _verify(session):
    """Read-only exact public handoff replay, failures included; no model reruns.

    An interrupted/integrity-failed allocation remains evidence but is not a
    verified delivery. This verifies engineering provenance/finite checks only.
    """
    from .worldbank import verify_call
    from . import preflight_v4
    from sciquest.state import get_record, authority_dir

    session.verify()
    with session.transaction() as (c, state):
        wrapped = session._events(c)
    events = [e["payload"] for e in wrapped]
    frozen = contract(events)
    policy = delivery.DeliveryPolicy(**frozen["policy"])
    limits = ResourceLimits(**state["comparison"]["limits"])
    policy.validate(limits)
    if (
        frozen["instructions_sha256"] != artifacts.sha(instructions(state["comparison"]).encode())
        or frozen["operation_schemas_hash"] != content_hash(delivery.VARIANTS)
        or frozen["public_software_contract_version"] != public_contract_v1.VERSION
        or state["information_mode"] != "black_box_v2"
    ):
        raise ValueError("Instrument contracts or information boundary changed")
    known_types = {
        "created","instrument_contract","query","stop","prediction_handoff",
        "instrument_analysis_started","instrument_analysis","instrument_notes",
        "instrument_archive","instrument_preflight","instrument_native_barrier",
        "instrument_preparation","instrument_dispatch","instrument_resource_stop","evaluation","final_failure",
    }
    if any(e["type"] not in known_types for e in events):
        raise ValueError("Unknown event under new instrument semantics")
    calls = [(index,e) for index,e in enumerate(events) if e["type"] == "instrument_preparation"]
    if len(calls) > limits.max_calls:
        raise ValueError("Instrument model call opportunities exceeded")
    covered, directories = {0,1}, set()
    terminal = False
    for number, (index, prep) in enumerate(calls, 1):
        cutoff = prep["author_cutoff"]
        if terminal or type(cutoff) is not int or not 2 <= cutoff <= index or any(i in covered for i in range(cutoff,index)):
            raise ValueError("Overlapping/stale call or continuation after terminal outcome")
        prior = events[:cutoff]
        between = events[cutoff:index]
        if sum(e["type"] == "instrument_preparation" for e in prior) != number-1:
            raise ValueError("Call prefix order changed")
        if any(e["type"] == "instrument_native_barrier" for e in prior):
            raise ValueError("Call after terminal native barrier")
        expected_call = f"instrument_calls/c{number:04d}"
        expected_dispatch = dict(type="instrument_dispatch", call=expected_call,
                                 prior_revision=cutoff-1, backend_spec_hash=content_hash(frozen["backend_spec"]))
        if prior[-1] != expected_dispatch:
            raise ValueError("Call has no exact preceding authorized dispatch")
        covered.add(cutoff-1)
        ctx = context(session, events=prior, state=state)
        ctl = ctx["controller"]
        prompt = instructions(ctx["comparison"])+_MARKER+json.dumps(ctx, ensure_ascii=False, separators=(",",":"))
        if ctl["terminal"] or campaign_v2.resource_stop(ctx["resource_usage"], limits, prompt):
            raise ValueError("Model call bypassed controller/resource stop")
        schema = delivery.response_schema(ctl["allowed_operations"])
        if prep["call"] != expected_call:
            raise ValueError("Unexpected call identity")
        directory = session.path/expected_call
        directories.add(directory.name)
        if prep.get("dispatch_exception"):
            if (
                prep["category"] != "service_failure" or not prep["error"] or between or index != calls[-1][0]
                or prep["prompt_sha256"] != artifacts.sha(prompt.encode()) or prep["schema_hash"] != content_hash(schema)
            ):
                raise ValueError("Missing-receipt attempt is not retained terminal service failure")
            failure = json.loads((directory/"dispatch_failure.json").read_text())
            if failure != prep:
                raise ValueError("Missing-receipt failure provenance changed")
            raw_hash = file_hash(directory/"receipt.json") if (directory/"receipt.json").is_file() else None
            if raw_hash != prep["raw_receipt_sha256"]:
                raise ValueError("Partial/late receipt bytes changed after dispatch exception")
            if prep["receipt_sha256"]:
                retained = verify_call(directory, expected_prompt=prompt, expected_schema=schema)
                if (
                    prep["receipt_sha256"] != raw_hash or prep["response_hash"] != content_hash(retained.get("response"))
                    or retained.get("backend_spec_hash") != content_hash(frozen["backend_spec"])
                    or retained.get("timeout_seconds") != frozen["timeout"]
                ):
                    raise ValueError("Retained late receipt provenance changed")
            covered.update(range(cutoff,index+1))
            terminal = True
            continue
        receipt = verify_call(directory, expected_prompt=prompt, expected_schema=schema)
        if (
            prep["receipt_sha256"] != file_hash(directory/"receipt.json")
            or prep["prompt_sha256"] != artifacts.sha(prompt.encode())
            or prep["schema_hash"] != content_hash(schema)
            or prep["response_hash"] != content_hash(receipt.get("response"))
            or receipt.get("backend_spec_hash") != content_hash(frozen["backend_spec"])
            or receipt.get("timeout_seconds") != frozen["timeout"]
        ):
            raise ValueError("Instrument model-call provenance changed")
        covered.update(range(cutoff,index+1))
        category = prep["category"]
        if category not in {"succeeded","format_error","offline_outcome","service_failure","integrity_failure","native_terminal_failure"}:
            raise ValueError("Unknown outcome accounting category")
        if (category == "succeeded") == bool(prep.get("error")):
            raise ValueError("Outcome category/error contradiction")
        if receipt["status"] != "succeeded":
            if category != "service_failure" or between:
                raise ValueError("Failed service receipt has research effects or retry classification")
            terminal = True
            continue
        if category == "service_failure":
            raise ValueError("Successful receipt labeled as service failure")
        invalid = False
        try:
            op, submitted = _validate_request(session, prior, state, ctx, receipt.get("response"))
        except RuntimeError:
            raise  # Owner evidence integrity cannot be converted into a format error.
        except Exception:
            invalid = True
        if invalid:
            if category != "format_error" or between:
                raise ValueError("Malformed request produced effects or ungrounded accounting")
            terminal = ctl["format_corrections_remaining"] == 0
            continue
        notes = [e for e in between if e["type"] == "instrument_notes"]
        expected_notes = []
        if category in ("succeeded","offline_outcome") and receipt["response"]["campaign_notes"]:
            expected_notes = [dict(type="instrument_notes", text=receipt["response"]["campaign_notes"],
                                   source_call=expected_call, author_cutoff=cutoff)]
        if notes != expected_notes:
            raise ValueError("Notes do not match their contemporaneous author response")
        effects = [e for e in between if e["type"] != "instrument_notes"]
        kind = op["kind"]
        if kind in ("query","stop"):
            if category != "succeeded" or len(effects) != 1 or effects[0]["type"] != kind:
                raise ValueError("Query/stop effects do not match the accepted operation")
            value = json.loads(op["query_json"] if kind == "query" else op["decision_json"])
            if kind == "query":
                if effects[0]["action"] != value["action"] or effects[0]["plan"] != value["plan"]:
                    raise ValueError("Recorded query differs from scientist request")
            elif effects[0]["decision"] != value["decision"]:
                raise ValueError("Frozen decision differs from scientist stop")
        elif kind in ("analysis","prepare"):
            _verify_analysis(session, prior, effects, op, ctx, expected_call, policy)
            outcome = effects[-1]
            expected_category = "succeeded" if outcome["status"] == "succeeded" else "offline_outcome"
            if category != expected_category:
                raise ValueError("Offline outcome consumed the wrong allowance")
        else:
            _verify_predict(session, prior, effects, op, submitted, ctx, directory, policy, category)
        terminal = category in ("integrity_failure","native_terminal_failure") or category == "format_error" and ctl["format_corrections_remaining"] == 0
    # The native quest is created once, between stop and the next handoff.
    handoffs = [(n,e) for n,e in enumerate(events) if e["type"] == "prediction_handoff"]
    if len(handoffs) > 1:
        raise ValueError("Repeated native quest handoff")
    for index, event in handoffs:
        if index in covered or not any(e["type"] == "stop" for e in events[:index]):
            raise ValueError("Native quest handoff outside the stopped public interface")
        covered.add(index)
    stops = [(n,e) for n,e in enumerate(events) if e["type"] == "instrument_resource_stop"]
    if len(stops) > 1:
        raise ValueError("Repeated hard resource stop")
    for index, event in stops:
        if index != len(events)-1 or index in covered or event["author_cutoff"] != index:
            raise ValueError("Resource stop has an invalid cutoff or continuation")
        ctx = context(session, events=events[:index], state=state)
        prompt = instructions(ctx["comparison"])+_MARKER+json.dumps(ctx, ensure_ascii=False, separators=(",",":"))
        if event.get("projected_dispatch"):
            dispatch = dict(type="instrument_dispatch", call=f"instrument_calls/c{ctx['resource_usage']['calls']+1:04d}",
                            prior_revision=index, backend_spec_hash=content_hash(frozen["backend_spec"]))
            projected = context(session, events=events[:index]+[dispatch], state=state)
            prompt = instructions(projected["comparison"])+_MARKER+json.dumps(projected, ensure_ascii=False, separators=(",",":"))
            reason = campaign_v2.resource_stop(projected["resource_usage"], limits, prompt)
        else:
            reason = campaign_v2.resource_stop(ctx["resource_usage"], limits, prompt)
        reason = reason or ("call_or_format_opportunities_exhausted" if ctx["controller"]["terminal"] else None)
        if not reason or event["reason"] != reason or event["ledger"] != ctx["resource_usage"]:
            raise ValueError("Resource stop does not follow the authorized prefix")
        covered.add(index)
    if covered != set(range(len(events))):
        raise ValueError("Unaccounted instrument side effects or incomplete attempt")
    calls_root = session.path/"instrument_calls"
    paths = list(calls_root.iterdir()) if calls_root.exists() else []
    if {p.name for p in paths} != directories or any(p.is_symlink() or not p.is_dir() for p in paths):
        raise ValueError("Unrecorded or unsafe instrument call directory")
    if (session.path/"model_calls").exists() and any((session.path/"model_calls").iterdir()):
        raise ValueError("Legacy/unrecorded model calls under new instrument")
    audit_store(session, events)
    if state["phase"] == "completed":
        if len([e for e in events if e["type"] == "instrument_native_barrier"]) != 1:
            raise ValueError("Completed campaign lacks its unique native barrier")
    return dict(
        verified=True, version=VERSION, phase=state["phase"], calls=len(calls),
        completed_delivery=state["phase"] == "completed" and bool(calls) and calls[-1][1]["category"] == "succeeded",
        scope="Exact authorized public handoffs, artifact bytes, resource accounting and declared finite checks. No inference, calibration, philosophical or arbitrary program-semantic certification.",
        usage=usage(events, session.path),
    )


def _verify_analysis(session, prior, effects, op, ctx, call, policy):
    if [e["type"] for e in effects] != ["instrument_analysis_started","instrument_analysis"]:
        raise ValueError("Offline launch/result sequence is incomplete or duplicated")
    started, result = effects
    aid = f"a{sum(e['type']=='instrument_analysis_started' for e in prior)+1:04d}"
    catalog = {o["id"]:o for record in authorized_artifacts(prior) for o in record["manifest"]["objects"]}
    code_sha = artifacts.sha(op["program"].encode())
    context_sha = artifacts.sha(artifacts.encoded(ctx))
    expected_start = dict(
        type="instrument_analysis_started", analysis_id=aid, kind=op["kind"],
        source_call=call, author_cutoff=ctx["revision"], code_sha256=code_sha,
        context_sha256=context_sha, input_ids=op["inputs"], outputs=op["outputs"],
        retrieved_bytes=sum(catalog[x]["bytes"] for x in op["inputs"]),
    )
    if started != expected_start or any(result[k] != started[k] for k in (
        "analysis_id","kind","source_call","author_cutoff","code_sha256","context_sha256"
    )):
        raise ValueError("Offline source/cutoff/input/output binding changed")
    if (
        result["code_file"] != call+"/analysis.py" or result["output_file"] != call+"/analysis.log"
        or file_hash(session.path/result["code_file"]) != code_sha
        or file_hash(session.path/call/"analysis_context.json") != context_sha
        or file_hash(session.path/result["output_file"]) != result["output_sha256"]
        or result["fitted_bundle"] != op["fitted_bundle"]
    ):
        raise ValueError("Offline retained code/context/output bytes changed")
    execution = result["execution"]
    if (
        not execution["blind_filesystem"] or not execution["isolated"] or execution["network"]
        or execution["timeout_seconds"] != policy.analysis_timeout_seconds
        or execution["command_hash"] != content_hash([sys.executable,"analysis.py"])
    ):
        raise ValueError("Offline execution boundary/command changed")
    record = result["artifacts"]
    if record:
        if execution["status"] != "succeeded" or result["status"] != "succeeded" or result["publication_error"]:
            raise ValueError("Failed offline work published a success artifact catalog")
        manifest = artifacts.verify(session.path/"artifacts", record, ctx["campaign_id"],
                                    limits=artifacts.ArtifactLimits(**contract(prior)["artifact_limits"]))
        if (
            manifest["analysis_id"] != aid or manifest["source_revision"] != ctx["revision"]
            or manifest["available_revision"] != ctx["revision"]+2
            or manifest["code_sha256"] != code_sha or manifest["context_sha256"] != context_sha
            or [o["name"] for o in manifest["objects"]] != [n[8:] for n in op["outputs"]]
        ):
            raise ValueError("Published artifact source/availability/output list changed")
    elif execution["status"] == "succeeded":
        if result["status"] != "publication_failed" or not result["publication_error"]:
            raise ValueError("Successful offline execution lost its publication outcome")
    elif result["status"] != execution["status"] or result["publication_error"]:
        raise ValueError("Offline runtime outcome changed")
    fidelity = result["fidelity"]
    if fidelity:
        directory = session.path/call/"fidelity"
        if fidelity != fidelity_v1._json(artifacts._read_regular(directory,"fidelity.json",1048576)):
            raise ValueError("Retained fitted-check record changed")
        for check in fidelity["checks"]:
            if file_hash(directory/(check["kind"]+".log")) != check["log_sha256"]:
                raise ValueError("Retained fitted-check log changed")
        for name, digest in fidelity["output_sha256"].items():
            if name not in {"live_reference.json","structure.json","reload.json"} or file_hash(directory/name) != digest:
                raise ValueError("Retained fitted-check output bytes changed")
        if fidelity["status"] == "succeeded":
            manifest = fidelity["declaration"]
            selected = [aid+"/"+n[8:] for n in [op["fitted_bundle"],*manifest["model_files"],manifest["reload_program"]]]
            bind_fitted_export(
                session, prior+effects,
                dict(fitted_bundle=aid+"/"+op["fitted_bundle"][8:], artifact_ids=selected),
                dict(model_export=dict(artifacts=["artifacts/"+x for x in selected[1:]],
                                       reload_notebook="reload.ipynb")), ctx,
            )
        elif fidelity["status"] != "failed" or fidelity["verified_quantities"] or not fidelity["error"]:
            raise ValueError("Fitted failure status is inconsistent")


def _verify_predict(session, prior, effects, op, submitted, ctx, directory, policy, category):
    from . import preflight_v4
    from sciquest.state import get_record, authority_dir

    types = [e["type"] for e in effects]
    if types[:2] != ["instrument_archive","instrument_preflight"]:
        raise ValueError("Predictor lacks frozen archive and retained public preflight")
    archive, event = effects[:2]
    frozen = contract(prior)
    expected_raw = delivery.encode_archive(
        archive_files(prior, session.path, op["artifact_ids"]),
        campaign_id=ctx["campaign_id"], cutoff=ctx["revision"],
        limits=delivery.ArchiveLimits(**frozen["archive_limits"]),
    )
    if (
        archive["source_call"] != str(directory.relative_to(session.path))
        or archive["author_cutoff"] != ctx["revision"] or archive["artifact_ids"] != op["artifact_ids"]
        or archive["raw_sha256"] != artifacts.sha(expected_raw) or archive["bytes"] != len(expected_raw)
        or archive["path"] != str(directory.relative_to(session.path))+"/instrument_archive.csv"
        or (session.path/archive["path"]).read_bytes() != expected_raw
    ):
        raise ValueError("Selected native archive differs from authorized prefix")
    native_plan = {
        **submitted, "demand_scientific_decision":ctx["decision"],
        "scientist_information_mode":"black_box_v2", "comparison_assignment":ctx["comparison"],
    }
    record = preflight_v4.verify_record(
        directory/"preflight", artifacts.sha(op["notebook_json"].encode()),
        content_hash(native_plan), ctx["revision"], context=ctx,
        expected_archive=list(csv.DictReader(io.StringIO(expected_raw.decode()))),
        expected_export=submitted.get("model_export"),
    )
    base = json.loads((directory/"preflight/preflight.json").read_text())
    if (
        event != dict(type="instrument_preflight", source_call=str(directory.relative_to(session.path)),
                      author_cutoff=ctx["revision"], record=record, record_hash=content_hash(record))
        or base["timeout"] != policy.preflight_timeout_seconds
    ):
        raise ValueError("Public preflight event/timeout changed")
    if record["status"] == "failed":
        if category != "format_error" or len(effects) != 2:
            raise ValueError("Failed preflight was not retained as a preparation error")
        return
    if category == "integrity_failure" and len(effects) == 2:
        return
    if len(effects) < 3 or effects[2]["type"] != "instrument_native_barrier":
        raise ValueError("Successful predictor has no terminal native barrier")
    barrier = effects[2]
    if barrier != dict(
        type="instrument_native_barrier", source_call=str(directory.relative_to(session.path)),
        author_cutoff=ctx["revision"], notebook_sha256=artifacts.sha(op["notebook_json"].encode()),
        plan_hash=content_hash(native_plan), archive_sha256=archive["raw_sha256"],
        preflight_hash=content_hash(record),
    ):
        raise ValueError("Native barrier differs from frozen public handoff")
    if types[3:] not in ([], ["evaluation"], ["final_failure"]):
        raise ValueError("Repeated or unexpected native outcomes")
    with session.transaction() as (_, state):
        quest, expid = Path(state["quest"]), state["experiment"]
    registered = get_record(quest,"registration",expid)
    if registered:
        name = ctx["prediction_files"]["campaign_archive"]
        snapshot = authority_dir(quest)/"snapshots"/expid
        if (
            registered["plan"] != native_plan
            or registered["file_hashes"][name] != archive["raw_sha256"]
            or file_hash(snapshot/name) != archive["raw_sha256"]
            or registered["file_hashes"]["notebook.ipynb"] != barrier["notebook_sha256"]
            or content_hash(registered["data_manifest"]) != archive["data_manifest_hash"]
        ):
            raise ValueError("Native registration differs from checked notebook/plan/archive")
    if category == "succeeded":
        if types[3:] != ["evaluation"] or not registered:
            raise ValueError("Successful predictor lacks a native evaluation")
        verify_native_predictions(session, ctx, directory, expected_export=submitted.get("model_export"))
    elif category != "native_terminal_failure":
        raise ValueError("Native handoff was treated as retryable preparation")
