"""Versioned delivery controller: compact evidence, exact archives, reserved finish.

No scientist-authored history is summarized by another agent. Resource reserves
are prospective controller thresholds, not an unverifiable backend token cap.
"""

from __future__ import annotations

import copy
import json
import hashlib
import fcntl
from dataclasses import asdict, dataclass
from pathlib import Path

from sciquest.io import content_hash, file_hash
from . import campaign_v2
from .apparatus import SEMANTICS, action_schemas, plan_schema
from .policies_v2 import COMMON, GUIDANCE, ResourceLimits
from .session import public_query

VERSION = "scientist-delivery-v3.2"


@dataclass(frozen=True)
class DeliveryPolicy:
    finalization_calls: int = 4  # stop, predict, up to two format corrections
    finalization_input_tokens: int = 60000
    finalization_output_tokens: int = 18000
    backend_overhead_tokens: int = 16000
    max_notes_bytes: int = 8000
    max_decision_bytes: int = 16000
    max_predictions_per_query: int = 8
    max_prediction_id_bytes: int = 128
    preflight_timeout_seconds: int = 120

    def validate(self, limits, repair_limit):
        if type(repair_limit) is not int or not 0 <= repair_limit <= 10:
            raise ValueError("Format-correction allowance must be an integer in 0..10")
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("Delivery reserves must be positive integers")
        if (
            self.finalization_calls < 2 + repair_limit
            or self.finalization_calls >= limits.max_calls
        ):
            raise ValueError(
                "Reserve stop + predictor + every allowed format correction; leave discovery calls"
            )
        if (
            self.finalization_input_tokens >= limits.input_token_stop
            or self.finalization_output_tokens >= limits.output_token_stop
        ):
            raise ValueError(
                "Delivery token reserves must leave a positive discovery budget"
            )
        return self


PROTOCOL = """You are an experimental scientist investigating a hidden apparatus.
The supervisor owns simulation, immutable evidence, resource limits and evaluation.
Return exactly the response schema. Put query/stop JSON in proposal_json, offline
Python in analysis_code, or complete notebook JSON and native plan YAML in
notebook_json/plan_yaml. Other payload strings must be empty. campaign_notes holds
your own bounded notes; empty retains the previous notes, nonempty replaces them.
Use context.json as supplied text; you have no direct tools or external memory.

Collection: query proposal is {kind:query,query:{revision:CURRENT,action:ACTION,plan:PLAN}}.
Use the exact apparatus and plan schemas in context. Optional methodological
records have identical validation in all arms. Stop proposal is
{kind:stop,revision:CURRENT,decision:DECISION}. DECISION has status
selected/inconclusive/not_identifiable/model_inadequate, nonempty reason,
explanation and limitations, and the claim records specified below. Stopping
freezes scientific conclusions and closes discovery permanently.

The history is compact: every observation and prospective threshold result stays
visible. Full original plans, including rivals, assumptions and abandoned claims,
are retrievable in campaign/queries.json. These are archival records, not new
observations. Offline analyses receive the exact files listed in workspace_files;
do not search for unlisted data files. They run without network or hidden-world
access. Prior analysis code and complete output are indexed under analysis/.
Use them directly instead of repeatedly printing entire programs or histories.
Full earlier notes and preparation errors are retained under campaign/. No owner
summary, rival catalog, likelihood or scientific solution is supplied.

When controller.stop_required is true, submit your scientific stop now. The
controller reserves calls and token opportunities for stopping, predictor delivery
and format corrections. After stopping, only predict is allowed. Produce an honest
provisional predictor from existing evidence; scientific uncertainty is acceptable.
Submit a native plan whose action is a STRING and predictions name demand_rmse or
sales_rmse. Its optional methodology records follow the same common validation;
do not use query IDs as native experiment prior_source IDs. Do not execute the
notebook. It receives the declared native training/feature files and the registered
campaign_archive CSV containing your own prior code, full output and records.
Read its path/part/parts/text/sha256 columns to reuse your own analysis instead of
rebuilding it: concatenate rows of the same path in numeric part order; sha256
hashes each text chunk's UTF-8 bytes. Chunking preserves complete large outputs.
Standalone model reload must use exported model artifacts, not training files.
Emit predictions.csv with every evaluation ID and
mean_demand, mean_sales, lower_sales, upper_sales. Means target physical demand and
capped sales; intervals target fresh standard-precision noisy sales readings.
No final scores are revealed until submission. Never manufacture scores or claim
scientific confirmation. Validation feedback concerns only your malformed inputs.
"""


PROTOCOL += """
Before native registration, the supervisor checks your predictor and optional clean
reload offline using ONLY the public inputs. Runtime, file-reference and public
output-format errors can use the same bounded pre-registration correction allowance
as syntax errors. No hidden labels, scientific scores or new observations are used.
Native registration/execution failure remains terminal and cannot be rescued.
On a rejected preparation, last_predictor_preparation identifies its exact archived
notebook and plan. The updated campaign archive includes submissions/pNNNN.ipynb,
submissions/pNNNN.yaml and submissions/pNNNN.error.txt. Reuse or revise your own
saved program rather than guessing a former filename/marker or retyping it all.
Your scientific decision stays frozen during these implementation corrections.
"""


def instructions(assignment):
    # COMMON's old memory paragraph is replaced explicitly, preserving uniform
    # scientific claim and optional-record rules without obsolete replay advice.
    start = COMMON.index("Use campaign_notes")
    end = COMMON.index("Freeze a final decision")
    common = COMMON[:start] + COMMON[end:]
    return (
        PROTOCOL
        + "\n"
        + common
        + "\nAPPARATUS SEMANTICS:\n"
        + SEMANTICS
        + "\nFIXED INQUIRY GUIDANCE:\n"
        + GUIDANCE[assignment["policy"]]
    )


def contract(session):
    records = [
        e for e in campaign_v2.events(session) if e["type"] == "delivery_contract"
    ]
    if len(records) > 1:
        raise ValueError("Multiple delivery contracts")
    return records[0] if records else None


def initialize(session, policy, spec, timeout, repair_limit):
    from .runner import RESPONSE_SCHEMA

    view = session.view(scientist=True)
    if not view.get("comparison"):
        raise ValueError("Delivery v3 requires a blind matched-guidance session")
    limits = ResourceLimits(**view["comparison"]["limits"])
    policy.validate(limits, repair_limit)
    expected = dict(
        type="delivery_contract",
        version=VERSION,
        policy=asdict(policy),
        backend_spec=spec,
        timeout=timeout,
        repair_limit=repair_limit,
        instructions_hash=content_hash(instructions(view["comparison"])),
        response_schema_hash=content_hash(campaign_v2.response_schema(RESPONSE_SCHEMA)),
    )
    with session.transaction() as (c, _):
        prior = [e["payload"] for e in session._events(c)]
        found = [e for e in prior if e["type"] == "delivery_contract"]
        if found:
            if found != [expected]:
                raise ValueError("Frozen delivery contract changed")
            raise ValueError(
                "Delivery attempt already started; preserve it and verify, never restart"
            )
        if len(prior) != 1:
            raise ValueError(
                "Delivery v3 requires a fresh session; do not salvage an existing campaign"
            )
        session._append(c, expected)
    return expected


def archive_records(session):
    history = campaign_v2.events(session)
    records = records_from_events(history)
    records["campaign/predictor_preparations.json"] = submission_records(
        history, session.path
    )
    return records


def submission_records(history, root):
    records = []
    for event in history:
        if event["type"] != "model_preparation":
            continue
        path = Path(root) / event["call"] / "receipt.json"
        if file_hash(path) != event["receipt_sha256"]:
            raise ValueError("Own preparation receipt changed")
        receipt = json.loads(path.read_text())
        response = receipt.get("response", {})
        if response.get("kind") == "predict":
            records.append(
                dict(
                    attempt=len(records) + 1,
                    notebook_json=response["notebook_json"],
                    plan_yaml=response["plan_yaml"],
                    error=event.get("error"),
                    call=event["call"],
                )
            )
    return records


def records_from_events(history):
    return {
        "campaign/queries.json": [
            public_query(e) for e in history if e["type"] == "query"
        ],
        "campaign/notes.json": [
            dict(text=e["text"], source_call=e["source_call"])
            for e in history
            if e["type"] == "campaign_notes"
        ],
        "campaign/preparation_errors.json": [
            dict(error=e["error"], call=e["call"], repairable=e["repairable"])
            for e in history
            if e["type"] == "model_preparation" and e.get("error")
        ],
    }


def native_archive_rows(history, root):
    """Serialize only allowlisted public/author-produced records as training data."""
    files = {
        name: json.dumps(data) for name, data in records_from_events(history).items()
    }
    submissions = submission_records(history, root)
    files["campaign/predictor_preparations.json"] = json.dumps(submissions)
    for item in submissions:
        prefix = f"submissions/p{item['attempt']:04d}"
        files[prefix + ".ipynb"] = item["notebook_json"]
        files[prefix + ".yaml"] = item["plan_yaml"]
        files[prefix + ".error.txt"] = item["error"] or ""
    number = 0
    for event in history:
        if event["type"] != "scientist_analysis":
            continue
        number += 1
        analysis = event["analysis"]
        files[f"analysis/a{number:04d}.py"] = analysis["code"]
        source = Path(root) / analysis["full_output_file"]
        if file_hash(source) != analysis["full_output_hash"]:
            raise ValueError("Own-campaign analysis output changed")
        files[f"analysis/a{number:04d}.txt"] = source.read_text(errors="replace")
    rows = []
    for name, text in sorted(files.items()):
        chunks = [text[i : i + 16000] for i in range(0, len(text), 16000)] or [""]
        for i, chunk in enumerate(chunks, 1):
            identity = "archive:" + name + (f"#{i:05d}" if len(chunks) > 1 else "")
            rows.append(
                dict(
                    id=identity,
                    path=name,
                    part=str(i),
                    parts=str(len(chunks)),
                    text=chunk,
                    sha256=hashlib.sha256(chunk.encode()).hexdigest(),
                )
            )
    return rows


def validate_payload(response, frozen):
    policy = DeliveryPolicy(**frozen["policy"])
    if len(response["campaign_notes"].encode()) > policy.max_notes_bytes:
        raise ValueError(
            f"campaign_notes exceeds {policy.max_notes_bytes} UTF-8 bytes; retain full work in indexed files"
        )
    if response["kind"] not in ("query", "stop"):
        return
    proposal = json.loads(response["proposal_json"])
    if response["kind"] == "stop":
        if len(json.dumps(proposal["decision"]).encode()) > policy.max_decision_bytes:
            raise ValueError(
                f"decision exceeds {policy.max_decision_bytes} serialized UTF-8 bytes; keep material claims and cite full archived work"
            )
    else:
        predictions = proposal["query"]["plan"]["predictions"]
        if (
            not isinstance(predictions, list)
            or len(predictions) > policy.max_predictions_per_query
        ):
            raise ValueError(
                f"query.plan.predictions must contain at most {policy.max_predictions_per_query} predictions"
            )
        for prediction in predictions:
            if (
                not isinstance(prediction, dict)
                or not isinstance(prediction.get("id"), str)
                or len(prediction["id"].encode()) > policy.max_prediction_id_bytes
            ):
                raise ValueError(
                    f"prediction.id must be a string within {policy.max_prediction_id_bytes} UTF-8 bytes"
                )


def transform_context(context, session, frozen):
    """Pure public-context transform, also used by exact historical replay."""
    from sciquest.validation import COMPARATORS

    context = copy.deepcopy(context)
    policy = DeliveryPolicy(**frozen["policy"])
    limits = ResourceLimits(**context["comparison"]["limits"])
    archive = archive_records(session)
    submissions = archive["campaign/predictor_preparations.json"]
    context["last_predictor_preparation"] = None
    if submissions:
        last = submissions[-1]
        excerpt = last["notebook_json"].encode()[:12000].decode(errors="ignore")
        context["last_predictor_preparation"] = dict(
            attempt=last["attempt"],
            notebook_file=f"submissions/p{last['attempt']:04d}.ipynb",
            plan_file=f"submissions/p{last['attempt']:04d}.yaml",
            notebook_excerpt=excerpt,
            excerpted=excerpt != last["notebook_json"],
            error=last["error"],
        )
    preflights = [
        e["record"]
        for e in campaign_v2.events(session)
        if e["type"] == "predictor_preflight"
    ]
    context["resource_usage"]["predictor_preflights"] = len(preflights)
    context["resource_usage"]["preflight_execution_seconds"] = sum(
        (r.get("execution") or {}).get("elapsed_seconds", 0)
        + (r.get("reload_execution") or {}).get("elapsed_seconds", 0)
        for r in preflights
    )
    context["archive_index"] = {
        name: dict(sha256=content_hash(data), records=len(data))
        for name, data in archive.items()
    }
    context["history"] = [
        dict(
            action=q["action"],
            design=q["design"],
            cost=q["cost"],
            observation=q["observation"],
            plan_reference=dict(
                file="campaign/queries.json", observation_id=q["observation"]["id"]
            ),
            predictions=[
                dict(
                    **p,
                    outcome=(
                        "supported"
                        if COMPARATORS[p["operator"]](
                            q["observation"]["value"], p["threshold"]
                        )
                        else "contradicted"
                    ),
                )
                for p in q["plan"]["predictions"]
            ],
        )
        for q in context["history"]
    ]
    if context["phase"] == "prediction_required":
        for entry in context["analysis_history"]:
            entry.pop("recent_output", None)
            entry["retrieval_only"] = True
        context["prediction_files"][
            "campaign_archive"
        ] = f"data/{2 if context['history'] else 1}/campaign_archive.csv"
        context["prediction_files"]["campaign_archive_columns"] = [
            "id",
            "path",
            "part",
            "parts",
            "text",
            "sha256",
        ]
    context["memory_rules"] = dict(
        max_notes_chars=limits.max_notes_chars,
        retrieval="Offline analysis receives the complete indexed same-campaign files. Original failed predictions, plans, prior notes and errors are preserved, not summarized away.",
    )
    context["apparatus_action_schemas"] = action_schemas(
        context["contract"], [q["observation"] for q in context["history"]]
    )
    context["collecting_plan_schema"] = plan_schema(["observation"], collecting=True)
    context["final_plan_schema"] = plan_schema(["demand_rmse", "sales_rmse"])
    context["workspace_files"] = dict(
        offline_analysis=[
            "context.json",
            *archive,
            "analysis/aNNNN.py",
            "analysis/aNNNN.txt",
        ],
        final_notebook=[
            "validation.yaml",
            "experiment_plan.yaml",
            "data/INDEX/train.csv (only when observations exist)",
            "data/INDEX/features.csv",
            "data/INDEX/campaign_archive.csv (id,path,part,parts,text,sha256; own prior work)",
        ],
        final_output="predictions.csv",
        notes="Native data/ files exist only in the final notebook workspace. Analyses use observations in context.json and indexed files. The final notebook has a registered training CSV containing the same prior code/output/records, not loose archive files. Standalone reload excludes training and must use your exported artifacts.",
    )
    controller = context["controller"]
    usage = context["resource_usage"]
    controller["delivery_version"] = VERSION
    controller["execution_limits"] = dict(
        offline_analysis_seconds=frozen["timeout"],
        predictor_preflight_seconds=policy.preflight_timeout_seconds,
        native_predictor_seconds=120,
    )
    controller["preparation_corrections_remaining"] = max(
        0,
        frozen["repair_limit"]
        - sum(
            bool(e.get("error"))
            for e in campaign_v2.events(session)
            if e["type"] == "model_preparation"
        ),
    )
    controller["reserves"] = asdict(policy)
    controller["reserve_reasons"] = []
    if controller["model_calls_remaining"] <= policy.finalization_calls:
        controller["reserve_reasons"].append("finalization_call_reserve")
    if (
        usage["input_tokens"]
        >= limits.input_token_stop - policy.finalization_input_tokens
    ):
        controller["reserve_reasons"].append("finalization_input_reserve")
    if (
        usage["output_tokens"]
        >= limits.output_token_stop - policy.finalization_output_tokens
    ):
        controller["reserve_reasons"].append("finalization_output_reserve")
    # Conservative prompt byte allowance plus registered backend overhead. This
    # is a forecast, not a bound on service-side token accounting or generation.
    estimated = (
        len(json.dumps(context).encode())
        + len(instructions(context["comparison"]).encode())
        + policy.backend_overhead_tokens
    )
    controller["estimated_next_input_tokens"] = estimated
    if (
        usage["input_tokens"] + estimated + policy.finalization_input_tokens
        >= limits.input_token_stop
    ):
        controller["reserve_reasons"].append("projected_input_reserve")
    controller["stop_required"] = context["phase"] == "collecting" and bool(
        controller["reserve_reasons"]
    )
    controller["predict_required"] = context["phase"] == "prediction_required"
    controller["allowed_kinds"] = (
        ["predict"]
        if controller["predict_required"]
        else ["stop"] if controller["stop_required"] else ["query", "analysis", "stop"]
    )
    controller["reserve_limits"] = (
        "Prospective thresholds and conservative prompt forecast; backend overhead/output are not hard capped. Actual hard stops still apply before every call, with possible one-call usage overshoot. Report exhaustion as failure, never increase limits or invent a predictor."
    )
    return context


def schema_for_context(base_schema, context):
    schema = copy.deepcopy(base_schema)
    schema["properties"]["kind"]["enum"] = context["controller"]["allowed_kinds"]
    return schema


def stage_archive(session, workspace, expected_index):
    archive = archive_records(session)
    actual = {
        name: dict(sha256=content_hash(data), records=len(data))
        for name, data in archive.items()
    }
    if actual != expected_index:
        raise ValueError("Own-campaign archive differs from model handoff")
    for name, data in archive.items():
        path = Path(workspace) / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data))


def refresh_native_archive(session, quest):
    from sciquest.state import get_record
    from .session import write_csv

    with session.transaction() as (c, state):
        if get_record(Path(quest), "registration", state["experiment"]):
            raise ValueError("Registered native inputs cannot be refreshed or repaired")
        prior = [e["payload"] for e in session._events(c)]
        rows = native_archive_rows(prior, session.path)
        path = Path(quest) / "data/raw/campaign_archive.csv"
        import csv

        if list(csv.DictReader(path.open())) != rows:
            write_csv(path, rows, ["id", "path", "part", "parts", "text", "sha256"])
            session._append(
                c,
                dict(
                    type="native_archive_refresh",
                    source_revision=len(prior),
                    archive_sha256=file_hash(path),
                ),
            )


def run(session, *, policy=None, repair_limit=2, **kwargs):
    from .runner import run_scientist

    with (session.path / ".demand/delivery.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This delivery attempt has a live supervisor") from exc
        return run_scientist(
            session,
            delivery_policy=policy or DeliveryPolicy(),
            repair_limit=repair_limit,
            **kwargs,
        )
