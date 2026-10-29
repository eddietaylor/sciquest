"""Prepare and verify matched v2 comparisons. Preparation never launches a model."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from sciquest.io import content_hash, file_hash, read_yaml, utc_now
from sciquest.runtime import runtime_manifest
from .adoption_v2 import load_adopted
from .assessment_v2 import PROMPT as REVIEW_PROMPT, SCHEMA as REVIEW_SCHEMA
from .campaign_v2 import (
    instructions,
    response_schema,
    usage_ledger,
    compact_memory,
    resource_stop,
)
from .policies_v2 import (
    POLICIES,
    ResourceLimits,
    assignment,
    instruction_audit,
    fidelity_audit,
)
from .runner import PROTOCOL, MODEL_INSTRUCTIONS, RESPONSE_SCHEMA, available_libraries
from .scoring_v2 import SCORING_CONTRACT
from .session import DemandSession, observation_fields, public_query
from .world import DemandConfig, EVAL_FEATURES, evaluation_rows
from .worldbank import dump, verify_call

VERSION = "matched-demand-comparison-v2.1"


def prepare_comparison(
    path,
    adoption,
    *,
    per_stratum=1,
    limits=None,
    model="gpt-6.1-sol",
    reasoning_effort="high",
):
    """Freeze a DEVELOPMENT protocol; running campaigns is a separate next phase."""
    if type(per_stratum) is not int or not 1 <= per_stratum <= 4:
        raise ValueError("Use 1..4 development worlds per stratum")
    record, manifest, private = load_adopted(adoption)
    limits = limits or ResourceLimits()
    strata = sorted({w["spec"]["stratum"] for w in private["worlds"]})
    selected = []
    for stratum in strata:
        candidates = [w for w in private["worlds"] if w["spec"]["stratum"] == stratum]
        candidates.sort(
            key=lambda w: content_hash(
                [manifest["private_commitment"], w["id"], "comparison-development-v2"]
            )
        )
        if len(candidates) < per_stratum:
            raise ValueError("Insufficient development worlds")
        selected.extend(candidates[:per_stratum])
    allocation = [dict(block=f"b{i:03d}", world=w) for i, w in enumerate(selected)]
    order = [dict(block=b["block"], policy=p) for b in allocation for p in POLICIES]
    seed = int(content_hash([manifest["private_commitment"], VERSION]), 16)
    random.Random(seed).shuffle(order)
    protocol = dict(
        version=VERSION,
        frozen_at=utc_now(),
        status="prepared_not_run",
        role="development",
        model=model,
        reasoning_effort=reasoning_effort,
        runtime=runtime_manifest(),
        adoption=str(Path(adoption).resolve()),
        adoption_hash=content_hash(record),
        manifest_hash=content_hash(manifest),
        per_stratum=per_stratum,
        limits=asdict(limits),
        policies=list(POLICIES),
        order=order,
        allocation_commitment=content_hash(allocation),
        prompt_prefixes={
            p: instructions(PROTOCOL, MODEL_INSTRUCTIONS, assignment(p, limits))
            for p in POLICIES
        },
        schema=response_schema(RESPONSE_SCHEMA),
        scoring=SCORING_CONTRACT,
        fairness=instruction_audit(),
        explanations=dict(
            prompt=REVIEW_PROMPT,
            schema=REVIEW_SCHEMA,
            masked_reviews=2,
            reporting="Retain both independent fresh assessments, failures, disagreement and claim-level evidence. No averaging away contradictions. Mask policy identity and numerical outcome scores; style masking is partial. Shared-model ratings are diagnostic, not human adjudication.",
        ),
        primary=None,
        readiness="Software/readiness only. Before ANY scientist call freeze backend executable/version, timeout, repair allowance, failure-inclusive primary, scales, meaningful target, outcome hierarchy and execution gates in a new pilot protocol. This preparation is NOT an executable pilot or confirmatory registration.",
        analysis="Matched independent-world paired contrasts. Report completion, query-setting versus transfer scores, interval scores, feasible decision regret, claim-level warrant/identification, behavioral fidelity and known resource use separately. No data-dependent winner/primary selection; no rows-as-replicates. Inspect all campaigns including failures. No confirmatory conclusions from these development worlds.",
        isolation="Fixed laws,seeds and apparatus per matched world; fresh memory-disabled model session per call; only explicit own-campaign observations/notes/analysis staged; no cross-campaign or owner-answer access.",
    )
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    dump(path / "protocol.json", protocol)
    (path / "protocol.sha256").write_text(content_hash(protocol) + "\n")
    dump(path / "private/allocation.json", allocation)
    (path / "private/allocation.json").chmod(0o600)
    return protocol


def load_comparison(path):
    path = Path(path).resolve()
    spec = json.loads((path / "protocol.json").read_text())
    allocation = json.loads((path / "private/allocation.json").read_text())
    record, manifest, private = load_adopted(spec["adoption"])
    if (
        spec["version"] != VERSION
        or content_hash(spec) != (path / "protocol.sha256").read_text().strip()
        or content_hash(allocation) != spec["allocation_commitment"]
        or spec["runtime"]["fingerprint"] != runtime_manifest()["fingerprint"]
        or spec["adoption_hash"] != content_hash(record)
        or spec["manifest_hash"] != content_hash(manifest)
    ):
        raise ValueError("Comparison preparation changed")
    if any(b["world"] not in private["worlds"] for b in allocation):
        raise ValueError("Comparison world not in adopted collection")
    expected = {(b["block"], p) for b in allocation for p in POLICIES}
    if {(r["block"], r["policy"]) for r in spec["order"]} != expected or len(
        spec["order"]
    ) != len(expected):
        raise ValueError("Unmatched comparison schedule")
    if (
        spec["role"] != "development"
        or spec["status"] != "prepared_not_run"
        or spec["primary"] is not None
        or spec["policies"] != list(POLICIES)
    ):
        raise ValueError(
            "Readiness preparation is not an executable or confirmatory registration"
        )
    selected = []
    for stratum in sorted({w["spec"]["stratum"] for w in private["worlds"]}):
        candidates = sorted(
            [w for w in private["worlds"] if w["spec"]["stratum"] == stratum],
            key=lambda w: content_hash(
                [manifest["private_commitment"], w["id"], "comparison-development-v2"]
            ),
        )
        selected.extend(candidates[: spec["per_stratum"]])
    if allocation != [dict(block=f"b{i:03d}", world=w) for i, w in enumerate(selected)]:
        raise ValueError(
            "Development allocation is not its registered balanced selection"
        )
    limits = ResourceLimits(**spec["limits"])
    if (
        spec["prompt_prefixes"]
        != {
            p: instructions(PROTOCOL, MODEL_INSTRUCTIONS, assignment(p, limits))
            for p in POLICIES
        }
        or spec["schema"] != response_schema(RESPONSE_SCHEMA)
        or spec["scoring"] != SCORING_CONTRACT
        or spec["fairness"] != instruction_audit()
        or spec["explanations"]["prompt"] != REVIEW_PROMPT
        or spec["explanations"]["schema"] != REVIEW_SCHEMA
    ):
        raise ValueError("Comparison contracts changed")
    return spec, allocation


def create_comparison_session(path, adoption, identity, policy, limits=None):
    _, _, private = load_adopted(adoption)
    selected = [w for w in private["worlds"] if w["id"] == identity]
    if len(selected) != 1:
        raise ValueError("Unknown adopted world")
    w = selected[0]
    return DemandSession.create(
        path,
        DemandConfig(
            scenario=(
                "observational_only"
                if w["spec"]["stratum"] == "observational_only"
                else "clean"
            ),
            seed=w["seed"],
            budget=24,
            price_limit=w["spec"]["price_limit"],
            world_spec=w["spec"],
        ),
        comparison_policy=policy,
        resource_limits=limits,
    )


def verify_comparison_session(session):
    """Replay exact allowlisted handoffs, own memory, choices, usage and scoring."""
    from contextlib import contextmanager
    from sciquest.methods.policies import policy_requirements

    session.verify()
    with session.transaction() as (c, state):
        events = session._events(c)
    if any(e["payload"]["type"] == "instrument_contract" for e in events):
        raise ValueError("Use instrument_v4.verify for the new prospective instrument")
    assigned = state.get("comparison")
    if assigned is None:
        raise ValueError("Session has no matched comparison assignment")
    limits = ResourceLimits(**assigned["limits"])
    prefix = instructions(PROTOCOL, MODEL_INSTRUCTIONS, assigned)
    schema = response_schema(RESPONSE_SCHEMA)
    calls = [e for e in events if e["payload"]["type"] == "model_preparation"]
    from . import delivery_v3

    delivery = delivery_v3.contract(session)
    if delivery is not None:
        if len(events) < 2 or events[1]["payload"] != delivery:
            raise ValueError("Delivery contract must precede every scientific action")
        policy = delivery_v3.DeliveryPolicy(**delivery["policy"])
        policy.validate(limits, delivery["repair_limit"])
        if (
            delivery["version"] != delivery_v3.VERSION
            or delivery["instructions_hash"]
            != content_hash(delivery_v3.instructions(assigned))
            or delivery["response_schema_hash"] != content_hash(schema)
        ):
            raise ValueError("Delivery contract changed")
    if len(calls) > limits.max_calls:
        raise ValueError("Model call limit exceeded")
    known_dirs = set()
    covered = set()
    for turn, event in enumerate(calls):
        prep = event["payload"]
        revision = prep["context_revision"]
        if type(revision) is not int or not 1 <= revision < event["revision"]:
            raise ValueError("Invalid call revision")
        prior = events[:revision]
        between = events[revision : event["revision"] - 1]
        if covered.intersection(e["revision"] for e in between) or any(
            e["payload"]["type"] == "model_preparation" for e in between
        ):
            raise ValueError("Overlapping or stale model handoff")
        covered.update(e["revision"] for e in between)
        queries = [e["payload"] for e in prior if e["payload"]["type"] == "query"]
        stops = [
            e["payload"]["decision"] for e in prior if e["payload"]["type"] == "stop"
        ]
        preps = [
            e["payload"] for e in prior if e["payload"]["type"] == "model_preparation"
        ]
        if len(preps) != turn or len(stops) > 1:
            raise ValueError("Model call order changed")

        class Replay:
            path = session.path

            @contextmanager
            def transaction(self):
                yield None, state

            def _events(self, _):
                return prior

        replay = Replay()
        memory = compact_memory(replay, assigned)
        ledger = usage_ledger(replay)
        phase = "prediction_required" if stops else "collecting"
        controller = dict(
            model_calls_remaining=limits.max_calls - turn,
            queries_remaining=limits.max_queries - len(queries),
            analyses_remaining=limits.max_analyses - len(memory["analysis_history"]),
        )
        controller["stop_required"] = (
            phase == "collecting" and controller["model_calls_remaining"] <= 2
        )
        public = {
            **state["public"],
            "price_limit": float(state["public"]["price_limit"]),
            "scoring": "Separate query-setting and transfer forecast errors, expected 90% sensor interval scores, feasible revenue/regret and claim-level explanation assessment. No composite scientific-success grade.",
        }
        context = dict(
            environment_version=public["environment_version"],
            information_mode="black_box_v2",
            phase=phase,
            revision=revision,
            contract=public,
            method_stack=state["method_stack"],
            policy_contract=policy_requirements(state["method_stack"]),
            remaining=public["budget"] - sum(q["cost"] for q in queries),
            spent=sum(q["cost"] for q in queries),
            history=[public_query(q) for q in queries],
            decision=stops[-1] if stops else None,
            failure=None,
            comparison=assigned,
            recent_preparations=[],
            **memory,
            resource_usage=ledger,
            validation_feedback=[
                p["error"] for p in preps if p.get("repairable") and p.get("error")
            ][-2:],
            available_libraries=available_libraries(),
            controller=controller,
        )
        if stops:
            context["evaluation_features"] = [
                {k: r[k] for k in EVAL_FEATURES}
                for r in evaluation_rows(state["world"])
            ]
            context["validation_suite"] = read_yaml(
                Path(state["quest"]) / "validation.yaml"
            )
            context["prediction_files"] = dict(
                training="data/0/train.csv" if queries else None,
                evaluation_features=f"data/{int(bool(queries))}/features.csv",
                output="predictions.csv",
                training_columns=list(observation_fields(public)) if queries else [],
                evaluation_columns=list(EVAL_FEATURES),
                public_validation="validation.yaml",
                registered_plan="experiment_plan.yaml",
            )
        expected_schema = schema
        expected_prefix = prefix
        if delivery is not None:
            context = delivery_v3.transform_context(context, replay, delivery)
            controller = context["controller"]
            expected_prefix = delivery_v3.instructions(assigned)
            expected_schema = delivery_v3.schema_for_context(schema, context)
        directory = session.path / prep["call"]
        known_dirs.add(directory.resolve())
        prompt = (directory / "prompt.txt").read_text()
        actual_prefix, actual_context = prompt.split("\ncontext.json:\n", 1)
        if (
            actual_prefix != expected_prefix
            or json.loads(actual_context) != context
            or resource_stop(ledger, limits, prompt) is not None
        ):
            raise ValueError(
                "Model handoff changed, leaked information or exceeded resources"
            )
        receipt = verify_call(
            directory, expected_prompt=prompt, expected_schema=expected_schema
        )
        if delivery is not None:
            from . import preflight_v3

            preflights = [
                e["payload"]
                for e in between
                if e["payload"]["type"] == "predictor_preflight"
            ]
            response = receipt.get("response", {})
            if len(preflights) > 1 or preflights and response.get("kind") != "predict":
                raise ValueError("Unexpected predictor preflight allocation")
            for p in preflights:
                import yaml
                import hashlib

                native_plan = yaml.safe_load(response["plan_yaml"])
                native_plan.update(
                    demand_scientific_decision=context["decision"],
                    scientist_information_mode="black_box_v2",
                    comparison_assignment=assigned,
                )
                record = preflight_v3.verify_record(
                    session.path / p["path"],
                    hashlib.sha256(response["notebook_json"].encode()).hexdigest(),
                    content_hash(native_plan),
                    revision,
                    context=context,
                    expected_archive=delivery_v3.native_archive_rows(
                        [e["payload"] for e in prior], session.path
                    ),
                )
                if (
                    p["record"] != record
                    or p["record_hash"] != content_hash(record)
                    or record["timeout"]
                    != delivery["policy"]["preflight_timeout_seconds"]
                ):
                    raise ValueError("Predictor preflight event/protocol changed")
                if record["status"] != "succeeded" and (
                    not prep.get("error") or not prep.get("repairable")
                ):
                    raise ValueError(
                        "Failed preflight was not retained as preparation feedback"
                    )
            if (
                response.get("kind") == "predict"
                and not prep.get("error")
                and (
                    len(preflights) != 1
                    or preflights[0]["record"]["status"] != "succeeded"
                )
            ):
                raise ValueError(
                    "Registered predictor lacks successful public-input preflight"
                )
        if delivery is not None and (
            receipt.get("timeout_seconds", delivery["timeout"]) != delivery["timeout"]
            or receipt.get("backend_spec_hash")
            != content_hash(delivery["backend_spec"])
            or prep.get("error")
            and not prep.get("repairable")
            and event != calls[-1]
        ):
            raise ValueError("Delivery timeout changed or terminal failure was retried")
        if (
            file_hash(directory / "receipt.json") != prep["receipt_sha256"]
            or prep["prompt_hash"] != content_hash(prompt)
            or prep["schema_hash"] != content_hash(expected_schema)
            or prep["response_hash"] != content_hash(receipt.get("response"))
        ):
            raise ValueError("Model call provenance changed")
        if receipt["status"] != "succeeded" or prep.get("error"):
            native_events = [
                e
                for e in between
                if e["payload"]["type"] in ("evaluation", "final_failure")
            ]
            if delivery is not None and native_events:
                import yaml
                from sciquest.state import get_record

                response = receipt.get("response", {})
                if response.get("kind") != "predict" or prep.get("repairable"):
                    raise ValueError(
                        "A registered execution failure cannot be treated as repairable preparation"
                    )
                registered = get_record(
                    Path(state["quest"]), "registration", state["experiment"]
                )
                expected = yaml.safe_load(response["plan_yaml"])
                expected.update(
                    demand_scientific_decision=context["decision"],
                    scientist_information_mode="black_box_v2",
                    comparison_assignment=assigned,
                )
                notebook = (
                    Path(state["quest"])
                    / "experiments"
                    / state["experiment"]
                    / "notebook.ipynb"
                )
                if (
                    registered is None
                    or registered["plan"] != expected
                    or json.loads(notebook.read_text())
                    != json.loads(response["notebook_json"])
                ):
                    raise ValueError(
                        "Failed registered predictor was not authored by the scientist"
                    )
            if any(
                e["payload"]["type"]
                in ("query", "stop", "scientist_analysis", "campaign_notes")
                for e in between
            ):
                raise ValueError("Research side effect on failed preparation")
        if receipt["status"] == "succeeded" and not prep.get("error"):
            r = receipt["response"]
            if delivery is not None and r["kind"] not in controller["allowed_kinds"]:
                raise ValueError("Model action bypassed delivery reserve")
            if delivery is not None:
                delivery_v3.validate_payload(r, delivery)
            action_events = [
                e["payload"]
                for e in between
                if e["payload"]["type"] != "campaign_notes"
            ]
            notes = [
                e["payload"]
                for e in between
                if e["payload"]["type"] == "campaign_notes"
            ]
            if notes != (
                [
                    dict(
                        type="campaign_notes",
                        text=r["campaign_notes"],
                        source_call=prep["call"],
                    )
                ]
                if r["campaign_notes"]
                else []
            ):
                raise ValueError("Campaign notes were not authored by this scientist")
            if len(r["campaign_notes"]) > limits.max_notes_chars:
                raise ValueError("Notes limit exceeded")
            if r["kind"] == "query":
                proposal = json.loads(r["proposal_json"])["query"]
                if (
                    len(action_events) != 1
                    or action_events[0]["type"] != "query"
                    or action_events[0]["action"] != proposal["action"]
                    or action_events[0]["plan"] != proposal["plan"]
                    or proposal["revision"] != revision
                    or len(queries) >= limits.max_queries
                    or controller["stop_required"]
                ):
                    raise ValueError("Query not chosen within limits by scientist")
            elif r["kind"] == "stop":
                proposal = json.loads(r["proposal_json"])
                if (
                    action_events != [dict(type="stop", decision=proposal["decision"])]
                    or proposal["revision"] != revision
                ):
                    raise ValueError("Decision not chosen by scientist")
            elif r["kind"] == "analysis":
                if (
                    len(action_events) != 1
                    or action_events[0]["type"] != "scientist_analysis"
                    or action_events[0]["analysis"]["code"] != r["analysis_code"]
                    or controller["analyses_remaining"] <= 0
                    or controller["stop_required"]
                ):
                    raise ValueError("Analysis not chosen within limits by scientist")
                analysis = action_events[0]["analysis"]
                log = session.path / analysis["full_output_file"]
                if (
                    file_hash(log) != analysis["full_output_hash"]
                    or log.read_text(errors="replace")[:65536] != analysis["output"]
                ):
                    raise ValueError("Analysis output changed")
            elif r["kind"] == "predict":
                import yaml
                from sciquest.state import get_record

                if not stops:
                    raise ValueError("Predictor before frozen decision")
                q = Path(state["quest"])
                registered = get_record(q, "registration", state["experiment"])
                expected = yaml.safe_load(r["plan_yaml"])
                expected.update(
                    demand_scientific_decision=stops[-1],
                    scientist_information_mode="black_box_v2",
                    comparison_assignment=assigned,
                )
                # nbformat.read[s] synthesizes random IDs for cells lacking an
                # ID. The native handoff copied the model file verbatim: compare
                # parsed JSON directly, preserving code, metadata and supplied
                # IDs. Session verification already validated execution/artifacts.
                if registered["plan"] != expected or json.loads(
                    (
                        q / "experiments" / state["experiment"] / "notebook.ipynb"
                    ).read_text()
                ) != json.loads(r["notebook_json"]):
                    raise ValueError("Frozen predictor was not chosen by scientist")
            else:
                raise ValueError("Unknown successful action")
    # Any attempted call must be accounted for; a crash is a visible interruption.
    actual_dirs = {
        p.parent.resolve() for p in (session.path / "model_calls").glob("*/prompt.txt")
    }
    if actual_dirs != known_dirs:
        raise ValueError("Unrecorded/interrupted model call; preserve and audit")
    if delivery is not None:
        handoffs = [e for e in events if e["payload"]["type"] == "prediction_handoff"]
        if handoffs:
            import csv
            import io
            import hashlib

            refreshes = [
                e["payload"]
                for e in events
                if e["payload"]["type"] == "native_archive_refresh"
            ]
            for refresh_event in [
                e for e in events if e["payload"]["type"] == "native_archive_refresh"
            ]:
                refresh = refresh_event["payload"]
                if refresh["source_revision"] != refresh_event["revision"] - 1:
                    raise ValueError("Native archive refresh has stale provenance")
                rows = delivery_v3.native_archive_rows(
                    [e["payload"] for e in events[: refresh["source_revision"]]],
                    session.path,
                )
                buffer = io.StringIO(newline="")
                writer = csv.DictWriter(
                    buffer, fieldnames=["id", "path", "part", "parts", "text", "sha256"]
                )
                writer.writeheader()
                writer.writerows(rows)
                if (
                    hashlib.sha256(buffer.getvalue().encode()).hexdigest()
                    != refresh["archive_sha256"]
                ):
                    raise ValueError("Native archive refresh content changed")
            archive_revision = (
                refreshes[-1]["source_revision"]
                if refreshes
                else handoffs[0]["revision"] - 1
            )
            before_handoff = [e["payload"] for e in events[:archive_revision]]
            expected_archive = delivery_v3.native_archive_rows(
                before_handoff, session.path
            )
            path = Path(state["quest"]) / "data/raw/campaign_archive.csv"
            if list(csv.DictReader(path.open())) != expected_archive:
                raise ValueError(
                    "Native own-campaign archive differs from preceding public evidence"
                )
        errors = 0
        for i, event in enumerate(calls):
            prep = event["payload"]
            if prep.get("error"):
                errors += 1
                receipt = json.loads(
                    (session.path / prep["call"] / "receipt.json").read_text()
                )
                terminal = (
                    not prep.get("repairable")
                    or receipt["status"] != "succeeded"
                    or errors > delivery["repair_limit"]
                )
                if terminal and i != len(calls) - 1:
                    raise ValueError("Terminal delivery failure was retried")
    if any(
        e["payload"]["type"]
        in (
            "query",
            "stop",
            "scientist_analysis",
            "campaign_notes",
            "predictor_preflight",
        )
        and e["revision"] not in covered
        for e in events
    ):
        raise ValueError("Research action outside model handoff")
    return dict(
        verified=True,
        calls=len(calls),
        phase=state["phase"],
        usage=usage_ledger(session),
        fidelity=fidelity_audit(session),
        scope="Exact allowlisted model handoffs, same-campaign notes/analysis provenance, scientific choices, budgets and independently recomputed scores. Not proof of philosophical sincerity or model-service isolation from pretrained knowledge.",
    )
