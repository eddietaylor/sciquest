"""Prospective blind discovery pilots; world is the matched analysis unit."""

from __future__ import annotations

import fcntl
import json
import math
import random
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from sciquest.io import content_hash, file_hash, read_yaml, utc_now
from sciquest.methods.policies import policy_requirements
from sciquest.methods.resolver import resolve_method_stack
from sciquest.runtime import runtime_manifest

from .backend import describe_backend, parse_events
from .runner import (
    RESPONSE_SCHEMA,
    available_libraries,
    run_scientist,
    scientist_instructions,
)
from .session import DemandSession, observation_fields, public_query
from .world import DemandConfig, EVAL_FEATURES, evaluation_rows
from .worldbank import dump, load_bank, verify_bank, verify_call

VERSION = "blind-demand-pilot-v1"
POLICIES = (
    "standard_empirical",
    "strong_inference",
    "bayesian",
    "popperian_falsificationist",
)


@dataclass(frozen=True)
class PilotConfig:
    model: str = "gpt-6.1-sol"
    reasoning_effort: str = "high"
    budget: int = 24
    max_queries: int = 12
    max_analyses: int = 8
    max_calls: int = 24
    repair_limit: int = 2
    timeout: int = 360
    concurrency: int = 4

    def __post_init__(self):
        for name in (
            "budget",
            "max_queries",
            "max_analyses",
            "max_calls",
            "timeout",
            "concurrency",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Pilot budgets must be positive integers")
        if (
            self.max_calls < 3
            or self.concurrency > 4
            or not 0 <= self.repair_limit <= 3
        ):
            raise ValueError("Invalid call, concurrency or format-repair allowance")


def freeze_pilot(
    path: Path,
    bank: Path,
    config=PilotConfig(),
    *,
    split="pilot",
    engineering_stratum=None,
    binary="codex",
):
    """Freeze all measurement/scoring choices before any pilot model invocation."""
    if split not in ("engineering", "pilot"):
        raise ValueError("Pilot execution cannot consume sealed evaluation worlds")
    if (
        json.loads((Path(bank) / "manifest.json").read_text()).get("version")
        == "private-demand-development-v2"
    ):
        raise ValueError(
            "V2 development worlds require a new scoring protocol; the frozen v1 comparison must not be reused"
        )
    verify_bank(bank)
    _, manifest, private = load_bank(bank)
    worlds = [
        w
        for w in private["worlds"]
        if w["split"] == split
        and (engineering_stratum is None or w["spec"]["stratum"] == engineering_stratum)
    ]
    if not worlds or (split == "pilot" and len(worlds) != 6):
        raise ValueError("The pilot needs all six reserved worlds")
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    (path / "private").mkdir(mode=0o700)
    schedule = [
        dict(block=f"b{i:03d}", policy=p) for i in range(len(worlds)) for p in POLICIES
    ]
    order_seed = secrets.randbits(128)
    random.Random(order_seed).shuffle(schedule)
    allocation = dict(
        blocks=[dict(block=f"b{i:03d}", world=w) for i, w in enumerate(worlds)],
        order_seed=order_seed,
    )
    from .grading import GRADING_PROMPT, GRADING_SCHEMA

    protocol = dict(
        version=VERSION,
        frozen_at=utc_now(),
        purpose="Feasibility/development only; no philosophy-superiority claim",
        split=split,
        **asdict(config),
        backend=describe_backend(binary),
        runtime=runtime_manifest(),
        bank_manifest_hash=content_hash(manifest),
        bank=str(Path(bank).resolve()),
        allocation_commitment=content_hash(allocation),
        worlds=len(worlds),
        campaigns=len(schedule),
        policies=list(POLICIES),
        order=schedule,
        prompt_prefixes={
            p: scientist_instructions(resolve_method_stack(p).to_dict())
            for p in POLICIES
        },
        response_schema=RESPONSE_SCHEMA,
        primary=dict(
            name="failure_inclusive_normalized_interventional_rmse",
            formula="min(1, RMSE of expected demand across all 60 sealed intervention feature rows / 100)",
            scale=100,
            failure_loss=1,
            aggregation="Equal weight to each independent matched world. Compare each instruction arm minus standard empirical within world. Failed/incomplete campaigns retain loss 1.",
        ),
        secondary=[
            "completion",
            "complete-case demand/sales RMSE",
            "interval coverage/score",
            "policy value/regret/constraints",
            "observation cost",
            "calls and known token usage",
            "blinded explanation rubric",
            "query/analysis/revision trace",
        ],
        explanations=dict(
            prompt=GRADING_PROMPT,
            schema=GRADING_SCHEMA,
            graders_per_available_decision=2,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            blinding="Mask policy labels, instructions and native policy extensions; judges see final explanation, observed evidence and private truth. Style may reveal approach. Two fresh model calls share a model and may share biases. These are diagnostic judgments, not certified scientific truth.",
        ),
        planning=dict(
            minimum_meaningful_loss_difference=0.10,
            familywise_alpha=0.05,
            comparisons=3,
            target_power=0.8,
            variance_upper_confidence=0.8,
            sd_floor=0.15,
            minimum_worlds=24,
            rule="Use paired WORLD variance, its one-sided 80% upper bound, and a preregistered SD floor; never the pilot's observed treatment effect. Round required N up to a multiple of six using noncentral-t planning. Include .05/.10/.15 effect sensitivity and assumptions.",
        ),
        feasibility=dict(
            minimum_completed_fraction=0.9,
            maximum_leakage_incidents=0,
            scientific_review="Inspect actual hypothesis development, experiment rationales, revision, calibrated stopping and predictors; do not infer discovery from forecast accuracy.",
        ),
        failure_policy="No replacement worlds or campaign retries. Up to two agent self-corrections of its own pre-execution format/plan errors count against the call budget. No final-score feedback or replacement predictor. All attempted calls and partial evidence remain in intent-to-run results.",
        future_evaluation="Engineering/pilot worlds are permanently development data after inspection. Untouched bank evaluation worlds are excluded from these runs; final study protocol/allocation must freeze before any evaluation-world scientist is launched.",
        randomness="Matched arms share deterministic query-index random streams (common random numbers), but use fresh model sessions. Independent worlds are the analysis units.",
    )
    dump(path / "protocol.json", protocol)
    (path / "protocol.sha256").write_text(content_hash(protocol) + "\n")
    dump(path / "private/allocation.json", allocation)
    (path / "private/allocation.json").chmod(0o600)
    return protocol


def load_pilot(path: Path):
    path = Path(path).resolve()
    spec = json.loads((path / "protocol.json").read_text())
    allocation = json.loads((path / "private/allocation.json").read_text())
    digest = (path / "protocol.sha256").read_text().strip()
    if (
        content_hash(spec) != digest
        or content_hash(allocation) != spec["allocation_commitment"]
    ):
        raise ValueError("Frozen pilot protocol/allocation changed")
    if spec["runtime"]["fingerprint"] != runtime_manifest()["fingerprint"]:
        raise ValueError(
            "Pilot source/dependencies changed; restore its original runtime"
        )
    _, manifest, private = load_bank(Path(spec["bank"]))
    if content_hash(manifest) != spec["bank_manifest_hash"]:
        raise ValueError("Pilot worldbank changed")
    for block in allocation["blocks"]:
        if (
            block["world"] not in private["worlds"]
            or block["world"]["split"] != spec["split"]
        ):
            raise ValueError(
                "Pilot references an unallocated or sealed evaluation world"
            )
    expected = {(b["block"], p) for b in allocation["blocks"] for p in POLICIES}
    if {(s["block"], s["policy"]) for s in spec["order"]} != expected or len(
        spec["order"]
    ) != len(expected):
        raise ValueError("Pilot matched schedule is incomplete or duplicated")
    return spec, allocation, digest


def _calls(session_path: Path):
    result = []
    for directory in (session_path / "model_calls").glob("*"):
        if not (directory / "prompt.txt").is_file():
            continue
        path = directory / "receipt.json"
        if not path.exists():
            usage = None
            trace = directory / "trace.jsonl"
            if trace.exists():
                try:
                    usage = parse_events(trace.read_text())["usage"]
                except ValueError:
                    pass
            result.append(
                dict(
                    path=str(directory.relative_to(session_path)),
                    receipt_sha256=None,
                    status="interrupted_no_receipt",
                    usage=usage,
                    started_at=None,
                    elapsed_seconds=None,
                    prompt_sha256=file_hash(directory / "prompt.txt"),
                    trace_sha256=file_hash(trace) if trace.exists() else None,
                )
            )
            continue
        receipt = json.loads(path.read_text())
        result.append(
            dict(
                path=str(path.parent.relative_to(session_path)),
                receipt_sha256=file_hash(path),
                status=receipt["status"],
                usage=receipt.get("usage"),
                started_at=receipt["started_at"],
                elapsed_seconds=receipt["elapsed_seconds"],
            )
        )
    return sorted(result, key=lambda r: (r["started_at"] or "", r["path"]))


def _config(world, spec):
    return DemandConfig(
        scenario="observational_only"
        if world["spec"]["stratum"] == "observational_only"
        else "clean",
        seed=world["seed"],
        budget=spec["budget"],
        world_spec=world["spec"],
    )


def _campaign(directory: Path, spec, block, policy, digest, progress=None):
    directory.mkdir(parents=True, exist_ok=False)
    session = DemandSession.create(
        directory / "session", _config(block["world"], spec), policy
    )
    error = None
    started = utc_now()
    try:
        result = run_scientist(
            session,
            model=spec["model"],
            reasoning_effort=spec["reasoning_effort"],
            max_turns=spec["max_calls"],
            timeout=spec["timeout"],
            backend_spec=spec,
            campaign_limits=dict(
                max_queries=spec["max_queries"], max_analyses=spec["max_analyses"]
            ),
            repair_limit=spec["repair_limit"],
            on_progress=(
                lambda event: progress(
                    dict(block=block["block"], policy=policy, **event)
                )
            )
            if progress
            else None,
        )
        error = result.get("agent_error") or result.get("runner_status")
        session.verify()
    except Exception as exc:
        error = str(exc)
    view = session.view()
    completed = view["phase"] == "completed" and error is None
    science = view.get("evaluation", {}).get("science") if completed else None
    loss = (
        min(1.0, science["interventional_demand_rmse"] / spec["primary"]["scale"])
        if completed
        else spec["primary"]["failure_loss"]
    )
    outcome = dict(
        block=block["block"],
        policy=policy,
        protocol_hash=digest,
        status="completed" if completed else "failed",
        phase=view["phase"],
        started_at=started,
        finished_at=utc_now(),
        primary_loss=loss,
        error=error,
        science=science,
        decision=view["decision"],
        cost_spent=view["spent"],
        query_count=len(view["history"]),
        calls=_calls(session.path),
    )
    dump(directory / "outcome.json", outcome)
    return outcome


def run_pilot(path: Path, progress=None, *, allow_evaluation=False):
    """Run once. A lost campaign is preserved as failed, never silently restarted."""
    path = Path(path).resolve()
    spec, allocation, digest = load_pilot(path)
    if spec["split"] == "evaluation" and (
        not allow_evaluation or spec.get("execution_gate") != "ready"
    ):
        raise ValueError(
            "Sealed evaluation execution requires explicit confirmatory selection and a passing feasibility gate"
        )
    blocks = {b["block"]: b for b in allocation["blocks"]}
    outcomes = {}
    with (path / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This pilot has a live supervisor") from exc
        pending = []
        for item in spec["order"]:
            key = item["block"] + "-" + item["policy"]
            directory = path / "campaigns" / key
            if (directory / "outcome.json").exists():
                outcome = json.loads((directory / "outcome.json").read_text())
                verify_campaign(directory, outcome, spec, blocks[item["block"]], digest)
                outcomes[key] = outcome
            elif directory.exists():
                # Exclusive OS lock proves no study supervisor still owns this
                # allocation. Preserve receipts and any partial native state.
                session = DemandSession(directory / "session")
                view = session.view()
                outcome = dict(
                    block=item["block"],
                    policy=item["policy"],
                    protocol_hash=digest,
                    status="failed",
                    phase=view["phase"],
                    started_at=None,
                    finished_at=utc_now(),
                    primary_loss=1.0,
                    error="Interrupted campaign; no replacement",
                    science=None,
                    decision=view["decision"],
                    cost_spent=view["spent"],
                    query_count=len(view["history"]),
                    calls=_calls(session.path),
                )
                dump(directory / "outcome.json", outcome)
                outcomes[key] = outcome
            else:
                pending.append((key, directory, item))
        with ThreadPoolExecutor(max_workers=spec["concurrency"]) as pool:
            futures = {
                pool.submit(
                    _campaign,
                    directory,
                    spec,
                    blocks[item["block"]],
                    item["policy"],
                    digest,
                    progress,
                ): key
                for key, directory, item in pending
            }
            for future in as_completed(futures):
                key = futures[future]
                outcomes[key] = future.result()
                dump(
                    path / "progress.json",
                    dict(
                        finished=len(outcomes),
                        total=spec["campaigns"],
                        last=outcomes[key],
                    ),
                )
                if progress:
                    progress(
                        dict(
                            campaign_finished=key,
                            status=outcomes[key]["status"],
                            primary_loss=outcomes[key]["primary_loss"],
                            finished=len(outcomes),
                            total=spec["campaigns"],
                        )
                    )
        ordered = [
            outcomes[item["block"] + "-" + item["policy"]] for item in spec["order"]
        ]
        dump(path / "outcomes.json", ordered)
        dump(path / "summary.json", summarize(spec, ordered))
    return ordered


def verify_campaign(directory: Path, outcome, spec, block, digest):
    session = DemandSession(directory / "session")
    proof = session.verify()
    view = session.view()
    with session.transaction() as (c, state):
        events = session._events(c)
        actual_config = state["world"]["config"]
        world = state["world"]
    if (
        actual_config != asdict(_config(block["world"], spec))
        or view["method_stack"] != resolve_method_stack(outcome["policy"]).to_dict()
    ):
        raise ValueError("Campaign world or philosophy differs from frozen allocation")
    if (
        outcome["protocol_hash"] != digest
        or outcome["phase"] != view["phase"]
        or outcome["decision"] != view["decision"]
        or outcome["cost_spent"] != view["spent"]
        or outcome["query_count"] != len(view["history"])
        or outcome["calls"] != _calls(session.path)
    ):
        raise ValueError("Campaign outcome differs from authoritative evidence")
    preps = [e for e in events if e["payload"]["type"] == "model_preparation"]
    if (
        len(outcome["calls"]) > spec["max_calls"]
        or len(view["history"]) > spec["max_queries"]
    ):
        raise ValueError("Campaign exceeded registered resource budget")
    if outcome["status"] == "completed" and (
        view["phase"] != "completed" or len(preps) != len(outcome["calls"])
    ):
        raise ValueError("Completed campaign lacks complete model-call provenance")
    for call in outcome["calls"]:
        directory_call = session.path / call["path"]
        if call["receipt_sha256"] is not None:
            verify_call(directory_call)
        elif (
            outcome["status"] == "completed"
            or file_hash(directory_call / "prompt.txt") != call["prompt_sha256"]
        ):
            raise ValueError("Interrupted model attempt was hidden or changed")
    for i, event in enumerate(preps):
        prep = event["payload"]
        call = session.path / prep["call"]
        receipt = verify_call(call, expected_schema=spec["response_schema"])
        if (
            file_hash(call / "receipt.json") != prep["receipt_sha256"]
            or receipt["prompt_hash"] != prep["prompt_hash"]
            or content_hash(receipt.get("response")) != prep["response_hash"]
        ):
            raise ValueError("Model-call transcript commitment changed")
        prompt = (call / "prompt.txt").read_text()
        prefix, encoded = prompt.split("\ncontext.json:\n", 1)
        context = json.loads(encoded)
        if prefix != spec["prompt_prefixes"][outcome["policy"]]:
            raise ValueError("Scientist instruction arm changed")
        revision = prep["context_revision"]
        prior = [e["payload"] for e in events if e["revision"] <= revision]
        history = [e for e in prior if e["type"] == "query"]
        analyses = [e["analysis"] for e in prior if e["type"] == "scientist_analysis"]
        stops = [e["decision"] for e in prior if e["type"] == "stop"]
        previous_preps = [e for e in prior if e["type"] == "model_preparation"]
        allowed = {
            "environment_version",
            "information_mode",
            "phase",
            "revision",
            "contract",
            "method_stack",
            "policy_contract",
            "remaining",
            "spent",
            "history",
            "decision",
            "failure",
            "recent_preparations",
            "analysis_history",
            "validation_feedback",
            "available_libraries",
            "controller",
        }
        if stops:
            allowed |= {"evaluation_features", "validation_suite", "prediction_files"}
        expected_phase = "prediction_required" if stops else "collecting"
        controller = dict(
            model_calls_remaining=spec["max_calls"] - i,
            queries_remaining=spec["max_queries"] - len(history),
            analyses_remaining=spec["max_analyses"] - len(analyses),
        )
        spent = sum(r["cost"] for r in history)
        controller["stop_required"] = not stops and (
            controller["model_calls_remaining"] <= 2
        )
        checks = dict(
            phase=expected_phase,
            revision=revision,
            contract=state["public"],
            method_stack=state["method_stack"],
            policy_contract=policy_requirements(state["method_stack"]),
            remaining=spec["budget"] - spent,
            spent=spent,
            history=[public_query(r) for r in history],
            decision=stops[-1] if stops else None,
            failure=None,
            recent_preparations=[],
            analysis_history=analyses,
            validation_feedback=[
                p["error"]
                for p in previous_preps
                if p.get("repairable") and p.get("error")
            ][-2:],
            controller=controller,
            information_mode="black_box_v2",
            environment_version=state["public"]["environment_version"],
            available_libraries=available_libraries(),
        )
        if set(context) != allowed or any(
            context.get(k) != v for k, v in checks.items()
        ):
            raise ValueError(
                "Scientist context contains leakage or differs from its actual preceding campaign"
            )
        if stops:
            if context["evaluation_features"] != [
                {k: r[k] for k in EVAL_FEATURES} for r in evaluation_rows(world)
            ]:
                raise ValueError("Prediction features changed")
            quest = Path(state["quest"])
            if context["validation_suite"] != read_yaml(
                quest / "validation.yaml"
            ) or context["prediction_files"] != dict(
                training="data/0/train.csv" if history else None,
                evaluation_features=f"data/{int(bool(history))}/features.csv",
                output="predictions.csv",
                training_columns=list(observation_fields(context["contract"]))
                if history
                else [],
                evaluation_columns=list(EVAL_FEATURES),
                public_validation="validation.yaml",
                registered_plan="experiment_plan.yaml",
            ):
                raise ValueError("Prediction handoff changed")
        if receipt["status"] == "succeeded" and prep.get("error") is None:
            response = receipt["response"]
            between = [
                e["payload"]
                for e in events
                if revision < e["revision"] < event["revision"]
            ]
            if response["kind"] == "query":
                proposal = json.loads(response["proposal_json"])["query"]
                queries = [e for e in between if e["type"] == "query"]
                if (
                    len(queries) != 1
                    or queries[0]["action"] != proposal["action"]
                    or queries[0]["plan"] != proposal["plan"]
                ):
                    raise ValueError("Executed query was not chosen by the model")
            elif response["kind"] == "stop" and (
                not any(
                    e["type"] == "stop"
                    and e["decision"]
                    == json.loads(response["proposal_json"])["decision"]
                    for e in between
                )
            ):
                raise ValueError("Scientific conclusion was not chosen by the model")
            elif response["kind"] == "analysis" and not any(
                e["type"] == "scientist_analysis"
                and e["analysis"]["code"] == response["analysis_code"]
                for e in between
            ):
                raise ValueError("Analysis program was not written by the model")
            elif response["kind"] == "predict":
                import nbformat
                import yaml
                from sciquest.state import get_record

                registered = get_record(
                    Path(state["quest"]), "registration", state["experiment"]
                )
                notebook = (
                    Path(state["quest"])
                    / "experiments"
                    / state["experiment"]
                    / "notebook.ipynb"
                )
                returned = nbformat.reads(response["notebook_json"], as_version=4)
                expected_plan = yaml.safe_load(response["plan_yaml"])
                expected_plan.update(
                    demand_scientific_decision=view["decision"],
                    scientist_information_mode="black_box_v2",
                )
                if [(c.cell_type, c.source) for c in returned.cells] != [
                    (c.cell_type, c.source)
                    for c in nbformat.read(notebook, as_version=4).cells
                ] or registered["plan"] != expected_plan:
                    raise ValueError(
                        "Registered predictor was not written by the model"
                    )
    if outcome["status"] == "completed":
        science = view["evaluation"]["science"]
        if outcome["science"] != science or outcome["primary_loss"] != min(
            1, science["interventional_demand_rmse"] / spec["primary"]["scale"]
        ):
            raise ValueError(
                "Primary outcome differs from independently recomputed truth"
            )
    elif outcome["primary_loss"] != 1 or outcome["science"] is not None:
        raise ValueError(
            "Failed campaign did not retain its registered primary penalty"
        )
    return proof


def summarize(spec, outcomes):
    arms = {}
    for policy in POLICIES:
        rows = [r for r in outcomes if r["policy"] == policy]
        known = [
            c["usage"] for r in rows for c in r["calls"] if c.get("usage") is not None
        ]
        complete = [r for r in rows if r["status"] == "completed"]
        arms[policy] = dict(
            campaigns=len(rows),
            completed=len(complete),
            failed=len(rows) - len(complete),
            mean_primary_loss=math.fsum(r["primary_loss"] for r in rows) / len(rows)
            if rows
            else None,
            mean_complete_case_rmse=math.fsum(
                r["science"]["interventional_demand_rmse"] for r in complete
            )
            / len(complete)
            if complete
            else None,
            model_calls=sum(len(r["calls"]) for r in rows),
            missing_usage_calls=sum(
                c.get("usage") is None for r in rows for c in r["calls"]
            ),
            known_input_tokens=sum(u["input_tokens"] for u in known),
            known_output_tokens=sum(u["output_tokens"] for u in known),
            observation_cost=sum(r["cost_spent"] for r in rows),
        )
    paired = {}
    for policy in POLICIES[1:]:
        pairs = []
        for block in sorted({r["block"] for r in outcomes}):
            by_arm = {r["policy"]: r for r in outcomes if r["block"] == block}
            if policy in by_arm and POLICIES[0] in by_arm:
                pairs.append(
                    dict(
                        block=block,
                        difference=by_arm[policy]["primary_loss"]
                        - by_arm[POLICIES[0]]["primary_loss"],
                    )
                )
        mean = math.fsum(p["difference"] for p in pairs) / len(pairs) if pairs else None
        sd = (
            math.sqrt(
                math.fsum((p["difference"] - mean) ** 2 for p in pairs)
                / (len(pairs) - 1)
            )
            if len(pairs) > 1
            else None
        )
        paired[policy] = dict(
            worlds=len(pairs), mean_difference=mean, paired_sd=sd, pairs=pairs
        )
    return dict(
        complete=len(outcomes) == spec["campaigns"],
        campaigns=len(outcomes),
        independent_worlds=spec["worlds"],
        arms=arms,
        paired_vs_empirical=paired,
        interpretation="Descriptive feasibility pilot. Campaigns and evaluation rows do not multiply the number of independent world comparisons. No philosophy-superiority conclusion.",
    )


def verify_pilot(path: Path):
    spec, allocation, digest = load_pilot(path)
    outcomes = json.loads((Path(path) / "outcomes.json").read_text())
    blocks = {b["block"]: b for b in allocation["blocks"]}
    if [(r["block"], r["policy"]) for r in outcomes] != [
        (r["block"], r["policy"]) for r in spec["order"]
    ]:
        raise ValueError("Saved outcomes omit or reorder registered campaigns")
    for outcome in outcomes:
        directory = (
            Path(path) / "campaigns" / (outcome["block"] + "-" + outcome["policy"])
        )
        if outcome != json.loads((directory / "outcome.json").read_text()):
            raise ValueError("Summary and campaign evidence differ")
        verify_campaign(directory, outcome, spec, blocks[outcome["block"]], digest)
    if summarize(spec, outcomes) != json.loads(
        (Path(path) / "summary.json").read_text()
    ):
        raise ValueError("Aggregate results changed")
    return dict(
        verified=True,
        protocol_hash=digest,
        campaigns=len(outcomes),
        completed=sum(r["status"] == "completed" for r in outcomes),
        model_calls=sum(len(r["calls"]) for r in outcomes),
        independent_worlds=spec["worlds"],
        scope="World/method allocation, private transcript replay, exact model inputs and outputs, no cross-campaign context, registered code, independent metrics and failure inclusion.",
    )
