"""Isolated external-agent handoffs for interactive scientific discovery."""

from __future__ import annotations

import json
import importlib.metadata
import os
import tempfile
import uuid
from pathlib import Path

from sciquest.agent import parse_agent_command
from sciquest.execution import run_command
from sciquest.io import read_yaml
from sciquest.io import content_hash, file_hash
from .world import EVAL_FEATURES
from .session import observation_fields
from sciquest.methods.prompt_builder import build_method_prompt_addendum
from sciquest.methods.resolver import resolve_method_stack
from sciquest.workflow import stage_data


def run_external(session, command=None, max_turns=64, timeout=180):
    command = command or os.environ.get("SCIQUEST_AGENT_COMMAND")
    if not command:
        raise ValueError("Configure SCIQUEST_AGENT_COMMAND or supply --agent-command")
    if type(max_turns) is not int or not 1 <= max_turns <= 10000:
        raise ValueError("Invalid turn limit")
    if session.view(scientist=True).get("comparison"):
        raise ValueError(
            "Matched guidance requires the tool-free scientist runner and its common memory/resource rules"
        )
    session.verify()
    for _ in range(max_turns):
        view = session.view(scientist=True)
        if view["phase"] in ("completed", "failed"):
            return view
        if view["phase"] == "prediction_required":
            q, expid = session.prepare_quest()
            view = session.view(scientist=True)
        with tempfile.TemporaryDirectory(prefix="sciquest-demand-agent-") as temp:
            workspace = Path(temp)
            (workspace / "context.json").write_text(json.dumps(view, indent=2))
            instructions = (
                PROTOCOL
                + "\n"
                + build_method_prompt_addendum(
                    resolve_method_stack(view["method_stack"])
                )
            )
            (workspace / "AGENTS.md").write_text(instructions)
            if view["phase"] == "prediction_required":
                stage_data(
                    q,
                    workspace,
                    read_yaml(q / "validation.yaml"),
                    read_yaml(q / "data_manifest.yaml"),
                )
                (workspace / "validation.yaml").write_text(
                    (q / "validation.yaml").read_text()
                )
            shown = "/tmp/sciquest-workspace"
            prompt = (
                f"Prepare one SciQuest demand-laboratory action in {shown}. Read AGENTS.md and context.json. Current phase: {view['phase']}. Do not execute notebooks or claim completion.\n"
                + instructions
            )
            argv = [
                part.replace("{prompt}", prompt).replace("{quest_path}", shown)
                for part in parse_agent_command(command)
            ]
            if "{prompt}" not in command:
                argv.append(prompt)
            log = session.path / "logs" / f"agent-{uuid.uuid4().hex}.log"
            execution = run_command(
                argv,
                workspace,
                log,
                hidden_root=session.path,
                isolated=True,
                network=False,
                blind=True,
                timeout=timeout,
            )
            error = None
            try:
                if execution["status"] != "succeeded":
                    raise ValueError("Agent preparation " + execution["status"])
                if view["phase"] == "collecting":
                    source = workspace / "proposal.json"
                    if (
                        not source.is_file()
                        or source.is_symlink()
                        or source.stat().st_size > 1024 * 1024
                    ):
                        raise ValueError("Agent must write a bounded proposal.json")
                    proposal = json.loads(source.read_text())
                    if proposal.get("kind") == "query":
                        session.query(proposal["query"])
                    elif proposal.get("kind") == "stop":
                        session.stop(proposal["decision"], proposal["revision"])
                    else:
                        raise ValueError("Proposal kind must be query or stop")
                else:
                    notebook = workspace / "notebook.ipynb"
                    plan = workspace / "experiment_plan.yaml"
                    if (
                        not notebook.is_file()
                        or notebook.is_symlink()
                        or not plan.is_file()
                        or plan.is_symlink()
                        or notebook.stat().st_size > 8 * 1024 * 1024
                        or plan.stat().st_size > 1024 * 1024
                    ):
                        raise ValueError(
                            "Final handoff needs notebook.ipynb and experiment_plan.yaml"
                        )
                    session.finish(notebook, read_yaml(plan))
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                error = str(exc)
            with session.transaction() as (c, s):
                session._append(
                    c,
                    dict(
                        type="agent_preparation",
                        execution=execution,
                        error=error,
                        log=str(log.relative_to(session.path)),
                    ),
                )
            if error:
                # A caller can inspect the bounded receipt and resume. Noop or
                # malformed preparation never counts as a completed experiment.
                return {**session.view(), "agent_error": error}
    final = session.view()
    return (
        final
        if final["phase"] in ("completed", "failed")
        else {**final, "runner_status": "turn_limit_reached"}
    )


PROTOCOL = """# Demand-laboratory scientist protocol
Read context.json. It contains the public observation contract, remaining cost
budget, fixed philosophy/phase assignment, and previous registered measurements.
You are an experimental scientist. The supervisor owns the hidden mechanism,
simulation, budgets, registration, final scoring and authoritative history.

In collecting phase, write proposal.json with either:
{"kind":"query","query":{"revision":CURRENT_REVISION,"action":ACTION,"plan":PLAN}}
or {"kind":"stop","revision":CURRENT_REVISION,"decision":DECISION}.
ACTION kind may be observe, intervene, replicate, or measure. An observe request
may set context and precision, but not historical price/capacity. An intervention
sets price, context, capacity and precision. Replicate and measure identify an
earlier observation_id. Replicate draws a new unit; measure audits the SAME unit's
latent demand, with correlated evidence. Respect capabilities and cost limits.
PLAN has nonempty text fields question, hypothesis, scope, assumptions,
data_access, stopping_rule,
action (hypothesis_test/rival_test/replication/measurement/data_acquisition),
confirmatory_status: exploratory, and predictions of the scalar observation
metric (id, metric: observation, operator, threshold). Apply the required plan
extensions in context.policy_contract; do not copy placeholder likelihoods.
Numerical Bayesian likelihoods must condition on the proposed measurement and
previous observations, including dependence of audits. Optional prior_source
references an earlier query ID and must match its recorded posterior.
DECISION has status selected/inconclusive/not_identifiable/model_inadequate,
reason, explanation and limitations (all nonempty text). Develop your own models,
parameters, rivals and uncertainty assessments from the evidence. Stopping freezes this
decision and permanently closes discovery. You may stop before spending budget.

In prediction_required phase, write notebook.ipynb and experiment_plan.yaml.
Read validation.yaml and the staged training/evaluation feature CSVs. Do not run
the notebook. Emit predictions.csv with every evaluation ID and mean_demand,
mean_sales, lower_sales, upper_sales. Means refer to expected latent demand and
physical capped sales under price interventions. Intervals are central 90%
forecasts for fresh NOISY sales sensor measurements, not intervals for the mean.
Use a concrete native SciQuest plan with registered predictions referring to
demand_rmse or sales_rmse, the required philosophy contracts, and assumptions.
The supervisor freezes the submitted notebook, executes it without network or
hidden-world access, and scores independent truth. Export a reloadable model when
possible. A new hypothesis is optional; honest abstention and stopping are valid.
Do not inspect private files, change history, manufacture scores or declare
scientific confirmation. No final-evaluation feedback is available until the
frozen predictor and scientific decision have both been submitted.
"""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["query", "stop", "analysis", "predict"]},
        "proposal_json": {"type": "string"},
        "analysis_code": {"type": "string"},
        "notebook_json": {"type": "string"},
        "plan_yaml": {"type": "string"},
    },
    "required": [
        "kind",
        "proposal_json",
        "analysis_code",
        "notebook_json",
        "plan_yaml",
    ],
    "additionalProperties": False,
}


def _analysis_history(session):
    with session.transaction() as (c, _):
        return [
            e["payload"]["analysis"]
            for e in session._events(c)
            if e["payload"]["type"] == "scientist_analysis"
        ]


MODEL_INSTRUCTIONS = """
Return the specified response object. For query/stop put the complete proposal
in proposal_json. For analysis put a Python program in analysis_code; the
supervisor runs it offline and returns stdout. It can read context.json and use
installed numerical libraries. This is exploratory analysis, never final scoring.
Each analysis has a fresh filesystem; retain useful code, fits and notes in its
printed output. For predict put the complete notebook JSON in notebook_json and
its native plan YAML in plan_yaml. All unused response strings must be empty.
No model catalog, likelihoods or fitted models are provided. Construct your own.
You have no direct tools. The complete context.json is embedded below; read it
as supplied text. Analysis code receives that JSON as an actual local file.
If controller.stop_required is true, submit your scientific stop decision now;
the next call is reserved for your final predictor. Respect remaining query and
analysis counts as well as observation cost. Exhausting observation cost or query
count closes further measurements, not numerical analysis. You may analyze your
existing data before stopping. After stopping, analysis may prepare the frozen
predictor using existing data/features, but cannot change the scientific decision
or acquire observations. With only one model call left, submit the predictor.
Validation feedback concerns only
your own invalid submissions, never hidden test scores. Correct your own proposal.
At final prediction, validation_suite and prediction_files below describe the
staged files your notebook will receive. You do not need tools to inspect them:
all training observations and evaluation feature rows are already in context.
The six basic plan text fields must be strings, not arrays. Native final plan
predictions use metric demand_rmse or sales_rmse; collecting predictions use
metric observation. Use the assigned philosophy's required extensions with your
own hypotheses and declared assumptions. A Bayesian prior_source linking a query
is optional during collection; omit it in the native final plan because query
IDs are not completed native experiments. An honest provisional model and
qualified explanation are valid; no mechanism identification is assumed.
"""


def scientist_instructions(method_stack):
    return (
        PROTOCOL
        + "\n"
        + build_method_prompt_addendum(resolve_method_stack(method_stack))
        + MODEL_INSTRUCTIONS
    )


def available_libraries():
    result = []
    for library in ("numpy", "scipy", "pandas", "scikit-learn", "nbformat", "PyYAML"):
        try:
            result.append(library + "==" + importlib.metadata.version(library))
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def run_scientist(
    session,
    *,
    model="gpt-6.1-sol",
    reasoning_effort="high",
    binary="codex",
    max_turns=64,
    timeout=360,
    backend_spec=None,
    campaign_limits=None,
    repair_limit=0,
    on_progress=None,
    delivery_policy=None,
):
    """Fresh tool-free model calls; only explicit same-campaign work is replayed.

    The model can design experiments, propose arbitrary explanations and request
    numerical analysis. Its generated Python runs offline with no supervisor
    imports, auth or host files. The client never executes researcher code.
    """
    from .backend import describe_backend, run_call
    from sciquest.io import utc_now

    if type(max_turns) is not int or not 1 <= max_turns <= 10000:
        raise ValueError("Invalid turn limit")
    session.view(scientist=True)  # fail before backend access for legacy sessions
    session.verify()
    with session.transaction() as (c, _):
        if any(e["payload"]["type"] == "instrument_contract" for e in session._events(c)):
            raise ValueError("Use the frozen v4 instrument; legacy runner cannot continue this campaign")
    spec = backend_spec or dict(
        backend=describe_backend(binary), model=model, reasoning_effort=reasoning_effort
    )
    comparison = session.view(scientist=True).get("comparison")
    frozen_delivery = None
    schema = RESPONSE_SCHEMA
    if comparison:
        from . import campaign_v2
        from .policies_v2 import ResourceLimits

        limits = ResourceLimits(**comparison["limits"])
        max_turns = limits.max_calls
        campaign_limits = dict(
            max_queries=limits.max_queries, max_analyses=limits.max_analyses
        )
        schema = campaign_v2.response_schema(RESPONSE_SCHEMA)
        if any(e["type"] == "resource_stop" for e in campaign_v2.events(session)):
            return {**session.view(), "runner_status": "resource_stopped"}
    if delivery_policy is not None:
        from . import delivery_v3

        frozen_delivery = delivery_v3.initialize(
            session, delivery_policy, spec, timeout, repair_limit
        )
    elif comparison:
        from .delivery_v3 import contract

        if contract(session) is not None:
            raise ValueError(
                "A v3 delivery attempt cannot continue under the legacy runner"
            )
    base_schema = schema
    initial_calls = 0
    if campaign_limits:
        with session.transaction() as (c, _):
            initial_calls = sum(
                e["payload"]["type"] == "model_preparation" for e in session._events(c)
            )
    for turn in range(initial_calls, max_turns):
        view = session.view(scientist=True)
        if view["phase"] in ("completed", "failed"):
            return session.view()
        if view["phase"] == "prediction_required":
            q, _ = session.prepare_quest()
            if frozen_delivery:
                delivery_v3.refresh_native_archive(session, q)
            view = session.view(scientist=True)
        context = {**view, "analysis_history": _analysis_history(session)}
        if comparison:
            context.update(campaign_v2.compact_memory(session, comparison))
            context["resource_usage"] = campaign_v2.usage_ledger(session)
        with session.transaction() as (c, _):
            preparations = [
                e["payload"]
                for e in session._events(c)
                if e["payload"]["type"] == "model_preparation"
            ]
        errors_so_far = sum(bool(p.get("error")) for p in preparations)
        context["validation_feedback"] = [
            p["error"] for p in preparations if p.get("repairable") and p.get("error")
        ][-2:]
        context["available_libraries"] = available_libraries()
        controller = None
        if campaign_limits:
            controller = dict(
                model_calls_remaining=max_turns - turn,
                queries_remaining=campaign_limits["max_queries"] - len(view["history"]),
                analyses_remaining=campaign_limits["max_analyses"]
                - len(context["analysis_history"]),
            )
            controller["stop_required"] = view["phase"] == "collecting" and (
                controller["model_calls_remaining"] <= 2
            )
            context["controller"] = controller
        instructions = scientist_instructions(view["method_stack"])
        if comparison:
            instructions = campaign_v2.instructions(
                PROTOCOL, MODEL_INSTRUCTIONS, comparison
            )
        if view["phase"] == "prediction_required":
            context["validation_suite"] = read_yaml(q / "validation.yaml")
            context["prediction_files"] = dict(
                training="data/0/train.csv" if view["history"] else None,
                evaluation_features=f"data/{int(bool(view['history']))}/features.csv",
                output="predictions.csv",
                training_columns=(
                    list(observation_fields(view["contract"]))
                    if view["history"]
                    else []
                ),
                evaluation_columns=list(EVAL_FEATURES),
                public_validation="validation.yaml",
                registered_plan="experiment_plan.yaml",
            )
        if frozen_delivery:
            context = delivery_v3.transform_context(context, session, frozen_delivery)
            controller = context["controller"]
            instructions = delivery_v3.instructions(comparison)
            schema = delivery_v3.schema_for_context(base_schema, context)
        prompt = (
            instructions
            + "\ncontext.json:\n"
            + json.dumps(context, ensure_ascii=not bool(frozen_delivery))
        )
        if comparison:
            reason = campaign_v2.resource_stop(
                context["resource_usage"], limits, prompt
            )
            if reason:
                with session.transaction() as (c, _):
                    session._append(
                        c,
                        dict(
                            type="resource_stop",
                            reason=reason,
                            ledger=context["resource_usage"],
                            prompt_bytes=len(prompt.encode()),
                        ),
                    )
                return {**session.view(), "runner_status": reason}
        call_dir = session.path / "model_calls" / uuid.uuid4().hex
        receipt = run_call(call_dir, prompt, schema, spec, session.path, timeout)
        error = None
        repairable = False
        try:
            repairable = True
            if receipt["status"] != "succeeded":
                raise ValueError("Model preparation failed; inspect the owner receipt")
            response = receipt["response"]
            if set(response) != set(schema["properties"]):
                raise ValueError("Unexpected model response fields")
            if comparison and (
                not isinstance(response["campaign_notes"], str)
                or len(response["campaign_notes"]) > limits.max_notes_chars
            ):
                raise ValueError("Campaign notes exceed the common bound")
            kind = response["kind"]
            if frozen_delivery and kind not in controller["allowed_kinds"]:
                raise ValueError(
                    "controller.allowed_kinds requires "
                    + ", ".join(controller["allowed_kinds"])
                )
            if frozen_delivery:
                delivery_v3.validate_payload(response, frozen_delivery)
            used = {
                "query": {"proposal_json"},
                "stop": {"proposal_json"},
                "analysis": {"analysis_code"},
                "predict": {"notebook_json", "plan_yaml"},
            }.get(kind)
            if used is None or any(
                not isinstance(response[k], str)
                or (bool(response[k].strip()) != (k in used))
                for k in RESPONSE_SCHEMA["required"]
                if k != "kind"
            ):
                raise ValueError("Invalid or ambiguous response payload")
            if kind in ("query", "stop"):
                if controller and controller["stop_required"] and kind != "stop":
                    raise ValueError("Controller requires your stop decision now")
                if view["phase"] != "collecting":
                    raise ValueError("Discovery is closed")
                proposal = json.loads(response["proposal_json"])
                if proposal["kind"] != kind:
                    raise ValueError("Proposal kind mismatch")
                if kind == "query":
                    if controller and (
                        controller["queries_remaining"] <= 0 or view["remaining"] == 0
                    ):
                        raise ValueError(
                            "No measurements remain; analyze existing data or stop"
                        )
                    session.query(proposal["query"])
                else:
                    session.stop(proposal["decision"], proposal["revision"])
            elif kind == "analysis":
                if controller and (
                    controller["stop_required"] or controller["analyses_remaining"] <= 0
                ):
                    raise ValueError(
                        "No further analysis calls are available; submit your decision"
                    )
                if view["phase"] not in ("collecting", "prediction_required"):
                    raise ValueError("Analysis is closed for this campaign")
                if (
                    controller
                    and view["phase"] == "prediction_required"
                    and controller["model_calls_remaining"] <= 1
                ):
                    raise ValueError("Submit the final predictor on the last call")
                with tempfile.TemporaryDirectory(prefix="sciquest-analysis-") as temp:
                    workspace = Path(temp)
                    (workspace / "context.json").write_text(json.dumps(context))
                    (workspace / "analysis.py").write_text(response["analysis_code"])
                    if comparison:
                        campaign_v2.stage_analyses(session, workspace)
                    if frozen_delivery:
                        delivery_v3.stage_archive(
                            session, workspace, context["archive_index"]
                        )
                    import sys

                    log = call_dir / "analysis.log"
                    execution = run_command(
                        [sys.executable, "analysis.py"],
                        workspace,
                        log,
                        hidden_root=session.path,
                        blind=True,
                        timeout=timeout,
                        max_file_bytes=1024 * 1024,
                    )
                    analysis = dict(
                        code=response["analysis_code"],
                        output=log.read_text(errors="replace")[:65536],
                        status=execution["status"],
                    )
                if comparison:
                    analysis.update(
                        full_output_file=str(log.relative_to(session.path)),
                        full_output_hash=file_hash(log),
                        full_output_excerpted=len(log.read_text(errors="replace"))
                        > 65536,
                    )
                with session.transaction() as (c, _):
                    session._append(
                        c, dict(type="scientist_analysis", analysis=analysis)
                    )
            elif kind == "predict":
                if view["phase"] != "prediction_required":
                    raise ValueError(
                        "Freeze the scientific decision before final prediction"
                    )
                with tempfile.TemporaryDirectory(prefix="sciquest-submission-") as temp:
                    workspace = Path(temp)
                    notebook = workspace / "notebook.ipynb"
                    notebook.write_text(response["notebook_json"])
                    plan = workspace / "plan.yaml"
                    plan.write_text(response["plan_yaml"])
                    from sciquest.methods.ledger import validate_plan
                    from sciquest.methods.policies import validate_policy_plan
                    import nbformat

                    try:
                        if frozen_delivery:
                            from .preflight_v3 import validate_notebook_json

                            validate_notebook_json(response["notebook_json"])
                        nbformat.validate(nbformat.read(notebook, as_version=4))
                    except nbformat.ValidationError as exc:
                        raise ValueError(
                            "Invalid notebook structure: " + str(exc)[:512]
                        ) from exc
                    submitted = validate_plan(
                        read_yaml(plan), context["validation_suite"]
                    )
                    validate_policy_plan(view["method_stack"], submitted)
                    if frozen_delivery:
                        from . import preflight_v3

                        preflight = preflight_v3.run(
                            session,
                            notebook,
                            submitted,
                            context,
                            call_dir / "preflight",
                            delivery_policy.preflight_timeout_seconds,
                        )
                        with session.transaction() as (c, _):
                            session._append(
                                c,
                                dict(
                                    type="predictor_preflight",
                                    path=str(
                                        (call_dir / "preflight").relative_to(
                                            session.path
                                        )
                                    ),
                                    record=preflight,
                                    record_hash=content_hash(preflight),
                                ),
                            )
                        if preflight["status"] != "succeeded":
                            raise ValueError(
                                "Predictor preflight failed before registration: "
                                + (
                                    preflight["error"]
                                    or "unknown public-input preparation error"
                                )
                            )
                    # Once registration/execution starts, no feedback-based
                    # replacement predictor is allowed under this campaign.
                    repairable = False
                    session.finish(notebook, submitted)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            error = str(exc)
        except Exception as exc:
            # Preserve unexpected ordinary failures as terminal attempts. Never
            # retry native execution or lose the model receipt on a CSV/library
            # error. Interruptions (BaseException) still require explicit audit.
            repairable = False
            error = type(exc).__name__ + ": " + str(exc)
        with session.transaction() as (c, _):
            if comparison and not error and response.get("campaign_notes"):
                session._append(
                    c,
                    dict(
                        type="campaign_notes",
                        text=response["campaign_notes"],
                        source_call=str(call_dir.relative_to(session.path)),
                    ),
                )
            session._append(
                c,
                dict(
                    type="model_preparation",
                    call=str(call_dir.relative_to(session.path)),
                    error=error,
                    repairable=bool(error and repairable),
                    context_revision=view["revision"],
                    prompt_hash=content_hash(prompt),
                    schema_hash=content_hash(schema),
                    response_hash=content_hash(receipt.get("response")),
                    receipt_sha256=(
                        file_hash(call_dir / "receipt.json")
                        if (call_dir / "receipt.json").is_file()
                        else None
                    ),
                    time=utc_now(),
                ),
            )
        if on_progress:
            on_progress(
                dict(
                    turn=turn + 1,
                    phase=session.view()["phase"],
                    error=error,
                    kind=receipt.get("response", {}).get("kind"),
                    usage=receipt.get("usage"),
                )
            )
        if error and (
            not repairable
            or errors_so_far >= repair_limit
            or receipt["status"] != "succeeded"
        ):
            return {**session.view(), "agent_error": error}
    final = session.view()
    return (
        final
        if final["phase"] in ("completed", "failed")
        else {**final, "runner_status": "turn_limit_reached"}
    )
