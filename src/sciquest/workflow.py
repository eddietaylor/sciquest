"""One supervisor lifecycle for registered research and external worker preparation."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import nbformat

from .core import create_scaffold, reserve_experiment
from .evaluation import evaluate_registered
from .execution import run_command
from .io import (
    append_text,
    atomic_text,
    contained_path,
    content_hash,
    experiment_dir,
    file_hash,
    read_yaml,
    utc_now,
    write_yaml,
)
from .methods.ledger import preregister_experiment
from .state import (
    acquire_lock,
    authority_dir,
    finalize_run,
    get_record,
    heartbeat,
    list_records,
    read_state,
    record,
    release_lock,
)


def stage_data(
    qpath: Path,
    workspace: Path,
    suite: dict[str, Any],
    manifest: dict[str, Any],
    *,
    include_training: bool = True,
) -> None:
    """Stage training data and explicitly whitelisted evaluation features only."""
    datasets = manifest.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("Datasets must be a list")
    training_hashes = {
        file_hash(contained_path(qpath, d["path"]))
        for d in datasets
        if d.get("role") == "training"
    }
    evaluation_hashes = {
        file_hash(contained_path(qpath, d["path"]))
        for d in datasets
        if d.get("role") == "evaluation"
    }
    if training_hashes.intersection(evaluation_hashes):
        raise ValueError("Training and evaluation data must not be identical copies")
    truth_targets = {}
    for metric in suite.get("metrics", []):
        cfg = metric.get("computation", {})
        truth_targets.setdefault(cfg.get("truth"), set()).add(
            cfg.get("target", "actual")
        )
    # Byte hashes miss subsets or reordered copies of protected observations.
    checked = set()
    for metric in suite.get("metrics", []):
        cfg = metric.get("computation", {})
        key, truth_name = cfg.get("key", "id"), cfg.get("truth")
        if (key, truth_name) in checked:
            continue
        checked.add((key, truth_name))
        matching = [
            d
            for index, d in enumerate(datasets)
            if f"data/{index}/{Path(d['path']).name}" == truth_name
            and d.get("role") == "evaluation"
        ]
        if len(matching) != 1:
            raise ValueError(
                "Every computed metric requires a protected evaluation dataset"
            )
        from .evaluation import _table

        evaluation_ids = _table(contained_path(qpath, matching[0]["path"]), key).keys()
        for dataset in datasets:
            if dataset.get("role") != "training":
                continue
            training = _table(contained_path(qpath, dataset["path"]), key)
            if evaluation_ids & training.keys():
                raise ValueError(
                    "Training and evaluation contain overlapping observation IDs"
                )
    for index, dataset in enumerate(datasets):
        source = contained_path(qpath, dataset.get("path"))
        name = f"data/{index}/{source.name}"
        target = contained_path(workspace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if dataset.get("role") == "training":
            if not include_training:
                continue
            if name in truth_targets:
                raise ValueError(
                    "Evaluation truth cannot also be exposed as training data"
                )
            shutil.copyfile(source, target)
        elif dataset.get("role") == "evaluation":
            columns = dataset.get("input_columns")
            if (
                not isinstance(columns, list)
                or not columns
                or len(set(columns)) != len(columns)
            ):
                raise ValueError(
                    "Evaluation data requires explicit unique input_columns"
                )
            if set(columns) & truth_targets.get(name, set()):
                raise ValueError("Evaluation targets must not be exposed as inputs")
            target = target.with_name("features.csv")
            with (
                source.open(newline="", encoding="utf-8") as stream,
                target.open("w", newline="", encoding="utf-8") as output,
            ):
                reader = csv.DictReader(stream)
                if not reader.fieldnames or any(
                    col not in reader.fieldnames for col in columns
                ):
                    raise ValueError("Evaluation feature columns are missing")
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                for row in reader:
                    writer.writerow({col: row[col] for col in columns})
        else:
            raise ValueError("Every dataset must declare role: training or evaluation")


def _collect(workspace: Path, exp: Path) -> None:
    count, size = 0, 0
    for source in workspace.rglob("*"):
        if source.is_symlink():
            raise ValueError("Worker outputs must not contain symlinks")
        if not source.is_file():
            continue
        relative = source.relative_to(workspace)
        if relative.parts[0] in {"data", ".ipython", ".jupyter"}:
            continue
        count += 1
        size += source.stat().st_size
        if count > 1000 or size > 256 * 1024 * 1024:
            raise ValueError("Worker output artifact budget exceeded")
        destination = contained_path(exp / "worker_artifacts", str(relative))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _notebook_sources(path: Path) -> list[tuple[str, str]]:
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    return [(c.cell_type, c.source) for c in nb.cells]


def _verify_model_export(
    qpath: Path,
    exp: Path,
    workspace: Path,
    snapshot: Path,
    registration: dict,
    manifest: dict,
    owner: str,
    isolated: bool,
    timeout: float,
    protected_root: Path | None = None,
) -> dict:
    from .evaluation import _number, _table

    contract = registration["plan"]["model_export"]
    from .demand.export_v1 import verify_export

    verify_export(workspace, contract)
    source = contained_path(workspace, contract["reload_notebook"])
    if not source.is_file() or source.is_symlink():
        raise ValueError("Standalone model reload notebook is missing")
    hashes = {
        name: file_hash(contained_path(workspace, name))
        for name in contract["artifacts"]
    }
    with tempfile.TemporaryDirectory(prefix="sciquest-model-reload-") as temp:
        fresh = Path(temp)
        stage_data(
            snapshot,
            fresh,
            registration["validation_suite"],
            manifest,
            include_training=False,
        )
        _stage_registered_context(fresh, registration)
        for name in contract["artifacts"]:
            target = contained_path(fresh, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(contained_path(workspace, name), target)
        shutil.copyfile(source, fresh / "reload.ipynb")
        result = run_command(
            [
                sys.executable,
                "-m",
                "sciquest.worker",
                "notebook",
                "reload.ipynb",
                "reload_executed.ipynb",
                str(max(1, int(timeout))),
            ],
            fresh,
            exp / "logs/model_reload.log",
            hidden_root=protected_root or qpath,
            isolated=isolated,
            timeout=timeout,
            on_tick=lambda: heartbeat(qpath, owner),
            blind=registration["plan"].get("scientist_information_mode")
            == "black_box_v2",
        )
        _collect(fresh, exp / "reload")
        if result["status"] != "succeeded":
            raise ValueError(
                "Standalone model reload failed; partial evidence preserved"
            )
        verify_export(fresh, contract)
        maximum_difference = 0.0
        prediction_hashes = {}
        for metric in registration["validation_suite"]["metrics"]:
            cfg = metric["computation"]
            name = cfg.get("artifact", "predictions.csv")
            key = cfg.get("key", "id")
            column = cfg.get("prediction", "prediction")
            original = _table(contained_path(workspace, name), key)
            reloaded = _table(contained_path(fresh, name), key)
            if original.keys() != reloaded.keys():
                raise ValueError(
                    "Reloaded prediction IDs differ from the original model"
                )
            difference = max(
                abs(
                    _number(original[i][column], column)
                    - _number(reloaded[i][column], column)
                )
                for i in original
            )
            maximum_difference = max(maximum_difference, difference)
            prediction_hashes[name] = file_hash(contained_path(fresh, name))
        if maximum_difference > contract.get("absolute_tolerance", 0):
            raise ValueError(
                "Reloaded model predictions differ beyond the registered tolerance"
            )
        return {
            "execution": result,
            "model_artifact_hashes": hashes,
            "prediction_artifact_hashes": prediction_hashes,
            "maximum_absolute_prediction_difference": maximum_difference,
            "training_data_available": False,
        }


def stage_research_context(qpath: Path, workspace: Path) -> None:
    """Expose bounded verified history, rather than restarting from the initial idea."""
    context = []
    for exp_id, completion in list_records(qpath, "completion", limit=100):
        registration = get_record(qpath, "registration", exp_id)
        evaluation = get_record(qpath, "evaluation", exp_id)
        if (
            not registration
            or not evaluation
            or completion.get("evaluation_hash") != content_hash(evaluation)
        ):
            raise ValueError("Prior research context has inconsistent evidence records")
        entry = {
            "experiment_id": exp_id,
            "plan": registration["plan"],
            "metrics": evaluation["metrics"],
            "metric_roles": evaluation.get("metric_roles", {}),
            "constraint_outcomes": evaluation.get("constraint_outcomes", {}),
            "undefined_metrics": evaluation.get("undefined_metrics", {}),
            "decision_evidence": {
                name: {
                    **detail,
                    "trace_sample": detail["trace_sample"][:5],
                    "feedback_trace_limit": 5,
                    "trace_truncated": detail["group_count"] > 5,
                }
                for name, detail in evaluation.get("decision_evidence", {}).items()
            },
            "prediction_outcomes": evaluation["prediction_outcomes"],
            "policy_evidence": evaluation["policy_evidence"],
            "registration_hash": registration["registration_hash"],
            "evaluation_mode": evaluation["confirmatory_status"],
        }
        if len(context) < 3:
            target = workspace / "history" / exp_id / "notebook.ipynb"
            target.parent.mkdir(parents=True, exist_ok=True)
            source = authority_dir(qpath) / "snapshots" / exp_id / "notebook.ipynb"
            if file_hash(source) != registration["file_hashes"]["notebook.ipynb"]:
                raise ValueError("Prior registered source code changed")
            shutil.copyfile(source, target)
            entry["source_notebook"] = str(target.relative_to(workspace))
        context.append(entry)
    failures = (read_state(qpath).get("failures") or [])[-5:]
    for failure in failures[-3:]:
        exp_id = failure.get("experiment")
        if not exp_id:
            continue
        failed = experiment_dir(qpath, exp_id)
        for name in [
            "notebook.ipynb",
            "executed_notebook.ipynb",
            "logs/execution.log",
            "logs/agent.log",
        ]:
            source = contained_path(failed, name)
            if source.is_file() and source.stat().st_size <= 1024 * 1024:
                target = contained_path(workspace, f"failures/{exp_id}/{name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    write_yaml(
        workspace / "research_state.yaml",
        {
            "schema_version": 1,
            "completed_research": context,
            "recent_execution_failures": failures,
            "latest_decision": read_state(qpath).get("last_decision"),
            "limits": "Most recent 100 completed records and three source notebooks. Feedback makes reused evaluation evidence exploratory.",
        },
    )


def _stage_registered_context(workspace: Path, registration: dict) -> None:
    """Stage public contracts from authoritative values, without truth or seeds."""
    write_yaml(workspace / "validation.yaml", registration["validation_suite"])
    write_yaml(workspace / "experiment_plan.yaml", registration["plan"])


def _execute_and_evaluate(
    qpath: Path,
    exp_id: str,
    owner: str,
    *,
    isolated: bool,
    timeout: float,
    protected_root: Path | None = None,
) -> dict[str, Any]:
    exp = experiment_dir(qpath, exp_id)
    registration = get_record(qpath, "registration", exp_id)
    if not registration:
        raise ValueError("Experiment has not been registered")
    from .runtime import runtime_manifest

    runtime = runtime_manifest()
    if runtime["fingerprint"] != registration.get("runtime", {}).get("fingerprint"):
        raise ValueError(
            "The registered code/dependency environment changed; create a new registration or restore the recorded runtime"
        )
    if get_record(qpath, "execution", exp_id):
        raise FileExistsError(
            "Registered execution cannot be repeated under the same ID"
        )
    snapshot = authority_dir(qpath) / "snapshots" / exp_id
    with tempfile.TemporaryDirectory(prefix="sciquest-execution-") as temp:
        workspace = Path(temp)
        for name, expected in registration["file_hashes"].items():
            if file_hash(contained_path(snapshot, name)) != expected:
                raise ValueError("Registered input hash mismatch")
        # Stage from captured data, never from mutable current quest data.
        captured_manifest = {"datasets": []}
        for index, dataset in enumerate(registration["data_manifest"]["datasets"]):
            captured_manifest["datasets"].append(
                {**dataset, "path": f"data/{index}/{Path(dataset['path']).name}"}
            )
        stage_data(
            snapshot, workspace, registration["validation_suite"], captured_manifest
        )
        shutil.copyfile(snapshot / "notebook.ipynb", workspace / "notebook.ipynb")
        # Use registered values, never mutable current metadata or truth data.
        _stage_registered_context(workspace, registration)
        args = [
            sys.executable,
            "-m",
            "sciquest.worker",
            "notebook",
            "notebook.ipynb",
            "executed_notebook.ipynb",
            str(max(1, int(timeout))),
        ]
        execution = run_command(
            args,
            workspace,
            exp / "logs/execution.log",
            hidden_root=protected_root or qpath,
            timeout=timeout,
            isolated=isolated,
            network=False,
            on_tick=lambda: heartbeat(qpath, owner),
            blind=registration["plan"].get("scientist_information_mode")
            == "black_box_v2",
        )
        execution.update(
            registration_hash=registration["registration_hash"],
            interpreter=sys.executable,
            python_version=sys.version,
            experiment_id=exp_id,
            runtime=runtime,
        )
        if execution["status"] == "succeeded" and registration["plan"].get(
            "model_export"
        ):
            remaining = timeout - execution["elapsed_seconds"]
            if remaining <= 0:
                execution["status"] = "timed_out"
            else:
                try:
                    execution["model_export_verification"] = _verify_model_export(
                        qpath,
                        exp,
                        workspace,
                        snapshot,
                        registration,
                        captured_manifest,
                        owner,
                        isolated,
                        remaining,
                        protected_root,
                    )
                except (ValueError, OSError, KeyError) as exc:
                    execution["status"] = "invalid_model_export"
                    execution["model_export_error"] = str(exc)
        output = workspace / "executed_notebook.ipynb"
        execution["prediction_artifact_hashes"] = {}
        for metric in registration["validation_suite"]["metrics"]:
            name = metric.get("computation", {}).get("artifact", "predictions.csv")
            artifact = contained_path(workspace, name)
            if artifact.is_file() and not artifact.is_symlink():
                execution["prediction_artifact_hashes"][name] = file_hash(artifact)
        execution["output_notebook_hash"] = (
            file_hash(output) if output.is_file() else None
        )
        if execution["status"] == "succeeded":
            if not output.is_file() or _notebook_sources(output) != _notebook_sources(
                snapshot / "notebook.ipynb"
            ):
                execution["status"] = "invalid_output"
            elif any(
                cell.get("execution_count") is None
                for cell in nbformat.read(output, as_version=4).cells
                if cell.cell_type == "code" and cell.source.strip()
            ):
                execution["status"] = "invalid_output"
        record(qpath, "execution", exp_id, execution, owner=owner)
        write_yaml(exp / "execution.yaml", execution, exclusive=True)
        _collect(workspace, exp)
        if output.is_file():
            shutil.copyfile(output, exp / "executed_notebook.ipynb")
        if execution["status"] != "succeeded":
            raise RuntimeError(
                f"Registered execution {execution['status']}; partial artifacts preserved"
            )
        evaluation = evaluate_registered(
            qpath, exp_id, workspace, execution, owner=owner
        )
        write_yaml(exp / "validation_results.yaml", evaluation, exclusive=True)
        return evaluation


def _finish(qpath: Path, exp_id: str, owner: str, result: dict[str, Any]) -> None:
    exp = experiment_dir(qpath, exp_id)
    meta = read_yaml(exp / "experiment.yaml", {})
    write_yaml(
        exp / "experiment.yaml",
        {
            **meta,
            "status": "evaluated",
            "registration_hash": result["registration_hash"],
            "comparison_group": result["comparison_group"],
        },
    )
    write_yaml(exp / "validation_results.yaml", result)
    outcome_text = "\n".join(
        f"- {key}: {value}" for key, value in result["prediction_outcomes"].items()
    )
    metric_text = "\n".join(
        f"- {name} ({result.get('metric_roles', {}).get(name, 'primary')}): {value}"
        for name, value in result["metrics"].items()
    )
    constraint_text = (
        "\n".join(
            f"- {name}: {outcome}"
            for name, outcome in result.get("constraint_outcomes", {}).items()
        )
        or "No separate constraint requirements registered."
    )
    undefined_text = (
        "\n".join(
            f"- {name}: {reason}"
            for name, reason in result.get("undefined_metrics", {}).items()
        )
        or "None."
    )
    policy_text = json.dumps(
        result.get("policy_evidence", {}), indent=2, sort_keys=True
    )
    atomic_text(
        exp / "experiment_report.md",
        f"# Registered experiment {exp_id}\n\n## Evidence\nMetrics computed by the supervisor from registered truth and prediction artifacts.\n\n{outcome_text}\n\n## Metrics\n{metric_text}\n\n## Constraints\n{constraint_text}\n\n## Undefined metrics\n{undefined_text}\n\n## Inquiry policy evidence\n```json\n{policy_text}\n```\n\n## Scope\n{result['evaluation_scope']}\nDecision metrics use owner-supplied expected intervention values on the registered finite action grid. Scientific understanding and causal identification require separate evidence.\n\n## Speculation\nNo broader causal or real-world conclusion follows automatically.\n",
    )
    marker = f"<!-- sciquest-evaluation:{exp_id} -->"
    if marker not in (qpath / "journal.md").read_text():
        append_text(
            qpath / "journal.md",
            f"\n\n{marker}\n## {exp_id}\nSupervisor evaluation recorded.\n{outcome_text}\n",
        )
    state = read_state(qpath)
    groups = state.get("score_groups", {})
    group = groups.get(result["comparison_group"], {})
    if result["aggregate_score"] is not None and (
        group.get("best_score") is None
        or result["aggregate_score"] > group["best_score"]
    ):
        groups[result["comparison_group"]] = {
            "best_score": result["aggregate_score"],
            "experiment": exp_id,
        }
    finalize_run(
        qpath,
        owner,
        exp_id,
        {
            "registration_hash": result["registration_hash"],
            "evaluation_hash": content_hash(result),
        },
        {
            "last_experiment": exp_id,
            "last_score": result["aggregate_score"],
            "score_groups": groups,
        },
    )


def _fail(qpath: Path, exp_id: str | None, owner: str, exc: BaseException) -> None:
    state = read_state(qpath)
    if state.get("lock_id") != owner:
        return
    failures = [
        *(state.get("failures") or []),
        {"experiment": exp_id, "error": str(exc), "time": utc_now()},
    ]
    release_lock(qpath, owner, "failed", changes={"failures": failures})


def execute_registered(
    qpath: Path,
    exp_id: str,
    *,
    isolated: bool = True,
    timeout: float = 600,
    protected_root: Path | None = None,
) -> dict[str, Any]:
    from .io import finite_number

    if finite_number(timeout, "timeout") <= 0:
        raise ValueError("timeout must be positive")
    if protected_root is not None:
        protected_root = Path(protected_root).resolve()
        if not qpath.resolve().is_relative_to(protected_root):
            raise ValueError("Protected root must contain the quest")
    completion = get_record(qpath, "completion", exp_id)
    evaluation = get_record(qpath, "evaluation", exp_id)
    if (
        completion
        and evaluation
        and completion.get("evaluation_hash") == content_hash(evaluation)
    ):
        from .logic import run_logic_check

        integrity = run_logic_check(qpath, exp_id)
        if not integrity["passed"]:
            raise ValueError(
                "Completed experiment artifacts changed: "
                + "; ".join(integrity["issues"])
            )
        return evaluation
    owner = acquire_lock(qpath)
    try:
        if evaluation:
            # Recover interrupted publication without rerunning the experiment.
            execution = get_record(qpath, "execution", exp_id)
            if (
                not execution
                or execution.get("status") != "succeeded"
                or execution.get("registration_hash")
                != evaluation.get("registration_hash")
            ):
                raise ValueError(
                    "Incomplete evaluation cannot be reconciled with its execution"
                )
            for name, expected in execution["prediction_artifact_hashes"].items():
                if (
                    file_hash(
                        contained_path(
                            experiment_dir(qpath, exp_id) / "worker_artifacts", name
                        )
                    )
                    != expected
                ):
                    raise ValueError("Prediction artifact changed before finalization")
            result = evaluation
        else:
            result = _execute_and_evaluate(
                qpath,
                exp_id,
                owner,
                isolated=isolated,
                timeout=timeout,
                protected_root=protected_root,
            )
        _finish(qpath, exp_id, owner, result)
        return result
    except BaseException as exc:
        _fail(qpath, exp_id, owner, exc)
        raise


def run_agent_iteration(
    qpath: Path,
    agent_command: str | None,
    *,
    isolated: bool = True,
    timeout: float = 600,
) -> str | None:
    from .io import finite_number

    if finite_number(timeout, "timeout") <= 0:
        raise ValueError("timeout must be positive")
    from .agent import launch_agent

    owner = acquire_lock(qpath)
    exp_id = None
    started = time.monotonic()
    try:
        exp_id, exp = reserve_experiment(qpath, owner)
        create_scaffold(qpath, exp_id, exp)
        with tempfile.TemporaryDirectory(prefix="sciquest-preparation-") as temp:
            workspace = Path(temp)
            for name in [
                "quest.yaml",
                "hypotheses.yaml",
                "validation.yaml",
                "data_manifest.yaml",
                "method_stack.yaml",
                "research_contract.yaml",
            ]:
                if (qpath / name).is_file():
                    shutil.copyfile(qpath / name, workspace / name)
            from .templates import AGENTS_MD

            atomic_text(workspace / "AGENTS.md", AGENTS_MD)
            suite, manifest = (
                read_yaml(qpath / "validation.yaml", {}),
                read_yaml(qpath / "data_manifest.yaml", {}),
            )
            ready = (
                suite.get("status") == "ready"
                and bool(suite.get("metrics"))
                and all(
                    isinstance(m.get("computation"), dict) for m in suite["metrics"]
                )
                and manifest.get("status") == "ready"
                and bool(manifest.get("datasets"))
                and all(
                    d.get("role") in {"training", "evaluation"}
                    for d in manifest["datasets"]
                )
            )
            write_yaml(
                workspace / "handoff.yaml",
                {
                    "schema_version": 1,
                    "mode": "prepare_experiment" if ready else "propose_setup",
                    "last_completed_experiment": read_state(qpath).get(
                        "last_experiment"
                    ),
                },
            )
            if ready:
                stage_data(qpath, workspace, suite, manifest)
            stage_research_context(qpath, workspace)
            from .methods.ledger import load_quest_method_stack
            from .methods.policies import policy_requirements

            write_yaml(
                workspace / "policy_contract.yaml",
                policy_requirements(load_quest_method_stack(qpath).to_dict()),
            )
            preparation = launch_agent(
                workspace,
                agent_command,
                hidden_root=qpath,
                isolated=isolated,
                timeout=timeout,
                log_path=exp / "logs/agent.log",
                on_tick=lambda: heartbeat(qpath, owner),
            )
            record(qpath, "preparation", exp_id, preparation, owner=owner)
            _collect(workspace, exp)
            if preparation["status"] != "succeeded":
                raise RuntimeError(f"Agent preparation {preparation['status']}")
            decision_path = workspace / "decision.yaml"
            if decision_path.exists() or not ready:
                decision = (
                    read_yaml(decision_path, {})
                    if decision_path.exists()
                    else read_yaml(workspace / "setup_proposal.yaml", {})
                )
                if (
                    not isinstance(decision, dict)
                    or not isinstance(decision.get("reason"), str)
                    or not decision["reason"].strip()
                ):
                    raise ValueError("A stop/setup decision requires a concrete reason")
                action = decision.get("action")
                if action not in {"stop", "needs_data", "propose_setup"} or (
                    not ready and action not in {"propose_setup", "needs_data", "stop"}
                ):
                    raise ValueError("Invalid research decision")
                record(
                    qpath,
                    "decision",
                    exp_id,
                    {**decision, "created_at": utc_now()},
                    owner=owner,
                )
                meta = read_yaml(exp / "experiment.yaml", {})
                write_yaml(
                    exp / "experiment.yaml",
                    {**meta, "status": "decision_recorded", "decision": decision},
                )
                append_text(
                    qpath / "journal.md",
                    f"\n\n## Research decision {exp_id}\n{action}: {decision['reason']}\nNo evaluated experiment is claimed.\n",
                )
                release_lock(
                    qpath,
                    owner,
                    changes={"last_decision": {"experiment": exp_id, **decision}},
                )
                return None
            for name in ["notebook.ipynb", "experiment_plan.yaml"]:
                source = workspace / name
                if not source.is_file() or source.is_symlink():
                    raise ValueError(f"Agent did not produce {name}")
                shutil.copyfile(source, exp / name)
        preregister_experiment(qpath, exp_id, owner=owner)
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(
                "Iteration wall-time budget exhausted during preparation"
            )
        result = _execute_and_evaluate(
            qpath, exp_id, owner, isolated=isolated, timeout=remaining
        )
        _finish(qpath, exp_id, owner, result)
        return exp_id
    except BaseException as exc:
        _fail(qpath, exp_id, owner, exc)
        raise
