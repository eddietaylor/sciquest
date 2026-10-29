"""Bounded public-input predictor preparation, before any native registration.

Never imports evaluator/world truth. It checks execution and public output
contracts, not prediction accuracy, interval calibration or scientific warrant.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile

from sciquest.execution import run_command
from sciquest.io import contained_path, content_hash, file_hash, write_yaml, read_yaml
from sciquest.state import get_record
from sciquest.workflow import stage_data

VERSION = "public-predictor-preflight-v1"
FIELDS = ("mean_demand", "mean_sales", "lower_sales", "upper_sales")


def validate_notebook_json(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"notebook_json: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("notebook_json must encode a notebook object")


def public_predictions(workspace, context):
    path = contained_path(workspace, "predictions.csv")
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("predictions.csv must be a regular file within 1 MiB")
    features = {str(r["id"]): r for r in context["evaluation_features"]}
    result = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", *FIELDS} <= set(reader.fieldnames or []):
            raise ValueError(
                "predictions.csv needs id, mean_demand, mean_sales, lower_sales, upper_sales"
            )
        for row in reader:
            identity = row.get("id")
            if identity not in features or identity in result:
                raise ValueError(
                    "predictions.csv has an unknown or duplicate public evaluation ID"
                )
            try:
                values = {key: float(row[key]) for key in FIELDS}
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"predictions.csv row {identity} needs numeric forecast fields"
                ) from exc
            if any(not math.isfinite(v) or abs(v) > 1e6 for v in values.values()):
                raise ValueError(
                    f"predictions.csv row {identity} exceeds finite public numerical bounds"
                )
            if (
                values["mean_demand"] < 0
                or not 0 <= values["mean_sales"] <= features[identity]["capacity"]
            ):
                raise ValueError(
                    f"predictions.csv row {identity} violates public physical mean bounds"
                )
            if values["lower_sales"] > values["upper_sales"]:
                raise ValueError(
                    f"predictions.csv row {identity} has a reversed interval"
                )
            result[identity] = values
    if result.keys() != features.keys():
        raise ValueError(
            "predictions.csv must cover every public evaluation feature ID"
        )
    return result


def _execute(workspace, directory, hidden_root, timeout, *, reload=False):
    log = directory / ("reload.log" if reload else "notebook.log")
    execution = run_command(
        [
            sys.executable,
            "-m",
            "sciquest.worker",
            "notebook",
            "notebook.ipynb",
            "executed.ipynb",
            str(max(1, int(timeout))),
        ],
        workspace,
        log,
        hidden_root=hidden_root,
        blind=True,
        timeout=timeout,
    )
    return execution, log


def run(session, notebook, submitted, context, directory, timeout, *, export_observer=None):
    """One preparation check; keeps failed code/logs, never native scores."""
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    with session.transaction() as (_, state):
        quest, expid = Path(state["quest"]), state["experiment"]
    if get_record(quest, "registration", expid):
        raise ValueError("Preflight must precede native registration")
    manifest = read_yaml(quest / "data_manifest.yaml")
    native_plan = {
        **submitted,
        "demand_scientific_decision": context["decision"],
        "scientist_information_mode": "black_box_v2",
        "comparison_assignment": context["comparison"],
    }
    record = dict(
        version=VERSION,
        notebook_sha256=file_hash(notebook),
        plan_hash=content_hash(native_plan),
        context_revision=context["revision"],
        timeout=timeout,
        status="failed",
        execution=None,
        reload_execution=None,
        error=None,
        scope="Public-input execution/output/reload checks only; no hidden labels, scoring, new observations or scientific-decision changes.",
    )
    shutil.copyfile(notebook, directory / "submitted.ipynb")
    write_yaml(directory / "submitted_plan.yaml", native_plan)
    try:
        with tempfile.TemporaryDirectory(
            prefix="sciquest-predictor-preflight-"
        ) as temp:
            work = Path(temp)
            stage_data(quest, work, context["validation_suite"], manifest)
            shutil.copyfile(notebook, work / "notebook.ipynb")
            write_yaml(work / "validation.yaml", context["validation_suite"])
            write_yaml(work / "experiment_plan.yaml", native_plan)
            record["public_input_hashes"] = {
                str(p.relative_to(work)): file_hash(p)
                for p in sorted(work.rglob("*"))
                if p.is_file()
            }
            for name in record["public_input_hashes"]:
                target = directory / "inputs" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(work / name, target)
            execution, log = _execute(work, directory, session.path, timeout)
            record["execution"] = execution
            if execution["status"] != "succeeded":
                detail = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))[
                    -2400:
                ]
                raise ValueError(
                    "Public-input notebook execution "
                    + execution["status"]
                    + ": "
                    + detail
                )
            original = public_predictions(work, context)
            shutil.copyfile(work / "predictions.csv", directory / "predictions.csv")
            record["predictions_sha256"] = file_hash(directory / "predictions.csv")
            export = submitted.get("model_export")
            if export:
                if export_observer is not None:
                    export_observer(work, export, directory, "original")
                source = contained_path(work, export["reload_notebook"])
                if not source.is_file():
                    raise ValueError("Declared standalone reload notebook is missing")
                with tempfile.TemporaryDirectory(
                    prefix="sciquest-preflight-reload-"
                ) as reload_temp:
                    fresh = Path(reload_temp)
                    stage_data(
                        quest,
                        fresh,
                        context["validation_suite"],
                        manifest,
                        include_training=False,
                    )
                    write_yaml(fresh / "validation.yaml", context["validation_suite"])
                    write_yaml(fresh / "experiment_plan.yaml", native_plan)
                    for name in export["artifacts"]:
                        origin = contained_path(work, name)
                        if not origin.is_file():
                            raise ValueError(
                                "Declared model artifact is missing: " + name
                            )
                        dest = contained_path(fresh, name)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(origin, dest)
                    shutil.copyfile(source, fresh / "notebook.ipynb")
                    execution, log = _execute(
                        fresh, directory, session.path, timeout, reload=True
                    )
                    record["reload_execution"] = execution
                    if execution["status"] != "succeeded":
                        detail = re.sub(
                            r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace")
                        )[-2400:]
                        raise ValueError(
                            "Public-input clean reload "
                            + execution["status"]
                            + ": "
                            + detail
                        )
                    if export_observer is not None:
                        export_observer(fresh, export, directory, "reload")
                    reloaded = public_predictions(fresh, context)
                    (directory / "reload_predictions.json").write_text(
                        json.dumps(reloaded, sort_keys=True) + "\n"
                    )
                    record["reload_predictions_sha256"] = file_hash(
                        directory / "reload_predictions.json"
                    )
                    tolerance = export.get("absolute_tolerance", 0)
                    if any(
                        abs(original[i][key] - reloaded[i][key]) > tolerance
                        for i in original
                        for key in FIELDS
                    ):
                        raise ValueError(
                            "Public-input clean reload changes forecast values beyond declared tolerance"
                        )
            record["status"] = "succeeded"
    except (ValueError, OSError, RuntimeError, csv.Error) as exc:
        record["error"] = str(exc)
    record["log_hashes"] = {p.name: file_hash(p) for p in directory.glob("*.log")}
    (directory / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def verify_record(
    path,
    expected_notebook,
    expected_plan_hash,
    revision,
    *,
    context=None,
    expected_archive=None,
):
    path = Path(path)
    record = json.loads((path / "preflight.json").read_text())
    if (
        record["version"] != VERSION
        or record["notebook_sha256"] != expected_notebook
        or file_hash(path / "submitted.ipynb") != expected_notebook
        or record["plan_hash"] != expected_plan_hash
        or content_hash(read_yaml(path / "submitted_plan.yaml")) != expected_plan_hash
        or record["context_revision"] != revision
    ):
        raise ValueError("Predictor preflight source/provenance changed")
    for name, digest in record["log_hashes"].items():
        if file_hash(path / name) != digest:
            raise ValueError("Predictor preflight log changed")
    for execution in (record.get("execution"), record.get("reload_execution")):
        if execution and (
            not execution.get("blind_filesystem")
            or not execution.get("isolated")
            or execution.get("network")
            or execution.get("timeout_seconds") != record["timeout"]
        ):
            raise ValueError("Predictor preflight isolation or timeout changed")
    if (
        record["status"] not in ("succeeded", "failed")
        or record["status"] == "failed"
        and not record.get("error")
    ):
        raise ValueError("Predictor preflight failure evidence is incomplete")
    if (
        record.get("predictions_sha256")
        and file_hash(path / "predictions.csv") != record["predictions_sha256"]
    ):
        raise ValueError("Predictor preflight output changed")
    for name, digest in record.get("public_input_hashes", {}).items():
        if file_hash(contained_path(path / "inputs", name)) != digest:
            raise ValueError("Predictor preflight public input snapshot changed")
    if context is not None and record.get("public_input_hashes"):
        inputs = path / "inputs"
        allowed = {
            "notebook.ipynb",
            "validation.yaml",
            "experiment_plan.yaml",
            context["prediction_files"]["evaluation_features"],
            context["prediction_files"]["campaign_archive"],
        }
        if context["prediction_files"]["training"]:
            allowed.add(context["prediction_files"]["training"])
        if set(record["public_input_hashes"]) != allowed:
            raise ValueError(
                "Unexpected or missing preflight input; only public files are allowed"
            )
        if (
            read_yaml(inputs / "validation.yaml") != context["validation_suite"]
            or content_hash(read_yaml(inputs / "experiment_plan.yaml"))
            != expected_plan_hash
        ):
            raise ValueError("Preflight public contracts changed")
        if file_hash(inputs / "notebook.ipynb") != expected_notebook:
            raise ValueError("Preflight executed another notebook")

        def table(name):
            with (inputs / name).open(newline="") as stream:
                return list(csv.DictReader(stream))

        feature_rows = [
            {k: str(v) for k, v in row.items()}
            for row in context["evaluation_features"]
        ]
        if table(context["prediction_files"]["evaluation_features"]) != feature_rows:
            raise ValueError("Preflight features differ from public handoff")
        if (
            expected_archive is not None
            and table(context["prediction_files"]["campaign_archive"])
            != expected_archive
        ):
            raise ValueError("Preflight own-work archive changed")
        if context["prediction_files"]["training"]:
            fields = context["prediction_files"]["training_columns"]
            expected_training = [
                {
                    k: (
                        ""
                        if q["observation"].get(k) is None
                        else str(q["observation"][k])
                    )
                    for k in fields
                }
                for q in context["history"]
            ]
            if table(context["prediction_files"]["training"]) != expected_training:
                raise ValueError(
                    "Preflight training differs from preceding public observations"
                )
        if record["status"] == "succeeded":
            original = public_predictions(path, context)
            if record.get("reload_predictions_sha256"):
                if (
                    file_hash(path / "reload_predictions.json")
                    != record["reload_predictions_sha256"]
                ):
                    raise ValueError("Preflight reload output changed")
                reloaded = json.loads((path / "reload_predictions.json").read_text())
                tolerance = read_yaml(path / "submitted_plan.yaml")["model_export"].get(
                    "absolute_tolerance", 0
                )
                if original.keys() != reloaded.keys() or any(
                    not math.isfinite(reloaded[i][key])
                    or abs(original[i][key] - reloaded[i][key]) > tolerance
                    for i in original
                    for key in FIELDS
                ):
                    raise ValueError("Preflight clean reload changed forecast values")
    if record["status"] == "succeeded" and (
        record["error"] is not None
        or record["execution"]["status"] != "succeeded"
        or record.get("reload_execution")
        and record["reload_execution"]["status"] != "succeeded"
    ):
        raise ValueError("Predictor preflight success contradicts execution")
    return record
