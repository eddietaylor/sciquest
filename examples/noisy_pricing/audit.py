"""Independent arithmetic and provenance checks of a saved benchmark allocation.

Does not call the generator or SciQuest scoring functions. No files are changed.
"""
from __future__ import annotations

import argparse
import sys

# Preserved source is evidence: importing its helpers must not create caches.
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sciquest.io import content_hash, read_yaml
from sciquest.state import get_record
from sciquest.readonly import database_reads_only


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-8):
        raise ValueError(f"Arithmetic mismatch: {actual} != {expected}")


def validate_evaluation_grid(truth: list[dict], protocol: dict) -> None:
    contexts = defaultdict(list)
    for row in truth:
        contexts[row["context_id"]].append(row)
    if len(contexts) != protocol["evaluation_contexts"] or any(
        len(group) != len(protocol["evaluation_prices"]) or
        sorted(float(row["price"]) for row in group) != protocol["evaluation_prices"] or
        len({row["x"] for row in group}) != 1 for group in contexts.values()):
        raise ValueError("Wrong evaluation contexts, actions, or context features")


def independent_metrics(truth, predictions, training_prices):
    if len({r["id"] for r in truth}) != len(truth):
        raise ValueError("Duplicate truth IDs")
    pred = {r["id"]: r for r in predictions}
    if len(pred) != len(predictions) or set(pred) != {r["id"] for r in truth}:
        raise ValueError("Predictions must match complete unique truth IDs")
    by_context = defaultdict(list)
    errors, unit_errors, seen_errors, unseen_errors = [], [], [], []
    for r in truth:
        p = pred[r["id"]]
        price, mean, mu = float(r["price"]), float(p["predicted_mean_demand"]), float(r["mean_demand"])
        close(p["predicted_revenue"], price * mean)
        close(r["expected_revenue"], price * mu)
        error = (mean - mu)**2
        errors.append(error)
        unit_errors.append((mean - float(r["realized_demand"]))**2)
        (seen_errors if price in training_prices else unseen_errors).append(error)
        by_context[r["context_id"]].append((price, float(p["predicted_revenue"]), float(r["expected_revenue"])))
    values, gaps = [], []
    for group in by_context.values():
        if len({r[0] for r in group}) != len(group):
            raise ValueError("Duplicate action in context")
        selected = sorted(group, key=lambda r: (-r[1], r[0]))[0]
        values.append(selected[2])
        gaps.append(max(r[2] for r in group) - selected[2])
    rms = lambda values: math.sqrt(math.fsum(values) / len(values))
    return dict(regret=math.fsum(gaps)/len(gaps), policy_value=math.fsum(values)/len(values),
                mean_demand_rmse=rms(errors), individual_demand_rmse=rms(unit_errors),
                seen_price_rmse=rms(seen_errors), unseen_price_rmse=rms(unseen_errors))


def audit(output: Path) -> dict:
    with database_reads_only():
        return _audit(output)


def authority_record(quest: Path, kind: str):
    if not (quest / ".sciquest/state.sqlite3").is_file():
        return None
    return get_record(quest, kind, "exp_001")


def validate_model(model: dict, baseline: dict, protocol: dict) -> None:
    dimension = 3 if baseline["model"] == "linear" else 6
    if (model.get("model") != baseline["model"] or model.get("ridge_sum_loss") != protocol["ridge_penalty"]
        or model.get("n") != protocol["training_units"]
        or not isinstance(model.get("coefficients"), list) or len(model["coefficients"]) != dimension
        or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in model["coefficients"])):
        raise ValueError("Model does not match allocated baseline/ridge/dimension/sample size")


def _audit(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())["sha256"]
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p != output / "manifest.json"}
    if actual != set(manifest):
        raise ValueError("Manifest file inventory mismatch")
    for rel, digest in manifest.items():
        if hashlib.sha256((output / rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Manifest hash mismatch: {rel}")
    protocol = json.loads((output / "allocation.json").read_text())
    results = json.loads((output / "results.json").read_text())
    outcomes = json.loads((output / "outcomes.json").read_text())
    if outcomes != results["outcomes"]:
        raise ValueError("Incremental and final outcomes differ")
    expected = {(f,s,b["name"]) for f in protocol["families"] for s in protocol["replicate_seeds"] for b in protocol["baselines"]}
    if len(outcomes) != len(expected) or {(r["family"],r["seed"],r["baseline"]) for r in outcomes} != expected:
        raise ValueError("Allocation rows missing or duplicated")
    barrier = json.loads((output / "before_execution.json").read_text())
    for name, filename in [("source_commitment_sha256", "source_commitment.json"), ("allocation_sha256", "allocation.json")]:
        if barrier[name] != hashlib.sha256((output / filename).read_bytes()).hexdigest():
            raise ValueError("Barrier commitment mismatch")
    for name, digest in json.loads((output / "source_commitment.json").read_text()).items():
        if hashlib.sha256((output / "owner_source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen source mismatch")
    registrations, starts, eval_hashes, random_hashes = [], [], defaultdict(set), defaultdict(set)
    checked = 0
    for r in outcomes:
        quest = output / "quests" / r["slot"]
        registration = authority_record(quest, "registration")
        if registration:
            if registration["registration_hash"] != barrier["registrations"][r["slot"]]:
                raise ValueError("Registration/barrier mismatch")
            if content_hash({k:v for k,v in registration.items() if k != "registration_hash"}) != registration["registration_hash"]:
                raise ValueError("Invalid authoritative registration hash")
            for rel, digest in registration["file_hashes"].items():
                snapshot=quest / ".sciquest/snapshots/exp_001" / rel
                if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                    raise ValueError("Registered snapshot hash mismatch")
            registrations.append(registration["created_at"])
            if len(registration["generation_replays"]) != 2 or not all(v["execution"]["isolated"] for v in registration["generation_replays"].values()):
                raise ValueError("Generator replay isolation not verified")
        execution = authority_record(quest, "execution")
        if execution:
            starts.append(execution["started_at"])
        if r["status"] != "completed":
            continue
        if not registration or not execution or not execution["isolated"] or not execution["blind_filesystem"] or execution["network"]:
            raise ValueError("Completed slot lacks blind isolated execution")
        exp = quest / "experiments/exp_001"
        if execution["status"] != "succeeded" or execution["registration_hash"] != registration["registration_hash"]:
            raise ValueError("Execution not bound to successful registration")
        if hashlib.sha256((exp / "executed_notebook.ipynb").read_bytes()).hexdigest() != execution["output_notebook_hash"]:
            raise ValueError("Executed notebook hash mismatch")
        for name,digest in execution["prediction_artifact_hashes"].items():
            if hashlib.sha256((exp / "worker_artifacts" / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Authoritative prediction artifact hash mismatch")
        truth = rows(quest / "data/raw/eval.csv")
        predictions = rows(exp / "worker_artifacts/predictions.csv")
        if len(truth) != protocol["evaluation_contexts"] * len(protocol["evaluation_prices"]):
            raise ValueError("Incomplete evaluation contexts/actions")
        validate_evaluation_grid(truth, protocol)
        metrics = independent_metrics(truth, predictions, protocol["training_prices"])
        for key, value in metrics.items():
            close(r["metrics"][key], value)
        evaluation = get_record(quest, "evaluation", "exp_001")
        if evaluation != read_yaml(exp / "validation_results.yaml") or evaluation["evidence_source"] != "supervisor_computed":
            raise ValueError("Not authoritative supervisor evaluation")
        completion=authority_record(quest,"completion")
        if (not completion or completion["evaluation_hash"] != content_hash(evaluation)
            or completion["registration_hash"] != registration["registration_hash"]
            or evaluation["registration_hash"] != registration["registration_hash"]):
            raise ValueError("Invalid completion/evaluation/registration binding")
        expected_metrics={"regret", "policy_value", "mean_demand_rmse", "individual_demand_rmse"}
        if set(evaluation["metrics"]) != expected_metrics:
            raise ValueError("Unexpected supervisor metrics")
        roles={m["name"]:m["role"] for m in registration["validation_suite"]["metrics"]}
        if roles != {m:("primary" if m == "regret" else "diagnostic") for m in expected_metrics}:
            raise ValueError("Metric roles differ from the allocation")
        for key, value in evaluation["metrics"].items():
            close(value, metrics[key])
        training = rows(quest / "data/raw/train.csv")
        if len(training) != protocol["training_units"] or len({v["id"] for v in training}) != len(training) or {v["id"] for v in training} & {v["id"] for v in truth}:
            raise ValueError("Training/evaluation identity boundary")
        if not {float(v["price"]) for v in training} <= set(protocol["training_prices"]):
            raise ValueError("Training prices outside declared support")
        model = json.loads((exp / "worker_artifacts/model.json").read_text())
        baseline = next(b for b in protocol["baselines"] if b["name"] == r["baseline"])
        validate_model(model, baseline, protocol)
        expected_source = (f"MODEL = {baseline['model']!r}\nRIDGE = {protocol['ridge_penalty']!r}\n"
                           + (output / "owner_source/analysis.py").read_text())
        source_notebook=json.loads((exp / "notebook.ipynb").read_text())
        executed_notebook=json.loads((exp / "executed_notebook.ipynb").read_text())
        cell_source=lambda cell: ''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
        if (len(source_notebook["cells"]) != len(executed_notebook["cells"]) or
            hashlib.sha256((exp / "notebook.ipynb").read_bytes()).hexdigest() != registration["file_hashes"]["notebook.ipynb"]):
            raise ValueError("Source notebook differs from registration")
        if cell_source(source_notebook['cells'][2]) != expected_source or any(
            cell_source(a) != cell_source(b) for a,b in zip(source_notebook['cells'],executed_notebook['cells'])):
            raise ValueError("Notebook source does not implement the allocated estimator")
        def vector(v):
            t,x=(float(v["price"])-40)/20,float(v["x"])
            return [1,t,x] if baseline["model"] == "linear" else [1,t,x,t*x,t*t,x*x]
        X=np.array([vector(v) for v in training]); y=np.array([float(v["demand"]) for v in training])
        # Independent augmented least-squares solver, rather than normal equations.
        penalty=np.eye(X.shape[1])*math.sqrt(protocol["ridge_penalty"]); penalty[0,0]=0
        beta=np.linalg.lstsq(np.vstack([X,penalty]), np.r_[y,np.zeros(X.shape[1])], rcond=None)[0]
        for actual, wanted in zip(model["coefficients"], beta):
            close(actual, wanted)
        pred_by_id={v["id"]:v for v in predictions}
        for v in truth:
            close(pred_by_id[v["id"]]["predicted_mean_demand"], np.array(vector(v)) @ beta)
        campaign=(r["family"],r["seed"])
        eh=hashlib.sha256((quest / "data/raw/eval.csv").read_bytes()).hexdigest()
        th=hashlib.sha256((quest / "data/raw/train.csv").read_bytes()).hexdigest()
        if (eh != r["evaluation_sha256"] or th != r["training_sha256"]
            or eh != registration["file_hashes"]["data/1/eval.csv"]
            or th != registration["file_hashes"]["data/0/train.csv"]):
            raise ValueError("Outcome data hash mismatch")
        eval_hashes[campaign].add(eh)
        if r["baseline"].startswith("randomized_"):
            random_hashes[campaign].add(th)
        checked += 1
    if starts and registrations and max(registrations) > min(starts):
        raise ValueError("An execution preceded allocation registration barrier")
    if any(len(v) != 1 for v in [*eval_hashes.values(), *random_hashes.values()]):
        raise ValueError("Paired baselines do not share evaluation/randomized training")
    # Require current summary implementation to equal the frozen source; arithmetic is also
    # independently exercised by hand-computable statistical regression tests.
    import summarize as summary_module
    commitment=json.loads((output / "source_commitment.json").read_text())
    if hashlib.sha256(Path(summary_module.__file__).read_bytes()).hexdigest() != commitment['summarize.py']:
        raise ValueError("Current summarizer differs from frozen source; use owner_source/audit.py")
    from summarize import summarize
    if results["summary"] != summarize(protocol,outcomes):
        raise ValueError("Summary/outcome mismatch")
    return dict(planned=len(outcomes), completed=checked, all_planned_completed=checked==len(outcomes),
                manifest_files_verified=len(manifest), independently_recomputed_metrics=True,
                independent_augmented_ridge_solver=True, external_agent_calls=results["external_agent_calls"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    result=audit(a.output.resolve())
    print(json.dumps(result,indent=2))
    if not result["all_planned_completed"]:
        raise SystemExit(1)
