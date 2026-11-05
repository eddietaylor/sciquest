"""Frozen finite noisy benchmark, actual isolated SciQuest execution, no LLM calls."""
from __future__ import annotations

import argparse
import sys

# Preserved source is evidence: importing its helpers must not create caches.
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import shutil
from pathlib import Path

import nbformat

from sciquest.core import create_quest, run_next
from sciquest.dashboard import build_dashboard
from sciquest.io import append_text, atomic_text, read_yaml, write_yaml
from sciquest.logic import run_logic_check
from sciquest.methods.ledger import preregister_experiment
from sciquest.runtime import runtime_manifest
from sciquest.state import get_record
from sciquest.workflow import execute_registered

from generate import generate
from summarize import summarize

HERE = Path(__file__).resolve().parent


def dump(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocation(profile: str) -> dict:
    protocol = json.loads((HERE / "protocol.json").read_text())
    if profile == "smoke":
        protocol["families"] = ["affine", "threshold"]
        protocol["replicate_seeds"] = protocol["replicate_seeds"][:2]
    elif profile != "full":
        raise ValueError(profile)
    protocol["profile"] = profile
    return protocol


def prepare(root: Path, slot: dict, baseline: dict, protocol: dict) -> tuple[Path, str]:
    quest = create_quest(root, dict(hero_statement="Noisy pricing benchmark: prescribed estimator",
        problem_statement="Measure prediction and pricing performance from a finite noisy sample.",
        initial_hypothesis="Fixed estimators have measurable prediction and pricing errors across declared synthetic worlds.",
        method="standard_empirical"), slug=slot["slot"])
    shutil.copyfile(HERE / "generate.py", quest / "data/generate.py")
    datasets = []
    for kind, name in [(baseline["design"], "train.csv"), ("evaluation", "eval.csv")]:
        generate(quest / "data/raw" / name, slot["family"], slot["seed"], kind,
                 protocol["training_units"], protocol["evaluation_contexts"])
        args = ["--kind", kind, "--family", slot["family"], "--seed", str(slot["seed"]),
                "--training-units", str(protocol["training_units"]), "--evaluation-contexts", str(protocol["evaluation_contexts"]), "--output", "output.csv"]
        entry = dict(path=f"data/raw/{name}", role="training" if name == "train.csv" else "evaluation",
                     provenance="Owner-generated synthetic finite sample; complete source committed before execution.",
                     generator="data/generate.py", generator_args=args, generated_output="output.csv", seed=slot["seed"])
        if name == "eval.csv":
            entry["input_columns"] = ["id", "context_id", "x", "price"]
        datasets.append(entry)
    write_yaml(quest / "data_manifest.yaml", dict(status="ready", datasets=datasets))
    common = dict(truth="data/1/eval.csv", key="id")
    decision = dict(**common, target="expected_revenue", prediction="predicted_revenue", group_by=["context_id"],
                    action="price", objective="maximize", tie_break="lowest_action", value_semantics="expected_interventional")
    write_yaml(quest / "validation.yaml", dict(status="ready", metrics=[
        dict(name="regret", role="primary", direction="minimize", weight=1,
             normalization=dict(type="minmax", min=0, max=10000), computation=dict(kind="regret", **decision)),
        dict(name="policy_value", role="diagnostic", direction="maximize", computation=dict(kind="policy_value", **decision)),
        dict(name="mean_demand_rmse", role="diagnostic", direction="minimize", computation=dict(kind="rmse", **common, target="mean_demand", prediction="predicted_mean_demand")),
        dict(name="individual_demand_rmse", role="diagnostic", direction="minimize", computation=dict(kind="rmse", **common, target="realized_demand", prediction="predicted_mean_demand")),
    ]))
    write_yaml(quest / "research_contract.yaml", dict(status="ready",
        question="Measure error and finite-grid regret of this prewritten estimator; no confirmatory performance threshold.",
        claim_scope="Descriptive synthetic benchmark; same context population and bounded price domain; no autonomous-discovery or philosophy claim.",
        estimand="Conditional mean demand given observed context under price intervention; latent type and noise marginalized. Finite-grid expected revenue.",
        rivals=["Alternative response shapes and latent selection can invalidate a simple regression."],
        assumptions=["Independent training/evaluation units from the same observed-context population.", "Owner's synthetic intervention values define the oracle; no latent-aware policy oracle."],
        observation_access=["This slot's training rows and evaluation features only."],
        intervention_access=["Prescribed training design, no adaptive experiment choice."],
        uncertainty_and_stopping="One fit; all planned outcomes retained. Campaign-level variability reported separately by the owner.",
        resource_budget=dict(experiments=1, timeout_seconds=120, external_agent_calls=0), evaluation_mode="descriptive_synthetic_benchmark"))
    append_text(quest / "journal.md", "\n## Scripted protocol before execution\n\n"
                "Fixed estimator and design supplied by the benchmark. No external scientific agent. "
                "Performance is descriptive; the sole registered prediction is the mathematical nonnegative-regret sanity check.\n")
    expid = run_next(quest, agent_stub=True)
    exp = quest / "experiments" / expid
    metadata = read_yaml(exp / "experiment.yaml")
    metadata.update(task_type="scripted_noisy_benchmark", model_architecture=f"{baseline['model']} ridge, fixed sum-loss penalty",
                    input_features=["x", "price"], target_features=["conditional mean demand"],
                    validation_technique="Protected oracle means and independent realized outcomes on fresh contexts")
    write_yaml(exp / "experiment.yaml", metadata)
    write_yaml(exp / "experiment_plan.yaml", dict(
        question="What prediction and decision errors does this prescribed estimator incur?",
        hypothesis="Finite-grid regret is nonnegative (mathematical sanity check only, no performance claim).",
        scope="One finite noisy training sample and independent evaluation contexts on a declared bounded price domain.",
        assumptions="Mean-zero disturbances; synthetic oracle marginalizes variables unavailable to the estimator.",
        data_access="Training x/price/demand; evaluation context_id/x/price only. No generator, family, seed, oracle labels, or prior campaigns staged.",
        stopping_rule="One fit, no tuning/retries/replacement. Retain a failed slot as failed.",
        action="hypothesis_test", confirmatory_status="exploratory", scientist_information_mode="black_box_v2",
        predictions=[dict(id="regret_nonnegative_sanity_only", metric="regret", operator=">=", threshold=0)]))
    excluded = [str(quest / "data/raw/eval.csv"), str(HERE), str(root.resolve())]
    boundary = ("from pathlib import Path\nimport importlib.util\n" +
                f"assert all(not Path(p).exists() for p in {excluded!r})\n" +
                "assert importlib.util.find_spec('sciquest') is None\n" +
                "assert sorted(p.name for p in Path('data/1').iterdir()) == ['features.csv']\n" +
                "print('owner source, labels, campaign history and supervisor excluded')\n")
    source = f"MODEL = {baseline['model']!r}\nRIDGE = {protocol['ridge_penalty']!r}\n" + (HERE / "analysis.py").read_text()
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Prewritten estimator\nFits only the current staged training data. No external-agent reasoning."),
        nbformat.v4.new_code_cell(boundary), nbformat.v4.new_code_cell(source)])
    for i, cell in enumerate(notebook.cells):
        cell["id"] = f"noisy-pricing-{i}"
    nbformat.write(notebook, exp / "notebook.ipynb")
    preregister_experiment(quest, expid)
    return quest, expid


def finish_slot(quest: Path, expid: str, slot: dict) -> None:
    result = execute_registered(quest, expid, timeout=120)
    check = run_logic_check(quest, expid)
    if not check["recorded_integrity_verified"] or not result["isolated_execution"]:
        raise RuntimeError(check)
    exp = quest / "experiments" / expid
    with (exp / "worker_artifacts/predictions.csv").open(newline="") as f:
        predictions = {r["id"]: r for r in csv.DictReader(f)}
    with (quest / "data/raw/eval.csv").open(newline="") as f:
        truth = list(csv.DictReader(f))
    # Owner-side subset diagnostics, explicitly separate from supervisor metrics.
    metrics = dict(result["metrics"])
    for seen in (True, False):
        errors = [(float(predictions[r["id"]]["predicted_mean_demand"]) - float(r["mean_demand"]))**2
                  for r in truth if (float(r["price"]) in (20, 30, 40, 50, 60)) == seen]
        metrics["seen_price_rmse" if seen else "unseen_price_rmse"] = (sum(errors) / len(errors))**.5
    slot.update(status="completed", metrics=metrics, isolated_execution=result["isolated_execution"],
                recorded_integrity_verified=check["recorded_integrity_verified"], scientific_verified=check["scientific_verified"],
                supervisor_metric_names=sorted(result["metrics"]),
                owner_diagnostic_names=["seen_price_rmse", "unseen_price_rmse"],
                training_sha256=sha(quest / "data/raw/train.csv"), evaluation_sha256=sha(quest / "data/raw/eval.csv"))
    dump(exp / "logic_check.json", check)


def present_slot(quest: Path, expid: str, slot: dict) -> None:
    metrics = slot["metrics"]
    append_text(quest / "journal.md", "\n## Scripted interpretation after evaluation\n\n"
                f"Supervisor regret: {metrics['regret']}; mean-demand RMSE: {metrics['mean_demand_rmse']}. "
                "These describe one synthetic campaign, not scientific confirmation or agent reasoning. "
                "Seen/unseen-price RMSE subsets are additional owner-computed diagnostics.\n")
    build_dashboard(quest)


def execute_slots(output: Path, outcomes: list[dict], prepared: dict) -> None:
    """One opportunity per registered slot. Ordinary failures do not erase later slots."""
    for slot in outcomes:
        if slot["status"] != "registered":
            continue
        slot["status"] = "running"
        dump(output / "outcomes.json", outcomes)
        try:
            finish_slot(*prepared[slot["slot"]], slot)
        except Exception as exc:
            slot.update(status="execution_failed", error_type=type(exc).__name__, error=str(exc))
        else:
            try:
                present_slot(*prepared[slot["slot"]], slot)
            except Exception as exc:
                slot["reporting_error"] = dict(error_type=type(exc).__name__, error=str(exc))
        dump(output / "outcomes.json", outcomes)
        print(f"{slot['slot']}: {slot['status']}", flush=True)


def run(output: Path, profile: str = "full") -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = allocation(profile)
    dump(output / "allocation.json", protocol)
    source = output / "owner_source"
    source.mkdir()
    for file in sorted(HERE.iterdir()):
        if file.is_file() and file.suffix in (".py", ".json", ".md"):
            shutil.copyfile(file, source / file.name)
    dump(output / "source_commitment.json", {p.name: sha(p) for p in sorted(source.iterdir())})
    dump(output / "runtime.json", runtime_manifest())
    outcomes = []
    for family in protocol["families"]:
        for seed in protocol["replicate_seeds"]:
            for baseline in protocol["baselines"]:
                outcomes.append(dict(slot=f"slot-{len(outcomes)+1:04d}", family=family, seed=seed,
                                     baseline=baseline["name"], status="planned"))
    dump(output / "outcomes.json", outcomes)
    prepared = {}
    for slot in outcomes:
        baseline = next(b for b in protocol["baselines"] if b["name"] == slot["baseline"])
        try:
            prepared[slot["slot"]] = prepare(output, slot, baseline, protocol)
            slot["status"] = "registered"
        except Exception as exc:
            slot.update(status="preparation_failed", error_type=type(exc).__name__, error=str(exc))
        dump(output / "outcomes.json", outcomes)
    registrations = {name: get_record(q, "registration", e)["registration_hash"] for name, (q, e) in prepared.items()}
    if any(get_record(q, "execution", e) for q, e in prepared.values()):
        raise RuntimeError("Execution occurred before the allocation registration barrier")
    dump(output / "before_execution.json", dict(registrations=registrations, planned=len(outcomes),
         preparation_failures=sum(s["status"] == "preparation_failed" for s in outcomes), executions_started=0,
         source_commitment_sha256=sha(output / "source_commitment.json"), allocation_sha256=sha(output / "allocation.json")))
    execute_slots(output, outcomes, prepared)
    results = dict(format=protocol["version"], profile=profile, external_agent_calls=0,
                   genuine_external_agent_reasoning_demonstrated=False, outcomes=outcomes,
                   summary=summarize(protocol, outcomes))
    dump(output / "results.json", results)
    from report import render
    render(output, results)
    hashes = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
              if p.is_file() and p != output / "manifest.json"}
    dump(output / "manifest.json", dict(sha256=hashes, scope="Immutable per-run files, including hidden supervisor records; runtime paths/timestamps may vary."))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    result = run(args.output, args.profile)
    print(json.dumps(dict(planned=result["summary"]["planned"], completed=result["summary"]["completed"])))
    if result["summary"]["completed"] != result["summary"]["planned"]:
        raise SystemExit(1)
