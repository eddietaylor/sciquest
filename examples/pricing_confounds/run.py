"""Run two prescribed experiments through SciQuest, without an external agent."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import nbformat

from sciquest.core import create_quest, run_next
from sciquest.dashboard import build_dashboard
from sciquest.io import append_text, read_yaml, write_yaml
from sciquest.logic import run_logic_check
from sciquest.methods.ledger import preregister_experiment
from sciquest.runtime import runtime_manifest
from sciquest.state import get_record
from sciquest.workflow import execute_registered

from generate import generate
from plots import render

HERE = Path(__file__).resolve().parent
ARMS = ("observational", "randomized")


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(root: Path, arm: str, seed: int) -> tuple[Path, str]:
    hypothesis = "Price-only OLS recovers population intervention means (RMSE <= 1e-10) and selects a zero-regret listed price."
    quest = create_quest(root, dict(
        hero_statement=f"Pricing under confounding: {arm}",
        problem_statement="Can the prescribed price-only estimator recover intervention means and select a price on {30,40}?",
        initial_hypothesis=hypothesis, method="standard_empirical",
    ), slug=arm)
    shutil.copyfile(HERE / "generate.py", quest / "data/generate.py")
    datasets = []
    for index, (kind, name) in enumerate([(arm, "train.csv"), ("evaluation", "eval.csv")]):
        generate(quest / "data/raw" / name, kind, seed)
        dataset = dict(path=f"data/raw/{name}", role="training" if index == 0 else "evaluation",
                       provenance="Synthetic noiseless analytic world; see the published generator and protocol. Eight training units; evaluation rows are population means, not sampled individuals.",
                       generator="data/generate.py", generator_args=["--kind", kind, "--seed", str(seed), "--output", "output.csv"],
                       generated_output="output.csv", seed=seed)
        if index == 1:
            dataset["input_columns"] = ["id", "population", "price"]
        datasets.append(dataset)
    write_yaml(quest / "data_manifest.yaml", dict(status="ready", datasets=datasets))
    decision = dict(truth="data/1/eval.csv", key="id", target="expected_revenue",
                    prediction="predicted_revenue", group_by=["population"], action="price",
                    objective="maximize", tie_break="lowest_action", value_semantics="expected_interventional")
    write_yaml(quest / "validation.yaml", dict(status="ready", metrics=[
        dict(name="mean_demand_rmse", role="primary", direction="minimize", weight=1,
             normalization=dict(type="minmax", min=0, max=40),
             computation=dict(kind="rmse", truth="data/1/eval.csv", key="id", target="mean_demand", prediction="predicted_mean_demand")),
        dict(name="policy_value", role="diagnostic", direction="maximize", computation=dict(kind="policy_value", **decision)),
        dict(name="regret", role="diagnostic", direction="minimize", computation=dict(kind="regret", **decision)),
    ]))
    write_yaml(quest / "research_contract.yaml", dict(
        status="ready", question=hypothesis,
        claim_scope="Known noiseless synthetic generator, equal Z population, exactly prices30/40; no autonomous-discovery, real-market or philosophy comparison claim.",
        estimand="E[D|do(P=p)] and p*E[D|do(P=p)] on the two listed prices.",
        rivals=["The observationally equivalent demand=-20+2*price law has the opposite intervention slope."],
        assumptions=["Equal population proportions Z=0/1.", "No noise, interference, capacity limits, costs, or dynamics.", "Balanced stratified assignment within each simulator-known Z in the randomized arm."],
        observation_access=["Eight price/demand training rows for this arm."],
        intervention_access=["Prescribed design only; no adaptive choice."],
        uncertainty_and_stopping="One registered fit per arm. No sampling uncertainty, p-value, or coverage estimate.",
        resource_budget=dict(experiments=1, timeout_seconds=120, external_agent_calls=0),
        evaluation_mode="analytic_reproduction",
    ))
    append_text(quest / "journal.md", "\n## Scripted protocol, before execution\n\n"
                "The owner supplies both designs and this OLS analysis. No external agent reasons or chooses experiments. "
                "Analytic expectations are known in advance; this is a transparent reproduction, not a blind discovery. "
                "Both arms will be registered before either executes.\n")
    expid = run_next(quest, agent_stub=True)
    exp = quest / "experiments" / expid
    metadata = read_yaml(exp / "experiment.yaml")
    metadata.update(task_type="scripted_synthetic_research", model_architecture="Price-only ordinary least squares with intercept",
                    input_features=["price"], target_features=["population mean demand"],
                    validation_technique="Protected analytic intervention means on the same two-price grid")
    write_yaml(exp / "experiment.yaml", metadata)
    (exp / "hypothesis.md").write_text("# Prescribed hypothesis\n\n" + hypothesis + "\n")
    write_yaml(exp / "experiment_plan.yaml", dict(
        question="Does price-only OLS fitted to this prescribed design recover the intervention means and best listed price?",
        hypothesis=hypothesis, scope="Noiseless balanced-Z synthetic population; finite prices30/40; one fixed analysis per design.",
        assumptions="UniformZ, affine mean model, fixed sampling/assignment; no statistical uncertainty or real-market validity implied.",
        data_access="This arm's training price/demand and evaluation price/population features. Evaluation labels and generator are not staged. Owner and public reader know the generator.",
        stopping_rule="One fit and one supervisor evaluation; no tuning or replacement.",
        action="hypothesis_test", confirmatory_status="exploratory", scientist_information_mode="black_box_v2",
        predictions=[dict(id="mean_recovery", metric="mean_demand_rmse", operator="<=", threshold=1e-10),
                     dict(id="optimal_listed_price", metric="regret", operator="<=", threshold=1e-10)],
    ))
    # These known file paths are probes, not scientific evidence. The worker must
    # not be able to read owner truth or the public repository's generator.
    excluded = [str(quest / "data/raw/eval.csv"), str(HERE / "generate.py")]
    boundary = "from pathlib import Path\nimport importlib.util\n" + f"assert all(not Path(p).exists() for p in {excluded!r})\n" + "assert importlib.util.find_spec('sciquest') is None\nprint('owner files and supervisor package excluded')\n"
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Prescribed price-only OLS\nThis notebook was assembled by the example script, not an external agent. It fits only the staged training rows."),
        nbformat.v4.new_code_cell(boundary),
        nbformat.v4.new_code_cell((HERE / "analysis.py").read_text()),
    ])
    for i, cell in enumerate(notebook.cells):
        cell["id"] = f"pricing-cell-{i}"
    nbformat.write(notebook, exp / "notebook.ipynb")
    preregister_experiment(quest, expid)
    return quest, expid


def run(output: Path, seed: int = 42) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # never replace a previous run
    prepared = [prepare(output, arm, seed) for arm in ARMS]
    # Durable barrier: both insert-only registrations exist before any notebook.
    registrations = {}
    for quest, expid in prepared:
        record = get_record(quest, "registration", expid)
        if not record or get_record(quest, "execution", expid):
            raise RuntimeError("Both registrations must precede either execution")
        registrations[quest.name] = record["registration_hash"]
    dump(output / "before_execution.json", dict(registrations=registrations, executions_started=0))
    summaries = []
    for quest, expid in prepared:
        result = execute_registered(quest, expid, timeout=120)
        check = run_logic_check(quest, expid)
        if not check["recorded_integrity_verified"] or not result["isolated_execution"]:
            raise RuntimeError(check)
        exp = quest / "experiments" / expid
        with (exp / "worker_artifacts/predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        selected = result["decision_evidence"]["policy_value"]["trace_sample"][0]
        summary = dict(arm=quest.name, metrics=result["metrics"],
                       fitted_intercept=float(rows[0]["fitted_intercept"]), fitted_slope=float(rows[0]["fitted_slope"]),
                       selected_price=selected["action"], prediction_outcomes=result["prediction_outcomes"],
                       recorded_integrity_verified=check["recorded_integrity_verified"], scientific_verified=check["scientific_verified"],
                       isolated_execution=result["isolated_execution"])
        summaries.append(summary)
        interpretation = ("\n## Scripted interpretation\n\n"
                          "The model is fitted inside the executed notebook; metrics above come from the supervisor. "
                          f"Actual mean-demand RMSE: {result['metrics']['mean_demand_rmse']}; "
                          f"actual finite-grid regret: {result['metrics']['regret']}. "
                          "Population-mean prediction is distinct from predicting an individual unit's Z-dependent demand. "
                          "The comparison figure is generated owner-side after both evaluations; it is not a worker artifact. "
                          "No external agent chose the hypothesis, design, model or interpretation.\n")
        append_text(exp / "experiment_report.md", interpretation)
        append_text(quest / "journal.md", interpretation)
        dump(exp / "logic_check.json", check)
    results = dict(format="sciquest-pricing-confounds-v1", seed=seed, price_grid=[30, 40],
                   proposal_author="prewritten example script", external_agent_calls=0,
                   genuine_external_agent_reasoning_demonstrated=False, experiments=summaries)
    dump(output / "results.json", results)
    render(output)
    for quest, _ in prepared:
        build_dashboard(quest)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>SciQuest pricing reproduction</title>'
        '<h1>Pricing under confounding</h1><p>Scripted synthetic research reproduction; zero external-agent calls.</p>'
        '<p><a href="report.md">Research report</a> · <a href="results.json">Machine-readable results</a></p>'
        '<img src="comparison.svg" alt="Actual demand estimates and finite-grid revenue">'
        '<ul>' + ''.join(f'<li><a href="quests/{arm}/reports/dashboard/index.html">{arm} SciQuest dashboard</a></li>' for arm in ARMS) + '</ul>'
    )
    dump(output / "runtime.json", runtime_manifest())
    hashes = {str(f.relative_to(output)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(output.rglob("*")) if f.is_file() and f.name != "manifest.json"}
    dump(output / "manifest.json", dict(sha256=hashes, scope="Per-run artifact hashes. Timestamps, paths and runtime records may differ between reproductions."))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are refused")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed), indent=2))
