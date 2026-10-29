"""Owner-side plots and commentary derived from actual executed artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def table(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def render(root: Path) -> None:
    results = json.loads((root / "results.json").read_text())
    colors = {"observational": "#bb4430", "randomized": "#2364aa"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.hashsalt": "sciquest-pricing-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    report = ["# Pricing under confounding: executed reproduction", "",
              "This is a prescribed, noiseless synthetic experiment. All hypotheses, designs, OLS code and commentary are scripted. No external agent was called.", "",
              "Metrics below are read from actual SciQuest supervisor evaluations, not filled from an expected-results table.", "",
              "| Training design | Fitted slope | Mean-demand RMSE | Selected price | Policy value | Regret |",
              "|---|---:|---:|---:|---:|---:|"]
    truth = table(root / "quests/observational/data/raw/eval.csv")
    prices = [float(r["price"]) for r in truth]
    means = [float(r["mean_demand"]) for r in truth]
    revenues = [float(r["expected_revenue"]) for r in truth]
    axes[0].plot(prices, means, "k--o", label="Protected intervention mean")
    axes[1].plot(prices, revenues, "k--o", label="Protected expected revenue")
    for entry in results["experiments"]:
        arm = entry["arm"]
        quest = root / "quests" / arm
        exp = quest / "experiments/exp_001"
        predictions = {r["id"]: r for r in table(exp / "worker_artifacts/predictions.csv")}
        y = [float(predictions[r["id"]]["predicted_mean_demand"]) for r in truth]
        rev = [float(predictions[r["id"]]["predicted_revenue"]) for r in truth]
        axes[0].plot(prices, y, "o-", color=colors[arm], label=f"{arm} OLS")
        axes[1].plot(prices, rev, "o-", color=colors[arm], label=f"{arm} prediction")
        chosen = entry["selected_price"]
        i = prices.index(chosen)
        axes[1].scatter([chosen], [revenues[i]], marker="*", s=190, color=colors[arm], zorder=5,
                        label=f"{arm} choice (actual value)")
        metrics = entry["metrics"]
        report.append(f"| {arm} | {entry['fitted_slope']:g} | {metrics['mean_demand_rmse']:g} | {chosen:g} | {metrics['policy_value']:g} | {metrics['regret']:g} |")
        plot_dir = exp / "artifacts/plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        # Both dashboards show the same explicitly owner-side comparison figure.
        # It is explanatory output after evaluation, never a worker input.
    for axis, title, ylabel in zip(axes,
                                  ["Population-mean demand", "Price selection on the registered grid"],
                                  ["Expected units", "Expected revenue (synthetic units)"]):
        axis.set(xlabel="Price", ylabel=ylabel, title=title, xticks=prices)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Owner-side comparison after supervisor evaluation\nSame estimator and prices; different assignment mechanisms", fontsize=12)
    fig.savefig(root / "comparison.svg", metadata={"Date": None})
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    for entry in results["experiments"]:
        destination = root / "quests" / entry["arm"] / "experiments/exp_001/artifacts/plots/comparison.svg"
        destination.write_bytes((root / "comparison.svg").read_bytes())
    report.extend(["", "## Interpretation and limits", "",
                   "The observational assignment entangles price with the simulator's demand stratum. Balanced stratified price assignment breaks that association in this constructed example. Both arms use eight units, identical price support and the same fitted estimator.", "",
                   "RMSE is over two exact population intervention means, not eight individual outcomes. Even the correct population-mean model leaves individual stratum deviations of ±20 units. No confidence interval, p-value, uncertainty calibration or performance distribution is estimated.", "",
                   "Policy value and regret use one common population price and the finite action set {30,40}, with deterministic lowest-price tie-breaking. They are not a sequential policy simulation, personalized policy or general market optimum.", "",
                   "Pre-execution registration supplies an auditable ordering and frozen inputs; the generator and expected results were already known to the authors. This is analytic reproduction rather than a fresh confirmatory finding. Integrity checks do not establish scientific novelty or general validity.", "",
                   "See each quest's journal.md, registered method_ledger.yaml, source/executed notebooks, worker_artifacts/predictions.csv, validation_results.yaml and reports/dashboard/index.html. Runtime and per-run hashes are in runtime.json and manifest.json."])
    (root / "report.md").write_text("\n".join(report) + "\n")
