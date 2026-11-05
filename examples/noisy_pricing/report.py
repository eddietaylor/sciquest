"""Owner-side figures and report after scoring; never estimator inputs."""
from __future__ import annotations

import csv
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize import METRICS


def render(output: Path, results: dict) -> None:
    outcomes, summary = results["outcomes"], results["summary"]
    fields = ["slot", "family", "seed", "baseline", "status", *METRICS, "error_type", "error"]
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in outcomes:
            writer.writerow({k: r.get("metrics", {}).get(k, r.get(k, "")) for k in fields})
    families = list(dict.fromkeys(r["family"] for r in outcomes))
    baselines = list(dict.fromkeys(r["baseline"] for r in outcomes))
    fig, axes = plt.subplots(2, len(families), figsize=(4*len(families), 7), squeeze=False)
    for j, family in enumerate(families):
        for row, metric in enumerate(("regret", "mean_demand_rmse")):
            ax = axes[row, j]
            for i, baseline in enumerate(baselines):
                values = [r["metrics"][metric] for r in outcomes if r["family"] == family and
                          r["baseline"] == baseline and r["status"] == "completed"]
                xs = [i + (k-(len(values)-1)/2)*.025 for k in range(len(values))]
                ax.scatter(xs, values, alpha=.7, s=22)
                if values:
                    ax.plot([i-.2, i+.2], [sum(values)/len(values)]*2, color="black", linewidth=2)
            ax.set_title(family)
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_xticks(range(len(baselines)), [b.replace("_", "\n") for b in baselines], fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Owner-side benchmark results: dots = campaign draws; bars = means\nFixed scripts; no external scientific agent", fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(output / "comparison.svg")
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)
    lines = ["# Noisy pricing benchmark v1", "",
             f"Profile: **{results['profile']}**. Completed **{summary['completed']}/{summary['planned']}** planned slots.", "",
             "Fixed scripted estimators; zero external-agent calls. No philosophy comparison or autonomous discovery demonstrated.", "",
             "The supervisor computes regret, policy value, mean-demand RMSE and individual-outcome RMSE from protected labels. Seen/unseen-price errors are owner-side subset diagnostics. Figures are generated owner-side after evaluation.", "",
             "Each campaign draws a fresh world and noisy finite training sample; evaluation uses independent contexts. The expected-revenue oracle marginalizes hidden type and noise, using only the context available to the fitted policy.", "",
             "## Per-family campaign summaries", "",
             "Available-case descriptive summaries; denominators are explicit. Missing slots prohibit a complete-allocation claim. No pooled winner, p-value, confidence interval, or row-level replication claim.", "",
             "| Family | Baseline | Completed/planned | Mean regret | Regret MCSE | Mean demand RMSE | Individual RMSE |", "|---|---|---:|---:|---:|---:|---:|"]
    def number(v):
        return "unavailable" if v is None else f"{v:.4f}"
    for g in summary["groups"]:
        m = g["metrics"]
        lines.append(f"| {g['family']} | {g['baseline']} | {g['completed']}/{g['planned']} | {number(m['regret']['mean'])} | {number(m['regret']['mcse'])} | {number(m['mean_demand_rmse']['mean'])} | {number(m['individual_demand_rmse']['mean'])} |")
    lines += ["", "## Prespecified paired differences", "",
              "Positive regret difference means the right estimator had less regret. Differences are left minus right within the same world/data campaign. MCSE = sample SD of paired differences / sqrt(complete pairs); unavailable for fewer than two pairs.", "",
              "| Family | Left minus right | Complete/planned pairs | Mean regret difference | Paired MCSE |", "|---|---|---:|---:|---:|"]
    for p in summary["paired_comparisons"]:
        m=p["metrics"]["regret"]
        lines.append(f"| {p['family']} | {p['left']} − {p['right']} | {m['n']}/{p['planned_pairs']} | {number(m['mean'])} | {number(m['mcse'])} |")
    lines += ["", "## Limits", "",
              "Four hand-specified families and 12 draws per family characterize this recipe, not real markets or unseen response families. Intermediate prices test model-based prediction within the price domain; their means are not nonparametrically identified by training prices alone. Independent evaluation x can fall beyond the realized training x range. Noise is bounded and additive, prices have no capacity/cost/competition/dynamic effects. No tuning, replacement seeds, or outcome-based selection is allowed. Public-source transparency cannot guarantee a future external model has no prior knowledge of this benchmark.", "",
              "A zero regret can coexist with forecast error. Both forecast and policy metrics must be inspected. MCSE describes the Monte Carlo mean under this recipe; it is not an interval or estimate of real-market uncertainty.", "",
              "See `allocation.json`, `source_commitment.json`, `before_execution.json`, `outcomes.json`, `metrics.csv`, all per-slot journals/notebooks/dashboards under `quests/`, and `manifest.json`. `owner_source/` preserves the exact source used for this allocation.", ""]
    (output / "report.md").write_text("\n".join(lines))
    rows = "".join(f'<tr><td>{html.escape(r["slot"])}</td><td>{html.escape(r["family"])}</td><td>{r["seed"]}</td><td>{html.escape(r["baseline"])}</td><td>{html.escape(r["status"])}</td><td><a href="quests/{r["slot"]}/reports/dashboard/index.html">dashboard</a></td></tr>' for r in outcomes)
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Noisy pricing benchmark</title>'
        '<style>body{font:16px system-ui;margin:2rem;max-width:1500px}table{border-collapse:collapse}td,th{padding:.4rem;border:1px solid #ccc}img{max-width:100%}</style>'
        '<h1>Noisy pricing benchmark v1</h1><p>Fixed scripted baselines; zero external-agent calls. Owner-side report after evaluation.</p>'
        f'<p>Completed {summary["completed"]}/{summary["planned"]} planned slots. Profile: {results["profile"]}.</p>'
        '<p><a href="report.md">Research report</a> · <a href="results.json">Results and paired summaries</a> · <a href="metrics.csv">All outcome rows</a> · <a href="allocation.json">Frozen allocation</a></p>'
        '<img src="comparison.svg" alt="Campaign-level regret and prediction errors by family and baseline">'
        '<table><tr><th>Slot</th><th>Family</th><th>Seed</th><th>Baseline</th><th>Status</th><th>Evidence</th></tr>'+rows+'</table>')
