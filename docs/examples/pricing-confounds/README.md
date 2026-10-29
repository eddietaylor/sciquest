# Executed pricing example

This directory preserves an actual run of the
[reproduction protocol](../../../examples/pricing_confounds/README.md), using a
built SciQuest wheel installed in a fresh pinned Python3.12 environment. It is
a prescribed, noiseless synthetic comparison with **zero external-agent calls**.

![Owner-side comparison generated after supervisor evaluation](comparison.png)

| Training design | Mean-demand RMSE | Chosen price | Expected revenue | Regret |
|---|---:|---:|---:|---:|
| Confounded observations | 20 | 40 | 1600 | 200 |
| Balanced randomized assignment | 0 | 30 | 1800 | 0 |

The observational hypothesis is contradicted and the balanced-design hypothesis
meets its registered thresholds. These statements concern exact population means
and the finite action set {30,40}. They are not novel scientific findings,
uncertainty-calibration results, real-market evidence, or an agent benchmark.

## Inspect the evidence

- [Executed report](run/report.md) and [machine-readable results](run/results.json)
- [Both registration hashes, captured before execution](run/before_execution.json)
- [Actual runtime and package provenance](run/runtime.json)
- [Per-run artifact hash manifest](run/manifest.json)
- [Observational journal](run/quests/observational/journal.md),
  [executed notebook](run/quests/observational/experiments/exp_001/executed_notebook.ipynb),
  [predictions](run/quests/observational/experiments/exp_001/worker_artifacts/predictions.csv),
  [supervisor metrics](run/quests/observational/experiments/exp_001/validation_results.yaml)
- [Randomized journal](run/quests/randomized/journal.md),
  [executed notebook](run/quests/randomized/experiments/exp_001/executed_notebook.ipynb),
  [predictions](run/quests/randomized/experiments/exp_001/worker_artifacts/predictions.csv),
  [supervisor metrics](run/quests/randomized/experiments/exp_001/validation_results.yaml)

Complete raw supervisor databases and frozen snapshots are retained under each
quest's `.sciquest/` directory. Paths and interpreter locations in those records
describe the original execution. Reproduce into a **new** output directory using
the protocol rather than attempting to rewrite or resume this saved run.

## View the actual dashboards

GitHub displays HTML files as source. After cloning, serve this saved output:

```bash
python -m http.server 8000 --directory docs/examples/pricing-confounds/run
```

Open `http://localhost:8000/`; its overview links both generated SciQuest
dashboards. HTML sources are also available directly:
[observational](run/quests/observational/reports/dashboard/index.html) and
[randomized](run/quests/randomized/reports/dashboard/index.html).
Plots were assembled owner-side after both supervisor evaluations and are
labeled accordingly. Equation text uses the bundled offline TeX fallback.

## Verification and critical review

[Local verification](VALIDATION.json) records 185 passing core tests and two
passing installed-wheel integration tests. The latter run the entire research
workflow twice from outside the checkout and independently recalculate metrics
from complete prediction/truth tables. All 94 source/wheel/installed package
files matched. An initial integration assertion expected underscores in a
human-readable dashboard label; its failed log is retained alongside the
corrected passing result. No application result was rewritten to fix that test.

See the [scientific review](../../reviews/pricing_simulator.md),
[design review](../../reviews/pricing_design.md),
[evaluation review](../../reviews/pricing_evaluation.md), and
[issue dispositions](../../reviews/pricing_issues.json). These are code-exposed
critical-agent reviews, not independent human scientific validation. A
metric-tolerance objection was withdrawn after checking pytest's actual behavior;
review agreement was not treated as proof.

The GitHub Actions `Installed-wheel scientific example` job independently runs
the reproduction and uploads its own artifacts. Local validation records describe
the local run; they do not assert a remote CI outcome before one exists.
