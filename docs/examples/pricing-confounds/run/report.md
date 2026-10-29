# Pricing under confounding: executed reproduction

This is a prescribed, noiseless synthetic experiment. All hypotheses, designs, OLS code and commentary are scripted. No external agent was called.

Metrics below are read from actual SciQuest supervisor evaluations, not filled from an expected-results table.

| Training design | Fitted slope | Mean-demand RMSE | Selected price | Policy value | Regret |
|---|---:|---:|---:|---:|---:|
| observational | 2 | 20 | 40 | 1600 | 200 |
| randomized | -2 | 0 | 30 | 1800 | 0 |

## Interpretation and limits

The observational assignment entangles price with the simulator's demand stratum. Balanced stratified price assignment breaks that association in this constructed example. Both arms use eight units, identical price support and the same fitted estimator.

RMSE is over two exact population intervention means, not eight individual outcomes. Even the correct population-mean model leaves individual stratum deviations of ±20 units. No confidence interval, p-value, uncertainty calibration or performance distribution is estimated.

Policy value and regret use one common population price and the finite action set {30,40}, with deterministic lowest-price tie-breaking. They are not a sequential policy simulation, personalized policy or general market optimum.

Pre-execution registration supplies an auditable ordering and frozen inputs; the generator and expected results were already known to the authors. This is analytic reproduction rather than a fresh confirmatory finding. Integrity checks do not establish scientific novelty or general validity.

See each quest's journal.md, registered method_ledger.yaml, source/executed notebooks, worker_artifacts/predictions.csv, validation_results.yaml and reports/dashboard/index.html. Runtime and per-run hashes are in runtime.json and manifest.json.
