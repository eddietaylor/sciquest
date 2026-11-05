# Noisy pricing benchmark v1

Profile: **full**. Completed **144/144** planned slots.

Fixed scripted estimators; zero external-agent calls. No philosophy comparison or autonomous discovery demonstrated.

The supervisor computes regret, policy value, mean-demand RMSE and individual-outcome RMSE from protected labels. Seen/unseen-price errors are owner-side subset diagnostics. Figures are generated owner-side after evaluation.

Each campaign draws a fresh world and noisy finite training sample; evaluation uses independent contexts. The expected-revenue oracle marginalizes hidden type and noise, using only the context available to the fitted policy.

## Per-family campaign summaries

Available-case descriptive summaries; denominators are explicit. Missing slots prohibit a complete-allocation claim. No pooled winner, p-value, confidence interval, or row-level replication claim.

| Family | Baseline | Completed/planned | Mean regret | Regret MCSE | Mean demand RMSE | Individual RMSE |
|---|---|---:|---:|---:|---:|---:|
| affine | observational_linear | 12/12 | 53.3921 | 12.7295 | 4.9788 | 10.5360 |
| affine | randomized_linear | 12/12 | 0.7649 | 0.4832 | 0.9332 | 9.3533 |
| affine | randomized_quadratic | 12/12 | 0.8072 | 0.4580 | 1.4077 | 9.4367 |
| quadratic | observational_linear | 12/12 | 81.1938 | 19.4246 | 6.7172 | 11.6929 |
| quadratic | randomized_linear | 12/12 | 0.8450 | 0.4841 | 3.7679 | 10.2203 |
| quadratic | randomized_quadratic | 12/12 | 1.4010 | 0.9120 | 1.5342 | 9.5443 |
| interaction | observational_linear | 12/12 | 104.9664 | 11.6108 | 6.3799 | 11.7663 |
| interaction | randomized_linear | 12/12 | 19.6163 | 2.9219 | 2.9515 | 10.3490 |
| interaction | randomized_quadratic | 12/12 | 2.5766 | 0.5654 | 1.8453 | 10.1223 |
| threshold | observational_linear | 12/12 | 85.1023 | 15.3038 | 5.7609 | 10.9557 |
| threshold | randomized_linear | 12/12 | 12.3636 | 5.2132 | 2.8819 | 9.7315 |
| threshold | randomized_quadratic | 12/12 | 10.3022 | 4.6103 | 1.8763 | 9.4059 |

## Prespecified paired differences

Positive regret difference means the right estimator had less regret. Differences are left minus right within the same world/data campaign. MCSE = sample SD of paired differences / sqrt(complete pairs); unavailable for fewer than two pairs.

| Family | Left minus right | Complete/planned pairs | Mean regret difference | Paired MCSE |
|---|---|---:|---:|---:|
| affine | observational_linear − randomized_linear | 12/12 | 52.6272 | 12.7540 |
| affine | randomized_linear − randomized_quadratic | 12/12 | -0.0424 | 0.6044 |
| quadratic | observational_linear − randomized_linear | 12/12 | 80.3489 | 19.6407 |
| quadratic | randomized_linear − randomized_quadratic | 12/12 | -0.5561 | 1.0145 |
| interaction | observational_linear − randomized_linear | 12/12 | 85.3501 | 12.2609 |
| interaction | randomized_linear − randomized_quadratic | 12/12 | 17.0397 | 3.0799 |
| threshold | observational_linear − randomized_linear | 12/12 | 72.7387 | 14.9616 |
| threshold | randomized_linear − randomized_quadratic | 12/12 | 2.0614 | 1.1318 |

## Limits

Four hand-specified families and 12 draws per family characterize this recipe, not real markets or unseen response families. Intermediate prices test model-based prediction within the price domain; their means are not nonparametrically identified by training prices alone. Independent evaluation x can fall beyond the realized training x range. Noise is bounded and additive, prices have no capacity/cost/competition/dynamic effects. No tuning, replacement seeds, or outcome-based selection is allowed. Public-source transparency cannot guarantee a future external model has no prior knowledge of this benchmark.

A zero regret can coexist with forecast error. Both forecast and policy metrics must be inspected. MCSE describes the Monte Carlo mean under this recipe; it is not an interval or estimate of real-market uncertainty.

See `allocation.json`, `source_commitment.json`, `before_execution.json`, `outcomes.json`, `metrics.csv`, all per-slot journals/notebooks/dashboards under `quests/`, and `manifest.json`. `owner_source/` preserves the exact source used for this allocation.
