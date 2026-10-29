# Pricing example: simulator and identification review

Reviewed 2026-10-29, source plus the actual first completed run at `/tmp/sciquest-pricing-development-v1`. No implementation edits, program/model calls, new notebook execution or old-record mutations. Checks used file/CSV/JSON reads and immutable read-only SQLite. I previously critiqued the proposed DGP/design; I did not author this example implementation.

**Disposition:** no numerical or identification blocker for the stated prescribed, noiseless, two-action teaching example. Two presentation clarifications should be resolved before publication; neither changes the computed result.

1. `generate.py:28` places `z0`/`z1` in row IDs. A worker can read stratum from metadata even though `analysis.py` ignores IDs and uses only price/demand. The README correctly says stratum is omitted from *estimator inputs*, but any stronger hidden-stratum interpretation would be false. Use opaque IDs or explicitly disclose that ID metadata carries stratum and is not used by this prescribed estimator. This is not evidence that the actual OLS fit read hidden truth.
2. `plots.py:29-66` reads owner truth after evaluation and copies a cross-arm comparison into both experiment dashboards. Source comments and README disclose this, but the standalone figure/dashboard calls it merely “Comparison”/“Result Graphs.” Label the figure or attached dashboard narrative “Owner-side comparison after evaluation.” A standalone viewer should not infer that a worker accessed protected truth or produced this cross-arm analysis. Plotting does not enter the registered fit or supervisor metrics.

## Actual evidence and independent arithmetic

All 87 per-run manifest entries match current bytes. Both immutable SQLite supervisor databases contain registration/execution/evaluation/completion records. Registration hashes match `before_execution.json`; prediction CSV hashes match execution metadata; authoritative evaluation metrics equal published `results.json`. Both notebooks have completed code cells and retained successful owner-file/package exclusion probes. This checks actual local evidence, not universal sandbox security. I inspected the integration-test assertions but did not rerun the test or certify current CI/wheel results.

Direct CSV recomputation gives:

| Arm | Fitted intercept/slope | Population-mean predictions | Mean RMSE | Selected price | Protected value | Regret |
|---|---|---|---:|---:|---:|---:|
| Observational | −20, +2 | 40, 60 | 20 | 40 | 1600 | 200 |
| Stratified randomized | 120, −2 | 60, 40 | 0 | 30 | 1800 | 0 |

Each arm has eight training rows, the same price support {30,40}, disjoint training/evaluation IDs and a separate quest. Randomized cells balance the two Z strata at **each** price. Protected evaluation rows are the two population means, not individual outcomes; pooled-model unit deviations remain ±20 even when mean RMSE is zero. Predicted revenue is price times predicted demand; selection uses one population group and explicit lowest-action ties. The finite-grid oracle and zero costs/censoring/dynamics are stated. Both published `scientific_verified` values are false.

## Scientific limits and discriminating counterexamples

The observational twin `D=-20+2P` reproduces all observational cells but reverses intervention slope. Observing Z would not resolve that arm's absent within-stratum price variation. The revised common-price grid removes the earlier coverage/extrapolation confound. Conversely, even all randomized cells cannot distinguish the published DGP from `D+c(P-30)(P-40)` at the two registered prices; at P=35 their means differ by −25c. Thus these data establish neither a mechanism nor off-grid validity. The README/report appropriately restrict claims to the listed population interventions and prescribed estimator.

Plans, estimator, interpretation and expected arithmetic are known in advance. Registration is execution ordering and frozen evidence, not an unanticipated confirmatory discovery. Reproductions are software checks, not independent statistical replicates; no external reasoning, philosophy comparison, real-market conclusion or semantic-calibration claim is supported. The actual report preserves those limits.

Reviewed source SHA256: `generate.py` 1e2c299cad16831e2361eecf3f15c0423e8532b012ef9f65afe5d63ed918b81d; `analysis.py` 85b95fc1047de21c9d70a2e03975a75b377d684640d085e04942b02c2861a109; `run.py` 63e56d8c0079fabadcd7310aebfb20f3170e186b0dac4ae173452c85b0e2eac1; `plots.py` ad09e1eb932fe426a501c074935a21f200f98afd808b295906b279f7f6cd528c. Later edits require their own disposition.

## Source-only fix disposition, 2026-10-29

Both presentation findings are addressed in the current candidate source. `generate.py` now creates arm/unit IDs only after the row shuffle; it no longer serializes Z in the identifier. The fixed estimator remains unchanged and still reads only price/demand. This removes the direct ID disclosure, not a promise that an informed owner could never infer stratum from known deterministic outcomes. The integration test now checks eight rows and reconstructs owner-side P/Z cell counts: four each of (30,0)/(40,1) observationally, and two each of all four cells randomly. Owner-law arithmetic belongs in this independent check, not the analysis.

`plots.py` now visibly titles the figure “Owner-side comparison after supervisor evaluation”; the scripted report narrative identifies it as owner-side after both evaluations. The figure is still created only after both supervisor executions and never staged into either registered notebook. Generic dashboard method explanations explicitly avoid implying that a method was used; the next-step suggestion explicitly says it is a template, not an agent decision. These disclosures address the provenance ambiguity without changing the scientific estimand or scores.

No further concrete scientific blocker found in these changes. This disposition is source-only: I did not rerun tests or inspect a new completed snapshot. The original development-v1 evidence reviewed above remains unchanged; final source/wheel tests and the future publication snapshot need their own recorded results.

Current SHA256:
- `examples/pricing_confounds/generate.py`: `3895cf0fc0d6e87370d3c769a8772a5b322debfbf7999b71a03f4214567c078d`
- `examples/pricing_confounds/plots.py`: `8a188b8fee7acded22088715ad8b262b6ba6f9f601aaafdd761903aa05ba9c59`
- `examples/pricing_confounds/run.py`: `f4c220466da3aa166d69434881e459179a481b061d2a0651f569b5156db80dea`
- `examples/pricing_confounds/analysis.py`: `85b95fc1047de21c9d70a2e03975a75b377d684640d085e04942b02c2861a109`
- `examples/pricing_confounds/test_end_to_end.py`: `a403edcebef3df913a9cda36d5f2b9508b43465e1b18e6c4e51090577b1794bb`
- `src/sciquest/dashboard.py`: `1e53561ae283f418b89045378c6fa73efda13c56c8870d7d30e6e32207dc6b73`

## Final published-artifact audit, 2026-10-29

**No scientific publication blocker found within the stated teaching-example scope.** Independently read the final `docs/examples/pricing-confounds/run` snapshot and retained validation files. This was an inert review, with no new tests, notebook/native executions, model calls or old-evidence changes.

- All 87 manifest entries match. The complete 88-file published tree (manifest included) has exactly the same file set and bytes as `/tmp/sciquest-pricing-published-v1`.
- All eight example source-input hashes match `VALIDATION.json` (raw SHA256 `5fb21c6eaf8ead56ee5f7d7afbbdeea53f89686e4c92204d804c8f42934b81d4`). The actual wheel SHA256 is `282158dd914d900a39a9386a6236ad65400129da5c730640816d9298d859e28f`. Independently compared all 94 packaged files byte for byte across current source, wheel members and `/tmp/sciquest-example-wheel-v2/lib/python3.12/site-packages/sciquest`: no mismatch.
- Retained core XML records 185 tests, zero failure/error/skip; final external integration XML records two tests, zero failure/error/skip, 7.509 seconds. Those test sources execute two full reproductions and an altered-training discriminator; I reviewed their actual recorded results rather than rerunning them. The initial integration XML retains one failure and one pass; its failure is the literal dashboard label assertion `mean_demand_rmse`, consistent with the declared harness correction. This is local installed-wheel evidence, not a claim that remote CI passed.
- Final published databases, opened immutable/read-only, retain authoritative registration/execution/evaluation/completion records. Registrations match the before-execution barrier; evaluation metrics equal published results; prediction CSV hashes match actual execution records. Both source notebooks contain the exact unchanged `analysis.py` cell, and executed notebooks retain completed analysis/exclusion-probe outputs. Direct CSV recalculation again gives observational intercept/slope −20/+2, RMSE20, action40, value1600, regret200; randomized 120/−2, RMSE0, action30, value1800, regret0. Both `scientific_verified` fields remain false.
- Both final training tables contain opaque unit IDs, with no Z identifier. The randomized design has **four P/Z cells with two units in each: eight units total**, not eight cells. The observational design has two occupied P/Z cells with four units each. The final SVG explicitly says “Owner-side comparison after supervisor evaluation”; each actual experiment narrative identifies plotting as owner-side after both evaluations. Both earlier presentation findings are resolved in actual published artifacts, not merely candidate source.

The final README correctly distinguishes the local TeX-text fallback from a full MathJax typesetter, and AI-assisted development/review from zero scientific-agent campaign calls. No uncertainty calibration, external scientific-agent reasoning, causal mechanism discovery, off-grid validity, real-market performance, philosophy comparison or repaired private-review qualification follows from this reproduction. Those bounds and the counterexamples above continue to apply.
