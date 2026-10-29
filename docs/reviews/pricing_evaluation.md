# Pricing example: evaluation review

Reviewed source and `/tmp/sciquest-pricing-development-v1` on 2026-10-29.
This is a code-exposed agent review, not independent human validation. No model,
agent, notebook, test or backend was executed by this reviewer. Standard-library
CSV/Fraction/JSON/hash inspection and read-only SQLite SELECTs were used.

## Concrete finding before publication

The development dashboards unconditionally include glossary statements that
“Ridge regression is the actual prediction model” and “The model predicts the
logarithm of demand.” Neither describes this example's price-only, unregularized
OLS on raw demand. The rank-aware-objective entry similarly describes an unused
metric. Evidence: `src/sciquest/dashboard.py:20–48`, unconditional glossary loop
at `:120`, inclusion at `:883`, and actual observational dashboard's
`operator-explanations` section. This is misleading run narration, even though
the example's report and Task + Model panel are correct.

Narrow correction: make these descriptions explicitly generic/conditional and
label them as definitions that may not apply, or show only applicable terms.
A regression should verify that a raw-demand OLS dashboard does not assert that
its model uses ridge/log-demand/rank-aware scoring. No scientific rerun or new
research gate is needed for that presentation correction.

## Independently corroborated development outcome

- All 87 entries in the run's manifest match retained file bytes.
- Both arms have eight training units and identical price support {30,40}.
  Exact Fraction OLS on actual CSV rows gives observational intercept/slope
  -20/+2 and randomized 120/-2. The balanced assignment is two units per
  price/stratum cell; the generator omits stratum from estimator inputs.
- Truth means are 60 and 40, with revenues 1800 and 1600. Actual observational
  predicted means/revenues are (40,1200) and (60,2400), selecting price40:
  RMSE20, value1600, regret200. Randomized predictions equal the two means and
  choose price30: RMSE0, value1800, regret0.
- Read-only supervisor records contain registration, execution, evaluation and
  completion for each arm. Recorded metrics equal the reported values;
  completion hashes bind the evaluations, and execution hashes bind prediction
  CSV bytes. Source control flow registers both arms before either execution;
  retained timestamp ordering is consistent (second-resolution ties alone are
  not proof of strict cross-quest ordering).
- Executed analysis cell source exactly equals `analysis.py`; both code cells
  have execution counts. Original JSON notebook sources are arrays of strings:
  joining them recovers the exact cell text. Filesystem probes and retained
  execution records support the stated bounded owner-file/package exclusion.
- Report, journal and README distinguish scripted authorship, empirical
  prediction from intervention truth, population means from individual outcomes,
  finite-grid regret from a continuous/sequential optimum, and integrity from
  scientific validity. No statistical or autonomous-discovery claim is made.

## CI, packaging and remaining evidence boundary

The proposed job installs a built wheel into a fresh pinned Python3.12 venv,
runs from outside the checkout with PYTHONPATH removed, and asserts the imported
package is under that interpreter's prefix. Its two full reproductions exercise
actual generator replay, registration, allowlisted isolated execution, evaluation,
journaling and dashboards. Raw-table arithmetic is independent of SciQuest's
metric implementation, and the altered-training test discriminates fitted
coefficients from copied expectations. Existing-output refusal preserves a run.
The upload explicitly includes hidden supervisor records. Matplotlib is declared;
Google font imports were removed and local MathJax is retained.

These are source-reviewed checks and one inspected development run. They are not
a completed installed-wheel CI result or a published artifact inventory. Review
the actual final wheel-run snapshot and test evidence before stating those
outcomes. Keep generated plot/report/journal/dashboard bytes with their actual
supervisor records; do not present a handwritten expected-results table as an
executed artifact. Ignore-rule `quests/` also matches nested published quest
directories, so the final Git-tracked inventory must actually retain the intended
published evidence rather than assuming recursive git add included it.

Scientific reproduction means fixed generated tables/predictions/results, with
per-run timestamps/runtime and plotting variation explicitly distinguished. The
example does not qualify experimental demand-assessment code merely because the
runtime contains it. Publication omits unrelated historical campaigns appropriately.
No additional scientific gate or agent call is requested by this review.

## Correction to an early review message

I initially alleged that `pytest.approx(..., abs=1e-10)` still used default
relative tolerance. That was incorrect: pytest's actual ApproxScalar.tolerance
returns the supplied absolute tolerance when rel is omitted. The existing metric
checks therefore implement the documented absolute tolerance; that objection is
withdrawn. Coefficient-only default tolerances do not invalidate the independently
checked metrics. An initial direct cell-source comparison also compared a JSON
string-array to a string; the corrected exact comparison joins the array and
passes. Neither false probe is retained as an implementation defect.

## Source-fix re-review

The revised glossary now says “In workflows using this estimator” for ridge,
“A model using log-demand” and “A rank-aware objective”; it explicitly labels
itself “Generic glossary” and directs readers to recorded methods/metrics.
This resolves the demonstrated false run-narration issue at source level.
Template recommendations are explicitly not agent decisions. Example commentary
now identifies the comparison figure as owner-generated after evaluation.
Opaque training IDs avoid embedding the omitted stratum as a trivial identifier.
The metric/coefficient assertions now consistently specify rel=0,abs=1e-10;
this is consistency, not repair of the withdrawn abs-only metric objection.

The new end-to-end assertions reject the old false ridge/log statements and
require the generic/template labels. Two old test_dashboard expectations remain
stale in the source inspected for this addendum: the exact old “Some observations
are more likely to appear...” sentence at line233 and “not only judged by
absolute error” at line242 no longer occur in the replacement descriptions.
Update those expectations to the actual generic wording before reporting a
passing suite. No test was run by this reviewer. Final installed-wheel results
and final published snapshot remain pending this separate source disposition.

## Final local result and publication-artifact review

Inspected `docs/examples/pricing-confounds/VALIDATION.json`, retained XML/logs,
final example source, final wheel/installed bytes and the saved `run/` artifacts.
No tests, model calls or executions were performed for this review.

- Retained XML independently contains 185 core cases and 2 installed-integration
  cases, each with zero failures/errors/skips. The initial integration XML has
  two cases with exactly one failure: the old underscore dashboard-label
  assertion. Its log shows that assertion rather than a metric/execution error.
  The final integration source exercises two complete reproductions plus the
  altered-training check; 2 passing tests are not two scientific replications.
- Wheel SHA256 is
  `282158dd914d900a39a9386a6236ad65400129da5c730640816d9298d859e28f`.
  All 94 actual packaged files, including the TeX fallback JavaScript, match
  source and the declared installed package byte-for-byte. The 93 Python/YAML
  files alone would not account for the full 94-file statement. All eight
  example source-input hashes in VALIDATION.json match current files.
- All 87 saved run-manifest entries match. The top-level published comparison
  PNG equals its saved-run counterpart. Actual saved supervisor metrics are
  20/1600/200 for observational RMSE/value/regret and 0/1800/0 for randomized;
  prediction CSV hashes match each execution record. Both executions record
  the same package identity and allowlisted filesystem mode.
- Both final dashboards contain the generic applicability notice and omit the
  false actual-ridge statement; the figure labels owner-side authorship. The
  earlier stale dashboard test expectations were updated to new generic text.
  The substantive presentation finding is resolved and has passing local
  regression evidence.
- Published documentation discloses scripted scientific authorship, zero
  experimental external-agent calls, AI-assisted development, exact/noiseless
  finite-grid scope, TeX-text fallback rather than full typesetting, and the
  separation of local validation from remote CI. These are appropriately bounded
  claims. Source/installed agreement is software evidence, not scientific proof
  or validation of the unrelated experimental assessment modules.

No material remaining implementation or claim blocker was found within this
review's scope. Before the publication commit, include the intended nested
`run/quests/` and hidden `.sciquest` records despite ignore rules; the coordinator
has explicitly planned that force-add. This review verifies working-tree saved
bytes, not a yet-uncreated Git commit or a remote CI run. Do not claim observed
GitHub Actions success until its actual run completes.

Reviewed VALIDATION.json SHA256:
`5fb21c6eaf8ead56ee5f7d7afbbdeea53f89686e4c92204d804c8f42934b81d4`.
