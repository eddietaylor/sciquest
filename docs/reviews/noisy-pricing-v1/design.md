# Noisy pricing v1: prospective design review

Reviewed the implemented protocol, generator, estimator, runner, summaries,
independent audit, reporting and tests before the full allocation. Read-only
source review only: no allocations, imports, tests, notebooks or backend calls.
This file is the only write. No benchmark outcomes were inspected or used to
choose a model, sample size, metric or seed.

The revised scientific design addresses the earlier prospective gaps. Exact
equations, parameter ranges, assignment probabilities and stream identities are
specified; ridge uses summed loss with an unpenalized intercept. Within a
family/seed, all baselines share the world/evaluation units, both randomized
models share training data, and assignment designs share potential training
units. Training and evaluation use different component streams, conditional on
the common world. Evaluation integrates hidden type/noise for mean and policy
targets, while its separate realized outcomes retain unit variation. Shared
potential-price residuals are correctly treated as context dependence.

Per-family paired left-minus-right summaries use campaign-level SD/sqrt(n), not
action rows or a pooled family mixture. Planned/completed denominators and
available-case status are explicit. Twelve draws per family and 64 contexts per
draw are reasonable for this bounded descriptive development benchmark; this
review supplies no powered superiority claim or rationale to increase them.

Two concrete issues should be resolved before the full run:

1. **Postprocessing can select which scientific outcomes enter summaries.**
   `run.py:138–148` installs completed status and valid supervisor metrics before
   writing the logic report/journal and dashboard. `execute_slots` catches any
   later exception at `run.py:158–161` and changes status to `execution_failed`;
   `summarize.py:30` then excludes those metrics. Counterexample: dashboard
   generation raises after successful notebook/evaluation. The benchmark retains
   the slot and metrics but reports an execution failure and omits its usable
   primary endpoint. Separate scientific scoring availability from presentation
   status; preserve postprocessing failures without turning them into missing
   scientific measurements. A scripted dashboard-failure regression should
   verify both endpoint inclusion and honest presentation-error accounting.

2. **The independent fit audit trusts the model's self-description.**
   `audit.py:118–125` selects linear versus quadratic features using
   `model.json['model']`, rather than the frozen baseline assignment. It does not
   require the declared ridge, training count or coefficient-vector length to
   match that assignment. Counterexample: the randomized-quadratic slot emits a
   valid linear fit and labels it linear. The independent solver and scoring
   checks can agree, although the intended model comparison was not executed.
   Bind the artifact's exact model kind/ridge/n/coefficient length to the
   corresponding protocol baseline, with an assigned-quadratic/returned-linear
   negative test. Current runner source does pass the intended model; this is a
   demonstrated audit false-admission path, not evidence an actual fit is wrong.

The README describes noisy smoke/full CI jobs in the present tense, but the
inspected `.github/workflows/ci.yml` has no noisy-pricing job yet. Wiring may be
forthcoming; distinguish the implemented workflow and actual observed results
from prospective CI plans. This is a reporting/integration gap, not a reason to
expand the scientific allocation.

Other limits are retained rather than treated as new blockers. Unseen prices are
interpolation within the declared domain, conditional on shape assumptions: a
term proportional to the product of `(p-q)` over the five training prices changes
intermediate truth without changing any training response. Four declared family
recipes do not sample a population of scientific problems. Zero regret can still
coexist with prediction error; both forecast and policy metrics must be shown.
MCSE describes this generator's Monte Carlo mean, not real-market uncertainty;
with failures it describes the explicitly selected available-case set, not an
unbiased full-allocation performance estimate. Fixed scripts and same-model
development review demonstrate neither autonomous discovery nor independent
human scientific validation. No new optional allocation is requested.

The tests contain useful hand-computable paired-statistic and failure-denominator
cases, noisy-oracle/potential-unit checks, a wrong-forecast/right-decision example,
and a real installed-wheel smoke reproduction. Their source presence is not a
claim they have passed; execution evidence remains the coordinator's next check.

Reviewed raw SHA256 snapshot:

```text
protocol.json 41009f21dad22374997c40c9d63a637dec3929e41ccad6110a643ae02fe8865b
README.md     fced356a3835b90a0d0726533cef9b666334c3f400a630129ee4962d6e67b8dc
generate.py   eb95ec3062471276195c15d06bc5f2ee962a154c27609311988d0c8379ad99d7
analysis.py   d0e73baef8c89bbbb7634446c998d394a39d18b4a1a35aecaf9b9282a6eec641
run.py        967b6dcbefd27bd8391229b64d103cda484b0a7b9de3f695146090ea589fda51
summarize.py  60b9ebde2dbde3deae1caf7acf9289da262b89489fd65365d434d7c54161a2e1
audit.py      6f7e43bdceef9aca5aae96fed36a730d948e17670c433366a84cd3f5d3ec4a37
tests         debeab6c5f004970e10a17a863184f73b7b5cbae605dc33974b7e5242f63393d
report.py     ef34a14824466385956776597197056e6116c326b3d7a31f637912cd3f92c077
```

## Bounded fix re-review, before the full allocation

Both reported counterexamples are resolved on current source inspection; no
remaining concrete design blocker was demonstrated in this bounded re-review.

- `run.py:146–170` separates dashboard/journal presentation from scientific
  execution. A presentation exception now adds `reporting_error` while retaining
  completed status and metrics, so the existing summary includes that scored
  slot. The scripted failure test injects a dashboard error and checks completed
  status plus its retained error, alongside ordinary execution/preparation
  failures and no repeated calls. This resolves the specific presentation-based
  selection path without treating a missing dashboard as successful reporting.
- `audit.py:84–90` requires the allocated model kind, ridge penalty, sample count,
  exact coefficient dimension and finite non-boolean coefficients. The fit basis
  at `audit.py:191–197` now comes from the assigned baseline. It additionally binds
  the notebook's exact MODEL/RIDGE/analysis cell to that assignment and frozen
  source, with source/registration/executed-notebook commitments. The tests now
  reject assigned-quadratic/returned-linear, wrong ridge/count/dimension and NaN
  alternatives. Thus the earlier valid-linear-fit false admission is closed.
- The CI workflow now contains the noisy benchmark job: installed-wheel checks,
  two smoke reproductions, a full fixed allocation, separate audit and retained
  hidden supervisor evidence. The previous missing-job observation is resolved
  as a source-integration issue. It remains a prospective workflow; this review
  establishes no remote CI result or full-allocation outcome.

The added grid check also requires the declared number of contexts, the complete
price list in each context and one consistent x per context. Its negative test
keeps the same total row count while substituting a wrong context/action layout,
which discriminates a genuine omission from a mere row-count check.

The coordinator reports two earlier six-test smoke suites passed, each containing
two 12-slot reproductions. I did not rerun them or independently inspect their
retained result artifacts in this source-only re-review. Final seven-test and
full-allocation evidence remain separate, pending checks. Original finite-world,
available-case and scientific-claim limitations above still apply; no additional
allocation or expanded criterion is requested.

Fix re-review raw SHA256:

```text
run.py    6a5003473bca33b4a8a121ab39238cc9638d251bd002a914b4ae3917bc563c84
audit.py  df21f11308ac88cbea0f107170bd882477573e9d29822de6e6465d3653a00a46
tests     b2a7f873f55c56fbb474110f4a6fa81e235e83531391384bb080e5c7be6ea570
ci.yml    dca140964eeacf3c8b82fb2705987cbdf8f28b5e371662fa66c1a561337db04a
```

## Post-run interpretation review

Accepted the completed run's narrow descriptive interpretation. No concrete
result-interpretation blocker was found. Inert reads of
`/tmp/sciquest-noisy-pricing-full-v1` show all 144 planned rows completed, no
reporting errors, zero declared external-agent calls and no claimed genuine
external-agent reasoning. All 5,780 manifest entries rehash exactly, and frozen
owner-source commitments match their files. I independently recomputed every
per-family baseline summary's means, SDs, MCSEs and interpolated quantiles from
the retained outcomes, plus all eight paired regret means/MCSEs. These agree.
Four selected raw prediction/truth-table checks, covering both randomized models
at seed 1101 in the quadratic and interaction families, also agree with recorded
RMSE/regret and price-times-demand arithmetic. This is not an independent rerun
of the notebooks or an independent refit of all 144 models; the coordinator's
full augmented-ridge audit is separate evidence.

The scientifically consequential counterexample is retained, not explained away:
in the quadratic family, randomized quadratic ridge improves mean forecast RMSE
from 3.7679 to 1.5342 but increases mean regret from 0.8450 to 1.4010. The
prespecified linear-minus-quadratic regret contrast is -0.5561, with MCSE 1.0145.
For seed 1101 specifically, the raw tables give RMSE 4.33355 versus 1.27771 but
regret 1.23638 versus 4.24493. The linear fit chooses prices 40/45 across contexts;
the quadratic fit chooses 40 everywhere. Whole-grid forecast accuracy and errors
near a decision boundary need not rank models the same way. This case disproves
a claim that added flexibility necessarily improves pricing, even in the named
quadratic family; it does not establish that linear models are generally better.

Across the retained draws, observational-minus-randomized-linear mean regret is
positive in all four families (52.6272, 80.3489, 85.3501, 72.7387). Those are
family-specific averages, not a claim of improvement in every campaign. The
linear-minus-quadratic randomized mean differences are -0.0424, -0.5561,
17.0397 and 2.0614 for affine/quadratic/interaction/threshold respectively. These
eight prespecified contrasts should remain together; no pooled winning model or
post-hoc subgroup headline is warranted. MCSE is variability of this finite
Monte Carlo mean under the specified recipe, not a confidence interval or
guarantee that a small observed contrast persists in another allocation.

There are 45 exact-zero-regret slots but no exact-zero mean-demand-RMSE slots.
This is compatible with noisy finite training: finite-grid regret depends on
action ranking, while RMSE measures forecast error. The oracle evaluates exact
conditional expected revenue, not independent realized profit. The new benchmark
therefore supplies genuine finite-noisy-sample performance variation within its
declared generator, while retaining model-based oracle assumptions and a small
finite action grid. It establishes no real-market performance, unseen-family
generalization, uncertainty calibration, autonomous discovery or philosophical
comparison. The report makes these boundaries clear.

The coordinator reports final seven-test local evidence (41.09 seconds) and
pre-completion source freezes. This post-run review did not independently inspect
that XML or remote CI state, and adds no claim that they passed beyond that
reported evidence. No execution, allocation, backend call or scientific input was
changed; only this review was appended. No new study or acceptance criterion is
requested.

Reviewed run raw SHA256:

```text
report.md    e02bcb6a38743851850efddf43101a6a0146c44718352e6eab1e0b446ff65011
results.json 9c2b6804685daac9ea0a3329f0a4fa306d451c170b22be57595afec95f2623d4
```
