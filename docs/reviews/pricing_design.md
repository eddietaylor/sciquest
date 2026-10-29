# Pricing example: design review

Reviewed the publication worktree's generator, analysis, runner, plotting,
README, installed-wheel integration test and CI workflow. Read-only source and
inert CSV/JSON inspection; no model calls, notebook execution, test reruns or
historical changes. This review file is the only write.

No remaining scientific-design blocker was demonstrated. The source implements
the claimed narrow example: two prescribed noiseless designs, the same univariate
OLS estimator and sample size, matched prices 30/40, protected population-mean
evaluation and a finite listed-action policy. It is a deterministic reproduction,
not a new scientific discovery or demonstration of agent reasoning.

Independent arithmetic from `/tmp/sciquest-pricing-development-v1` training,
prediction and truth CSVs agrees:

| Arm | Intercept | Slope | RMSE | Price | Value | Regret |
|---|---:|---:|---:|---:|---:|---:|
| Observational | -20 | 2 | 20 | 40 | 1600 | 200 |
| Randomized | 120 | -2 | 0 | 30 | 1800 | 0 |

Concrete design/implementation checks:

- `run.py:52` correctly gives the decision evaluator **predicted revenue**,
  separately from demand predictions. The evaluator maximizes that column; it
  does not multiply demand by price itself. On this particular two-price example,
  demand and revenue happen to rank actions identically; that coincidence is not
  the general revenue criterion. For example means `[60,50]` at prices `[30,40]`
  rank demand and revenue differently. The actual configuration uses revenue,
  and actual value/regret are read from supervisor evaluation.
- `run.py:115` prepares both separate quests and their actual registrations before
  the first `execute_registered`. Each immutable snapshot has only its own
  training dataset. The known analytic expectations are disclosed, plans are
  exploratory, and registration is not presented as an unknown-result guarantee.
- `analysis.py` fits coefficients only from staged training price/demand; the
  changed-training check discriminates a hardcoded expected fit. The runner does
  not read `expected.json` to generate results. Fitted slope remains an
  execution-bound artifact diagnostic rather than a fictitious generic metric.
- `black_box_v2` selects the allowlisted worker runtime. The actual notebook
  probes owner truth, repository generator and supervisor package exclusion,
  while the estimator checks exact training/evaluation feature columns. These
  are bounded checks of this worker boundary, not secrecy from authors/readers
  or proof of universal sandbox security. Registered plans/validation are public
  worker context; this is not hidden-answer discovery.
- RMSE alone is primary; value/regret remain diagnostics. The joint predeclared
  recovery/zero-regret criterion legitimately fails in the observational arm.
  Two protected population-mean rows are not eight individual outcomes, and
  duplicate noiseless strata/reproductions do not supply statistical inference.
- Plots and commentary use actual artifacts after evaluation and are identified
  as owner-side/scripted. Journals, dashboards, registration and scientific
  validity are distinguished. No private-review-service or philosophy-comparison
  qualification follows from this example.

Correction to the initial review: the objection that
`pytest.approx(..., abs=1e-10)` retains its default relative tolerance was wrong
and is withdrawn; the original metric checks already enforced the specified
absolute tolerance. The original coefficient checks omitted an explicit tolerance
and did use the default relative allowance. Current checks consistently use
`rel=0, abs=1e-10`, including coefficients. Final execution results remain separate.

The subsequent bounded fixes are coherent on source review: training IDs are
assigned after shuffling and no longer encode Z; the comparison plot visibly
states that it is owner-side after evaluation; generic glossary definitions are
conditional rather than statements that every experiment used those methods;
next-experiment text is labeled a template suggestion, not an agent decision.
The integration test also checks actual registration timestamps against the
first execution and independently reconstructs the intended price/Z cell counts.
Two existing dashboard-test substring assertions still name the old glossary
wording and need their expected text aligned; this is a test-harness issue, not
a new scientific requirement. Final core/wheel/end-to-end outcomes are pending.

CI source builds/installs a wheel into a fresh environment, runs outside the
checkout with `PYTHONPATH` removed, checks installed provenance, executes the
actual workflow twice and independently recomputes metrics from raw tables.
It checks immutable registrations, generator replay, executed notebooks, artifact
IDs, journal/dashboard/plot output and refuses overwriting an existing run.
Passing local development output is not evidence the newly added GitHub job has
passed: preserve/report the actual final source/build/fresh-wheel/CI outcomes.

Scope counterexample: adding `k*(p-30)*(p-40)` to the demand law leaves every
listed training/evaluation response unchanged while altering unlisted prices.
Thus these two actions cannot establish a response curve or a general continuous
optimum. The README already confines its claims to the registered grid and known
affine teaching world; no broader identification claim is warranted.

Reviewed key raw SHA256:

```text
run.py          f4c220466da3aa166d69434881e459179a481b061d2a0651f569b5156db80dea
generate.py     3895cf0fc0d6e87370d3c769a8772a5b322debfbf7999b71a03f4214567c078d
analysis.py     85b95fc1047de21c9d70a2e03975a75b377d684640d085e04942b02c2861a109
plots.py        8a188b8fee7acded22088715ad8b262b6ba6f9f601aaafdd761903aa05ba9c59
test_end_to_end a403edcebef3df913a9cda36d5f2b9508b43465e1b18e6c4e51090577b1794bb
README.md       3550c17799abd34d752b8e14f08bf6caa115568b6efd7b4c2fb79d06ac8b5714
ci.yml          a4eadde67e8e3a861f74a74e414a158c7c69b2c6768cec388b543efde3d86f64
```

Same-model development review and matching arithmetic are not independent human
scientific validation. No new optional study, model allocation or old-record edit
is required by this disposition.

## Final local evidence audit, 2026-10-29

The earlier pending-test statements above are superseded by this result audit.
No remaining concrete publication-design blocker was found. This was an inert
standard-library XML/JSON/CSV/SQLite-read-only/ZIP/hash inspection; no tests,
notebooks, package verifiers or backend calls were run.

- Retained core XML contains 185 test cases, zero failures/errors/skips; its log
  reports 54.59 seconds. Final installed-wheel integration XML contains two test
  cases, zero failures/errors/skips; its log reports 7.51 seconds. The earlier
  integration XML still records one failure and one pass, specifically the
  dashboard-label harness mismatch. This is not an all-attempts-clean claim.
  The two stale core dashboard assertions have also been aligned in the passing
  final source run. The metric-tolerance objection remains withdrawn.
- All eight `VALIDATION.json` source-input hashes match current files. The wheel
  raw SHA256 matches the report; all 94 package files are byte-identical across
  current source, wheel ZIP and the named installed environment. These are local
  environment checks, not evidence of a completed remote GitHub matrix.
- Every one of the published run's 87 manifest entries matches both its copied
  artifact and the original `/tmp/sciquest-pricing-published-v1` artifact. Both
  actual integration reproductions remain present under
  `/tmp/sciquest-pricing-integration-v3`; each has 87 matching manifest entries.
  Their scientific JSON, training tables, truth tables and prediction tables
  match one another and the published run byte-for-byte. Full-directory identity
  is correctly not promised, because runtime evidence includes paths/timestamps.
- Actual read-only supervisor records bind the barrier's two registration hashes
  to execution, evaluation and completion. Every registered file hash matches
  its frozen snapshot; executed-notebook and prediction hashes match execution
  records. Both registration timestamps are no later than the earliest notebook
  start; second-resolution timestamps alone cannot establish a strict ordering.
  The inspected runner supplies that ordering with both registrations and its
  durable no-execution barrier before the execution loop.
- Independent arithmetic from the published raw tables again yields OLS
  `(-20, +2)` and `(120, -2)`, RMSE `20/0`, selected prices `40/30`, values
  `1600/1800` and regrets `200/0`. Eight training rows have precisely the declared
  price/stratum cell counts. Actual evaluations retain primary RMSE and diagnostic
  value/regret, exploratory status and the observational negative outcome.
  Both executed notebooks have completed code cells, no error outputs, the
  successful boundary-probe message and the exact reviewed analysis source.
  Execution records specify isolated, blind-filesystem, network-disabled runs.

The added CI workflow is a source-reviewed plan: Python 3.10/3.11/3.12 core jobs
and a separate Python 3.12 installed-wheel reproduction job. Local evidence here
is Python 3.12.3 only; no remote CI result has yet been observed. Two tests include
two complete workflow reproductions plus a changed-training estimator check;
they are not two independent scientific replications or agent trajectories.
The published report retains zero external-agent calls and
`scientific_verified=false`. Known analytic expectations, same-model development
review and matching deterministic outputs establish no novel discovery,
philosophical superiority or repair of the historical reviewer-service failure.

Final evidence raw SHA256:

```text
VALIDATION.json 5fb21c6eaf8ead56ee5f7d7afbbdeea53f89686e4c92204d804c8f42934b81d4
core.xml        8914db9ee4efd784df4b7fb371be921fc70c9c1bc7215a061668018dcce14843
integration.xml d7a18635e9b57a2cd361cb0a69c004c7bc1a58a221702fa21362eafef6d9d6ec
initial.xml     f9994d64b83bc21bbbe2fe2b5a0a1cc443a1b9026b7ef91075c254d49e72125e
wheel           282158dd914d900a39a9386a6236ad65400129da5c730640816d9298d859e28f
final test      305fb558358d289fd45df5595221271224dc9549afb5af3a644508c53737f7b1
final README    96c6545f78d4de675174b31983a2625f9ec5449bb7b67cbcc1fd7c5f65b73d0a
```
