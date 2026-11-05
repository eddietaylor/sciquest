# Noisy pricing v1: prospective evaluation review

Source-only review before full-allocation outcomes. No allocation, notebook,
test, model or backend was run by this reviewer; no implementation was changed.
This is code-exposed critical-agent review, not human scientific validation.

## Coherent parts

The conditional oracle integrates hidden type and noise, while decision scoring
uses the observed context and one common finite action grid. Expected revenues
are price times the conditional mean. Campaign-level paired differences are
left minus right, with separate family summaries, complete-pair denominators,
sample SD/sqrt(n), and no pooled winner. The sum-loss ridge objective and
unpenalized intercept agree with analysis.py's normal equations; audit.py uses
an augmented least-squares solver as a genuinely different numerical route.

The generator gives paired observational/randomized potential units and identical
randomized training sets across estimators. Its family/component PRNG separation,
positive-demand bound, no clipping, independent evaluation units and within-unit
potential-outcome correlation match the written design. The tests independently
exercise the oracle equation and a forecast-error4/regret0 counterexample. That
counterexample distinguishes prediction from decisions; it is not uncertainty
calibration evidence. No performance ranking or nonzero result is required.

## Necessary bounded fixes before the full allocation

### 1. Bind independently scored files to supervisor execution

`audit.py:100–114` checks isolation flags, metric arithmetic and the evaluation
record, but never binds the prediction CSV's raw bytes to
`execution.prediction_artifact_hashes`. The top-level manifest is an internally
stored checksum list, not a substitute for the execution record.

Concrete counterexample: append a newline to predictions.csv, update its
manifest digest, and retain every numeric row and supervisor record. Current
numeric/ridge checks accept the changed CSV, although it is no longer the
recorded execution artifact. The existing negative test only changes the CSV
without updating its self-manifest, so it does not discriminate this boundary.

Required check: compare actual CSV raw SHA to the execution record and retain
that record's registration binding; require succeeded execution and valid
completion/evaluation binding for a completed slot. A read-only existing
integrity check can supplement the independent numeric calculation. Add the
same-numeric/updated-self-manifest negative. This does not require external
attestation or a new scientific acceptance threshold.

### 2. Bind baseline identity and exact grid to the allocation

`audit.py:119–130` chooses its independent feature matrix using the returned
model.json model label, rather than the baseline's frozen model specification.
An accidental linear implementation in randomized_quadratic, accompanied by
honest linear model.json/predictions, can satisfy this audit. Also, coefficient
comparison uses zip without a dimension check, allowing a truncated reported
coefficient list to escape that artifact check.

Required: derive the expected model/design from the allocation baseline; require
its declared model, ridge metadata and exact coefficient dimension to match.
Bind the executed model configuration to the actual registered notebook rather
than treating a returned self-label as the prescribed estimator.

`audit.py:105–106` verifies only total evaluation-row count. For example, 32
contexts with 18 distinct actions each have the prescribed 576 rows and pass
that count plus the duplicate-action guard, despite violating64×9. Require64
unique contexts and exactly the declared nine distinct prices in each; validate
training support against the declared five prices. These are finite schema/
allocation checks, not semantic validation of arbitrary claims.

### 3. Preserve evaluation availability separately from presentation failure

`run.py:138–148` sets completed and stores valid metrics before writing the
journal/dashboard; `:158–161` catches any subsequent exception and relabels the
slot execution_failed. A dashboard exception therefore excludes an already
successfully evaluated fit from completed-case scientific summaries and falsely
names its failure phase. No retry is needed or appropriate.

Required: retain actual execution/evaluation completion and a separately named
postprocessing/presentation failure, or explicitly prescribe whole-pipeline
completion while retaining a separate metric-availability flag and reporting its
denominator. Test a successful evaluation followed by a deliberate dashboard
failure. Do not silently alter the prespecified available-case estimand after
observing such a failure.

### 4. Make durability and read-only custody claims true

`run.py:28–29` dump uses plain write_text; incremental outcomes updates at
`:157/:162/:193` can leave truncated JSON if interrupted mid-write. Use the
existing atomic-writing utility for the durable slot inventory and commitments;
retain interruption without retry or resume.

`audit.py` calls get_record outside `database_reads_only()`. Current
`state.py:54–96,273–280` consequently opens BEGIN IMMEDIATE and schema-creation
transactions rather than fail-closed read-only access. Healthy databases may
remain byte-identical, but malformed/missing-table evidence can be initialized
before the audit rejects it. Wrap the audit in the existing read-only context.
An unchanged-tree before/after check and malformed-record refusal are sufficient;
there is no need to promise arbitrary filesystem-security guarantees.

### 5. Bind or qualify the frozen-summary claim

`audit.py:144–148` says it recomputes summaries from “exact frozen source”, but
imports summarize from the current working-tree module. Verifying archived
owner_source bytes does not prove those are the imported bytes. Bind the current
summarizer hash to the saved source commitment before use, or explicitly describe
this as a current-version recomputation and report that version. Do not execute
arbitrary archived code merely to satisfy the wording. The summary recomputation
is shared implementation consistency; hand-computable tests provide its separate
arithmetic check, not an independent statistical oracle.

## Scope and disposition

The proposed installed-wheel smoke test is a meaningful real workflow check,
including alternate training, actual isolation, complete IDs, journals and
reproduction. Its mocked slot-failure test is correctly limited to controller
continuation; it is not evidence of an actual failed notebook's full custody.
No full-allocation outcome or passing test result is asserted in this review.

The findings above are software/evidence-boundary fixes; none calls for a new
control bank, more campaigns, new baselines, agent calls, a desired ranking or a
performance threshold. Additional confidence intervals, larger seeds, noisy
families beyond the frozen four, and publication generalization studies remain
optional extensions. Re-review the bounded fixes and their actual test evidence
before presenting the 144-slot run as completely audited.

## Bounded source-fix re-review

Re-read final audit.py, run.py and test_noisy_pricing.py before the full allocation.
No executions/tests/imports were performed by this reviewer. The coordinator's
reported earlier six-test passes are not independently reclassified as the
pending final seven-test result.

The five named findings are resolved at source level:

1. Completed slots now require succeeded isolated execution bound to the actual
   registration; executed notebook and prediction bytes are checked against
   execution hashes. Registration content and every frozen snapshot file are
   rehashed. Completion/evaluation/registration hashes and metric roles are bound,
   and current raw train/eval CSVs match registered hashes. The integration test
   includes the discriminating newline change followed by a refreshed self-
   manifest: the authoritative execution hash must still reject it.
2. Model type comes from the allocated baseline; declared model, ridge, n, finite
   coefficients and exact dimension are validated. The registered analysis cell
   must equal frozen analysis plus the allocated MODEL/RIDGE; source and executed
   cells agree. The alternative least-squares matrix likewise uses the allocated
   model. Exact context count, per-context declared action grid, constant context
   features and allowed training support are checked. Synthetic tests cover wrong
   model, empty/extra coefficient dimensions, bad ridge/n/nonfinite coefficients,
   and the same-row-count/wrong-grid counterexample.
3. finish_slot retains completed scored results; present_slot is a separate
   operation. A dashboard/journal exception records reporting_error without
   changing scientific metric availability. The controller test now explicitly
   checks successful completion survives a dashboard failure, failed execution
   remains failed, later registered slots execute once, and preparation failure
   stays in the inventory. A completed metric allocation is therefore distinct
   from an assertion that every dashboard was generated.
4. dump now uses existing atomic_text for incremental JSON inventories and
   commitments. audit wraps its entire operation in database_reads_only. Missing
   authority is not created; the malformed-existing-empty-database negative uses
   query-only access and requires unchanged empty bytes. This closes the named
   write-mode audit concern without inventing a wider security guarantee.
5. Current summarize.py bytes must match the saved source commitment before its
   output is accepted. The report correctly relies on independent hand-computable
   summary tests as the separate arithmetic check; reuse of the summarizer is
   consistency evidence, not an independent statistical implementation.

Disposition: no concrete remaining blocker within the reviewed evaluation scope.
The planned final seven-test run and ordinary source/installed evidence review
remain the existing validation steps, not new gates requested by this critic.
Do not describe those pending results or the unexecuted 144-slot allocation as
already complete. The actual full run still requires failure-inclusive reporting,
raw saved-artifact audit and the declared scientific claim limits. No result is
required to favor a baseline or be nonzero; no additional campaign, model call,
control bank or universal semantic-validation requirement is introduced here.

## Completed-allocation outcome assessment

Read-only assessment of `/tmp/sciquest-noisy-pricing-full-v1`, its separate audit
JSON and the retained final test XML. No notebook, backend, allocation or project
verifier was executed by this reviewer. Standard-library file/CSV/hash/statistical
calculations supplied the checks below.

- Actual inventory and all 5,780 manifest hashes match. All 144 outcome rows are
  completed, none has reporting_error, and incremental/final outcome lists agree.
  Every reported metric is finite. The nine frozen owner-source files match
  current source and Git commit4fbcf5f exactly.
- Independently recomputed all six scores for every slot from the saved truth and
  prediction CSVs. There are no mismatches at the audit's declared tolerance.
  Independently recomputed every per-family/baseline mean, sample SD, MCSE and
  interpolated quantile, plus paired means/MCSE: all agree with results.json.
  The coordinator's separate audit records the additional augmented-ridge and
  authoritative custody checks across all144; that solver was not rerun here.
- Final retained XML has seven test cases and zero failures/errors/skips, with
  the real smoke-workflow test included. These are software checks, not seven
  scientific replications. The final no-bytecode setting protects archived helper
  imports from creating cache artifacts; no new scientific construct is claimed.

No outcome/report arithmetic inconsistency or claim-invalidating blocker was
found. The descriptive report correctly presents family-separated results,
paired complete/planned denominators, no pooled winner and no causal claim about
real markets, autonomous reasoning or philosophical policies.

Consequential interpretations to preserve:

1. Randomized linear has lower observed mean regret than observational linear
   in all four families. This is descriptive evidence for this recipe and fixed
   estimators, not a universal ranking or a statistical-significance claim.
2. More flexible prediction is not automatically better pricing. In the
   quadratic family, randomized quadratic reduces mean-demand RMSE from3.7679
   to1.5342, yet mean regret rises from0.8450 to1.4010. The paired regret
   difference is -0.5561 with MCSE1.0145. Do not turn the forecast improvement
   into a claim of improved decision quality.
3. In the threshold family, quadratic lowers mean regret from12.3636 to10.3022
   but has lower paired regret in only3/12campaigns. Its few larger gains can
   outweigh more frequent losses. “Consistently superior” would misdescribe the
   observed distribution; the existing table does not make that claim.
4. There are45exact-zero-regret slots while all144slots have nonzero mean-demand
   RMSE. This is a finite observed count, not an independent-replicate success
   rate. A coarse finite grid permits correct decisions despite prediction error.
5. Full and smoke profiles intentionally share four family/seed campaigns; their
   repeated deterministic reproductions are not new independent draws. Nothing
   here should be relabeled a blinded holdout or fresh human validation. The
   existing development/descriptive framing is appropriate; no replacement bank
   or additional allocation is requested.

One narrow wording clarification is warranted in README's sentence that oracle
means “avoid adding Monte Carlo noise to policy scoring”: they avoid extra
realized-outcome noise CONDITIONAL ON the64sampled evaluation contexts. Those
contexts remain a Monte Carlo sample, so their finite average still varies.
The campaign-level MCSE includes that variation along with world/training draws.
This is a scope clarification, not a scoring defect or reason to repeat results.

The remaining publication step is to preserve these actual full-run artifacts,
audit and review with their original hashes/statuses. Do not infer an observed
remote CI outcome from the local tests or claim validity beyond this transparent,
synthetic, descriptive benchmark.
