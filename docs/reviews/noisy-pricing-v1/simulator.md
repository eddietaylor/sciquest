# Noisy pricing v1: pre-full-run simulator review

Source-only review, 2026-11-05. No benchmark allocation, backend/model call, estimator execution or test run; no implementation or historical-artifact edits. I reviewed the earlier design and deterministic example, and authored earlier unrelated instrument interfaces. This is a bounded development critique, not blind human scientific validation.

**Scientific disposition:** the generator, marginal oracle and fixed estimator now implement the proposed noisy descriptive task coherently. No demonstrated DGP/oracle numerical defect or staged truth disclosure found. Two audit requirements and one machine-readable scope clarification should be resolved before publishing full-run audit assurances.

## Required corrections

1. **Enforce the advertised read-only audit.** `audit.py:88,95,110` calls `state.get_record` outside the existing `database_reads_only()` context. In `state.py:273`, missing DB files correctly return None; my early message suggesting a missing file would initialize was incorrect. An *existing zero-byte/uninitialized* SQLite file in a retained failed quest passes that file guard, however, and the default `transaction` (`state.py:53-110`) creates tables and seeds state. This can change bytes after the audit already checked the manifest. Healthy authority reads also use a writer-capable `BEGIN IMMEDIATE`, not query-only mode. Smallest repair: wrap authority reads in the existing read-only context; missing/incomplete failed-slot authority must remain unavailable or raise without initialization. Discriminating regression: a failure-inclusive saved allocation containing an uninitialized authority file must not acquire new schema, state, files or changed bytes during audit. This is a source-derived counterexample, not an executed reproduction.

2. **Bind fitted-artifact checks to the allocated estimator and exact shape.** `audit.py:118-127` selects linear versus quadratic features from `model.json`, and compares coefficients with truncating `zip`. Replacing its coefficients list by `[]`, while refreshing the unanchored per-run manifest, skips every coefficient comparison; prediction checks still use independently recomputed beta and can pass unchanged. Extra trailing coefficients are similarly ignored. The model's declared `n` and ridge tag are not checked. Smallest repair: derive expected model/features from the frozen baseline declaration; require exact model, penalty, sample count, finite coefficient list and length (3 or6), then compare all coordinates. An empty/truncated/extended vector or wrong model/ridge/n must reject. This does not allege that actual fits used the wrong coefficients; it limits the current claim that audit independently verifies the retained fitted artifact. A self-generated manifest is integrity evidence against accidental changes, not external authenticity.

3. **Align the machine-readable extrapolation statement.** `protocol.json:73` still says “No extrapolation” without qualification. README correctly explains that fresh x can exceed finite training extrema, and intermediate-price means require model assumptions. State no price-domain extrapolation/new-family generalization claim, while retaining empirical-context-range and intermediate-price limitations. No new experiments are needed.

## Source-confirmed scientific repairs

`generate.py:21-37` declares independent world ranges and an exact conditional mean m(p,x). Hidden residual is L(Z−1/2)+sigma*e, with independent balanced Z and uniform e. Its conditional expectation is zero, so the oracle marginalizes variables unavailable to the policy rather than rewarding a latent-aware policy. The full-domain lower bound is conservatively 108−52−6−20−10−10=10, because at most one family-specific reduction is active. Thus there is no clipping/truncation bias. Observational softmax assignment is strictly positive at all five prices for both latent types; that positivity does not remove latent confounding.

SHA-derived streams include family/seed/component. Observational and randomized samples share training x/Z/e units and differ only in assignment; evaluation uses a distinct stream. One evaluation unit's Z/e are shared across its nine potential prices, which is an explicit valid dependence structure. The exact oracle is m(p,x), while individual-outcome RMSE uses a separate noisy realized label. Physical disturbance variance L²/4+sigma²/3 remains nonzero even for an exact mean predictor. These repairs avoid the old deterministic two-price benchmark's forced exact mean recovery.

`analysis.py` fits the declared sum-loss ridge with intercept unpenalized, using only staged training x/price/demand and evaluation ID/context_id/x/price. There are no simulator equations, family/seed, expected coefficients or oracle constants in the fitted source. `run.py` gives slots opaque IDs, stages black_box_v2 notebooks with explicit feature whitelists, and supplies generic plans/contracts rather than the owner generator. Its path/package probes are bounded checks, not universal security or absence-of-model-memory proof. Current source supplies no external agent. Actual execution of these boundaries remains to be evidenced by retained installed runs.

## Scope and verification limits

Full allocation is four fixed families ×12 seeds ×3 baselines =144 slots. Source retains planned/running/failure rows, continues ordinary failures and uses a before-notebook registration barrier. Summaries separate supervisor metrics from owner-only seen/unseen-price RMSE, pair within campaign and report available-case denominators rather than treating action rows as independent replicates. I inspected test assertions; no test outcomes are certified here.

Intermediate-price identification remains assumption-dependent: adding c times the product of (p−q) over all five training prices leaves every possible training row unchanged but alters intermediate-price means. An exact mean oracle does not certify discovery of that law. Zero regret can still accompany prediction error, and nonzero sampling/misspecification error is not required in every campaign. The four constructed families, bounded physical noise and fixed support do not establish broad market, unseen-family, adaptive-policy, philosophical or autonomous-discovery performance. Additional worlds/baselines/CIs/model calls are optional extensions, not required repairs.

Reviewed source SHA256 (candidate state; subsequent repairs need a separate disposition):
- `examples/noisy_pricing/generate.py`: `eb95ec3062471276195c15d06bc5f2ee962a154c27609311988d0c8379ad99d7`
- `examples/noisy_pricing/analysis.py`: `d0e73baef8c89bbbb7634446c998d394a39d18b4a1a35aecaf9b9282a6eec641`
- `examples/noisy_pricing/run.py`: `6a5003473bca33b4a8a121ab39238cc9638d251bd002a914b4ae3917bc563c84`
- `examples/noisy_pricing/README.md`: `fced356a3835b90a0d0726533cef9b666334c3f400a630129ee4962d6e67b8dc`
- `examples/noisy_pricing/protocol.json`: `565614592de282f3ab1d252d5ae625056aea3bb2a0877429fb3c044b5ed0f0fd`
- `examples/noisy_pricing/audit.py`: `49fbd3112607bcb99504ee3e5b43afe660717d0027cba01eab5c13ec053cbb99`
- `examples/noisy_pricing/test_noisy_pricing.py`: `468bdbb83f87bb62ea0f0e4a2a3b32bca4880d13330d5c9c0999120012c8fe91`
- `examples/noisy_pricing/summarize.py`: `60b9ebde2dbde3deae1caf7acf9289da262b89489fd65365d434d7c54161a2e1`

## Bounded source re-review, 2026-11-05

The three requested corrections are addressed in the current candidate. No remaining scientific/source blocker found within this review's bounded scope; no full allocation or test results are certified here.

- `audit()` now enters `database_reads_only()` for the complete audit. Missing authority returns None; existing malformed authority is read without initialization. The new regression distinguishes these cases and requires a zero-byte DB to remain unchanged after its read-only failure. This repairs the corrected existing-uninitialized-file counterexample; it does not revive my erroneous initial missing-file counterexample.
- `validate_model` binds model name, ridge tag, sample count, coefficient-list length and finite nonboolean coordinates to the allocated baseline. Feature construction derives from that baseline; exact length makes the later zip comparison complete. Tests reject wrong basis, empty/extended vector, wrong ridge/n and nonfinite values. The audit also binds source notebook bytes to registration, checks its actual estimator cell against frozen analysis plus allocated model/penalty, and requires executed/source cell text agreement. The earlier fitted-artifact verification holes are closed at this declared representation level, without a universal semantic-fidelity claim.
- The protocol now restricts the claim to no **price-domain** extrapolation and explicitly admits fresh x beyond finite training extrema. This matches the README and retains the intermediate-price nonidentification limitation.

Additional bounded checks are sound: exact context count and full declared action set per context replace an insufficient total-row-count check; raw prediction hashes bind to authoritative execution; raw training/evaluation bytes bind to registered snapshots; completion/evaluation/registration hashes are checked. The independent augmented ridge solver and separate supervisor/owner metrics remain intact. A separate reporting-error field now preserves completed scientific metrics when later dashboard generation fails, rather than relabeling that as a fitting failure.

I inspected the new negative-test assertions, not their execution. Parent reports prior smoke validations; current final seven-test validation and any future full run need their own retained results. Generator and analysis hashes below match the initial review; the scientific law/model were not changed by these repairs. This source disposition is prospective, not outcome validation or a claim that a baseline wins.

Current reviewed SHA256:
- `examples/noisy_pricing/audit.py`: `df21f11308ac88cbea0f107170bd882477573e9d29822de6e6465d3653a00a46`
- `examples/noisy_pricing/test_noisy_pricing.py`: `b2a7f873f55c56fbb474110f4a6fa81e235e83531391384bb080e5c7be6ea570`
- `examples/noisy_pricing/protocol.json`: `565614592de282f3ab1d252d5ae625056aea3bb2a0877429fb3c044b5ed0f0fd`
- `examples/noisy_pricing/generate.py`: `eb95ec3062471276195c15d06bc5f2ee962a154c27609311988d0c8379ad99d7`
- `examples/noisy_pricing/analysis.py`: `d0e73baef8c89bbbb7634446c998d394a39d18b4a1a35aecaf9b9282a6eec641`
- `examples/noisy_pricing/run.py`: `6a5003473bca33b4a8a121ab39238cc9638d251bd002a914b4ae3917bc563c84`
