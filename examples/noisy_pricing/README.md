# Noisy pricing benchmark v1 — prospective protocol

This separate benchmark preserves the exact two-price sanity check. It tests
fixed estimators under finite noisy data, imperfect realized treatment balance,
latent confounding, model misspecification, and independently sampled evaluation
contexts. It is a synthetic instrument benchmark, not evidence of autonomous
scientific discovery or a comparison of philosophies. No external model is called.

The machine-readable allocation is `protocol.json`: four declared world families,
12 campaign seeds per family, three fixed baselines, 160 training units per design,
and 64 new evaluation contexts with nine candidate prices. Every campaign draws
new world parameters and samples. Seeds do not merely permute fixed rows. Within
a campaign, all baselines share the same world and evaluation set; randomized
baselines share identical training data. Observational and randomized designs
share unit covariates, latent variables, and disturbances (paired potential units).
Comparisons are paired by campaign, never across individual action rows.

Training prices are 20,30,40,50,60. Evaluation includes those endpoints/support
points plus unseen intermediate prices25,35,45,55. Context x is Uniform[-1,1]
in both samples. The owner knows latent z~Bernoulli(1/2); the estimator sees x,
price, and observed demand, with opaque unit IDs. Independent uniform price
assignment is not forced to balance z or x. Observational assignment favors high
prices for z=1 and low prices for z=0, with positive probability of all five prices.

Owner-only demand is an affine price/context response plus a latent intercept,
zero-mean bounded observation noise, and family-specific curvature, interaction,
or a hinge. Parameters vary across campaigns. Equations and ranges are specified below. All demand is positive on the declared domain;
there is no clipping or mismatch between oracle means and truncated noise.

Baselines: observational linear ridge, randomized linear ridge, randomized
quadratic ridge. Both include context; linear features are 1,t,x and quadratic
features are 1,t,x,t*x,t*t,x*x, with t=(price-40)/20. Penalty0.01 is fixed,
intercept unpenalized, and no tuning uses evaluation labels. Features are generic
model choices, not fitted simulator constants. No estimator receives family,
seed, generator, oracle, other campaigns, or earlier outcomes. Public source
transparency is not secrecy against a model previously trained on this repository;
fresh process/filesystem isolation does not certify absence of model memory.

Regret is primary: mean across contexts of best listed expected revenue minus
revenue of the predicted best price, with lowest-price tie breaking. Additional
metrics: mean-demand RMSE, individual-outcome RMSE, policy value, and separate
mean errors on seen versus unseen prices. Oracle expected means avoid adding
Monte Carlo noise to policy scoring; training remains noisy and finite. Zero
regret remains possible when an imperfect estimator gets every decision right.

Report all family/seed/baseline outcomes, both prespecified paired comparisons,
mean, SD, MCSE, and empirical quantiles. MCSE describes variability of this
specific synthetic recipe's finite Monte Carlo mean, not uncertainty about real
markets. There is no confirmatory threshold, winner selection, or p-value. The
small CI smoke suite checks software, not scientific performance; the full frozen
allocation supplies the descriptive benchmark results. Failures remain failures.

## Exact owner-side world and oracle

Let t=(p−40)/20. Draw independent world parameters B~U[108,120],
S~U[40,52], C~U[2,6], L~U[12,20], σ~U[6,10]. Set

`m(p,x) = B − S*t + C*x − Q*t² + I*x*t − H*max(t,0)`.

Affine has Q=I=H=0. Quadratic has Q~U[6,12] and I=H=0.
Interaction has I~U[5,9] and Q=H=0. Threshold has H~U[12,20] and
Q=I=0. Additional active parameters are independent uniform draws.
`D(p,x,z,e) = m(p,x) + L*(z−1/2) + σ*e`, with x~U[−1,1],
z~Bernoulli(1/2), e~U[−1,1] independent. Disturbances are physical
unit variation; no separate measurement error is modeled. No clipping is used.

The worst-case demand is at least10 across all declared families and domains:
base≥108, price reduction≤52, context reduction≤6, latent reduction≤10,
noise reduction≤10, and the largest active family reduction≤20. Thus additive
mean-zero disturbances remain mean-zero in the observed response.
The oracle is **m(p,x)=E[D|do(p),x,world]**, integrating latent z and e;
it cannot condition policy decisions on the realized hidden type.

Observational assignment on the five training prices has probabilities
`Pr(P=p|z) = exp(1.5*(2z−1)*t) / sum_q exp(1.5*(2z−1)*t_q)`.
All probabilities exceed0.026, but latent selection still confounds regression.
Randomized assignment independently samples each training price with probability1/5.
Neither design enforces exact balance. Both use the same independent x/z/e units
within a campaign; only their assignment streams differ. Evaluation samples fresh
x/z/e units; each context's z/e are shared across all its potential prices.
Individual-outcome RMSE therefore has correlated rows within context, which are
never counted as independent statistical replications.

The generator derives each PRNG from SHA256(version/family/seed/component).
World, training units, observational assignment, randomized assignment, and
evaluation have separate stream identities. Distinct families include family in
the stream key. No rejection sampling, resampling, or favorable-seed selection.
The latent/noise variance conditional on x is L²/4+σ²/3; the oracle-mean predictor
thus has nonzero population individual-outcome RMSE even when mean prediction is exact.

Intermediate-price behavior is a model assumption being evaluated. Adding a term
proportional to the product of `(p−q)` over all five training prices would leave
all possible training rows unchanged while altering intermediate-price truth.
This benchmark does not claim nonparametric causal identification there. Fresh x
is from the same population, though it may exceed realized training extrema.

Ridge minimizes the **sum** of squared residuals plus0.01 times the squared
non-intercept coefficients. The linear model has features[1,t,x]; quadratic has
[1,t,x,t*x,t²,x²]. No scaling, tuning, clipping, or constraint fitting follows evaluation.

## Reproduce and inspect

Use the same Linux/CPython3.12/Bubblewrap environment and pinned requirements as
the original example. From the repository root:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r examples/pricing_confounds/requirements.txt
python -m pip install --no-deps -e .
python examples/noisy_pricing/run.py --profile smoke --output ./noisy-pricing-smoke
python examples/noisy_pricing/run.py --profile full --output ./noisy-pricing-full
python examples/noisy_pricing/audit.py ./noisy-pricing-full
python -m http.server 8000 --directory ./noisy-pricing-full
```

Smoke runs affine/threshold, first two campaign seeds:12 slots. Full runs144.
The timeout is120 seconds per notebook with one execution opportunity per slot;
there is no model/API cost. Preparation replays both generators in isolated
processes. Every available slot registration and the complete source/allocation
commitment precede the first notebook. Ordinary preparation/execution failures
are retained and other slots continue. A partial allocation exits nonzero and
reports denominators; interruptions leave the last atomically persisted statuses.
Dashboard/journal failures are separate reporting errors and do not discard valid
supervisor-scored outcomes. No resume,
overwrite, retry, replacement, or outcome-based stopping.

SciQuest requires a measurable registered prediction: here it is only the
mathematical check `regret>=0`. Its supported status is **not a performance claim**.
All experimental plans are exploratory/descriptive. Regret is the sole primary
metric; other supervisor metrics are diagnostics. Seen/unseen RMSE subsets are
explicitly computed by the owner after evaluation. Aggregate normalized scores
are framework plumbing and are not the benchmark's scientific endpoint.

`owner_source/` preserves exact code/protocol, `source_commitment.json` hashes it,
`before_execution.json` records registrations and allocation hash, `outcomes.json`
retains all slots incrementally, and `manifest.json` hashes every final file.
Each quest retains its actual executed notebook, full prediction CSV, model
coefficients, supervisor database, journal, integrity audit, and dashboard.
The owner report links them all. Source and oracle are public for reproducibility,
but excluded from each notebook's fresh filesystem/network sandbox. Notebook
probes verify source, labels, prior campaigns and supervisor imports are absent.
They do not certify absence of future external models' prior training knowledge.

The source protocol and all code are fixed before a run. This is development of
a transparent benchmark, not an independently preregistered confirmatory study.
CI runs meaningful oracle/scoring/failure tests and repeats the complete smoke
allocation from an installed wheel; the full allocation is separately scheduled
in the benchmark CI job. No result is required to be nonzero or favor a baseline.
