# Pricing under confounding: a reproducible SciQuest example

This is a transparent analytic reproduction, not a novel finding or an autonomous
agent discovery. Two prescribed experiments use the same estimator, eight units
per arm and the same prices. SciQuest supplies registration, isolated execution,
frozen evidence, protected-label evaluation, journaling and dashboards.

## Scientific question and ground truth

Can price-only ordinary least squares recover population intervention means and
select the best listed price from confounded versus randomized training data?

The owner-side [generator](generate.py) defines individual demand

\[
D(p,z)=100-2p+40z,\qquad z\in\{0,1\},\quad p\in\{30,40\}.
\]

There is no observation noise. The target population has equal proportions of
the two demand strata, so

\[
E[D\mid do(P=p)]=120-2p,\qquad R(p)=p(120-2p).
\]

The truth rows are population expectations, not sampled individual outcomes:

| Price | Mean demand | Expected revenue |
|---:|---:|---:|
| 30 | 60 | 1800 |
| 40 | 40 | 1600 |

Revenue has no costs, capacity constraints, competitor behavior or time dynamics.
The oracle considers exactly these two actions. It is not a general pricing
simulator or a claim about a continuous or sequential optimum.

## Prescribed experiments and hypothesis

Both arms fit an intercept and slope using only their own training `price` and
`demand` columns. The [analysis source](analysis.py) contains neither the simulator
law nor expected coefficients, revenues or scores. It becomes an actual
registered notebook cell and runs inside SciQuest's allowlisted worker runtime.

1. **Observational:** four units in each stratum receive price `30+10*z`.
   This confounds price with stratum. The observed pairs are `(30,40)` and
   `(40,60)`, giving the marginal fit `-20+2*price`.
2. **Randomized:** within each simulator-known stratum, assign two units to each
   price, shuffling assignment with seed 42. This is a balanced stratified
   randomized design, not an unconstrained finite random sample. Its pooled fit
   is `120-2*price` because stratum proportions balance at **each** price.

Stratum is known to the generator/design and omitted from the estimator inputs.
The observational law `D=-20+2*P` with no stratum effect reproduces all observed
training cells but predicts the opposite intervention slope. Merely fitting the
observational pairs therefore does not identify the causal effect. Observing
stratum would not fix the lack of within-stratum price variation in that arm.

Both experiments register the same falsifiable criterion before either notebook
executes: population-mean RMSE ≤ `1e-10` **and** finite-grid pricing regret ≤
`1e-10`. The observational arm is expected to fail; the randomized arm is expected
to meet it. Registration is an audit of before-execution commitments, not evidence
of novelty or a result unknown to the authors. Plans are marked exploratory.

## Metrics and expected arithmetic

Predictions include demand and **price times predicted demand**. SciQuest chooses
the action with highest predicted revenue, breaking ties toward the lower price;
it then evaluates the choice using protected expected-revenue truth.

| Design | Intercept | Slope | Demand RMSE | Chosen price | Policy value | Regret |
|---|---:|---:|---:|---:|---:|---:|
| Observational | −20 | +2 | 20 | 40 | 1600 | 200 |
| Randomized | 120 | −2 | 0 | 30 | 1800 | 0 |

For observational OLS, predicted means are `[40,60]` versus truth `[60,40]`;
RMSE is `sqrt(((-20)^2+20^2)/2)=20`. Predicted revenues `[1200,2400]` select
price 40, whose actual value is 1600; regret is `1800-1600=200`.
The balanced fit reproduces the two population means and selects price 30.

RMSE is the primary metric. Policy value and regret remain separate diagnostics;
the fitted slope is an execution-bound artifact diagnostic, not an invented
supervisor metric. Expected values are in [expected.json](expected.json) for
independent checking; the runner never reads that file to manufacture results.

Zero population-mean error does not imply zero individual error: each stratum
differs from the population mean by ±20 units. Eight noiseless units, two truth
rows, and repeated reproductions are not independent statistical replications.
There are no p-values, confidence intervals or calibration claims.
An added term `c*(price-30)*(price-40)` vanishes at both tested prices, so these
experiments cannot identify behavior outside the registered grid.

## Reproduce

Use Linux, CPython 3.12 and Bubblewrap with working user namespaces. On
Debian/Ubuntu, install Bubblewrap with `sudo apt-get install bubblewrap`. Containers
or hardened hosts that prohibit user namespaces must permit them; execution fails
explicitly rather than falling back to an unisolated run.

From a checkout of this repository:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r examples/pricing_confounds/requirements.txt
python -m pip install --no-deps -e .
python examples/pricing_confounds/run.py --output ./pricing-run
python -m http.server 8000 --directory ./pricing-run
```

Open `http://localhost:8000/`. Both complete SciQuest dashboards are linked from
the overview. Setup downloads dependencies; experiment execution makes no model
calls and requires no network access. Preserve the dashboard directory including
its local MathJax asset. System font fallbacks are used.
The bundled equation asset is a readable TeX-text fallback, not a full MathJax
typesetter; scientific plots are rendered separately by Matplotlib.

The runner refuses an existing output path. Use another new directory to repeat
the run. Reproduction means identical generated CSVs, predictions and scientific
`results.json` for the same seed/environment, with numeric checks at `1e-10`
absolute tolerance. Whole directories and hashes vary with timestamps, absolute
paths, notebook provenance, runtime versions and execution records. PNG/SVG bytes
can vary across plotting-library/font/platform versions; compare their underlying
tables, not image hashes, across environments.

The pinned [requirements](requirements.txt) record the resolved Python3.12
runtime/test/plot packages. `runtime.json` records the actual interpreter,
installed packages and SciQuest source identity. The OS/kernel/Bubblewrap and
CPU architecture are external requirements, not fully pinned by pip.

## Output and audit trail

```text
pricing-run/
  results.json, report.md, comparison.svg, comparison.png, index.html
  before_execution.json          # both registration hashes, before any execution
  runtime.json, manifest.json    # runtime provenance and per-run artifact hashes
  quests/{observational,randomized}/
    hypotheses.yaml, research_contract.yaml, validation.yaml, journal.md
    data/{generate.py,raw/{train.csv,eval.csv}}
    .sciquest/                   # supervisor database and frozen snapshots
    experiments/exp_001/
      experiment_plan.yaml, method_ledger.yaml, notebook.ipynb
      executed_notebook.ipynb, execution.yaml, validation_results.yaml
      worker_artifacts/predictions.csv, experiment_report.md, logic_check.json
      artifacts/plots/comparison.svg, logs/
    reports/dashboard/{index.html,assets/mathjax/tex-mml-chtml.js}
```

Report metrics come from actual supervisor results. Plotting is a separate
owner-side explanatory step after evaluation, using actual prediction/truth
tables. Journal entries distinguish the prewritten protocol, automatic supervisor
evaluation, and scripted interpretation. Integrity checks verify records and
frozen inputs; `scientific_verified` remains false. Independent arithmetic and
criticism are still necessary.

## Installed-wheel integration test and CI

CI builds a wheel, installs it into a fresh environment, changes directory outside
the repository and runs this exact test:

```bash
python -m build --wheel
python3.12 -m venv /tmp/sciquest-example-wheel
/tmp/sciquest-example-wheel/bin/python -m pip install -r examples/pricing_confounds/requirements.txt
/tmp/sciquest-example-wheel/bin/python -m pip install --no-deps dist/sciquest-0.2.0-py3-none-any.whl
REPRO_CHECKOUT="$PWD"
cd /tmp
/tmp/sciquest-example-wheel/bin/python -m pytest \
  "$REPRO_CHECKOUT/examples/pricing_confounds/test_end_to_end.py" -v
```

The test asserts installed-module provenance, actual isolated execution and
generator replay, completed notebooks, authoritative registration/evaluation,
complete prediction IDs, independent metric arithmetic, hypothesis outcomes,
journal entries, plots and dashboard assets. It repeats the full workflow twice
and compares scientific outputs. An altered-training test checks that coefficients
are fitted rather than copied from analytic expectations. There are no execution
mocks or isolation skips. CI uploads run artifacts, including hidden `.sciquest`
records, for inspection.

## What this says about agents

All scientific choices here are prescribed: hypothesis, data designs, estimator,
metrics, stopping rule and interpretation. Fitting, execution, scoring,
journaling, plotting and dashboard assembly are deterministic software operations.
`external_agent_calls=0` and `genuine_external_agent_reasoning_demonstrated=false`
are explicit output fields.

SciQuest separately supports external workers, but no genuine external agent
reasoning is demonstrated here. Such a run would be a separate experiment
requiring original prompts, responses, model/version, access, budget, failures
and provenance. This example does not validate the experimental private-review
service or compare philosophy-inspired policies. It does not prove absence of
pretrained knowledge or universal sandbox security. The worker checks that this
run's owner truth, published generator and supervisor package are inaccessible
in its actual runtime; the author and reader still know the world by design.

Code and protocol preparation used AI-assisted development and critical-agent
review. That development assistance is distinct from a recorded SciQuest
scientific-agent campaign and is not counted as an experimental observation.
