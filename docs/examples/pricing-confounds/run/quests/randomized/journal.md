# Research Journal: randomized

Created: 2026-10-29T19:35:01+00:00

## Quest Started

Hero Statement: Pricing under confounding: randomized

Problem Statement: Can the prescribed price-only estimator recover intervention means and select a price on {30,40}?

Initial Hypothesis: Price-only OLS recovers population intervention means (RMSE <= 1e-10) and selects a zero-regret listed price.

## Scripted protocol, before execution

The owner supplies both designs and this OLS analysis. No external agent reasons or chooses experiments. Analytic expectations are known in advance; this is a transparent reproduction, not a blind discovery. Both arms will be registered before either executes.


## Experiment 1

* Hypothesis: See experiments/exp_001/hypothesis.md
* Method: See experiments/exp_001/notebook.ipynb
* Results: Pending or recorded in validation_results.yaml
* Validation scores: Pending
* Aggregate score: Pending
* Interpretation: Agent stub created experiment scaffold.
* Debug notes: None recorded
* Next direction: Agent to decide
* User injections: None


<!-- sciquest-evaluation:exp_001 -->
## exp_001
Supervisor evaluation recorded.
- mean_recovery: supported
- optimal_listed_price: supported

## Scripted interpretation

The model is fitted inside the executed notebook; metrics above come from the supervisor. Actual mean-demand RMSE: 0.0; actual finite-grid regret: 0.0. Population-mean prediction is distinct from predicting an individual unit's Z-dependent demand. The comparison figure is generated owner-side after both evaluations; it is not a worker artifact. No external agent chose the hypothesis, design, model or interpretation.
