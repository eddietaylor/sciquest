# Registered experiment exp_001

## Evidence
Metrics computed by the supervisor from registered truth and prediction artifacts.

- mean_recovery: contradicted
- optimal_listed_price: contradicted

## Metrics
- mean_demand_rmse (primary): 20.0
- policy_value (diagnostic): 1600.0
- regret (diagnostic): 200.0

## Constraints
No separate constraint requirements registered.

## Undefined metrics
None.

## Inquiry policy evidence
```json
{
  "contract": {
    "enforced_plan_contracts": [],
    "guidance_only_methods": [
      "standard_empirical"
    ],
    "inactive_plan_extensions": [],
    "phase_governance": {
      "evidence_evaluation": "standard_empirical",
      "experiment_design": "standard_empirical",
      "hypothesis_generation": "standard_empirical",
      "long_term_tracking": "standard_empirical",
      "problem_framing": "standard_empirical",
      "stress_testing": "standard_empirical"
    },
    "requirements": {},
    "scope": "Registered planning constraints and numerical evidence updates; not a guarantee of philosophical fidelity or test severity",
    "version": 2
  },
  "observed_prediction_outcomes": {
    "mean_recovery": "contradicted",
    "optimal_listed_price": "contradicted"
  }
}
```

## Scope
Registered decision rules; not automatic causal identification or statistical error control
Decision metrics use owner-supplied expected intervention values on the registered finite action grid. Scientific understanding and causal identification require separate evidence.

## Speculation
No broader causal or real-world conclusion follows automatically.

## Scripted interpretation

The model is fitted inside the executed notebook; metrics above come from the supervisor. Actual mean-demand RMSE: 20.0; actual finite-grid regret: 200.0. Population-mean prediction is distinct from predicting an individual unit's Z-dependent demand. The comparison figure is generated owner-side after both evaluations; it is not a worker artifact. No external agent chose the hypothesis, design, model or interpretation.
