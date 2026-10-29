# SciQuest research worker protocol

The supervisor owns registration, execution, evaluation, state, and research history.
The worker prepares scientific proposals in its assigned workspace. Do not update
state.yaml, the supervisor database, registered inputs, or previous experiments.

1. Read handoff.yaml, quest.yaml, research_contract.yaml, hypotheses.yaml,
   method_stack.yaml, validation.yaml, and data_manifest.yaml when present.
2. If handoff mode is propose_setup, write setup_proposal.yaml with action:
   propose_setup, a concrete reason, candidate data sources, metric definitions,
   assumptions, rivals, intervention access, resource budget, and stopping rules.
   A setup proposal is not an experiment or confirmation.
3. Otherwise choose one useful research action. A new hypothesis is optional:
   rival tests, replication, measurement, data acquisition, and stopping are valid.
   Preserve the user's hypothesis as a candidate that evidence may contradict.
4. Apply the assigned philosophy-inspired inquiry policy to observable decisions.
   Maintain explicit rivals and discriminating predictions where appropriate.
   Numerical Bayesian updates require an explicit likelihood and model assumptions.
5. Write experiment_plan.yaml with concrete question, hypothesis, scope,
   assumptions, data_access, stopping_rule, action, confirmatory_status, and
   measurable predictions (id, metric, operator, threshold). Use existing metric
   definitions; propose future amendments separately.
6. Write notebook.ipynb using staged training data under data/INDEX/FILENAME and
   evaluation features under data/INDEX/features.csv. Evaluation labels are hidden.
   Emit the complete prediction CSV required by validation.yaml. Preserve model
   parameters, feature definitions, dependencies, and reloadable prediction code.
   Decision metrics require predicted objective values for every listed action;
   protected truth determines policy value, regret and constraint violations.
   Treat primary, diagnostic and constraint roles separately. Decision utility
   does not establish scientific understanding or causal identification.
7. Do not execute the notebook during preparation. The supervisor freezes inputs,
   runs the notebook in a bounded isolated process, and computes metrics from
   registered truth. Agent-written scores do not establish results.
8. Reports, plots, and diagrams are optional explanatory artifacts, generated when
   scientifically useful. Distinguish observations, assumptions, and speculation.
9. If the question is unidentifiable with current evidence, or further work is
   unwarranted, write decision.yaml with action: needs_data or stop and a concrete
   reason. Inconclusive and negative outcomes are legitimate.
10. Stop after the proposal. Never declare scientific confirmation or completion;
    those depend on the supervisor's recorded evidence and the claim's limits.

For package development, preserve legacy research artifacts. Add meaningful
regression, integration, and adversarial checks for observed failure modes.

## Critical review of development and research

The user requests a team of critical agents throughout SciQuest development and
research. At material changes to simulators, study protocols, assessment, or paper
arguments, use independent reviewers covering simulator/identification validity,
experimental/statistical design, and evaluation/argument validity. Give reviewers
the question and authoritative artifacts before suggesting a preferred answer.

Require concrete evidence and a counterexample or discriminating check. Distinguish
demonstrated defects, limits of a validator, untested assumptions, and optional
extensions. Keep an issue ledger with evidence, consequence, proposed resolution,
verification, and unresolved disagreements. The coordinating researcher checks
findings against source or execution; agreement among agents is not proof and
same-model reviewers are not independent human validation. Re-review material fixes.

This team reviews DEVELOPMENT and research design. Do not add these instructions
to experimentalist prompts or generated worker templates. Truth-aware critics must
never supply simulator laws, private rivals, answers, cross-campaign examples, or
owner summaries to experimental subjects. Preserve the approved public interface
and each campaign's own evidence boundary. Do not alter live campaigns, frozen
protocols, admitted historical worlds, scores, or old artifacts to address critique.
Changes require a new prospective version/allocation where applicable.

Before a scientific allocation, resolve findings that invalidate its intended
claim, or explicitly narrow the claim and record the remaining limitation. Do not
treat every reviewer suggestion as a mandatory feature, invent guarantees about
free-text semantic validation, or use review consensus as a substitute for external
evidence and eventual human scientific/philosophical assessment.
