AGENTS_MD = """# SciQuest Agent Protocol

SciQuest Core is deterministic infrastructure. The agent supplies scientific reasoning and creativity.

For every `sciquest run-next --quest <quest_slug>` iteration:

1. Read all quest files first: `quest.yaml`, `state.yaml`, `hypotheses.yaml`, `validation.yaml`, `data_manifest.yaml`, and `journal.md`.
2. Generate exactly one evolved, testable hypothesis for this iteration.
3. If `data_manifest.yaml` status is `missing_user_data`, infer required data, find or generate suitable open data, store it under `data/raw/`, and document schema, meaning, provenance, and limitations.
4. If `validation.yaml` status is `agent_required`, define a domain-appropriate validation suite with metric name, description, direction, weight, and normalization.
5. Create a new `experiments/exp_NNN/` folder without overwriting history.
6. Write `experiment.yaml`, `hypothesis.md`, and a Jupyter `notebook.ipynb`.
7. The notebook must show a dataset preview, report dataset size before/after transforms, document preprocessing and features, define target and inputs, include docstrings, generate plots, save metrics to `validation_results.yaml`, save artifacts to `artifacts/`, and run in a clean environment.
8. Execute the notebook. Debug failures with a bounded retry loop and preserve logs.
9. Run `sciquest validate --quest <quest_slug> --experiment exp_NNN`.
10. Write `experiment_report.md` separating evidence from speculation.
11. Append a journal entry with hypothesis, method, results, scores, interpretation, debug notes, next direction, and user injections.
12. Update `state.yaml` and stop after one iteration unless an external scheduler invokes another run.

Never overwrite prior experiments. Preserve failed experiments and logs.
"""
