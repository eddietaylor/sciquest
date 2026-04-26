AGENTS_MD = """# SciQuest Agent Protocol

SciQuest Core is deterministic infrastructure. The agent supplies scientific reasoning and creativity.

For every `sciquest run-next --quest <quest_slug>` iteration:

1. Read all quest files first: `quest.yaml`, `state.yaml`, `hypotheses.yaml`, `validation.yaml`, `data_manifest.yaml`, and `journal.md`.
2. Generate exactly one evolved, testable hypothesis for this iteration. For the first experiment, operationalize the user's initial hypothesis rather than discarding it.
3. If `data_manifest.yaml` status is `missing_user_data`, infer required data, find or generate suitable open data, store it under `data/raw/`, and document schema, meaning, provenance, and limitations. If synthetic/generated data is used, preserve the generator code under `experiments/exp_NNN/artifacts/` as `data_generation_script.py` or `data_generator.py`.
4. If `validation.yaml` status is `agent_required`, define a domain-appropriate validation suite with metric name, description, direction, weight, and normalization.
5. Create a new `experiments/exp_NNN/` folder without overwriting history, including `logs/`, `artifacts/plots/`, and `artifacts/diagrams/`.
6. Write `experiment.yaml`, `hypothesis.md`, and a Jupyter `notebook.ipynb`.
7. In `experiment.yaml`, document dashboard metadata: `task_type` (supervised, unsupervised, generative, simulation, causal, etc.), `model_architecture`, `input_features`, `target_features`, `validation_technique`, and `technical_diagrams`.
8. Create technical SVG diagrams under `artifacts/diagrams/`:
   - `model_architecture.svg`: data/features → model components → outputs.
   - `validation_technique.svg`: train/validation/test/counterfactual flow, metrics, aggregate score.
9. The notebook must be split into readable sections: setup, data loading/generation, dataset preview, preprocessing/features, model/baseline, validation, plots/artifacts, and interpretation handoff.
10. The notebook must show a dataset preview, report dataset size before/after transforms, document preprocessing and features, define target and inputs, include docstrings, generate plots, save metrics to `validation_results.yaml`, save artifacts to `artifacts/`, and run in a clean environment.
11. Compare against the strongest reasonable baseline available for the iteration, not only a trivial baseline. If a weak baseline is used, explicitly justify it and propose the stronger next baseline.
12. Execute the notebook. Debug failures with a bounded retry loop and preserve logs.
13. Run `sciquest validate --quest <quest_slug> --experiment exp_NNN`.
14. Run `sciquest logic-check --quest <quest_slug> --experiment exp_NNN` and fix any issues before stopping.
15. Write `experiment_report.md` separating evidence from speculation and including graph interpretation.
16. Append a journal entry with hypothesis, method, results, scores, interpretation, debug notes, next direction, and user injections.
17. Update `state.yaml` and stop after one iteration unless an external scheduler invokes another run.

Never overwrite prior experiments. Preserve failed experiments and logs.
"""
