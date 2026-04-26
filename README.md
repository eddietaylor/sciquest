# SciQuest

SciQuest is an open-source, domain-agnostic Python package for agent-operable autonomous research programs. It provides deterministic infrastructure for iterative hypothesis evolution, sandboxed Jupyter experiments, validation, reporting, and scientific journaling.

Architecture principle:

- SciQuest Core = deterministic infrastructure
- Agent (Hermes/Codex/etc.) = scientific reasoning and creativity

SciQuest intentionally does not hard-code any research domain, dataset, model architecture, or proprietary concept.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e '.[test]'
pytest
```

## Commands

```bash
sciquest new
sciquest new --start-agent --agent-command "hermes chat -q"
sciquest continue --quest <quest_slug> --idea "new idea" --data "new data note"
sciquest list
sciquest status --quest <quest_slug>
sciquest run-next --quest <quest_slug>
sciquest run-loop --quest <quest_slug> --start-agent --agent-command "hermes chat -q"
sciquest validate --quest <quest_slug> --experiment exp_001
sciquest logic-check --quest <quest_slug> --experiment exp_001
sciquest dashboard --quest <quest_slug>
sciquest journal --quest <quest_slug>
sciquest journal --quest <quest_slug> --append "manual note"
```

Every command accepts `--root PATH`; quests are stored under `PATH/quests/`.

## New quest flow

`sciquest new` prompts for:

1. Hero Statement
2. Problem Statement
3. Initial Hypothesis (conceptual + observational)
4. Subjective Priors (optional)
5. Core Data Description (optional)
6. Validation Suite (optional)
7. Validation weighting preferences (optional, natural language)
8. Confirmation to start quest

Inputs are stored in structured YAML and Markdown. By default, `sciquest new` also shows a small Newton/SciQuest terminal splash. Use `--no-splash` to suppress it.

To have quest creation immediately hand off to an agent, run:

```bash
sciquest new --start-agent --agent-command "hermes chat -q"
```

or configure a default command:

```bash
export SCIQUEST_AGENT_COMMAND="hermes chat -q"
sciquest new --start-agent
```

SciQuest passes a complete one-iteration protocol prompt to the external agent on stdin. The command can be any local/open agent executable; SciQuest does not embed a proprietary agent.

If no core data is supplied, SciQuest creates `data_manifest.yaml` with `status: missing_user_data`. The agent must infer required data, find or generate a dataset, store it in `data/raw/`, and write schema, meaning, provenance, and limitations.

If no validation suite is supplied, SciQuest marks `validation.yaml` as `agent_required`. The agent must formalize metric name, description, direction, weight, and normalization.

## Quest layout

```text
quests/
  <quest_slug>/
    quest.yaml
    state.yaml
    hypotheses.yaml
    validation.yaml
    data_manifest.yaml
    journal.md
    AGENTS.md
    data/
      raw/
      processed/
    experiments/
      exp_001/
        experiment.yaml
        hypothesis.md
        notebook.ipynb
        executed_notebook.ipynb
        experiment_report.md
        validation_results.yaml
        logs/
        artifacts/
          plots/
          diagrams/
    reports/
    artifacts/
    logs/
```

## Experiment lifecycle

Each iteration is designed to be run by an agent or external scheduler:

1. Agent evolves hypothesis
2. Creates experiment notebook
3. Executes notebook
4. If failure, debug and retry while preserving logs
5. Run validation
6. Generate experiment report
7. Append journal entry
8. Update `state.yaml`

The built-in `run-next` command creates a deterministic experiment scaffold and lock-protected state transition. It does not invent scientific content; agents do that by following `AGENTS.md`.

To run a scheduler-style agent iteration directly instead of creating only the scaffold:

```bash
sciquest run-next --quest <quest_slug> --start-agent --agent-command "hermes chat -q"
```

## Notebook requirements

Experiment notebooks must:

- show dataset preview
- report dataset size before/after transforms
- document preprocessing and features
- define target and inputs clearly
- include docstrings
- generate plots
- save metrics to `validation_results.yaml`
- save artifacts to `artifacts/`
- run in a clean environment

SciQuest includes an nbclient-based execution wrapper.

## Validation system

`validation.yaml` metrics use this shape:

```yaml
status: ready
metrics:
  - name: accuracy
    description: Example metric
    direction: maximize
    weight: 1.0
    normalization:
      type: minmax
      min: 0
      max: 1
```

Supported directions are `maximize`, `minimize`, and `target`. Supported normalization types are `identity`, `minmax`, and `target`.

Aggregate score is the weighted mean of normalized metric scores.

## Dashboard

Build a dynamic static HTML dashboard for a quest:

```bash
sciquest dashboard --quest <quest_slug>
```

The dashboard is written to:

```text
quests/<quest_slug>/reports/dashboard/index.html
```

It lets you click through experiment iterations and displays, when documented by the agent:

- task type: supervised, unsupervised, generative, simulation, causal, etc.
- model architecture
- input features and target features
- model architecture diagram
- validation technique diagram
- result graphs from `artifacts/plots/`
- validation metrics and aggregate score
- interpretation extracted from `experiment_report.md`

Agents should place technical SVG diagrams under:

```text
experiments/exp_NNN/artifacts/diagrams/
  model_architecture.svg
  validation_technique.svg
```

## Logic check

`sciquest logic-check` verifies, as far as deterministic infrastructure can, that:

- a hypothesis exists and can be tested in a notebook
- notebook artifacts exist and mention required elements
- validation metrics exist and align structurally with the problem
- conclusions can be checked against results
- data manifest status is visible
- no missing validation results create silent failures
- reports separate speculation from evidence
- required experiment subdirectories exist, including `artifacts/plots/` and `artifacts/diagrams/`
- generated/synthetic datasets preserve a data generation artifact
- experiment metadata includes dashboard fields such as task type, features, targets, model architecture, and validation technique

## State and locking

`state.yaml` tracks:

- `quest_status`: idle, running, or failed
- `current_experiment`
- `last_experiment`
- `last_score`
- `best_score`
- `failures`
- `last_updated`
- `lock_id`

`run-next` acquires a lock to prevent concurrent runs and releases it after success or failure.

## Iteration loop mode

Run multiple autonomous iterations sequentially:

```bash
sciquest run-loop --quest <quest_slug> --start-agent --agent-command "hermes chat -q"
```

By default, `run-loop` runs at most 3 iterations. Override this with:

```bash
sciquest run-loop --quest <quest_slug> --max-iterations 10 --start-agent --agent-command "hermes chat -q"
```

The loop is foreground and deterministic: it starts the next iteration only after the prior `run-next`/agent process returns and `state.yaml` is idle/unlocked. If an iteration fails or the quest remains locked/running, the loop stops rather than stacking concurrent runs.

For package tests or scaffold-only dry runs:

```bash
sciquest run-loop --quest <quest_slug> --agent-stub --max-iterations 3
```

## Scheduling

SciQuest does not implement cron. Use an external scheduler to call:

```bash
sciquest run-next --quest <quest_slug>
```

## License

MIT
