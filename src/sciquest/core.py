from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import (
    quest_dir,
    quests_dir,
    read_yaml,
    write_yaml,
    append_text,
    utc_now,
    slugify,
)
from .templates import AGENTS_MD
from .notebooks import create_stub_notebook
from .methods.ledger import append_method_sections_to_report, load_quest_method_stack
from .methods.resolver import auto_recommend_method_stack, resolve_method_stack
from .state import (
    acquire_lock,
    release_lock,
    read_state,
    update_state,
    transaction,
    event,
    _read,
    _save,
)
from .io import canonical_json, experiment_dir


def create_quest(root: Path, answers: dict[str, Any], slug: str | None = None) -> Path:
    for field in [
        "hero_statement",
        "problem_statement",
        "initial_hypothesis",
        "subjective_priors",
        "core_data_description",
        "validation_suite",
        "validation_weighting_preferences",
    ]:
        if field in answers and not isinstance(answers[field], str):
            raise ValueError(f"{field} must be text")
    method_selection = answers.get("method_stack") or answers.get("method")
    if answers.get("auto_method"):
        method_stack = auto_recommend_method_stack(
            {**answers, "description": answers.get("problem_statement", "")}
        )
    else:
        method_stack = resolve_method_stack(method_selection)
    qslug = slugify(slug or answers.get("hero_statement", "quest"))
    qpath = quest_dir(root, qslug)
    qpath.mkdir(parents=True, exist_ok=False)
    for sub in [
        "data/raw",
        "data/processed",
        "experiments",
        "reports",
        "artifacts",
        "logs",
    ]:
        (qpath / sub).mkdir(parents=True, exist_ok=True)
    now = utc_now()
    write_yaml(
        qpath / "quest.yaml",
        {
            "slug": qslug,
            "created_at": now,
            "hero_statement": answers.get("hero_statement", ""),
            "problem_statement": answers.get("problem_statement", ""),
            "ontology": [
                "Quest",
                "Hypothesis",
                "Experiment",
                "Validation Suite",
                "Research Journal",
                "Agent Protocol",
            ],
        },
    )
    write_yaml(
        qpath / "state.yaml",
        {
            "schema_version": 2,
            "quest_status": "idle",
            "current_experiment": None,
            "last_experiment": None,
            "last_score": None,
            "best_score": None,
            "failures": [],
            "last_updated": now,
            "lock_id": None,
        },
    )
    write_yaml(
        qpath / "hypotheses.yaml",
        {
            "hypotheses": [
                {
                    "id": "h001",
                    "version": 1,
                    "created_at": now,
                    "claim": answers.get("initial_hypothesis", ""),
                    "conceptual": answers.get("initial_hypothesis", ""),
                    "observational": answers.get("initial_hypothesis", ""),
                    "subjective_priors": answers.get("subjective_priors", ""),
                    "status": "initial",
                }
            ]
        },
    )
    core_data = answers.get("core_data_description", "").strip()
    write_yaml(
        qpath / "data_manifest.yaml",
        {
            "status": "provided_description" if core_data else "missing_user_data",
            "description": core_data,
            "datasets": [],
            "agent_instructions": "Infer required data, find or generate it, store under data/raw/, and document schema, meaning, provenance, and limitations."
            if not core_data
            else "Document concrete data files when added.",
        },
    )
    val = answers.get("validation_suite", "").strip()
    write_yaml(
        qpath / "validation.yaml",
        {
            "status": "provided_description" if val else "agent_required",
            "description": val,
            "weighting_preferences_raw": answers.get(
                "validation_weighting_preferences", ""
            ),
            "metrics": [],
            "agent_instructions": "Infer appropriate metrics and formalize this file with name, description, direction, weight, normalization."
            if not val
            else "Formalize provided validation description into metrics before scoring.",
        },
    )
    write_yaml(qpath / "method_stack.yaml", method_stack.to_dict())
    write_yaml(
        qpath / "research_contract.yaml",
        {
            "schema_version": 1,
            "status": "draft",
            "question": answers.get("problem_statement", ""),
            "initial_hypothesis": answers.get("initial_hypothesis", ""),
            "claim_scope": "",
            "estimand": "",
            "rivals": [],
            "assumptions": [],
            "observation_access": [],
            "intervention_access": [],
            "uncertainty_and_stopping": "",
            "resource_budget": {},
            "evaluation_mode": "exploratory",
            "instructions": "Draft a concrete evidence contract before confirmation. Replication, measurement, data acquisition, and stopping are valid research actions.",
        },
    )
    (qpath / "journal.md").write_text(
        f"# Research Journal: {qslug}\n\nCreated: {now}\n\n## Quest Started\n\nHero Statement: {answers.get('hero_statement', '')}\n\nProblem Statement: {answers.get('problem_statement', '')}\n\nInitial Hypothesis: {answers.get('initial_hypothesis', '')}\n",
        encoding="utf-8",
    )
    (qpath / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    return qpath


def list_quests(root: Path) -> list[dict[str, Any]]:
    directory = quests_dir(root)
    out = []
    for q in sorted(directory.glob("*")) if directory.exists() else []:
        if q.is_dir() and (q / "quest.yaml").exists():
            meta = read_yaml(q / "quest.yaml", {})
            state = read_state(q)
            out.append(
                {
                    "slug": meta.get("slug", q.name),
                    "status": state.get("quest_status"),
                    "best_score": state.get("best_score"),
                }
            )
    return out


def next_experiment_id(qpath: Path) -> str:
    nums = []
    for p in (qpath / "experiments").glob("exp_*"):
        if p.is_dir():
            try:
                nums.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return f"exp_{(max(nums) + 1) if nums else 1:03d}"


def reserve_experiment(qpath: Path, owner: str) -> tuple[str, Path]:
    with transaction(qpath) as connection:
        state = _read(connection)
        if state.get("lock_id") != owner:
            raise RuntimeError("Experiment allocation requires the active owner")
        reserved = [
            row[0]
            for row in connection.execute(
                "SELECT id FROM records WHERE kind='reservation'"
            )
        ]
        candidate = int(next_experiment_id(qpath).split("_")[1])
        number = max([candidate, *[int(item.split("_")[1]) + 1 for item in reserved]])
        exp_id = f"exp_{number:03d}"
        payload = {"experiment_id": exp_id, "run_id": owner, "created_at": utc_now()}
        connection.execute(
            "INSERT INTO records VALUES('reservation',?,?)",
            (exp_id, canonical_json(payload)),
        )
        event(connection, "experiment_reserved", payload)
        state["current_experiment"] = exp_id
        _save(connection, qpath, state)
    exp = experiment_dir(qpath, exp_id)
    exp.mkdir(exist_ok=False, parents=True)
    return exp_id, exp


def create_scaffold(qpath: Path, exp_id: str, exp: Path) -> None:
    for sub in ["logs", "artifacts/plots", "artifacts/diagrams"]:
        (exp / sub).mkdir(parents=True, exist_ok=False)
    write_yaml(
        exp / "experiment.yaml",
        {
            "id": exp_id,
            "created_at": utc_now(),
            "status": "draft",
            "task_type": "agent_required",
            "model_architecture": "agent_required",
            "input_features": [],
            "target_features": [],
            "validation_technique": "agent_required",
            "technical_diagrams": [],
        },
    )
    stack = load_quest_method_stack(qpath)
    from .methods.ledger import create_draft_ledger

    create_draft_ledger(qpath, exp_id, exp)
    (exp / "hypothesis.md").write_text(
        "# Research action\n\nSpecify a hypothesis test, rival test, replication, measurement, data acquisition, or stop decision.\n",
        encoding="utf-8",
    )
    create_stub_notebook(exp / "notebook.ipynb", exp_id)
    (exp / "experiment_report.md").write_text(
        "# Experiment Report\n\n## Evidence\nUnexecuted draft.\n\n## Speculation\nNo scientific result is established.\n",
        encoding="utf-8",
    )
    append_method_sections_to_report(exp / "experiment_report.md", stack)


def run_next(qpath: Path, agent_stub: bool = False, execute: bool = False) -> str:
    if execute:
        raise ValueError(
            "Scaffolds cannot be executed as research; use register and execute for a concrete experiment"
        )
    lock = acquire_lock(qpath)
    exp_id = None
    try:
        exp_id, exp = reserve_experiment(qpath, lock)
        create_scaffold(qpath, exp_id, exp)
        append_journal_entry(qpath, exp_id, "Agent stub created experiment scaffold.")
        release_lock(qpath, lock, "idle", changes={"last_prepared_experiment": exp_id})
        return exp_id
    except Exception as exc:
        state = read_state(qpath)
        failures = state.get("failures") or []
        failures.append({"experiment": exp_id, "error": str(exc), "time": utc_now()})
        release_lock(qpath, lock, "failed", changes={"failures": failures})
        raise


def append_journal_entry(qpath: Path, exp_id: str, note: str = "") -> None:
    append_text(
        qpath / "journal.md",
        f"\n\n## Experiment {exp_id.split('_')[-1].lstrip('0') or '0'}\n\n* Hypothesis: See experiments/{exp_id}/hypothesis.md\n* Method: See experiments/{exp_id}/notebook.ipynb\n* Results: Pending or recorded in validation_results.yaml\n* Validation scores: Pending\n* Aggregate score: Pending\n* Interpretation: {note}\n* Debug notes: None recorded\n* Next direction: Agent to decide\n* User injections: None\n",
    )


def continue_quest(
    qpath: Path, idea: str | None = None, data: str | None = None
) -> None:
    with transaction(qpath) as connection:
        if _read(connection).get("lock_id"):
            raise RuntimeError("Queue user input after the active run finishes")
        injection = {"time": utc_now(), "idea": idea, "data": data}
        path = qpath / "user_injections.yaml"
        existing = read_yaml(path, {"injections": []})
        existing.setdefault("injections", []).append(injection)
        write_yaml(path, existing)
        event(connection, "user_injection", injection)
        append_text(
            qpath / "journal.md",
            f"\n\n## User Injection\n\nTime: {injection['time']}\n\nIdea: {idea or ''}\n\nData: {data or ''}\n",
        )


def update_state_with_validation(
    qpath: Path, exp_id: str, validation: dict[str, Any]
) -> None:
    state = read_state(qpath)
    score = validation.get("aggregate_score")
    groups = state.get("score_groups", {})
    group_id = validation.get("comparison_group")
    changes = {"last_scored_experiment": exp_id, "last_score": score}
    if group_id and validation.get("evidence_source") == "supervisor_computed":
        group = groups.get(group_id, {})
        if group.get("best_score") is None or score > group["best_score"]:
            group = {"best_score": score, "experiment": exp_id}
        groups[group_id] = group
        changes["score_groups"] = groups
    update_state(qpath, changes)
