from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

from .io import quest_dir, quests_dir, read_yaml, write_yaml, append_text, utc_now, slugify
from .templates import AGENTS_MD
from .notebooks import create_stub_notebook, execute_notebook
from .methods.ledger import append_method_sections_to_report, load_quest_method_stack, preregister_experiment
from .methods.resolver import auto_recommend_method_stack, resolve_method_stack


def create_quest(root: Path, answers: dict[str, str], slug: str | None = None) -> Path:
    qslug = slugify(slug or answers.get("hero_statement", "quest"))
    qpath = quest_dir(root, qslug)
    if qpath.exists():
        raise FileExistsError(f"Quest already exists: {qslug}")
    for sub in ["data/raw", "data/processed", "experiments", "reports", "artifacts", "logs"]:
        (qpath / sub).mkdir(parents=True, exist_ok=True)
    now = utc_now()
    write_yaml(qpath / "quest.yaml", {
        "slug": qslug,
        "created_at": now,
        "hero_statement": answers.get("hero_statement", ""),
        "problem_statement": answers.get("problem_statement", ""),
        "ontology": ["Quest", "Hypothesis", "Experiment", "Validation Suite", "Research Journal", "Agent Protocol"],
    })
    write_yaml(qpath / "state.yaml", {
        "quest_status": "idle", "current_experiment": None, "last_experiment": None,
        "last_score": None, "best_score": None, "failures": [], "last_updated": now, "lock_id": None,
    })
    write_yaml(qpath / "hypotheses.yaml", {"hypotheses": [{
        "id": "h001", "version": 1, "created_at": now, "claim": answers.get("initial_hypothesis", ""),
        "conceptual": answers.get("initial_hypothesis", ""), "observational": answers.get("initial_hypothesis", ""),
        "subjective_priors": answers.get("subjective_priors", ""), "status": "initial",
    }]})
    core_data = answers.get("core_data_description", "").strip()
    write_yaml(qpath / "data_manifest.yaml", {
        "status": "provided_description" if core_data else "missing_user_data",
        "description": core_data,
        "datasets": [],
        "agent_instructions": "Infer required data, find or generate it, store under data/raw/, and document schema, meaning, provenance, and limitations." if not core_data else "Document concrete data files when added.",
    })
    val = answers.get("validation_suite", "").strip()
    write_yaml(qpath / "validation.yaml", {
        "status": "provided_description" if val else "agent_required",
        "description": val,
        "weighting_preferences_raw": answers.get("validation_weighting_preferences", ""),
        "metrics": [],
        "agent_instructions": "Infer appropriate metrics and formalize this file with name, description, direction, weight, normalization." if not val else "Formalize provided validation description into metrics before scoring.",
    })
    method_selection = answers.get("method_stack") or answers.get("method")
    if answers.get("auto_method"):
        method_stack = auto_recommend_method_stack({**answers, "description": answers.get("problem_statement", "")})
    else:
        method_stack = resolve_method_stack(method_selection)
    write_yaml(qpath / "method_stack.yaml", method_stack.to_dict())
    (qpath / "journal.md").write_text(f"# Research Journal: {qslug}\n\nCreated: {now}\n\n## Quest Started\n\nHero Statement: {answers.get('hero_statement','')}\n\nProblem Statement: {answers.get('problem_statement','')}\n\nInitial Hypothesis: {answers.get('initial_hypothesis','')}\n", encoding="utf-8")
    (qpath / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    return qpath


def list_quests(root: Path) -> list[dict[str, Any]]:
    directory = quests_dir(root)
    out = []
    for q in sorted(directory.glob("*")) if directory.exists() else []:
        if q.is_dir() and (q / "quest.yaml").exists():
            meta = read_yaml(q / "quest.yaml", {})
            state = read_yaml(q / "state.yaml", {})
            out.append({"slug": meta.get("slug", q.name), "status": state.get("quest_status"), "best_score": state.get("best_score")})
    return out


def acquire_lock(qpath: Path) -> str:
    state = read_yaml(qpath / "state.yaml", {})
    if state.get("lock_id"):
        raise RuntimeError(f"Quest locked: {state['lock_id']}")
    lock = uuid.uuid4().hex
    state.update({"lock_id": lock, "quest_status": "running", "last_updated": utc_now()})
    write_yaml(qpath / "state.yaml", state)
    return lock


def release_lock(qpath: Path, lock_id: str, status: str = "idle") -> None:
    state = read_yaml(qpath / "state.yaml", {})
    if state.get("lock_id") == lock_id:
        state["lock_id"] = None
    state["quest_status"] = status
    state["last_updated"] = utc_now()
    write_yaml(qpath / "state.yaml", state)


def next_experiment_id(qpath: Path) -> str:
    nums = []
    for p in (qpath / "experiments").glob("exp_*"):
        if p.is_dir():
            try:
                nums.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return f"exp_{(max(nums) + 1) if nums else 1:03d}"


def run_next(qpath: Path, agent_stub: bool = False, execute: bool = False) -> str:
    lock = acquire_lock(qpath)
    exp_id = next_experiment_id(qpath)
    exp = qpath / "experiments" / exp_id
    try:
        for sub in ["logs", "artifacts/plots", "artifacts/diagrams"]:
            (exp / sub).mkdir(parents=True, exist_ok=False)
        write_yaml(exp / "experiment.yaml", {
            "id": exp_id,
            "created_at": utc_now(),
            "status": "agent_stub" if agent_stub else "agent_required",
            "task_type": "agent_required",
            "model_architecture": "agent_required",
            "input_features": [],
            "target_features": [],
            "validation_technique": "agent_required",
            "technical_diagrams": [
                "artifacts/diagrams/model_architecture.svg",
                "artifacts/diagrams/validation_technique.svg",
            ],
        })
        stack = load_quest_method_stack(qpath)
        preregister_experiment(qpath, exp_id, exp)
        (exp / "hypothesis.md").write_text("# Hypothesis\n\nAgent must evolve one testable hypothesis for this iteration.\n", encoding="utf-8")
        create_stub_notebook(exp / "notebook.ipynb", exp_id)
        if execute:
            execute_notebook(exp / "notebook.ipynb", exp / "executed_notebook.ipynb", exp)
        (exp / "experiment_report.md").write_text("# Experiment Report\n\n## Evidence\nAgent stub only.\n\n## Speculation\nAgent must interpret substantive results.\n", encoding="utf-8")
        append_method_sections_to_report(exp / "experiment_report.md", stack)
        append_journal_entry(qpath, exp_id, "Agent stub created experiment scaffold.")
        state = read_yaml(qpath / "state.yaml", {})
        state.update({"current_experiment": None, "last_experiment": exp_id, "last_updated": utc_now()})
        write_yaml(qpath / "state.yaml", state)
        release_lock(qpath, lock, "idle")
        return exp_id
    except Exception as exc:
        state = read_yaml(qpath / "state.yaml", {})
        failures = state.get("failures") or []
        failures.append({"experiment": exp_id, "error": str(exc), "time": utc_now()})
        state.update({"failures": failures, "last_updated": utc_now()})
        write_yaml(qpath / "state.yaml", state)
        release_lock(qpath, lock, "failed")
        raise


def append_journal_entry(qpath: Path, exp_id: str, note: str = "") -> None:
    append_text(qpath / "journal.md", f"\n\n## Experiment {exp_id.split('_')[-1].lstrip('0') or '0'}\n\n* Hypothesis: See experiments/{exp_id}/hypothesis.md\n* Method: See experiments/{exp_id}/notebook.ipynb\n* Results: Pending or recorded in validation_results.yaml\n* Validation scores: Pending\n* Aggregate score: Pending\n* Interpretation: {note}\n* Debug notes: None recorded\n* Next direction: Agent to decide\n* User injections: None\n")


def continue_quest(qpath: Path, idea: str | None = None, data: str | None = None) -> None:
    injection = {"time": utc_now(), "idea": idea, "data": data}
    path = qpath / "user_injections.yaml"
    existing = read_yaml(path, {"injections": []})
    existing.setdefault("injections", []).append(injection)
    write_yaml(path, existing)
    append_text(qpath / "journal.md", f"\n\n## User Injection\n\nTime: {injection['time']}\n\nIdea: {idea or ''}\n\nData: {data or ''}\n")


def update_state_with_validation(qpath: Path, exp_id: str, validation: dict[str, Any]) -> None:
    state = read_yaml(qpath / "state.yaml", {})
    score = validation.get("aggregate_score")
    state["last_experiment"] = exp_id
    state["last_score"] = score
    if score is not None and (state.get("best_score") is None or score > state.get("best_score")):
        state["best_score"] = score
    state["last_updated"] = utc_now()
    write_yaml(qpath / "state.yaml", state)
