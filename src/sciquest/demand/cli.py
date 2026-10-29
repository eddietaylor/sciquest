"""Command-line access to persistent demand discovery sessions."""

from __future__ import annotations

import csv
import json
import secrets
from pathlib import Path

import typer

from .session import DemandSession
from .world import DemandConfig, SCENARIOS

app = typer.Typer(
    help="Interactive demand discovery: budgeted experiments and sealed evaluation."
)


def emit(value):
    typer.echo(json.dumps(value, indent=2))


@app.command("create")
def create(
    path: Path,
    scenario: str = "clean",
    seed: int | None = None,
    budget: int = 32,
    method: str = "standard_empirical",
    family: str | None = None,
    price_limit: float = 1.5,
    assisted: bool = False,
):
    """Create a new session. Owner-only seed/family arguments are never handed to the scientist."""
    session = DemandSession.create(
        path,
        DemandConfig(
            scenario,
            secrets.randbits(32) if seed is None else seed,
            budget,
            family,
            price_limit,
        ),
        method,
        assisted=assisted,
    )
    emit(session.view())


@app.command("inspect")
def inspect(path: Path):
    """Read the public contract and observations without revealing private truth."""
    emit(DemandSession(path).view())


@app.command("request")
def request(path: Path, proposal: Path):
    """Register and perform one query from a JSON proposal."""
    emit(DemandSession(path).query(json.loads(proposal.read_text())))


@app.command("stop")
def stop(path: Path, decision: Path, revision: int):
    """Freeze a scientific decision and receive sealed prediction features."""
    emit(DemandSession(path).stop(json.loads(decision.read_text()), revision))


@app.command("finish")
def finish(path: Path, notebook: Path, plan: Path):
    """Execute a frozen final notebook and independently evaluate it once."""
    from sciquest.io import read_yaml

    emit(DemandSession(path).finish(notebook, read_yaml(plan)))


@app.command("reference")
def reference(path: Path):
    """Run the registered inspectable reference strategy through a full session."""
    from .reference import run_reference

    emit(run_reference(DemandSession(path)))


@app.command("run-agent")
def run_agent(
    path: Path,
    agent_command: str | None = None,
    max_turns: int = 64,
    timeout: float = 360,
    model: str = "gpt-6.1-sol",
    reasoning_effort: str = "high",
    binary: str = "codex",
):
    """Run fresh memory-disabled model calls. An optional external command runs offline only."""
    from .runner import run_external, run_scientist

    session = DemandSession(path)
    if agent_command:
        emit(run_external(session, agent_command, max_turns, timeout))
    else:
        emit(
            run_scientist(
                session,
                model=model,
                reasoning_effort=reasoning_effort,
                binary=binary,
                max_turns=max_turns,
                timeout=timeout,
            )
        )


@app.command("verify")
def verify(path: Path, reveal: bool = False):
    """Replay the transcript and budget. Reveal truth only after completion."""
    emit(DemandSession(path).verify(reveal))


@app.command("benchmark")
def benchmark(path: Path, seeds: int = 3, budget: int = 32, scenario: str = "all"):
    """Matched-seed pilot of four experimental-design strategies; never overwrites runs."""
    from .reference import POLICIES, run_reference

    if not 1 <= seeds <= 1000:
        raise typer.BadParameter("seeds must be in [1,1000]")
    if scenario != "all" and scenario not in SCENARIOS:
        raise typer.BadParameter("Unknown scenario")
    path.mkdir(parents=True, exist_ok=False)
    results = []
    for case in SCENARIOS if scenario == "all" else [scenario]:
        for seed in range(seeds):
            for policy in POLICIES:
                session = DemandSession.create(
                    path / f"{case}-{seed}-{policy}",
                    DemandConfig(case, seed, budget),
                    policy,
                    assisted=True,
                )
                result = run_reference(session)
                proof = session.verify()
                row = dict(
                    scenario=case,
                    seed=seed,
                    policy=policy,
                    **result["science"],
                    verification=proof,
                )
                results.append(row)
                (path / "results.json").write_text(json.dumps(results, indent=2))
                typer.echo(
                    f"{case} seed={seed} {policy}: cost={row['cost_spent']} demand_rmse={row['interpolation_demand_rmse']:.3f} regret={row['unconstrained_grid_regret']:.3f}"
                )
    columns = [
        "scenario",
        "seed",
        "policy",
        "cost_spent",
        "query_count",
        "mechanism_recovered",
        "correct_identifiability_abstention",
        "correct_model_inadequacy_abstention",
        "family_brier",
        "interpolation_demand_rmse",
        "extrapolation_demand_rmse",
        "predictive_interval_coverage_90",
        "mean_interval_score_90",
        "policy_value",
        "unconstrained_grid_regret",
        "constraint_violation_rate",
    ]
    with (path / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


study_app = typer.Typer(help="Freeze and run an LLM experimental-design comparison.")
app.add_typer(study_app, name="study")


@study_app.command("freeze")
def study_freeze(
    path: Path,
    model: str = "gpt-6.1-sol",
    reasoning_effort: str = "high",
    budget: int = 12,
    max_queries: int = 6,
    timeout: int = 120,
    repeats: int = 1,
    binary: str = "codex",
    concurrency: int = 4,
    assisted: bool = False,
):
    """Save prompts, settings, primary outcome and private matched-world allocation before evaluation."""
    from .study import StudyConfig, freeze_study

    emit(
        freeze_study(
            path,
            StudyConfig(
                model,
                reasoning_effort,
                budget,
                max_queries,
                timeout,
                repeats,
                concurrency,
            ),
            binary,
            assisted=assisted,
        )
    )


@study_app.command("run")
def study_run(path: Path):
    """Execute the frozen schedule once, preserving failures and model usage."""
    from .study import run_study

    for outcome in run_study(path):
        typer.echo(
            f"{outcome['block']} {outcome['policy']}: {outcome['status']} calls={outcome.get('model_calls', len(outcome['calls']))}"
        )


@study_app.command("verify")
def study_verify(path: Path):
    """Verify the complete protocol, model-call receipts, campaigns and summary."""
    from .study import verify_study

    emit(verify_study(path))


worlds_app = typer.Typer(
    help="Create private agent-designed world collections before experiments."
)
app.add_typer(worlds_app, name="worlds")


@worlds_app.command("create")
def worlds_create(
    path: Path,
    batches: int = 32,
    model: str = "gpt-6.1-sol",
    reasoning_effort: str = "high",
    binary: str = "codex",
    timeout: int = 360,
    concurrency: int = 4,
    import_attempt: Path | None = None,
):
    from .worldbank import create_bank_parallel

    emit(
        create_bank_parallel(
            path,
            batches=batches,
            model=model,
            reasoning_effort=reasoning_effort,
            binary=binary,
            timeout=timeout,
            concurrency=concurrency,
            import_attempt=import_attempt,
            progress=emit,
        )
    )


@worlds_app.command("verify")
def worlds_verify(path: Path):
    from .worldbank import verify_bank

    emit(verify_bank(path))


@worlds_app.command("create-v2")
def worlds_create_v2(
    path: Path,
    per_stratum: int = 4,
    concurrency: int = 4,
    maximum_attempts: int = 8,
    model: str = "gpt-6.1-sol",
    reasoning_effort: str = "high",
    binary: str = "codex",
    timeout: int = 360,
):
    """Create a private v2 DEVELOPMENT collection with decision/access/budget gates."""
    from .collection_v2 import create_collection

    emit(
        create_collection(
            path,
            per_stratum=per_stratum,
            concurrency=concurrency,
            maximum_attempts=maximum_attempts,
            model=model,
            reasoning_effort=reasoning_effort,
            binary=binary,
            timeout=timeout,
            progress=emit,
        )
    )


@worlds_app.command("verify-v2")
def worlds_verify_v2(path: Path):
    from .collection_v2 import verify_collection

    emit(verify_collection(path))


@worlds_app.command("session-v2")
def worlds_session_v2(
    path: Path, collection: Path, identity: str, method: str = "standard_empirical"
):
    """Create a blind session from an admitted world; does not launch a scientist."""
    from .collection_v2 import create_session

    emit(create_session(path, collection, identity, method).view(scientist=True))


blind_app = typer.Typer(
    help="Freeze, run and audit fully blind scientific-discovery pilots."
)
app.add_typer(blind_app, name="blind-study")


@blind_app.command("freeze")
def blind_freeze(
    path: Path,
    bank: Path,
    split: str = "pilot",
    engineering_stratum: str | None = None,
    budget: int = 24,
    max_queries: int = 12,
    max_analyses: int = 8,
    max_calls: int = 24,
    timeout: int = 360,
    concurrency: int = 4,
    model: str = "gpt-6.1-sol",
    reasoning_effort: str = "high",
    binary: str = "codex",
):
    from .research import PilotConfig, freeze_pilot

    emit(
        freeze_pilot(
            path,
            bank,
            PilotConfig(
                model=model,
                reasoning_effort=reasoning_effort,
                budget=budget,
                max_queries=max_queries,
                max_analyses=max_analyses,
                max_calls=max_calls,
                timeout=timeout,
                concurrency=concurrency,
            ),
            split=split,
            engineering_stratum=engineering_stratum,
            binary=binary,
        )
    )


@blind_app.command("run")
def blind_run(path: Path, confirmatory: bool = False):
    from .research import run_pilot

    outcomes = run_pilot(path, progress=emit, allow_evaluation=confirmatory)
    emit(
        dict(
            finished=len(outcomes),
            completed=sum(r["status"] == "completed" for r in outcomes),
        )
    )


@blind_app.command("grade")
def blind_grade(path: Path):
    from .grading import grade_pilot, verify_grades

    grade_pilot(path, progress=emit)
    emit(verify_grades(path))


@blind_app.command("verify")
def blind_verify(path: Path):
    from .research import verify_pilot

    emit(verify_pilot(path))


@blind_app.command("analyze")
def blind_analyze(path: Path):
    from .planning import analyze_pilot

    emit(analyze_pilot(path))


@blind_app.command("freeze-next")
def blind_freeze_next(path: Path, pilot: Path):
    from .planning import freeze_confirmatory

    emit(freeze_confirmatory(path, pilot))


comparison_app = typer.Typer(
    help="Prepare matched v2 scoring/guidance comparisons; no campaign is launched by preparation."
)
app.add_typer(comparison_app, name="comparison-v2")

workflow_v3_app = typer.Typer(
    help="Opt-in delivery reserves and separately versioned scientific review. Does not rerun old campaigns."
)
app.add_typer(workflow_v3_app, name="workflow-v3")


@workflow_v3_app.command("run")
def workflow_v3_run(
    path: Path,
    binary: str = "codex",
    model: str = "gpt-6.1-sol",
    reasoning_effort: str = "high",
    timeout: float = 360,
    repair_limit: int = 2,
    reserve_calls: int = 4,
    reserve_input_tokens: int = 60000,
    reserve_output_tokens: int = 18000,
):
    """Freeze and run one fresh matched-guidance session; existing attempts cannot restart."""
    from .delivery_v3 import DeliveryPolicy, run

    result = run(
        DemandSession(path),
        policy=DeliveryPolicy(
            finalization_calls=reserve_calls,
            finalization_input_tokens=reserve_input_tokens,
            finalization_output_tokens=reserve_output_tokens,
        ),
        binary=binary,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        repair_limit=repair_limit,
    )
    emit(
        {k: result[k] for k in ("phase", "agent_error", "runner_status") if k in result}
    )


@workflow_v3_app.command("verify")
def workflow_v3_verify(path: Path):
    from .delivery_v3 import contract
    from .comparison_v2 import verify_comparison_session

    source = DemandSession(path)
    if contract(source) is None:
        raise typer.BadParameter("Session has no v3 delivery contract")
    emit(verify_comparison_session(source))


@workflow_v3_app.command("review")
def workflow_v3_review(
    path: Path,
    session: Path,
    binary: str = "codex",
    model: str = "gpt-6.1-sol",
    reasoning_effort: str = "high",
    timeout: float = 600,
):
    """One new diagnostic assessment, preserving raw failure evidence without retries."""
    from .assessment_v3 import review_packet, assess_once
    from .comparison_v2 import verify_comparison_session
    from .backend import describe_backend

    source = DemandSession(session)
    verify_comparison_session(source)
    packet = review_packet(source)
    if packet is None:
        raise typer.BadParameter("No frozen scientific decision is available")
    result = assess_once(
        path,
        packet,
        dict(
            backend=describe_backend(binary),
            model=model,
            reasoning_effort=reasoning_effort,
        ),
        timeout,
    )
    emit(
        {k: result[k] for k in ("version", "status", "error", "reason") if k in result}
    )


@workflow_v3_app.command("verify-review")
def workflow_v3_verify_review(path: Path, session: Path):
    from .assessment_v3 import review_packet, verify_assessment
    from .comparison_v2 import verify_comparison_session

    source = DemandSession(session)
    verify_comparison_session(source)
    result = verify_assessment(path, review_packet(source))
    emit(dict(version=result["version"], status=result["status"], verified=True))


assessment_v4_app = typer.Typer(
    help="Prospective diagnostic review with separate truth and warrant provenance."
)
app.add_typer(assessment_v4_app, name="assessment-v4")


@assessment_v4_app.command("review")
def assessment_v4_review(
    path: Path,
    session: Path,
    binary: str = "codex",
    model: str = "gpt-6.1-sol",
    reasoning_effort: str = "high",
    timeout: float = 900,
):
    """One fresh assessment; structural acceptance is not semantic validation."""
    from .assessment_v4 import review_packet, assess_once
    from .comparison_v2 import verify_comparison_session
    from .backend import describe_backend

    source = DemandSession(session)
    verify_comparison_session(source)
    packet = review_packet(source)
    if packet is None:
        raise typer.BadParameter("No frozen scientific decision is available")
    result = assess_once(
        path,
        packet,
        dict(
            backend=describe_backend(binary),
            model=model,
            reasoning_effort=reasoning_effort,
        ),
        timeout,
    )
    emit(
        {k: result[k] for k in ("version", "status", "error", "reason") if k in result}
    )


@assessment_v4_app.command("verify")
def assessment_v4_verify(path: Path, session: Path):
    from .assessment_v4 import review_packet, verify_assessment
    from .comparison_v2 import verify_comparison_session

    source = DemandSession(session)
    verify_comparison_session(source)
    packet = review_packet(source)
    if packet is None:
        raise typer.BadParameter("No frozen scientific decision is available")
    result = verify_assessment(path, packet)
    emit(
        dict(
            version=result["version"],
            status=result["status"],
            verified=True,
            interpretation="Provenance/structure only; not a proof of semantic correctness.",
        )
    )


@comparison_app.command("adopt")
def comparison_adopt(path: Path, bank: Path):
    """Replay the archived verifier and preserve a new unchanged-physics adoption."""
    from .adoption_v2 import adopt_collection

    record = adopt_collection(path, bank)
    emit(
        dict(
            version=record["version"],
            manifest_hash=record["manifest_hash"],
            verified=True,
        )
    )


@comparison_app.command("prepare")
def comparison_prepare(path: Path, adoption: Path, per_stratum: int = 1):
    """Freeze readiness contracts only; a new executable pilot registration is still required."""
    from .comparison_v2 import prepare_comparison

    spec = prepare_comparison(path, adoption, per_stratum=per_stratum)
    emit(
        dict(
            version=spec["version"],
            status=spec["status"],
            planned_campaigns=len(spec["order"]),
            campaigns_launched=0,
        )
    )


@comparison_app.command("verify")
def comparison_verify(path: Path):
    from .comparison_v2 import load_comparison

    spec, blocks = load_comparison(path)
    emit(
        dict(
            verified=True,
            status=spec["status"],
            development_worlds=len(blocks),
            campaigns_launched=0,
        )
    )


@comparison_app.command("session")
def comparison_session(
    path: Path, adoption: Path, identity: str, policy: str = "standard_empirical"
):
    """Create an independently pinned, matched-guidance session without invoking a model."""
    from .comparison_v2 import create_comparison_session

    emit(
        create_comparison_session(path, adoption, identity, policy).view(scientist=True)
    )


@comparison_app.command("verify-session")
def comparison_verify_session(path: Path):
    from .comparison_v2 import verify_comparison_session

    emit(verify_comparison_session(DemandSession(path)))


@comparison_app.command("review")
def comparison_review(
    path: Path,
    session: Path,
    binary: str = "codex",
    model: str = "gpt-6.1-sol",
    reasoning_effort: str = "high",
):
    """Create one retained private policy-masked diagnostic review; never reruns an existing path."""
    from .assessment_v2 import assess_once, review_packet
    from .backend import describe_backend
    from .comparison_v2 import verify_comparison_session

    source = DemandSession(session)
    verify_comparison_session(source)
    packet = review_packet(source)
    if packet is None:
        raise typer.BadParameter("No frozen scientific decision is available to review")
    result = assess_once(
        path,
        packet,
        dict(
            backend=describe_backend(binary),
            model=model,
            reasoning_effort=reasoning_effort,
        ),
    )
    emit(
        dict(
            status=result["status"],
            version=result["version"],
            error=result.get("error"),
            reason=result.get("reason"),
        )
    )
