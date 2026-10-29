"""Pilot analysis and prospective selection of an untouched evaluation study."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from sciquest.io import content_hash, utc_now

from .designed import STRATA
from .grading import AXES, verify_grades
from .research import POLICIES, load_pilot, summarize, verify_pilot
from .worldbank import dump, load_bank


def paired_power(worlds, difference, sd, alpha=0.05 / 3):
    from scipy.stats import nct, t

    critical = t.ppf(1 - alpha / 2, worlds - 1)
    noncentrality = difference * math.sqrt(worlds) / sd
    return float(
        nct.sf(critical, worlds - 1, noncentrality)
        + nct.cdf(-critical, worlds - 1, noncentrality)
    )


def required_worlds(difference, sd, target=0.8, alpha=0.05 / 3, minimum=24):
    for n in range(6 * math.ceil(minimum / 6), 10002, 6):
        if paired_power(n, difference, sd, alpha) >= target:
            return n
    raise ValueError("Planning assumptions require more than 10000 independent worlds")


def write_table(path, rows, fields=None):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields or list(rows[0]), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def analyze_pilot(path: Path):
    from scipy.stats import chi2, t

    path = Path(path).resolve()
    proof = verify_pilot(path)
    grade_proof = verify_grades(path)
    spec, allocation, digest = load_pilot(path)
    if spec["split"] != "pilot":
        raise ValueError("Main pilot planning uses the six reserved pilot worlds")
    outcomes = json.loads((path / "outcomes.json").read_text())
    summary = summarize(spec, outcomes)
    grades = json.loads((path / "explanation_assessments.json").read_text())
    mapping = json.loads((path / "private/grading_allocation.json").read_text())[
        "mapping"
    ]
    by_campaign = {r["campaign"]: r["anonymous"] for r in mapping}
    stratum = {b["block"]: b["world"]["spec"]["stratum"] for b in allocation["blocks"]}
    campaign_rows = []
    grade_rows = []
    for outcome in outcomes:
        key = outcome["block"] + "-" + outcome["policy"]
        campaign_rows.append(
            dict(
                block=outcome["block"],
                stratum=stratum[outcome["block"]],
                policy=outcome["policy"],
                status=outcome["status"],
                primary_loss=outcome["primary_loss"],
                demand_rmse=(outcome["science"] or {}).get(
                    "interventional_demand_rmse"
                ),
                cost=outcome["cost_spent"],
                queries=outcome["query_count"],
                model_calls=len(outcome["calls"]),
                error=outcome["error"],
                interval_coverage=(outcome["science"] or {}).get(
                    "predictive_interval_coverage_90"
                ),
                regret=(outcome["science"] or {}).get("unconstrained_grid_regret"),
            )
        )
        ratings = [
            r["grade"]
            for r in grades
            if r["anonymous"] == by_campaign[key] and r["status"] == "completed"
        ]
        grade_rows.append(
            dict(
                block=outcome["block"],
                policy=outcome["policy"],
                available_ratings=len(ratings),
                **{
                    a: sum(r[a] for r in ratings) / len(ratings) if ratings else None
                    for a in AXES
                },
                disagreement_flag=any(
                    abs(ratings[0][a] - ratings[1][a]) >= 2 for a in AXES
                )
                if len(ratings) == 2
                else None,
            )
        )
    comparisons = {}
    planning_sd = spec["planning"]["sd_floor"]
    paired_rows = []
    for policy, result in summary["paired_vs_empirical"].items():
        n, sd, mean = result["worlds"], result["paired_sd"], result["mean_difference"]
        critical = float(
            t.ppf(
                1
                - spec["planning"]["familywise_alpha"]
                / (2 * spec["planning"]["comparisons"]),
                n - 1,
            )
        )
        half = critical * sd / math.sqrt(n)
        upper_sd = sd * math.sqrt(
            (n - 1)
            / float(chi2.ppf(1 - spec["planning"]["variance_upper_confidence"], n - 1))
        )
        planning_sd = max(planning_sd, upper_sd)
        comparisons[policy] = dict(
            independent_worlds=n,
            mean_loss_difference=mean,
            paired_sd=sd,
            bonferroni_interval=[mean - half, mean + half],
            sd_upper80=upper_sd,
            interpretation="Exploratory interval under an approximate independent-world t model; six heterogeneous worlds are a weak variance estimate.",
        )
        paired_rows.extend(
            dict(policy=policy, stratum=stratum[p["block"]], **p)
            for p in result["pairs"]
        )
    target = spec["planning"]["minimum_meaningful_loss_difference"]
    alpha = spec["planning"]["familywise_alpha"] / spec["planning"]["comparisons"]
    n = required_worlds(
        target,
        planning_sd,
        spec["planning"]["target_power"],
        alpha,
        spec["planning"]["minimum_worlds"],
    )
    complete_fraction = proof["completed"] / spec["campaigns"]
    ready = complete_fraction >= spec["feasibility"]["minimum_completed_fraction"]
    power = dict(
        target_difference=target,
        planning_sd=planning_sd,
        minimum_sd_floor=spec["planning"]["sd_floor"],
        method="Two-sided paired t approximation with noncentral-t power, Bonferroni alpha across three comparisons, one-sided 80% upper SD estimate and preregistered SD floor. No pilot effect estimate is used as a target.",
        assumptions="Approximate normal independent WORLD differences; the fixed stratified, designer-generated bank and model stochasticity can depart from this assumption. No finite-sample power guarantee is claimed. No finite-population correction is used.",
        planned_independent_worlds=n,
        planned_campaigns=4 * n,
        target_power=spec["planning"]["target_power"],
        nominal_power=paired_power(n, target, planning_sd, alpha),
        sensitivity=[
            dict(
                minimum_difference=delta,
                required_worlds=required_worlds(delta, planning_sd, alpha=alpha),
            )
            for delta in (0.05, 0.10, 0.15)
        ],
        caveat="Six pilot worlds cannot support a precise effect or variance estimate. Include conservative sensitivity; future model/runner changes require a new protocol and fresh development evidence.",
    )
    analysis = dict(
        protocol_hash=digest,
        verification=proof,
        grading_verification=grade_proof,
        summary=summary,
        comparisons=comparisons,
        power=power,
        feasibility=dict(
            completed_fraction=complete_fraction,
            technical_readiness=ready,
            verified_information_boundary=True,
            scientific_validity="Review the recorded scientific traces and explanation assessments. Completion and forecasting do not certify mechanism discovery or philosophical fidelity.",
        ),
        explanation_rubric={
            p: {
                a: sum(
                    r[a] for r in grade_rows if r["policy"] == p and r[a] is not None
                )
                / sum(r[a] is not None for r in grade_rows if r["policy"] == p)
                if any(r[a] is not None for r in grade_rows if r["policy"] == p)
                else None
                for a in AXES
            }
            for p in POLICIES
        },
        explanation_disagreements=sum(
            r["disagreement_flag"] is True for r in grade_rows
        ),
        explanation_coverage={
            p: dict(
                ratings=sum(
                    r["available_ratings"] for r in grade_rows if r["policy"] == p
                ),
                potential_ratings=2 * sum(r["policy"] == p for r in grade_rows),
            )
            for p in POLICIES
        },
        resource_accounting=dict(
            experimentalist_calls=proof["model_calls"],
            experimentalist_known_input_tokens=sum(
                a["known_input_tokens"] for a in summary["arms"].values()
            ),
            experimentalist_known_output_tokens=sum(
                a["known_output_tokens"] for a in summary["arms"].values()
            ),
            grader_calls=sum(r["status"] != "decision_unavailable" for r in grades),
            grader_known_input_tokens=sum(
                (r.get("usage") or {}).get("input_tokens", 0) for r in grades
            ),
            grader_known_output_tokens=sum(
                (r.get("usage") or {}).get("output_tokens", 0) for r in grades
            ),
            scope="Separate scientist and assessment resources; missing usage is reported by coverage, not inferred as zero. Designer-generation receipts are preserved in the independent bank. No dollar-cost claim.",
        ),
        literature=[
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC3081994/",
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html",
        ],
        interpretation="This pilot tests a bounded static demand-discovery workflow and estimates feasibility/cost. It cannot establish superiority of a philosophy or industry validity.",
    )
    report = path / "analysis"
    report.mkdir(exist_ok=True)
    dump(report / "analysis.json", analysis)
    dump(report / "power_plan.json", power)
    write_table(report / "campaigns.csv", campaign_rows)
    write_table(report / "paired_differences.csv", paired_rows)
    write_table(report / "explanation_scores.csv", grade_rows)
    write_table(
        report / "power_sensitivity.csv",
        [
            dict(
                worlds=k,
                difference=delta,
                power=paired_power(k, delta, planning_sd, alpha),
            )
            for k in (24, 48, 72, 120, 180)
            for delta in (0.05, 0.10, 0.15)
        ],
    )
    lines = [
        "BLIND SCIENTIFIC-DISCOVERY PILOT — DEVELOPMENT EVIDENCE",
        f"Protocol: {digest}",
        f"Worlds: {spec['worlds']}; campaigns: {spec['campaigns']}; completed: {proof['completed']}",
        "Primary: failure-inclusive min(1, all-intervention demand RMSE / 100). Lower is better.",
        "",
        "Arm                          completed  mean loss  complete-case RMSE  calls",
    ]
    for p, row in summary["arms"].items():
        rmse = (
            "NA"
            if row["mean_complete_case_rmse"] is None
            else f"{row['mean_complete_case_rmse']:.3f}"
        )
        lines.append(
            f"{p:30s} {row['completed']:3d}/{row['campaigns']}    {row['mean_primary_loss']:.4f}     {rmse:>8s}       {row['model_calls']}"
        )
    lines += [
        "",
        "Paired differences versus empirical; six world units, not 24 independent subjects:",
    ]
    for p, row in comparisons.items():
        lines.append(
            f"{p}: mean {row['mean_loss_difference']:.4f}; exploratory simultaneous interval {row['bonferroni_interval']}; SD {row['paired_sd']:.4f}"
        )
    lines += [
        "",
        f"Explanation assessments: {grade_proof['completed']}/{grade_proof['assessments']} available; {analysis['explanation_disagreements']} campaigns have >=2-point disagreement on an axis.",
        "Two fresh, policy-masked model judges share a base model and can share errors. Scores are diagnostic; exact quotations and private truth packets are preserved.",
        "",
        f"Planning target fixed BEFORE pilot: {target:.2f} mean loss units, alpha .05 familywise, 80% power.",
        f"Conservative planning SD: {planning_sd:.4f}; proposed N: {n} independent worlds / {4 * n} campaigns.",
        f"Technical scale-up gate: {'PASS' if ready else 'FAIL — fix feasibility before a large study'}.",
        "",
        "No philosophy-superiority or mechanism-recovery conclusion follows from this pilot. Scientific traces, assumptions, negative results and limitations must be reviewed. Pilot worlds are development data permanently.",
    ]
    (report / "REPORT.txt").write_text("\n".join(lines) + "\n")
    return analysis


def freeze_confirmatory(path: Path, pilot: Path):
    """Select an untouched balanced evaluation set; do not run its campaigns."""
    pilot = Path(pilot).resolve()
    path = Path(path).resolve()
    verify_pilot(pilot)
    verify_grades(pilot)
    analysis = json.loads((pilot / "analysis/analysis.json").read_text())
    spec, _, digest = load_pilot(pilot)
    # Recompute before using saved analysis; a tampered projection cannot choose N.
    if analyze_pilot(pilot) != analysis:
        raise ValueError("Pilot analysis differs from authoritative results")
    _, manifest, private = load_bank(Path(spec["bank"]))
    n = analysis["power"]["planned_independent_worlds"]
    selected = []
    for stratum in STRATA:
        candidates = [
            w
            for w in private["worlds"]
            if w["split"] == "evaluation" and w["spec"]["stratum"] == stratum
        ]
        candidates.sort(
            key=lambda w: content_hash(
                [manifest["private_commitment"], w["id"], "confirmatory-rank-v1"]
            )
        )
        if len(candidates) < n // 6:
            raise ValueError(
                f"Sealed bank has insufficient worlds for planned N={n}; extend the collection without exposing pilot results to its designer before freezing"
            )
        selected.extend(candidates[: n // 6])
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    (path / "private").mkdir(mode=0o700)
    blocks = [dict(block=f"b{i:03d}", world=w) for i, w in enumerate(selected)]
    order_seed = int(
        content_hash([manifest["private_commitment"], digest, "confirmatory-order-v1"]),
        16,
    )
    schedule = [dict(block=b["block"], policy=p) for b in blocks for p in POLICIES]
    import random

    random.Random(order_seed).shuffle(schedule)
    allocation = dict(blocks=blocks, order_seed=order_seed)
    final = {
        **spec,
        "version": "blind-demand-confirmatory-v1",
        "frozen_at": utc_now(),
        "purpose": "Prospective philosophy-inspired instruction comparison on previously untouched designed worlds",
        "split": "evaluation",
        "worlds": n,
        "campaigns": 4 * n,
        "order": schedule,
        "allocation_commitment": content_hash(allocation),
        "pilot_protocol_hash": digest,
        "pilot_analysis_hash": content_hash(analysis),
        "planning_result": analysis["power"],
        "execution_gate": "ready"
        if analysis["feasibility"]["technical_readiness"]
        else "not_ready_feasibility_failed",
        "confirmatory_analysis": "Equal-weight world-level paired primary-loss differences versus empirical. Three two-sided paired t comparisons with Bonferroni familywise alpha .05; report all outcomes, simultaneous intervals, raw failures and descriptive stratified results. Check sensitivity to bounded/skewed losses with world-level resampling; do not substitute a more favorable primary analysis. Explanation ratings remain separate, diagnostic and partially blinded.",
        "sampling": "Balanced stratum sampling; deterministic rankings bound to the pre-pilot bank commitment. No evaluation performance or law inspection selected worlds. All pilot/engineering IDs excluded.",
        "status": "frozen_not_run",
    }
    dump(path / "protocol.json", final)
    (path / "protocol.sha256").write_text(content_hash(final) + "\n")
    dump(path / "private/allocation.json", allocation)
    (path / "private/allocation.json").chmod(0o600)
    dump(
        path / "evaluation_collection.json",
        dict(
            bank_manifest_hash=content_hash(manifest),
            worlds=[
                dict(
                    block=b["block"],
                    id=b["world"]["id"],
                    commitment=content_hash(b["world"]),
                )
                for b in blocks
            ],
            independent_worlds=n,
            prior_scientist_campaigns_on_these_worlds=0,
        ),
    )
    return final
