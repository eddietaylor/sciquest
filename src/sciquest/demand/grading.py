"""Policy-masked, truth-grounded diagnostic explanation assessments.

These are fresh model judgments, not independent human adjudication or a proof
of understanding. The scorer never sends packets or grades to a scientist.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sciquest.io import content_hash, file_hash

from .backend import run_call
from .session import DemandSession, public_query
from .world import evaluation_rows
from .worldbank import dump, verify_call

AXES = (
    "mechanism_faithfulness",
    "evidence_use",
    "epistemic_calibration",
    "test_and_revision",
)
CLAIM_SCHEMA = dict(
    type="object",
    properties=dict(quote=dict(type="string"), assessment=dict(type="string")),
    required=["quote", "assessment"],
    additionalProperties=False,
)
GRADING_SCHEMA = dict(
    type="object",
    properties={
        **{a: dict(type="integer", enum=[0, 1, 2, 3, 4]) for a in AXES},
        "rationale": dict(type="string"),
        "supported_claims": dict(type="array", items=CLAIM_SCHEMA),
        "unsupported_claims": dict(type="array", items=CLAIM_SCHEMA),
    },
    required=[*AXES, "rationale", "supported_claims", "unsupported_claims"],
    additionalProperties=False,
)
GRADING_PROMPT = """Assess one anonymous scientific-discovery campaign against PRIVATE ground truth.
You are an evaluator, never the experimentalist. No philosophy identity,
instruction profile, other campaigns, numerical forecast scores or other
judges' assessments are provided. Do not infer or reward a philosophical style.
Judge the actual evidence, claims and stated limits under a restricted budget.

Return four separate integer ratings, 0..4:
mechanism_faithfulness: 0 main causal explanation contradicted; 1 little correct
mechanistic content; 2 partly correct provisional explanation with substantive
gaps; 3 substantially faithful price/context/capacity explanation; 4 faithful
mechanism at the resolution claimed, with correct relevant qualifications.
An honest 'unknown' can be calibrated but is not by itself mechanism recovery.
evidence_use: 0 unsupported assertion/no empirical work; 1 weak links between
readings and conclusion; 2 reasonable use of some evidence; 3 clear appropriate
comparison of predictions with measurements; 4 systematic, defensible evidence
integration including contradictory evidence and dependent/paired measurements.
epistemic_calibration: 0 unjustified certainty or major identification error;
1 substantial overclaiming; 2 mixed qualification; 3 appropriately provisional
claims; 4 precise distinctions among assumptions, observations and what can be
identified. Lack of causal access can justify abstention. Noise levels, nuisance
law and true model class were UNKNOWN to the scientist; do not penalize it merely
for not guessing an undisclosed equation or exact parameter.
test_and_revision: 0 no meaningful test; 1 weakly motivated sampling; 2 plausible
tests with limited interpretation; 3 discriminating experiments or justified
replication/measurement and appropriate updating; 4 convincing tests and actual
revision where warranted, or clearly justified stability where evidence agrees.
Do not reward revision merely because a belief changed.

Give a concise rationale and separate supported_claims/unsupported_claims. Each
claim needs an EXACT quote from the scientist's supplied decision or plan text,
plus a truth/evidence-based assessment. Use empty arrays if no such claims exist.
Never manufacture a quotation. A true guess unsupported by observations can get
faithfulness credit while receiving low evidence/calibration credit. Prediction
accuracy is not itself a mechanism-recovery criterion. These are diagnostic
model judgments with possible shared-model/style biases, not certified verdicts.
"""


def packet(session: DemandSession):
    with session.transaction() as (c, state):
        events = session._events(c)
        world, decision = state["world"], state["decision"]
    if decision is None:
        return None
    evidence = []
    for event in events:
        if event["payload"]["type"] != "query":
            continue
        query = public_query(event["payload"])
        # Explicit policy-specific contracts/labels are excluded. Their effects
        # may still be inferable from scientific prose; blinding is partial.
        evidence.append(
            dict(
                action=query["action"],
                observation=query["observation"],
                plan={
                    k: query["plan"][k]
                    for k in (
                        "question",
                        "hypothesis",
                        "scope",
                        "assumptions",
                        "data_access",
                        "stopping_rule",
                        "action",
                        "confirmatory_status",
                        "predictions",
                    )
                },
            )
        )
    grid = [
        {
            k: r[k]
            for k in (
                "price",
                "context",
                "capacity",
                "expected_demand",
                "expected_sales",
            )
        }
        for r in evaluation_rows(world)
    ]
    truth_contract = "The private expression is g(p,x,z), z uniform over -1,0,1. Demand=max(0,g). In confounded/observational_only strata add strength*z-g(1+z/4,x,z)+g(1,x,z) before truncation; historical price=1+z/4. Other historical prices are randomized. Historical capacity is 60 for censored, 180 otherwise. Intervention sets price/context/capacity on a fresh unit, preserving the nuisance adjustment. Sales=min(capacity,demand), then independent mean-zero Gaussian sensor error is added. Precise instrumentation uses precise_sigma. Audits measure the SAME unit's latent demand, with precise instrumentation; replications sample NEW units. Observational_only has no price intervention access and has an observationally equivalent no-price-effect twin g(1,x,z). Ground-truth tables are private and were unavailable to the scientist."
    if world["spec"]["version"] == "designed-demand-v2":
        truth_contract += " V2 additionally adds the private sensor_bias to every instrument reading, including audits and known-reference calibration. Calibration readings measure a known reference, not a demand unit; they share instrument error at the selected precision and can be replicated, but cannot receive demand audits. This truth packet and all oracle witnesses remain unavailable to the scientist."
    return dict(
        decision=decision,
        evidence=evidence,
        private_world=world["spec"],
        truth_contract=truth_contract,
        private_intervention_truth=grid,
    )


def validate_grade(grade, source_packet):
    if not isinstance(grade, dict) or set(grade) != set(GRADING_SCHEMA["properties"]):
        raise ValueError("Invalid explanation assessment fields")
    if (
        any(type(grade[a]) is not int or not 0 <= grade[a] <= 4 for a in AXES)
        or not isinstance(grade["rationale"], str)
        or not grade["rationale"].strip()
    ):
        raise ValueError("Invalid explanation ratings or rationale")
    texts = list(source_packet["decision"].values())
    for query in source_packet["evidence"]:
        texts.extend(v for v in query["plan"].values() if isinstance(v, str))
    for kind in ("supported_claims", "unsupported_claims"):
        if not isinstance(grade[kind], list):
            raise ValueError("Claims must be a list")
        for claim in grade[kind]:
            if (
                set(claim) != {"quote", "assessment"}
                or not isinstance(claim["quote"], str)
                or not claim["quote"].strip()
                or not isinstance(claim["assessment"], str)
                or not claim["assessment"].strip()
                or not any(claim["quote"] in text for text in texts)
            ):
                raise ValueError("Explanation judge supplied an ungrounded quotation")
    return grade


def grade_pilot(path: Path, progress=None):
    from .research import load_pilot, verify_pilot

    verify_pilot(path)
    spec, allocation, digest = load_pilot(path)
    path = Path(path).resolve()
    outcomes = json.loads((path / "outcomes.json").read_text())
    seed = allocation["order_seed"] ^ 0xD15C0A
    mapping, jobs = [], []
    for result in outcomes:
        key = result["block"] + "-" + result["policy"]
        anonymous = content_hash([seed, key])[:20]
        source = packet(DemandSession(path / "campaigns" / key / "session"))
        mapping.append(
            dict(anonymous=anonymous, campaign=key, packet_hash=content_hash(source))
        )
        for judge in range(spec["explanations"]["graders_per_available_decision"]):
            jobs.append((anonymous, judge, source))
    random.Random(seed).shuffle(jobs)
    dump(
        path / "private/grading_allocation.json",
        dict(protocol_hash=digest, mapping=mapping),
    )

    def run_one(job):
        anonymous, judge, source = job
        directory = path / "grades" / anonymous / f"judge-{judge}"
        if (directory / "assessment.json").exists():
            return json.loads((directory / "assessment.json").read_text())
        if directory.exists():
            raise ValueError(
                "Interrupted explanation assessment must be audited, not silently rerun"
            )
        directory.mkdir(parents=True)
        dump(directory / "packet.private.json", source)
        if source is None:
            result = dict(
                anonymous=anonymous,
                judge=judge,
                status="decision_unavailable",
                grade=None,
                usage=None,
            )
        else:
            prompt = (
                spec["explanations"]["prompt"]
                + "\nANONYMOUS PRIVATE PACKET\n"
                + json.dumps(source)
            )
            receipt = run_call(
                directory / "call",
                prompt,
                spec["explanations"]["schema"],
                spec,
                path,
                spec["timeout"],
            )
            error, grade = None, None
            try:
                if receipt["status"] != "succeeded":
                    raise ValueError("Explanation model call failed")
                grade = validate_grade(receipt["response"], source)
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
            result = dict(
                anonymous=anonymous,
                judge=judge,
                status="completed" if error is None else "failed",
                error=error,
                grade=grade,
                usage=receipt.get("usage"),
                receipt_sha256=file_hash(directory / "call/receipt.json"),
            )
        dump(directory / "assessment.json", result)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=spec["concurrency"]) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress:
                progress(
                    dict(
                        grades_finished=len(results),
                        grades_total=len(jobs),
                        status=result["status"],
                    )
                )
    results.sort(key=lambda r: (r["anonymous"], r["judge"]))
    dump(path / "explanation_assessments.json", results)
    return results


def verify_grades(path: Path):
    from .research import load_pilot

    spec, _, digest = load_pilot(path)
    allocation = json.loads(
        (Path(path) / "private/grading_allocation.json").read_text()
    )
    results = json.loads((Path(path) / "explanation_assessments.json").read_text())
    mapping = {r["anonymous"]: r for r in allocation["mapping"]}
    expected = {(key, judge) for key in mapping for judge in range(2)}
    if (
        allocation["protocol_hash"] != digest
        or {(r["anonymous"], r["judge"]) for r in results} != expected
        or len(results) != len(expected)
    ):
        raise ValueError("Explanation assessment allocation changed")
    for result in results:
        row = mapping[result["anonymous"]]
        source = packet(
            DemandSession(Path(path) / "campaigns" / row["campaign"] / "session")
        )
        directory = (
            Path(path) / "grades" / result["anonymous"] / f"judge-{result['judge']}"
        )
        if (
            content_hash(source) != row["packet_hash"]
            or source != json.loads((directory / "packet.private.json").read_text())
            or result != json.loads((directory / "assessment.json").read_text())
        ):
            raise ValueError("Anonymous explanation packet or assessment changed")
        if source is not None:
            prompt = (
                spec["explanations"]["prompt"]
                + "\nANONYMOUS PRIVATE PACKET\n"
                + json.dumps(source)
            )
            receipt = verify_call(
                directory / "call",
                expected_prompt=prompt,
                expected_schema=spec["explanations"]["schema"],
            )
            if (
                file_hash(directory / "call/receipt.json") != result["receipt_sha256"]
                or receipt.get("usage") != result["usage"]
            ):
                raise ValueError("Explanation receipt changed")
            if (
                result["status"] == "completed"
                and validate_grade(receipt["response"], source) != result["grade"]
            ):
                raise ValueError("Explanation ratings changed")
    return dict(
        verified=True,
        assessments=len(results),
        completed=sum(r["status"] == "completed" for r in results),
        scope="Two fresh policy-masked model assessments where a decision exists, with exact quotations, truth packets and receipts. Partial blinding; not independent human validation.",
    )
