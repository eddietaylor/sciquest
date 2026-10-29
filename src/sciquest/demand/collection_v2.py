"""Private v2 development collections, validated and sealed before scientist runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter
import json
import secrets
import threading

from sciquest.io import content_hash, file_hash, utc_now
from sciquest.runtime import runtime_manifest
from .admission import admission_report
from .backend import describe_backend, run_call
from .designed import STRATA, P_GRID, X_GRID, Z_GRID
from .designed_v2 import VERSION, FIELDS, latent
from .worldbank import dump, verify_call

COLLECTION_VERSION = "private-demand-development-v2"
SCHEMA = dict(
    type="object",
    properties={
        k: dict(
            type="number"
            if k
            in {
                "sensor_sigma",
                "precise_sigma",
                "sensor_bias",
                "confounding_strength",
                "price_limit",
            }
            else "string",
            **(
                {"enum": [VERSION]}
                if k == "version"
                else {"enum": list(STRATA)}
                if k == "stratum"
                else {}
            ),
        )
        for k in sorted(FIELDS)
    },
    required=sorted(FIELDS),
    additionalProperties=False,
)
PROMPT = """You privately design a static demand world for a scientific-discovery laboratory.
Create ONE original world in the requested stratum. You see no experimentalist
messages, philosophy identity, scientific results or grader feedback. Your law,
this prompt, all oracle checks and explanations stay private. Do not encode any
message, identifier, nonce or answer into a law or observation.

Return version designed-demand-v2 and fields in the response schema. Expression
g(p,x,z) uses p=price in [.4,1.6], x=context in [-1,1], latent z uniform {-1,0,1}.
Only numeric constants <=1000 in magnitude, p,x,z, +,-,*,/, powers **0..4,
exp,tanh,sin,cos,log1p,sqrt,abs,min,max, conditional expressions. Max1024characters,
140ASTnodes. Demand=max(0,g) with the following selection adjustment. No code,
attributes, strings or randomness in an expression. Physical demand must stay
0..300 on all apparatus/evaluation points. Mean demand must not increase with
price. Do not assume that passing checks establishes continuum validity.

In confounded and observational_only worlds, historical price=1+z/4 and demand
adds strength*z-g(1+z/4,x,z)+g(1,x,z) BEFORE truncation. Other historical prices
are randomized. Historical capacity=60 in censored,180 otherwise. Interventions
can set price [.5,.75,1,1.25,1.5], context [-1,0,1], capacity [60,180]. Fresh
replications redraw z; audits measure the SAME unit's demand. Physical sales
are min(capacity,demand). Instrument reads physical quantity +sensor_bias plus
independent Gaussian error, sensor_sigma in [.5,4], precise_sigma in [.2,min(2,sensor_sigma)],
bias in [-4,4]. confounding_strength in [0,20]. Known-reference calibration is
available and shares the same measurement law. These parameters are not told
to scientists. Observational_only permits history, audits, replications and
calibration, with NO price interventions.

Choose public operating price_limit=1.0 or1.25. Revenue evaluation compares grid
prices [.5,.75,1,1.25,1.5] at directly queryable contexts [-1,0,1] and capacities60,180. Unqueried intermediate/outside prices and contexts are separate transfer features, never decision actions.
Private feasibility uses prices <=price_limit. There must be at least TWO states
where this price limit excludes a better revenue action by >=1 unit.

Nonflat worlds must have >=2 different optimal FEASIBLE prices across states,
>=2 states losing >=2 revenue units by always choosing the highest feasible
price, mean loss from that highest-price baseline >=2, and mean regret of the
best single feasible price >=1. Design meaningful context/capacity variation,
not merely a monotone response with a trivial highest-price optimum. Steep price
responses and broad context variation can help, but invent diverse relationships.
Nonflat .5-to-1.5 expected-demand change >=40 somewhere. Nonlinear needs max
non-affine departure>=3 on the full checked grid, not just a midpoint signature.
Interaction needs price-response span to differ >=15 across contexts.
Flat must contain NO p variable and is an explicit invariant negative control,
exempt from the within-world varying-price gate. Censored needs demand>75 at
historical price1 in some states. Names are private strata, not industrial claims.

Private admission also compares against the reference-preserving invariant
rival g(1,x,z), or for flat truth against g(p,x,z)+100*(1-p). Both must have
physical demand0..300. Across feasible grid states, an agent that knows neither
world and assigns prior1/2 must have Bayes regret>=1 relative to knowing it;
optimal policies must differ in >=2states. With interventions, a threshold test
using five fresh readings each at two query prices, at a common context and
capacity180, plus two calibration readings (total22cost/12queries) must have
>=80%power at5%false-positive and reduce pairwise expected decision regret by
>=40%. This is an owner-only witness, never an inference catalog for the scientist.
Observational_only instead needs exact historical/audit equivalence with that
invariant rival and different counterfactual decisions, so abstention is correct.
All worlds also retain off-grid/whole-law ambiguity; do not claim complete
mechanism recovery from a finite budget. Write honest private description and
scientific_challenge. Rejections below contain only apparatus/gate feedback.
"""


def signature(spec):
    # Include observable acquisition kernels: equal interventional laws with
    # different historical capacity or access are scientifically distinct tasks.
    intervened = [
        [round(v, 8) for v in sorted(latent(spec, p, x, z) for z in Z_GRID)]
        for p in P_GRID
        for x in X_GRID
    ]
    endogenous = spec["stratum"] in ("confounded", "observational_only")
    cap = 60 if spec["stratum"] == "censored" else 180
    history = []
    for x in (-1, 0, 1):
        values = []
        for z in Z_GRID:
            for p in [1 + z / 4] if endogenous else [0.5, 0.75, 1, 1.25, 1.5]:
                values.append((p, cap, round(latent(spec, p, x, z), 8)))
        history.append(sorted(values))
    return content_hash(
        dict(
            interventions=spec["stratum"] != "observational_only",
            interventional=intervened,
            historical=history,
        )
    )


def prompt(job, feedback):
    return (
        PROMPT
        + "\nPRIVATE DESIGN JOB\n"
        + json.dumps({k: job[k] for k in ("id", "stratum", "nonce")})
        + "\nPRIVATE VALIDATION FEEDBACK\n"
        + json.dumps(feedback)
    )


def create_collection(
    path,
    *,
    per_stratum=4,
    concurrency=4,
    maximum_attempts=8,
    model="gpt-6.1-sol",
    reasoning_effort="high",
    binary="codex",
    timeout=360,
    progress=None,
):
    if (
        any(
            type(v) is not int
            for v in (per_stratum, concurrency, maximum_attempts, timeout)
        )
        or not 1 <= per_stratum <= 20
        or not 1 <= concurrency <= 4
        or not 1 <= maximum_attempts <= 10
        or timeout < 1
    ):
        raise ValueError("Invalid v2 generation limits")
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    (path / "private").mkdir(mode=0o700)
    jobs = [
        dict(
            id=f"{s}-{i:02d}",
            stratum=s,
            nonce=secrets.token_hex(8),
            seed=secrets.randbits(32),
        )
        for i in range(per_stratum)
        for s in STRATA
    ]
    protocol = dict(
        version=COLLECTION_VERSION,
        created_at=utc_now(),
        purpose="Pre-campaign DEVELOPMENT collection; no confirmatory allocation or scored scientist runs",
        runtime=runtime_manifest(),
        model=model,
        reasoning_effort=reasoning_effort,
        backend=describe_backend(binary),
        timeout=timeout,
        concurrency=concurrency,
        maximum_attempts=maximum_attempts,
        per_stratum=per_stratum,
        jobs=jobs,
        design_prompt=PROMPT,
        design_schema=SCHEMA,
        selection="Only frozen physics/decision/access/budget gates and duplicate-response rejection. No experimentalist outcomes or policy information. All attempts retained.",
    )
    dump(path / "generation_protocol.json", protocol)
    (path / "generation_protocol.sha256").write_text(content_hash(protocol) + "\n")
    accepted = []
    attempts = []
    lock = threading.Lock()

    def generate(job):
        feedback = []
        for attempt in range(maximum_attempts):
            relative = f"private/designer_calls/{job['id']}/attempt-{attempt:02d}"
            receipt = run_call(
                path / relative, prompt(job, feedback), SCHEMA, protocol, path, timeout
            )
            error = None
            report = None
            spec = receipt.get("response")
            sig = None
            try:
                if receipt["status"] != "succeeded":
                    raise ValueError("Designer call did not succeed")
                if not isinstance(spec, dict) or spec.get("stratum") != job["stratum"]:
                    raise ValueError("Wrong requested stratum")
                report = admission_report(spec, job["seed"])
                sig = signature(spec)
            except (ValueError, KeyError, TypeError, SyntaxError) as exc:
                error = str(exc)
            with lock:
                if error is None and sig in {w["signature"] for w in accepted}:
                    error = "Duplicate finite response-distribution signature; invent a different law"
                item = dict(
                    job=job["id"],
                    attempt=attempt,
                    call=relative,
                    feedback=json.loads(json.dumps(feedback)),
                    receipt_sha256=file_hash(path / relative / "receipt.json"),
                    accepted=error is None,
                    error=error,
                )
                attempts.append(item)
                if error is None:
                    accepted.append(
                        dict(
                            id="v2-" + job["nonce"],
                            job=job["id"],
                            seed=job["seed"],
                            spec=spec,
                            signature=sig,
                            validation=report,
                            designer_call=relative,
                            role="development",
                        )
                    )
                dump(
                    path / "private/generation_progress.json",
                    dict(accepted=accepted, attempts=attempts),
                )
                if progress:
                    progress(
                        dict(
                            job=job["id"],
                            attempt=attempt,
                            accepted=error is None,
                            error=error,
                            worlds=len(accepted),
                        )
                    )
            if error is None:
                return True
            feedback.append(dict(error=error, rejected_proposal=spec))
        return False

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = [
            f.result() for f in as_completed([pool.submit(generate, j) for j in jobs])
        ]
    if not all(results):
        dump(
            path / "terminal.json",
            dict(
                status="failed",
                accepted=len(accepted),
                reason="Frozen attempts exhausted; preserve this attempt; no scientist campaign was run",
            ),
        )
        raise ValueError(
            "V2 generation incomplete; all proposals and partial worlds preserved"
        )
    if runtime_manifest()["fingerprint"] != protocol["runtime"]["fingerprint"]:
        raise ValueError(
            "Runtime changed during generation; preserve the unsealed attempt"
        )
    private = dict(worlds=accepted, attempts=attempts)
    dump(path / "private/worlds.json", private)
    (path / "private/worlds.json").chmod(0o600)
    manifest = dict(
        version=COLLECTION_VERSION,
        generation_protocol_hash=content_hash(protocol),
        private_commitment=content_hash(private),
        world_count=len(accepted),
        role="development",
        worlds=[dict(id=w["id"], commitment=content_hash(w)) for w in accepted],
        scientist_campaigns_launched_during_generation=0,
        statement="Private v2 gates passed before any scientist campaign. Oracle laws, rivals and schedules stay private. No industrial or whole-law-learnability claim.",
    )
    dump(path / "manifest.json", manifest)
    (path / "manifest.sha256").write_text(content_hash(manifest) + "\n")
    dump(
        path / "terminal.json",
        dict(status="sealed", scientist_campaigns_launched_during_generation=0),
    )
    return manifest


def load_collection(path):
    path = Path(path).resolve()
    protocol = json.loads((path / "generation_protocol.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    private = json.loads((path / "private/worlds.json").read_text())
    if (
        protocol["version"] != COLLECTION_VERSION
        or content_hash(protocol)
        != (path / "generation_protocol.sha256").read_text().strip()
        or content_hash(manifest) != (path / "manifest.sha256").read_text().strip()
        or content_hash(protocol) != manifest["generation_protocol_hash"]
        or content_hash(private) != manifest["private_commitment"]
    ):
        raise ValueError("V2 collection commitment mismatch")
    if protocol["runtime"]["fingerprint"] != runtime_manifest()["fingerprint"]:
        raise ValueError(
            "V2 collection runtime changed; restore its frozen source and dependencies"
        )
    if manifest["worlds"] != [
        dict(id=w["id"], commitment=content_hash(w)) for w in private["worlds"]
    ]:
        raise ValueError("V2 public/private world mismatch")
    return protocol, manifest, private


def verify_collection(path):
    path = Path(path).resolve()
    protocol, manifest, private = load_collection(path)
    jobs = {j["id"]: j for j in protocol["jobs"]}
    feedback = {j: [] for j in jobs}
    seen = set()
    proven = set()
    completed = set()
    worlds = {w["designer_call"]: w for w in private["worlds"]}
    for item in private["attempts"]:
        job = jobs[item["job"]]
        if (
            item["job"] in completed
            or item["attempt"] != len(feedback[item["job"]])
            or item["feedback"] != feedback[item["job"]]
        ):
            raise ValueError("Designer correction sequence changed")
        if (
            item["attempt"] >= protocol["maximum_attempts"]
            or item["call"]
            != f"private/designer_calls/{job['id']}/attempt-{item['attempt']:02d}"
        ):
            raise ValueError("Designer call allocation changed")
        receipt = verify_call(
            path / item["call"],
            expected_prompt=prompt(job, item["feedback"]),
            expected_schema=SCHEMA,
        )
        if file_hash(path / item["call"] / "receipt.json") != item["receipt_sha256"]:
            raise ValueError("Designer receipt changed")
        expected_error = None
        candidate = receipt.get("response")
        try:
            if receipt["status"] != "succeeded":
                raise ValueError("Designer call did not succeed")
            if (
                not isinstance(candidate, dict)
                or candidate.get("stratum") != job["stratum"]
            ):
                raise ValueError("Wrong requested stratum")
            expected_report = admission_report(candidate, job["seed"])
            if signature(candidate) in seen:
                raise ValueError(
                    "Duplicate finite response-distribution signature; invent a different law"
                )
        except (ValueError, KeyError, TypeError, SyntaxError) as exc:
            expected_error = str(exc)
        if item["error"] != expected_error or item["accepted"] != (
            expected_error is None
        ):
            raise ValueError("Admission/rejection decision changed")
        if item["accepted"]:
            world = worlds[item["call"]]
            if (
                world["spec"] != receipt["response"]
                or world["seed"] != job["seed"]
                or world["spec"]["stratum"] != job["stratum"]
                or world["job"] != job["id"]
                or world["id"] != "v2-" + job["nonce"]
                or world["role"] != "development"
            ):
                raise ValueError(
                    "Accepted world differs from recorded proposal/allocation"
                )
            report = expected_report
            sig = signature(world["spec"])
            if (
                report != world["validation"]
                or sig != world["signature"]
                or sig in seen
            ):
                raise ValueError("Admission/novelty replay mismatch")
            seen.add(sig)
            proven.add(world["id"])
            completed.add(item["job"])
        else:
            if not item["error"]:
                raise ValueError("Rejected call needs an explicit reason")
            feedback[item["job"]].append(
                dict(error=item["error"], rejected_proposal=receipt.get("response"))
            )
    if (
        len(proven) != manifest["world_count"]
        or completed != set(jobs)
        or len(proven) != 6 * protocol["per_stratum"]
    ):
        raise ValueError("Missing, duplicate or unproven admitted worlds")
    counts = Counter(w["spec"]["stratum"] for w in private["worlds"])
    if counts != Counter({s: protocol["per_stratum"] for s in STRATA}):
        raise ValueError("V2 collection lost its frozen stratum balance")
    return dict(
        verified=True,
        worlds=len(proven),
        counts=dict(counts),
        designer_attempts=len(private["attempts"]),
        manifest_hash=content_hash(manifest),
        scientist_campaigns_launched_during_generation=0,
        role="development",
        scope="Replayed admitted physical/decision/access/budget checks, model provenance and correction histories. Does not certify industry validity, arbitrary-law recovery or future scientist performance.",
    )


def create_session(path, collection, identity, method="standard_empirical"):
    from .session import DemandSession
    from .world import DemandConfig

    verify_collection(collection)
    _, _, private = load_collection(collection)
    matches = [w for w in private["worlds"] if w["id"] == identity]
    if len(matches) != 1:
        raise ValueError("Unknown v2 world identity")
    world = matches[0]
    spec = world["spec"]
    return DemandSession.create(
        path,
        DemandConfig(
            scenario="observational_only"
            if spec["stratum"] == "observational_only"
            else "clean",
            seed=world["seed"],
            budget=24,
            price_limit=spec["price_limit"],
            world_spec=spec,
        ),
        method,
    )
