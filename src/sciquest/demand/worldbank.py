"""Private agent-designed collections, split before any scientific outcomes."""

from __future__ import annotations

import json
import random
import secrets
import sys
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from sciquest.io import content_hash, file_hash, utc_now
from sciquest.runtime import runtime_manifest

from .backend import describe_backend, parse_events, run_call
from .designed import STRATA, VERSION, design_report, validate_spec

BANK_VERSION = "private-demand-worldbank-v1"


def world_runtime():
    """Keep sealed laws portable across changes to scientist instructions.

    Actual campaigns still freeze the full package/dependency runtime. A world
    collection additionally pins the engine that defines its physical truth.
    """
    package = Path(__file__).resolve().parents[1]
    return dict(
        python_version=sys.version,
        files={
            name: file_hash(package / name)
            for name in ("demand/designed.py", "demand/world.py", "io.py")
        },
    )


FIELDS = {
    "version": {"type": "string", "enum": [VERSION]},
    "stratum": {"type": "string", "enum": list(STRATA)},
    "expression": {"type": "string"},
    "sensor_sigma": {"type": "number"},
    "precise_sigma": {"type": "number"},
    "confounding_strength": {"type": "number"},
    "description": {"type": "string"},
    "scientific_challenge": {"type": "string"},
}
DESIGN_SCHEMA = dict(
    type="object",
    properties=dict(
        worlds=dict(
            type="array",
            items=dict(
                type="object",
                properties=FIELDS,
                required=list(FIELDS),
                additionalProperties=False,
            ),
        )
    ),
    required=["worlds"],
    additionalProperties=False,
)
DESIGN_PROMPT = """You are a private simulator designer for a scientific-discovery laboratory.
Invent exactly SIX distinct static demand laws, one per stratum: flat, nonlinear,
interaction, confounded, censored, observational_only. You never see scientist
messages, philosophy labels, scientific results or grader feedback. Design a
diverse collection before experiments. Your laws and explanations remain private.
Your task is meaningful scientific challenges; overwhelming noise is not success.

Each law is a mathematical expression g(p,x,z): price p in [.4,1.6], context x
in [-1,1], latent customer state z uniform over {-1,0,1}. Use constants, p,x,z,
+,-,*,/, integer powers **0..4, exp,tanh,sin,cos,log1p,sqrt,abs,min,max, and
conditional expressions (A if p>c else B). No code, attributes, strings, arrays,
imports or randomness. Maximum 1024 characters and 140 AST nodes. Demand is
max(0,g), with an owner-controlled adjustment in confounded/observational worlds.
Keep all resulting demand between 0 and 300 throughout the finite apparatus
domain. Expected demand must be nonincreasing in price. Typical levels 80..140
and price effects of 15..70 are useful, but invent the relationships yourself.
Use additive and interacting latent heterogeneity so intervals can be learned.
Strata:
flat: no price variable at all; context and customer heterogeneity are allowed.
nonlinear: meaningful curvature in expected price response.
interaction: price response meaningfully changes with context.
confounded: a genuine price effect hidden by endogenous historical assignment.
censored: historical capacity 60 hides part of demand; interventions can set 180.
observational_only: genuine price effect, with only confounded historical access.
All other strata permit interventions. Do not modify the public apparatus,
evaluation, budgets or scientific-agent instructions.

For confounded and observational_only, the supervisor sets historical price
1+z/4 and adds strength*z - g(1+z/4,x,z)+g(1,x,z) to demand. This makes historical
readings compatible with multiple price responses, even with paired audits;
interventions preserve that adjustment. Ensure the corrected demand stays valid.
Other strata use historical randomized price. Physical sales=min(capacity,demand).
Independent mean-zero Gaussian instrument error is owner-side, added after sales;
sensor_sigma must be 1..8; precise_sigma .2..min(2,sensor_sigma). Choose moderate
values. confounding_strength must be 0..25. For nonflat strata, expected demand
at .5 versus 1.5 must differ by >=max(8,2*sensor_sigma) for some context.

Return version designed-demand-v1, stratum, expression, sensor_sigma,
precise_sigma, confounding_strength, description, scientific_challenge for each
world. Describe the true mechanism and the challenge honestly. Do not embed
messages, identifiers or answers for scientists in measurements or expressions.
Avoid repeating accepted expressions from the private design registry below.
"""


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify_call(directory: Path, *, expected_prompt=None, expected_schema=None):
    receipt = json.loads((directory / "receipt.json").read_text())
    prompt = (directory / "prompt.txt").read_text()
    schema = json.loads((directory / "schema.json").read_text())
    if receipt["prompt_hash"] != content_hash(prompt) or receipt[
        "schema_hash"
    ] != content_hash(schema):
        raise ValueError("Model prompt/schema hash mismatch")
    if expected_prompt is not None and prompt != expected_prompt:
        raise ValueError("Model prompt differs from its frozen handoff")
    if expected_schema is not None and schema != expected_schema:
        raise ValueError("Model schema differs from its frozen contract")
    trace_hash = (
        file_hash(directory / "trace.jsonl")
        if (directory / "trace.jsonl").exists()
        else None
    )
    if receipt.get("trace_sha256") != trace_hash:
        raise ValueError("Raw model trace changed")
    if receipt["status"] == "succeeded":
        response = json.loads((directory / "response.json").read_text())
        parsed = parse_events((directory / "trace.jsonl").read_text())
        if (
            response != receipt["response"]
            or content_hash(response) != receipt["response_hash"]
            or response != json.loads(parsed["final_message"])
            or parsed["usage"] != receipt["usage"]
        ):
            raise ValueError("Model response or usage provenance mismatch")
    return receipt


def create_bank(
    path: Path,
    *,
    batches=32,
    model="gpt-6.1-sol",
    reasoning_effort="high",
    binary="codex",
    timeout=360,
    progress=None,
):
    """192 worlds by default: 6 engineering, 6 pilot, 180 untouched evaluation."""
    if type(batches) is not int or batches < 3 or batches > 100:
        raise ValueError("Use 3..100 batches, each containing all six challenge strata")
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    (path / "private").mkdir(mode=0o700)
    spec = dict(
        version=BANK_VERSION,
        created_at=utc_now(),
        batches=batches,
        model=model,
        reasoning_effort=reasoning_effort,
        backend=describe_backend(binary),
        runtime=runtime_manifest(),
        world_runtime=world_runtime(),
        design_prompt=DESIGN_PROMPT,
        design_schema=DESIGN_SCHEMA,
        maximum_attempts_per_batch=3,
        designer_call_timeout_seconds=timeout,
        split_rule="After all accepted designs, independent OS-entropy stratified shuffle: one per stratum engineering, one pilot, all others evaluation. No scientific outcomes or selection by scores.",
    )
    dump(path / "generation_protocol.json", spec)
    (path / "generation_protocol.sha256").write_text(content_hash(spec) + "\n")
    accepted, attempts = [], []
    for batch in range(batches):
        feedback = []
        for attempt in range(spec["maximum_attempts_per_batch"]):
            registry = [w["spec"]["expression"] for w in accepted]
            prompt = (
                DESIGN_PROMPT
                + "\nPRIVATE DESIGN REGISTRY\n"
                + json.dumps(registry)
                + "\nVALIDATION FEEDBACK\n"
                + json.dumps(feedback)
            )
            relative = f"private/designer_calls/batch-{batch:02d}-attempt-{attempt:02d}"
            receipt = run_call(
                path / relative, prompt, DESIGN_SCHEMA, spec, path, timeout
            )
            item = dict(
                batch=batch,
                attempt=attempt,
                call=relative,
                receipt_sha256=file_hash(path / relative / "receipt.json"),
            )
            error = None
            try:
                if receipt["status"] != "succeeded":
                    raise ValueError("Designer model call did not succeed")
                worlds = receipt["response"]["worlds"]
                if len(worlds) != len(STRATA) or {w["stratum"] for w in worlds} != set(
                    STRATA
                ):
                    raise ValueError("One world per declared stratum is required")
                checked = [validate_spec(w) for w in worlds]
                expressions = registry + [w["expression"] for w in checked]
                if len(set(expressions)) != len(expressions):
                    raise ValueError("Exact expression duplicates are not allowed")
                new = []
                for world in checked:
                    seed = secrets.randbits(32)
                    new.append(
                        dict(
                            id="w-" + secrets.token_hex(8),
                            seed=seed,
                            spec=world,
                            designer_call=relative,
                            validation=design_report(world, seed),
                        )
                    )
                accepted.extend(new)
            except (ValueError, KeyError, TypeError, SyntaxError) as exc:
                error = str(exc)
                feedback.append(error)
            item.update(error=error, accepted=error is None)
            attempts.append(item)
            dump(
                path / "private/generation_progress.json",
                dict(accepted=accepted, attempts=attempts),
            )
            if progress:
                progress(
                    dict(
                        batch=batch,
                        attempt=attempt,
                        accepted=error is None,
                        error=error,
                        worlds=len(accepted),
                    )
                )
            if error is None:
                break
        else:
            raise ValueError(
                "Designer exhausted frozen validation attempts; preserve this bank attempt"
            )
    return _seal_bank(path, spec, accepted, attempts)


def _seal_bank(path, spec, accepted, attempts):
    split_seed = secrets.randbits(128)
    rng = random.Random(split_seed)
    for stratum in STRATA:
        group = [w for w in accepted if w["spec"]["stratum"] == stratum]
        rng.shuffle(group)
        for i, world in enumerate(group):
            world["split"] = (
                "engineering" if i == 0 else "pilot" if i == 1 else "evaluation"
            )
    private = dict(worlds=accepted, attempts=attempts, split_seed=split_seed)
    dump(path / "private/worlds.json", private)
    (path / "private/worlds.json").chmod(0o600)
    public = dict(
        version=BANK_VERSION,
        generation_protocol_hash=content_hash(spec),
        private_commitment=content_hash(private),
        counts={
            split: sum(w["split"] == split for w in accepted)
            for split in ("engineering", "pilot", "evaluation")
        },
        worlds=[
            dict(id=w["id"], split=w["split"], commitment=content_hash(w))
            for w in accepted
        ],
        statement="Private laws created and apparatus-validated before scientific campaigns; evaluation worlds have not been used for scientist runs.",
    )
    dump(path / "manifest.json", public)
    (path / "manifest.sha256").write_text(content_hash(public) + "\n")
    return public


PARALLEL_DESIGN_PROMPT = (
    DESIGN_PROMPT
    + """
The nonlinear stratum specifically needs abs(Eg(.5,x,z)+Eg(1.5,x,z)-2*Eg(1,x,z))
>=1 for at least one context. An odd tanh or cubic centered at p=1 can be nonlinear
but fail this particular signature; use asymmetric curvature for that stratum.
The interaction stratum needs the .5-to-1.5 expected demand change to differ by
>=3 across contexts. Other strata may include different monotone nonlinear shapes.
Independent batches can be proposed concurrently. A private diversity nonce
distinguishes the design job; do not encode that nonce in any expression or data.
If a proposal was rejected, its complete content and apparatus error are supplied
in validation feedback. Correct the error while preserving diverse valid laws.
"""
)


def _imported_worlds(path, metadata):
    if metadata is None:
        return [], 0
    source = Path(path) / metadata["path"]
    for name, digest in metadata["files"].items():
        if file_hash(source / name) != digest:
            raise ValueError("Imported generation evidence changed")
    protocol = json.loads((source / "generation_protocol.json").read_text())
    state = json.loads((source / "private/generation_progress.json").read_text())
    if (
        content_hash(protocol) != metadata["protocol_hash"]
        or protocol["world_runtime"] != world_runtime()
    ):
        raise ValueError("Imported generation engine/protocol mismatch")
    ancestors, ancestor_calls = _imported_worlds(
        source, protocol.get("imported_attempt")
    )
    if [
        {k: v for k, v in w.items() if k != "split"}
        for w in state["accepted"][: len(ancestors)]
    ] != ancestors:
        raise ValueError("Imported ancestor worlds were lost or modified")
    registry = [w["spec"]["expression"] for w in ancestors]
    feedback = []
    batch = -1
    by_batch = {}
    proven_worlds = len(ancestors)
    for attempt in state["attempts"]:
        if batch != attempt["batch"]:
            batch = attempt["batch"]
            feedback = []
        call = source / attempt["call"]
        if protocol.get("mode") == "parallel-v2":
            if registry[: len(attempt["registry"])] != attempt["registry"]:
                raise ValueError("Imported concurrent registry changed")
            prior = by_batch.setdefault(attempt["batch"], [])
            expected_feedback = [
                dict(
                    error=a["error"],
                    rejected_proposal=json.loads(
                        (source / a["call"] / "receipt.json").read_text()
                    ).get("response"),
                )
                for a in prior
                if a["error"]
            ]
            if attempt["feedback"] != expected_feedback or attempt["attempt"] != len(
                prior
            ):
                raise ValueError("Imported concurrent correction history changed")
            prior.append(attempt)
            visible_registry, visible_feedback = (
                attempt["registry"],
                attempt["feedback"],
            )
        else:
            visible_registry, visible_feedback = registry, feedback
        prompt = (
            protocol["design_prompt"]
            + "\nPRIVATE DESIGN REGISTRY\n"
            + json.dumps(visible_registry)
            + "\nVALIDATION FEEDBACK\n"
            + json.dumps(visible_feedback)
        )
        if protocol.get("mode") == "parallel-v2":
            prompt += "\nPRIVATE DESIGN JOB\n" + json.dumps(attempt["job"])
        receipt = verify_call(
            call, expected_prompt=prompt, expected_schema=protocol["design_schema"]
        )
        if file_hash(call / "receipt.json") != attempt["receipt_sha256"]:
            raise ValueError("Imported designer receipt changed")
        if attempt["accepted"]:
            proposed = [
                w for w in state["accepted"] if w["designer_call"] == attempt["call"]
            ]
            if [w["spec"] for w in proposed] != receipt["response"]["worlds"]:
                raise ValueError("Imported worlds differ from their actual designer")
            registry.extend(w["spec"]["expression"] for w in proposed)
            proven_worlds += len(proposed)
        else:
            feedback.append(attempt["error"])
    if proven_worlds != len(state["accepted"]):
        raise ValueError("Imported collection includes unproven worlds")
    worlds = json.loads(json.dumps(state["accepted"]))
    for world in worlds:
        world["designer_call"] = metadata["path"] + "/" + world["designer_call"]
        world.pop("split", None)
    return worlds, ancestor_calls + len(
        list((source / "private/designer_calls").glob("*/receipt.json"))
    )


def create_bank_parallel(
    path: Path,
    *,
    batches=32,
    concurrency=4,
    import_attempt=None,
    model="gpt-6.1-sol",
    reasoning_effort="high",
    binary="codex",
    timeout=360,
    progress=None,
):
    """Validated concurrent proposals, with explicit private context lineage."""
    if (
        type(batches) is not int
        or not 3 <= batches <= 100
        or type(concurrency) is not int
        or not 1 <= concurrency <= 4
    ):
        raise ValueError("Use 3..100 batches and 1..4 private design workers")
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    (path / "private").mkdir(mode=0o700)
    imported = None
    if import_attempt is not None:
        source = Path(import_attempt).resolve()
        if (source / "manifest.json").exists():
            raise ValueError(
                "Only an unallocated generation attempt can be imported; never reshuffle a sealed collection"
            )
        target = path / "private/imported-initial-attempt"
        shutil.copytree(source, target)
        original = json.loads((target / "generation_protocol.json").read_text())
        imported = dict(
            path="private/imported-initial-attempt",
            protocol_hash=content_hash(original),
            files={
                str(p.relative_to(target)): file_hash(p)
                for p in target.rglob("*")
                if p.is_file()
            },
            reason="Unallocated pre-science generation continuation; accepted proposals and all rejected/interrupted evidence preserved. No scientific performance selection.",
        )
    accepted, _ = _imported_worlds(path, imported)
    if len(accepted) % 6 or len(accepted) > 6 * batches:
        raise ValueError("Imported accepted batches do not fit the target collection")
    first_batch = len(accepted) // 6
    attempts = []
    mutex = threading.Lock()
    spec = dict(
        version=BANK_VERSION,
        mode="parallel-v2",
        created_at=utc_now(),
        batches=batches,
        concurrency=concurrency,
        model=model,
        reasoning_effort=reasoning_effort,
        backend=describe_backend(binary),
        runtime=runtime_manifest(),
        world_runtime=world_runtime(),
        design_prompt=PARALLEL_DESIGN_PROMPT,
        design_schema=DESIGN_SCHEMA,
        maximum_attempts_per_batch=3,
        designer_call_timeout_seconds=timeout,
        imported_attempt=imported,
        split_rule="After all validated designs, independent OS-entropy stratified shuffle: one engineering, one pilot and all others evaluation per stratum. No scientific results or performance selection.",
    )
    dump(path / "generation_protocol.json", spec)
    (path / "generation_protocol.sha256").write_text(content_hash(spec) + "\n")

    def generate(batch):
        feedback = []
        nonce = secrets.token_hex(12)
        for attempt in range(spec["maximum_attempts_per_batch"]):
            with mutex:
                registry = [w["spec"]["expression"] for w in accepted]
            job = dict(batch=batch, nonce=nonce)
            prompt = (
                PARALLEL_DESIGN_PROMPT
                + "\nPRIVATE DESIGN REGISTRY\n"
                + json.dumps(registry)
                + "\nVALIDATION FEEDBACK\n"
                + json.dumps(feedback)
                + "\nPRIVATE DESIGN JOB\n"
                + json.dumps(job)
            )
            relative = f"private/designer_calls/batch-{batch:02d}-attempt-{attempt:02d}"
            receipt = run_call(
                path / relative, prompt, DESIGN_SCHEMA, spec, path, timeout
            )
            item = dict(
                batch=batch,
                attempt=attempt,
                call=relative,
                receipt_sha256=file_hash(path / relative / "receipt.json"),
                registry=registry,
                feedback=json.loads(json.dumps(feedback)),
                job=job,
            )
            error = None
            with mutex:
                try:
                    if receipt["status"] != "succeeded":
                        raise ValueError("Designer model call did not succeed")
                    worlds = receipt["response"]["worlds"]
                    if len(worlds) != 6 or {w["stratum"] for w in worlds} != set(
                        STRATA
                    ):
                        raise ValueError(
                            "Exactly one law per declared stratum is required"
                        )
                    checked = [validate_spec(w) for w in worlds]
                    expressions = [w["spec"]["expression"] for w in accepted] + [
                        w["expression"] for w in checked
                    ]
                    if len(set(expressions)) != len(expressions):
                        raise ValueError(
                            "Exact expression duplicate, including a concurrent proposal; change the duplicated law"
                        )
                    new = []
                    for world in checked:
                        seed = secrets.randbits(32)
                        new.append(
                            dict(
                                id="w-" + secrets.token_hex(8),
                                seed=seed,
                                spec=world,
                                designer_call=relative,
                                validation=design_report(world, seed),
                            )
                        )
                    accepted.extend(new)
                except (ValueError, KeyError, TypeError, SyntaxError) as exc:
                    error = str(exc)
                item.update(error=error, accepted=error is None)
                attempts.append(item)
                dump(
                    path / "private/generation_progress.json",
                    dict(accepted=accepted, attempts=attempts),
                )
                if progress:
                    progress(
                        dict(
                            batch=batch,
                            attempt=attempt,
                            accepted=error is None,
                            error=error,
                            worlds=len(accepted),
                        )
                    )
            if error is None:
                return
            feedback.append(
                dict(error=error, rejected_proposal=receipt.get("response"))
            )
        raise ValueError(
            "Designer exhausted its frozen validation attempts; preserve all completed evidence"
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        next_batch = first_batch
        pending = set()
        while next_batch < batches or pending:
            while next_batch < batches and len(pending) < concurrency:
                pending.add(pool.submit(generate, next_batch))
                next_batch += 1
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                future.result()
    return _seal_bank(path, spec, accepted, attempts)


def load_bank(path: Path, *, check_runtime=True):
    path = Path(path).resolve()
    protocol = json.loads((path / "generation_protocol.json").read_text())
    public = json.loads((path / "manifest.json").read_text())
    private = json.loads((path / "private/worlds.json").read_text())
    if (
        content_hash(protocol)
        != (path / "generation_protocol.sha256").read_text().strip()
        or content_hash(public) != (path / "manifest.sha256").read_text().strip()
        or content_hash(private) != public["private_commitment"]
        or content_hash(protocol) != public["generation_protocol_hash"]
    ):
        raise ValueError("Worldbank commitment mismatch")
    if check_runtime and protocol["world_runtime"] != world_runtime():
        raise ValueError("Worldbank engine changed; restore its frozen world runtime")
    expected = [
        dict(id=w["id"], split=w["split"], commitment=content_hash(w))
        for w in private["worlds"]
    ]
    if expected != public["worlds"]:
        raise ValueError("Public/private world allocation mismatch")
    return protocol, public, private


def verify_bank(path: Path):
    protocol, public, private = load_bank(path)
    imported, imported_calls = _imported_worlds(path, protocol.get("imported_attempt"))
    if [
        {k: v for k, v in w.items() if k != "split"}
        for w in private["worlds"][: len(imported)]
    ] != imported:
        raise ValueError("Imported accepted worlds were lost or modified")
    registry, feedback, batch = [w["spec"]["expression"] for w in imported], [], -1
    by_batch = {}
    for attempt in private["attempts"]:
        if attempt["batch"] != batch:
            batch, feedback = attempt["batch"], []
        directory = Path(path) / attempt["call"]
        if file_hash(directory / "receipt.json") != attempt["receipt_sha256"]:
            raise ValueError("Designer receipt changed")
        if protocol.get("mode") == "parallel-v2":
            if registry[: len(attempt["registry"])] != attempt["registry"]:
                raise ValueError(
                    "Concurrent designer received an unrecorded private registry"
                )
            prior = by_batch.setdefault(attempt["batch"], [])
            expected_feedback = [
                dict(
                    error=a["error"],
                    rejected_proposal=json.loads(
                        (Path(path) / a["call"] / "receipt.json").read_text()
                    ).get("response"),
                )
                for a in prior
                if a["error"]
            ]
            if attempt["feedback"] != expected_feedback or attempt["attempt"] != len(
                prior
            ):
                raise ValueError("Designer correction history changed")
            prior.append(attempt)
            visible_registry = attempt["registry"]
            visible_feedback = attempt["feedback"]
        else:
            visible_registry = registry
            visible_feedback = feedback
        prompt = (
            protocol["design_prompt"]
            + "\nPRIVATE DESIGN REGISTRY\n"
            + json.dumps(visible_registry)
            + "\nVALIDATION FEEDBACK\n"
            + json.dumps(visible_feedback)
        )
        if protocol.get("mode") == "parallel-v2":
            prompt += "\nPRIVATE DESIGN JOB\n" + json.dumps(attempt["job"])
        receipt = verify_call(
            directory, expected_prompt=prompt, expected_schema=protocol["design_schema"]
        )
        if attempt["accepted"]:
            proposed = [
                w for w in private["worlds"] if w["designer_call"] == attempt["call"]
            ]
            if [w["spec"] for w in proposed] != receipt["response"]["worlds"]:
                raise ValueError(
                    "Accepted world batch differs from its designer proposal"
                )
            registry.extend(w["spec"]["expression"] for w in proposed)
        else:
            feedback.append(attempt["error"])
    rng = random.Random(private["split_seed"])
    for stratum in STRATA:
        group = [w for w in private["worlds"] if w["spec"]["stratum"] == stratum]
        if len(group) != protocol["batches"]:
            raise ValueError("World collection lost a challenge stratum")
        rng.shuffle(group)
        for i, world in enumerate(group):
            if world["split"] != (
                "engineering" if i == 0 else "pilot" if i == 1 else "evaluation"
            ):
                raise ValueError(
                    "World split does not match its frozen random allocation"
                )
    for world in private["worlds"]:
        if design_report(world["spec"], world["seed"]) != world["validation"]:
            raise ValueError("World validation replay mismatch")
        call = json.loads(
            (Path(path) / world["designer_call"] / "response.json").read_text()
        )
        if world["spec"] not in call["worlds"]:
            raise ValueError("World law was not proposed by its recorded designer")
    return dict(
        verified=True,
        counts=public["counts"],
        manifest_hash=content_hash(public),
        designer_attempts=len(private["attempts"]) + imported_calls,
        scope="Frozen agent proposals, private allocation commitments and apparatus checks; no claim of market validity or immunity to pretraining bias.",
    )
