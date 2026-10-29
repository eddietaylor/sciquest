"""Persistent supervisor state for discovery actions and one sealed final test."""

from __future__ import annotations

import csv
import fcntl
import json
import math
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from sciquest.core import create_quest, run_next
from sciquest.io import (
    canonical_json,
    content_hash,
    finite_number,
    file_hash,
    write_yaml,
)
from sciquest.methods.ledger import preregister_experiment, validate_plan
from sciquest.methods.policies import (
    policy_evidence,
    policy_requirements,
    validate_policy_plan,
)
from sciquest.methods.registry import MethodRegistry
from sciquest.methods.resolver import resolve_method_stack
from sciquest.runtime import runtime_manifest
from sciquest.workflow import execute_registered

from .world import (
    DemandConfig,
    EVAL_FEATURES,
    FAMILIES,
    assisted_contract,
    draw,
    evaluation_rows,
    normalize_action,
    private_world,
    public_contract,
)


def write_csv(path: Path, rows: list[dict], fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def validate_decision(decision: dict, *, assisted=True, claims=False) -> dict:
    if not assisted:
        if not isinstance(decision, dict) or set(decision) != (
            {
                "status",
                "reason",
                "explanation",
                "limitations",
            }
            | ({"claims"} if claims else set())
        ):
            raise ValueError(
                "Decision needs status, reason, explanation and limitations"
            )
        if decision["status"] not in (
            "selected",
            "inconclusive",
            "not_identifiable",
            "model_inadequate",
        ):
            raise ValueError("Unknown final decision status")
        if any(
            not isinstance(decision[k], str) or not decision[k].strip()
            for k in ("reason", "explanation", "limitations")
        ):
            raise ValueError("Decision text must be nonempty")
        return json.loads(canonical_json(decision))
    if not isinstance(decision, dict) or set(decision) - {
        "status",
        "reason",
        "mechanism_probabilities",
        "selected_family",
        "parameters",
    }:
        raise ValueError("Invalid final scientific decision fields")
    if decision.get("status") not in (
        "selected",
        "inconclusive",
        "not_identifiable",
        "model_inadequate",
    ):
        raise ValueError("Unknown final decision status")
    if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
        raise ValueError("Final decision needs a reason")
    p = decision.get("mechanism_probabilities")
    if not isinstance(p, dict) or set(p) != set(FAMILIES):
        raise ValueError(
            "Probabilities must cover all mechanism families, including other"
        )
    if any(not 0 <= finite_number(v) <= 1 for v in p.values()) or not math.isclose(
        math.fsum(p.values()), 1, abs_tol=1e-9
    ):
        raise ValueError("Mechanism probabilities must sum to one")
    if (
        decision["status"] == "selected"
        and decision.get("selected_family") not in FAMILIES[:-1]
    ):
        raise ValueError("A selected decision needs a candidate selected_family")
    if decision["status"] == "selected" and p[decision["selected_family"]] < max(
        p.values()
    ):
        raise ValueError("Selected family must have maximal reported probability")
    params = decision.get("parameters", {})
    if not isinstance(params, dict) or set(params) - {
        "baseline",
        "context_effect",
        "sensitivity",
    }:
        raise ValueError("Unknown model parameters")
    for value in params.values():
        finite_number(value, "model parameter")
    return json.loads(canonical_json(decision))


class DemandSession:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.db = self.path / ".demand/state.sqlite3"
        if not self.db.is_file():
            raise ValueError("Demand session does not exist")

    @classmethod
    def create(
        cls,
        path: Path,
        config: DemandConfig,
        method="standard_empirical",
        *,
        assisted=False,
        comparison_policy=None,
        resource_limits=None,
    ):
        if (
            assisted
            and config.world_spec
            and config.world_spec["version"] == "designed-demand-v2"
        ):
            raise ValueError("V2 worlds require blind sessions")
        path = Path(path).resolve()
        comparison = None
        if comparison_policy is not None:
            from .policies_v2 import assignment

            if (
                assisted
                or not config.world_spec
                or config.world_spec["version"] != "designed-demand-v2"
                or method != "standard_empirical"
            ):
                raise ValueError(
                    "Matched guidance requires a blind v2 world and common empirical validation"
                )
            comparison = assignment(comparison_policy, resource_limits)
        stack = resolve_method_stack(method)
        path.mkdir(parents=True, exist_ok=False)
        (path / ".demand").mkdir(mode=0o700)
        registry = MethodRegistry.default()
        initial = dict(
            world=private_world(config),
            public=assisted_contract(config) if assisted else public_contract(config),
            information_mode="assisted_v1" if assisted else "black_box_v2",
            method_stack=stack.to_dict(),
            profiles={m: asdict(registry.get(m)) for m in stack.methods_used},
            runtime=runtime_manifest(),
            phase="collecting",
            spent=0,
            decision=None,
            quest=None,
            experiment=None,
            result=None,
        )
        if comparison is not None:
            initial["comparison"] = comparison
        db = path / ".demand/state.sqlite3"
        with sqlite3.connect(db) as c:
            c.executescript(
                "CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL); CREATE TABLE events (revision INTEGER PRIMARY KEY, payload TEXT NOT NULL, previous_hash TEXT NOT NULL, hash TEXT NOT NULL);"
            )
            c.execute("INSERT INTO state VALUES(1,?)", (canonical_json(initial),))
        db.chmod(0o600)
        session = cls(path)
        with session.transaction() as (c, state):
            session._append(
                c,
                dict(
                    type="created",
                    initial_hash=content_hash(
                        {
                            k: state[k]
                            for k in (
                                "world",
                                "public",
                                "method_stack",
                                "profiles",
                                "runtime",
                                "information_mode",
                                *(("comparison",) if "comparison" in state else ()),
                            )
                        }
                    ),
                ),
            )
        return session

    @contextmanager
    def transaction(self):
        from sciquest.readonly import DATABASE_READS_ONLY

        if DATABASE_READS_ONLY.get():
            if self.db.is_symlink():
                raise ValueError("Read-only audit database cannot be a symlink")
            c = sqlite3.connect(self.db.resolve().as_uri()+"?mode=ro", uri=True, timeout=30, isolation_level=None)
            try:
                c.execute("PRAGMA query_only=ON")
                c.execute("BEGIN")
                initial = c.execute("SELECT payload FROM state WHERE id=1").fetchone()[0]
                state = json.loads(initial)
                yield c, state
                if canonical_json(state) != initial:
                    raise RuntimeError("Read-only audit attempted to change session state")
            finally:
                c.close()
            return
        c = sqlite3.connect(self.db, timeout=30, isolation_level=None)
        try:
            c.execute("PRAGMA synchronous=FULL")
            c.execute("BEGIN IMMEDIATE")
            state = json.loads(
                c.execute("SELECT payload FROM state WHERE id=1").fetchone()[0]
            )
            yield c, state
            c.execute("UPDATE state SET payload=? WHERE id=1", (canonical_json(state),))
            c.commit()
        except BaseException:
            c.rollback()
            raise
        finally:
            c.close()

    def _events(self, c):
        return [
            dict(revision=r, payload=json.loads(p), previous_hash=prev, hash=h)
            for r, p, prev, h in c.execute(
                "SELECT revision,payload,previous_hash,hash FROM events ORDER BY revision"
            )
        ]

    def _append(self, c, payload):
        last = c.execute(
            "SELECT revision,hash FROM events ORDER BY revision DESC LIMIT 1"
        ).fetchone()
        revision = 1 if last is None else last[0] + 1
        previous = "" if last is None else last[1]
        digest = content_hash(
            dict(revision=revision, previous_hash=previous, payload=payload)
        )
        c.execute(
            "INSERT INTO events VALUES(?,?,?,?)",
            (revision, canonical_json(payload), previous, digest),
        )
        return revision

    def _runtime_check(self, state):
        if runtime_manifest()["fingerprint"] != state["runtime"]["fingerprint"]:
            raise ValueError(
                "Demand session runtime changed; preserve this session and create a new one"
            )

    def view(self, *, scientist=False) -> dict:
        with self.transaction() as (c, s):
            blind = s.get("information_mode") == "black_box_v2"
            if scientist and not blind:
                raise ValueError(
                    "Assisted legacy sessions cannot be used for blind discovery"
                )
            events = self._events(c)
            history = [
                {k: v for k, v in e["payload"].items() if k != "private"}
                for e in events
                if e["payload"]["type"] == "query"
            ]
            if blind:
                history = [public_query(r) for r in history]
            result = dict(
                environment_version=s["public"]["environment_version"],
                information_mode=s.get("information_mode", "assisted_v1"),
                phase=s["phase"],
                revision=len(events),
                contract=s["public"],
                method_stack=s["method_stack"],
                policy_contract=policy_requirements(s["method_stack"]),
                remaining=s["public"]["budget"] - s["spent"],
                spent=s["spent"],
                history=history,
                decision=s["decision"],
                failure=(
                    ("Final test failed" if s.get("failure") else None)
                    if scientist
                    else s.get("failure")
                ),
            )
            if "comparison" in s:
                from .policies_v2 import validate_assignment

                result["comparison"] = validate_assignment(s["comparison"])
                result["contract"] = {
                    **s["public"],
                    "price_limit": float(s["public"]["price_limit"]),
                    "scoring": "Separate query-setting and transfer forecast errors, expected 90% sensor interval scores, feasible revenue/regret and claim-level explanation assessment. No composite scientific-success grade.",
                }
            result["recent_preparations"] = (
                []
                if scientist
                else [
                    e["payload"]
                    for e in events
                    if e["payload"]["type"] == "agent_preparation"
                ][-3:]
            )
            if s["phase"] != "collecting":
                result["evaluation_features"] = [
                    {k: r[k] for k in EVAL_FEATURES}
                    for r in evaluation_rows(s["world"])
                ]
            if s["phase"] == "completed" and not scientist:
                result["evaluation"] = s["result"]
            return result

    @staticmethod
    def _query_plan(stack, submitted, previous, *, comparison=False):
        plan = validate_plan(submitted, {"metrics": [{"name": "observation"}]})
        if plan["confirmatory_status"] != "exploratory":
            raise ValueError("Adaptive discovery actions must be exploratory")
        contract = validate_policy_plan(stack, plan)
        if comparison:
            from .policies_v2 import optional_contract

            contract = optional_contract(stack, plan)
        belief = plan.get("belief_model")
        if "bayesian" in contract["enforced_plan_contracts"] and belief.get(
            "prior_source"
        ):
            source = belief["prior_source"]["experiment_id"]
            matches = [r for r in previous if r["observation"]["id"] == source]
            if len(matches) != 1:
                raise ValueError("Query prior_source must identify a prior query")
            post = (
                matches[0]["policy_evidence"].get("belief_update", {}).get("posterior")
            )
            if (
                post is None
                or set(post) != set(belief["prior"])
                or any(
                    not math.isclose(v, belief["prior"][k], rel_tol=1e-12, abs_tol=0)
                    for k, v in post.items()
                )
            ):
                raise ValueError("Query prior does not match its source posterior")
            if (
                not isinstance(belief.get("conditional_evidence_assumptions"), str)
                or not belief["conditional_evidence_assumptions"].strip()
            ):
                raise ValueError(
                    "Linked query evidence needs conditional assumptions, especially for paired measurements"
                )
        return plan, contract

    def query(self, proposal: dict) -> dict:
        if not isinstance(proposal, dict) or set(proposal) != {
            "revision",
            "action",
            "plan",
        }:
            raise ValueError("Query proposal needs revision, action and plan")
        with self.transaction() as (c, s):
            self._runtime_check(s)
            if s["phase"] != "collecting":
                raise ValueError("Discovery is closed after the final decision")
            events = self._events(c)
            if type(proposal["revision"]) is not int or proposal["revision"] != len(
                events
            ):
                raise ValueError("Stale demand-lab revision")
            previous = [e["payload"] for e in events if e["payload"]["type"] == "query"]
            design, cost = normalize_action(proposal["action"], s["public"], previous)
            if s["spent"] + cost > s["public"]["budget"]:
                raise ValueError("Insufficient observation budget")
            plan, contract = self._query_plan(
                s["method_stack"],
                proposal["plan"],
                previous,
                comparison="comparison" in s,
            )
            observation, hidden = draw(s["world"], design, len(previous) + 1, previous)
            from sciquest.validation import COMPARATORS

            outcomes = {
                p["id"]: (
                    "supported"
                    if COMPARATORS[p["operator"]](observation["value"], p["threshold"])
                    else "contradicted"
                )
                for p in plan["predictions"]
            }
            evidence = policy_evidence(
                {"plan": plan, "policy_contract": contract}, outcomes
            )
            payload = dict(
                type="query",
                action=proposal["action"],
                design=design,
                plan=plan,
                policy_contract=contract,
                policy_evidence=evidence,
                cost=cost,
                observation=observation,
                private=hidden,
            )
            self._append(c, payload)
            s["spent"] += cost
            return (
                public_query(payload)
                if s.get("information_mode") == "black_box_v2"
                else {k: v for k, v in payload.items() if k != "private"}
            )

    def stop(self, decision: dict, revision: int) -> dict:
        with self.transaction() as (c, s):
            decision = validate_decision(
                decision,
                assisted=s.get("information_mode") != "black_box_v2",
                claims="comparison" in s,
            )
            if "comparison" in s:
                from .policies_v2 import validate_claims

                validate_claims(
                    decision,
                    [
                        e["payload"]["observation"]["id"]
                        for e in self._events(c)
                        if e["payload"]["type"] == "query"
                    ],
                )
            self._runtime_check(s)
            if s["phase"] != "collecting":
                raise ValueError("Final decision is already frozen")
            if type(revision) is not int or revision != len(self._events(c)):
                raise ValueError("Stale demand-lab revision")
            self._append(c, dict(type="stop", decision=decision))
            s.update(phase="prediction_required", decision=decision)
        return self.view()

    def prepare_quest(self) -> tuple[Path, str]:
        """Create the protected native SciQuest final experiment, once only."""
        with self.transaction() as (c, s):
            self._runtime_check(s)
            if s["phase"] != "prediction_required":
                raise ValueError("Freeze a final decision before prediction")
            if s["quest"]:
                return Path(s["quest"]), s["experiment"]
            q = create_quest(
                self.path,
                {
                    "hero_statement": s["public"]["question"],
                    "initial_hypothesis": "The learned mechanism predicts independent intervention outcomes.",
                    "method_stack": s["method_stack"],
                },
                slug="demand-confirmation",
            )
            events = self._events(c)
            observations = [
                e["payload"]["observation"]
                for e in events
                if e["payload"]["type"] == "query"
            ]
            # Empty discovery is valid; a header-only training file cannot pass the
            # native table contract, so it is omitted entirely.
            fields = (
                observation_fields(s["public"])
                if s.get("information_mode") == "black_box_v2"
                else TRAIN_FIELDS
            )
            bundle = dict(
                world=s["world"], observations=observations, train_fields=list(fields)
            )
            (q / "data/bundle.json").write_text(canonical_json(bundle))
            (q / "data/generator.py").write_text(GENERATOR)
            datasets = []
            if observations:
                write_csv(q / "data/raw/train.csv", observations, fields)
                datasets.append(
                    self._dataset("training", "train.csv", s["world"]["config"]["seed"])
                )
            write_csv(q / "data/raw/eval.csv", evaluation_rows(s["world"]))
            d = self._dataset("evaluation", "eval.csv", s["world"]["config"]["seed"])
            d["input_columns"] = list(EVAL_FEATURES)
            datasets.append(d)
            index = len(datasets) - 1
            if any(e["payload"]["type"] == "delivery_contract" for e in events):
                from .delivery_v3 import native_archive_rows

                rows = native_archive_rows([e["payload"] for e in events], self.path)
                write_csv(
                    q / "data/raw/campaign_archive.csv",
                    rows,
                    ["id", "path", "part", "parts", "text", "sha256"],
                )
                datasets.append(
                    dict(
                        path="data/raw/campaign_archive.csv",
                        role="training",
                        provenance="Immutable scientist-authored same-campaign code, output and records serialized by supervisor; not new observation units.",
                    )
                )
            write_yaml(
                q / "data_manifest.yaml", dict(status="ready", datasets=datasets)
            )
            suite = {
                "status": "ready",
                "metrics": [
                    dict(
                        name="demand_rmse",
                        direction="minimize",
                        role="primary",
                        normalization=dict(type="minmax", min=0, max=100),
                        computation=dict(
                            kind="rmse",
                            truth=f"data/{index}/eval.csv",
                            target="expected_demand",
                            prediction="mean_demand",
                            prediction_min=0,
                        ),
                    ),
                    dict(
                        name="sales_rmse",
                        direction="minimize",
                        role="diagnostic",
                        computation=dict(
                            kind="rmse",
                            truth=f"data/{index}/eval.csv",
                            target="expected_sales",
                            prediction="mean_sales",
                            prediction_min=0,
                            prediction_max_column="capacity",
                        ),
                    ),
                ],
            }
            if s["world"].get("spec", {}).get("version") == "designed-demand-v2":
                for metric in suite["metrics"]:
                    metric["role"] = "diagnostic"
                suite["research_scoring"] = (
                    "demand-scoring-v2.1; separate components, no native primary"
                )
            write_yaml(q / "validation.yaml", suite)
            expid = run_next(q, agent_stub=True)
            s.update(quest=str(q), experiment=expid)
            self._append(
                c,
                dict(
                    type="prediction_handoff",
                    quest="demand-confirmation",
                    experiment=expid,
                ),
            )
            return q, expid

    @staticmethod
    def _dataset(role, filename, seed):
        return dict(
            path=f"data/raw/{filename}",
            role=role,
            provenance="Simulated demand-lab data. Training replay serializes the frozen collected observations; session verification separately replays their generating actions. Evaluation replay generates fresh protected outcomes.",
            generator="data/generator.py",
            generator_inputs=["data/bundle.json"],
            seed=seed,
            generator_args=[
                "--split",
                role,
                "--seed",
                str(seed),
                "--output",
                "output.csv",
            ],
            generated_output="output.csv",
        )

    def finish(self, notebook: Path, plan: dict) -> dict:
        # Serialize draft copying, registration and publication. A crash releases
        # flock; the next invocation can reconcile an already scored native run.
        with (self.path / ".demand/final.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("A final evaluation is already running") from exc
            if self.view()["phase"] == "completed":
                self.verify()
                return self.view()["evaluation"]
            return self._finish_locked(notebook, plan)

    def _finish_locked(self, notebook: Path, plan: dict) -> dict:
        from sciquest.state import get_record
        from .evaluation import evaluate_science

        with self.transaction() as (c, s):
            self._runtime_check(s)
            if s["phase"] == "completed":
                return s["result"]
            if s["phase"] == "failed":
                raise ValueError(
                    "Final test failed and is terminal; preserve it and start a new campaign"
                )
        q, expid = self.prepare_quest()
        with self.transaction() as (c, s):
            expected_decision = s["decision"]
            expected_mode = s.get("information_mode", "assisted_v1")
        exp = q / "experiments" / expid
        reg = get_record(q, "registration", expid)
        if reg is None:
            plan = json.loads(canonical_json(plan))
            plan["demand_scientific_decision"] = expected_decision
            plan["scientist_information_mode"] = expected_mode
            with self.transaction() as (_, state):
                if "comparison" in state:
                    plan["comparison_assignment"] = state["comparison"]
            shutil.copyfile(notebook, exp / "notebook.ipynb")
            write_yaml(exp / "experiment_plan.yaml", plan)
            preregister_experiment(q, expid)
        elif reg["plan"].get("demand_scientific_decision") != expected_decision:
            raise ValueError("Frozen decision does not match registration")
        if (
            reg is not None
            and reg["plan"].get("scientist_information_mode") != expected_mode
        ):
            raise ValueError(
                "Registered scientist information boundary does not match the session"
            )
        try:
            result = execute_registered(q, expid, timeout=120, protected_root=self.path)
            execution = get_record(q, "execution", expid)
            artifact = exp / "worker_artifacts/predictions.csv"
            if file_hash(artifact) != execution["prediction_artifact_hashes"].get(
                "predictions.csv"
            ):
                raise ValueError("Final predictions changed after execution")
            with self.transaction() as (c, s):
                science = evaluate_science(
                    s["world"],
                    assisted_contract(DemandConfig(**s["world"]["config"])),
                    expected_decision,
                    artifact,
                    s["spent"],
                    sum(e["payload"]["type"] == "query" for e in self._events(c)),
                )
            combined = dict(
                native_evaluation=result,
                science=science,
                quest=str(q),
                experiment=expid,
            )
        except Exception as exc:
            # Once a frozen execution exists, never allow another predictor to
            # exploit its feedback. Invalid final intervals also terminate a run.
            if get_record(q, "execution", expid) is not None:
                with self.transaction() as (c, s):
                    s.update(phase="failed", failure=str(exc))
                    self._append(c, dict(type="final_failure", error=str(exc)))
            raise
        with self.transaction() as (c, s):
            s.update(phase="completed", result=combined)
            self._append(c, dict(type="evaluation", result=combined))
        # Non-authoritative projection can be repaired without repeating science.
        write_yaml(self.path / "results.yaml", combined)
        return combined

    def verify(self, reveal=False) -> dict:
        with self.transaction() as (c, s):
            self._runtime_check(s)
            contract = (
                public_contract
                if s.get("information_mode") == "black_box_v2"
                else assisted_contract
            )
            if s["world"] != private_world(DemandConfig(**s["world"]["config"])) or s[
                "public"
            ] != contract(DemandConfig(**s["world"]["config"])):
                raise ValueError("Environment contract or world changed")
            previous = ""
            spent = 0
            history = []
            frozen_decision = None
            published = None
            failed = None
            events = self._events(c)
            initial_hash = content_hash(
                {
                    k: s[k]
                    for k in (
                        "world",
                        "public",
                        "method_stack",
                        "profiles",
                        "runtime",
                        "information_mode",
                        *(("comparison",) if "comparison" in s else ()),
                    )
                }
            )
            if not events or events[0]["payload"] != dict(
                type="created", initial_hash=initial_hash
            ):
                raise ValueError("Initial environment or method commitment changed")
            for expected_revision, e in enumerate(events, 1):
                if e["revision"] != expected_revision:
                    raise ValueError("Nonconsecutive transcript revisions")
                if e["previous_hash"] != previous or e["hash"] != content_hash(
                    dict(
                        revision=e["revision"],
                        previous_hash=previous,
                        payload=e["payload"],
                    )
                ):
                    raise ValueError("Demand transcript hash chain mismatch")
                previous = e["hash"]
                payload = e["payload"]
                if payload["type"] == "query":
                    if frozen_decision is not None:
                        raise ValueError("Query recorded after final decision")
                    plan, contract = self._query_plan(
                        s["method_stack"],
                        payload["plan"],
                        history,
                        comparison="comparison" in s,
                    )
                    design, cost = normalize_action(
                        payload["action"], s["public"], history
                    )
                    obs, hidden = draw(s["world"], design, len(history) + 1, history)
                    if (
                        obs != payload["observation"]
                        or hidden != payload["private"]
                        or cost != payload["cost"]
                        or design != payload["design"]
                    ):
                        raise ValueError("Demand generation replay mismatch")
                    from sciquest.validation import COMPARATORS

                    outcomes = {
                        p["id"]: (
                            "supported"
                            if COMPARATORS[p["operator"]](obs["value"], p["threshold"])
                            else "contradicted"
                        )
                        for p in plan["predictions"]
                    }
                    if (
                        contract != payload["policy_contract"]
                        or policy_evidence(
                            {"plan": plan, "policy_contract": contract}, outcomes
                        )
                        != payload["policy_evidence"]
                    ):
                        raise ValueError("Query policy evidence changed")
                    spent += cost
                    history.append(payload)
                elif payload["type"] == "stop":
                    if frozen_decision is not None:
                        raise ValueError("Multiple frozen decisions")
                    frozen_decision = validate_decision(
                        payload["decision"],
                        assisted=s.get("information_mode") != "black_box_v2",
                        claims="comparison" in s,
                    )
                    if "comparison" in s:
                        from .policies_v2 import validate_claims

                        validate_claims(
                            frozen_decision, [q["observation"]["id"] for q in history]
                        )
                elif payload["type"] == "evaluation":
                    if (
                        published is not None
                        or frozen_decision is None
                        or failed is not None
                    ):
                        raise ValueError("Invalid final publication sequence")
                    published = payload["result"]
                elif payload["type"] == "final_failure":
                    failed = payload["error"]
            if spent != s["spent"] or spent > s["public"]["budget"]:
                raise ValueError("Budget accounting mismatch")
            if (
                frozen_decision != s["decision"]
                or published != s["result"]
                or failed != s.get("failure")
            ):
                raise ValueError("Transcript and published state differ")
            expected_phase = (
                "completed"
                if published is not None
                else (
                    "failed"
                    if failed is not None
                    else (
                        "prediction_required"
                        if frozen_decision is not None
                        else "collecting"
                    )
                )
            )
            if s["phase"] != expected_phase:
                raise ValueError("Transcript phase mismatch")
            if s["phase"] == "completed":
                from sciquest.state import get_record
                from sciquest.logic import run_logic_check
                from .evaluation import evaluate_science

                q = Path(s["quest"])
                expid = s["experiment"]
                if s["result"]["native_evaluation"] != get_record(
                    q, "evaluation", expid
                ):
                    raise ValueError(
                        "Native final evaluation differs from published results"
                    )
                if (
                    get_record(q, "registration", expid)["plan"].get(
                        "demand_scientific_decision"
                    )
                    != s["decision"]
                ):
                    raise ValueError("Registered scientific decision changed")
                if get_record(q, "registration", expid)["plan"].get(
                    "scientist_information_mode"
                ) != s.get("information_mode"):
                    raise ValueError(
                        "Registered scientist information boundary changed"
                    )
                if (
                    "comparison" in s
                    and get_record(q, "registration", expid)["plan"].get(
                        "comparison_assignment"
                    )
                    != s["comparison"]
                ):
                    raise ValueError("Registered comparison assignment changed")
                execution = get_record(q, "execution", expid)
                if s.get("information_mode") == "black_box_v2" and (
                    not execution.get("blind_filesystem")
                    or (
                        execution.get("model_export_verification")
                        and not execution["model_export_verification"]["execution"].get(
                            "blind_filesystem"
                        )
                    )
                ):
                    raise ValueError(
                        "Final execution did not enforce the scientist information boundary"
                    )
                check = run_logic_check(q, expid)
                if not check["passed"]:
                    raise ValueError(
                        "Native final-evaluation integrity failure: "
                        + str(check["issues"])
                    )
                expected = evaluate_science(
                    s["world"],
                    assisted_contract(DemandConfig(**s["world"]["config"])),
                    s["decision"],
                    q / "experiments" / expid / "worker_artifacts/predictions.csv",
                    spent,
                    len(history),
                )
                if expected != s["result"]["science"]:
                    raise ValueError("Scientific evaluation changed")
            proof = dict(
                verified=True,
                query_count=len(history),
                cost_spent=spent,
                transcript_hash=previous,
                phase=s["phase"],
                scope="Initial world/method commitment, hash chain, deterministic query replay, policy evidence and budget accounting; completed runs also verify registered artifacts and recompute final scores. Not a proof of scientific validity or protection against an owner rewriting the whole database.",
            )
            if reveal:
                if s["phase"] != "completed":
                    raise ValueError("Truth is only disclosed after completion")
                proof["truth"] = s["world"]
            return proof


TRAIN_FIELDS = (
    "id",
    "kind",
    "price",
    "context",
    "capacity",
    "sigma",
    "quantity",
    "value",
    "paired_with",
    "replicates",
)
PUBLIC_OBSERVATION_FIELDS = tuple(k for k in TRAIN_FIELDS if k != "sigma")
CALIBRATED_OBSERVATION_FIELDS = (*PUBLIC_OBSERVATION_FIELDS, "precision", "reference")


def observation_fields(public):
    return (
        CALIBRATED_OBSERVATION_FIELDS
        if "calibration_references" in public
        else PUBLIC_OBSERVATION_FIELDS
    )


def public_query(payload):
    """Never copy arbitrary simulator fields or owner diagnostics into a handoff."""
    return {
        **{k: payload[k] for k in ("type", "action", "plan", "cost")},
        "design": {
            k: v
            for k, v in payload["design"].items()
            if k
            in {
                "kind",
                "context",
                "price",
                "capacity",
                "precision",
                "observation_id",
                "replicates",
                "reference",
            }
        },
        "observation": {
            k: payload["observation"][k]
            for k in CALIBRATED_OBSERVATION_FIELDS
            if k in payload["observation"]
        },
    }


GENERATOR = """import argparse,json
from pathlib import Path
from sciquest.demand.world import evaluation_rows
from sciquest.demand.session import write_csv,TRAIN_FIELDS
p=argparse.ArgumentParser();p.add_argument('--split');p.add_argument('--seed',type=int);p.add_argument('--output');a=p.parse_args()
b=json.loads(Path('data/bundle.json').read_text())
assert a.seed==b['world']['config']['seed']
rows=b['observations'] if a.split=='training' else evaluation_rows(b['world'])
write_csv(Path(a.output),rows,b.get('train_fields',TRAIN_FIELDS) if a.split=='training' else None)
"""
