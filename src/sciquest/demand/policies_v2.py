"""Matched guidance treatments and observable fidelity; no philosophical truth test."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from sciquest.io import content_hash
from sciquest.methods.policies import validate_policy_plan, policy_evidence
from sciquest.methods.resolver import resolve_method_stack

POLICIES = (
    "standard_empirical",
    "strong_inference",
    "bayesian",
    "popperian_falsificationist",
)
VERSION = "matched-guidance-v2.2"
# Similar length, one paragraph per treatment; no padding and no repeated profiles
# by phase. Exact bytes/words are disclosed, not represented as equal tokenizer cost.
GUIDANCE = {
    "standard_empirical": "Follow an empirical inquiry cycle. State a testable hypothesis and its scope, plan an observation, compare the result with the prediction, and choose a useful next step. Consider plausible alternatives and measurement limitations. Revise provisional conclusions when warranted by observations, and replicate or improve measurement where useful. Explain what each test changed in your understanding. Keep exploratory findings distinct from confirmation, acknowledge unresolved questions, and stop when additional inquiry is unlikely to help within the available resources.",
    "strong_inference": "Follow a strong-inference inquiry cycle. Develop multiple substantive rival explanations before selecting observations, and seek tests on which their predictions differ. Record rival-specific predictions and the observation that would discriminate them. Compare the measured result with these predictions, retaining rivals where uncertainty prevents elimination. Revise or expand the rival set when necessary, and design the next discriminating test. Explain what each test changed in your understanding. A comparison among stated rivals does not establish that the surviving rival exhausts possible explanations.",
    "bayesian": "Follow a Bayesian inquiry cycle. Represent uncertainty across substantive explanations using explicit prior beliefs and a predictive observation model. Update beliefs by conditioning on the actual evidence, stating likelihood assumptions and dependencies. Compare prospective observations by expected information or decision value where feasible, and explain approximations. Track posterior provenance rather than resetting beliefs or counting evidence twice. Explain what each test changed in your understanding. Check whether all stated models may be inadequate; posterior confidence remains conditional on priors, likelihoods and the rival set.",
    "popperian_falsificationist": "Follow a falsification-inspired inquiry cycle. Formulate risky predictions and seek tests capable of exposing specified inadequacies rather than merely accumulating agreement. Before observing, state the failure criterion, auxiliary assumptions, measurement checks and what failure would change. After a failed prediction, examine the hypothesis and auxiliaries, then implement a justified rejection, revision or replication. Explain what each test changed in your understanding. Surviving a test does not verify a theory; uncertainty, weak tests and untested alternatives limit what can be inferred.",
}

COMMON = """You investigate a hidden apparatus using only your own observations and analysis.
The supplied inquiry guidance is fixed for this campaign. Every treatment has the
same action space, common plan format, validation, memory and resource rules.
Here method_stack and policy_contract describe COMMON VALIDATION only;
comparison.policy and the fixed paragraph below govern inquiry in every phase.
Do not interpret the common empirical validation stack as replacing that guidance.
If the public contract lists calibration_references, an apparatus ACTION can be
{kind:calibrate,reference:REFERENCE,precision:standard/precise}, using the listed
calibration cost and precision surcharge. This is in addition to the other listed
actions; a known-reference reading is not a demand unit.
Plans may include rivals, discriminating_prediction_id, belief_model and
severe_test, but none is required for format acceptance in this guidance comparison.
Optional records, when supplied, receive the SAME validation in every arm:
rivals is a list of {id,claim,predictions:{PREDICTION_ID:number}} (predictions optional
except for a discriminator). A discriminating_prediction_id names a registered
prediction whose rival values imply different threshold-rule outcomes.
belief_model has prediction_id, prior:{RIVAL_ID:probability},
likelihood:{RIVAL_ID:{supported:probability,contradicted:probability}}, assumptions;
probabilities normalize. Optional prior_source:{experiment_id:QUERY_ID} links the
previous posterior, with conditional_evidence_assumptions for dependence.
severe_test has prediction_id, failure_action (reject/revise/replicate),
auxiliary_assumptions (list of strings), diagnostic_rationale (string).
No likelihood values or scientific rivals are provided. Record substantive method-specific work
where appropriate so it can be audited. Do not manufacture numbers to fill fields.
These are bounded operationalizations, not complete philosophical traditions.

Use campaign_notes to retain your own useful code, fits, uncertainty, contradictions
and evidence references. Empty notes retain the preceding notes. New notes replace
them and must fit the stated character bound. Original observations and plans stay
in history. Prior analysis code and complete bounded output are in same-campaign
analysis/aNNNN.py and analysis/aNNNN.txt during offline analysis. The compact index
records hashes and statuses; omitted stdout is not missing evidence. You may inspect
those files in an analysis call, which uses the ordinary analysis allowance. No
owner-authored summary, model catalog, equations or solution is supplied.

Freeze a final decision with the usual status/reason/explanation/limitations plus
claims: a nonempty list of {id,quote,kind,scope,epistemic_status,assumptions,evidence_ids}.
Quote must appear exactly in your explanation, reason or limitations. Kind is
causal/descriptive/predictive/measurement/decision; scope is query_settings/transfer/
full_law; epistemic_status is identified/assumption_dependent/unresolved. Assumptions
is a string and evidence_ids lists your observation IDs. Record material positive
claims and unresolved questions. These fields record your judgments; labels alone
earn no credit. All decisive prose claims remain assessable even if omitted here.

Final forecasts have physical means and central 90% intervals for fresh STANDARD
precision noisy sales readings. Pricing recommendations use only decision_eligible
rows at or below the public price_limit, maximizing price*mean_sales, ties at the
lowest price. Limits apply to deployment, not listed experimental controls. Query
settings and transfer rows, physical predictions and sensor intervals, and scientific
claims and decision utility are assessed separately. No final score feedback is
available before the complete frozen submission.
"""


@dataclass(frozen=True)
class ResourceLimits:
    max_queries: int = 12
    max_analyses: int = 8
    max_calls: int = 24
    max_prompt_bytes: int = 98304
    max_notes_chars: int = 8000
    recent_analysis_chars: int = 6000
    input_token_stop: int = 200000
    output_token_stop: int = 60000

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("Resource limits must be positive integers")
        if (
            self.max_calls < 3
            or self.max_notes_chars > 16000
            or self.recent_analysis_chars > 16000
        ):
            raise ValueError("Invalid resource limits")


def assignment(policy, limits=None):
    if policy not in POLICIES:
        raise ValueError("Unknown comparison policy")
    return dict(
        version=VERSION,
        policy=policy,
        validation="common_empirical_schema",
        guidance_hash=content_hash(GUIDANCE[policy]),
        limits=asdict(limits or ResourceLimits()),
    )


def validate_assignment(value):
    if value != assignment(value["policy"], ResourceLimits(**value["limits"])):
        raise ValueError("Comparison assignment changed")
    return value


def instruction_audit():
    stats = {
        p: dict(
            words=len(t.split()), utf8_bytes=len(t.encode()), sha256=content_hash(t)
        )
        for p, t in GUIDANCE.items()
    }
    return dict(
        version=VERSION,
        treatments=stats,
        max_word_ratio=max(s["words"] for s in stats.values())
        / min(s["words"] for s in stats.values()),
        validation="Identical common plan and final submission schema for every arm; no arm-specific mandatory methodological records. Supplied optional records receive identical validation.",
        computation="Same model/reasoning/call/analysis/observation and prompt-size opportunities. Cumulative usage stopping is checked before the next call and can overshoot by one call; CLI has no verified output-token cap. Not exact FLOP or token matching; report raw totals, missing usage, overshoot and efficiency separately. Do not adjust the primary effect post hoc for realized compute, a possible treatment mediator.",
        estimand="Effect of these four guidance paragraphs with common validation, not effect of the older instruction-plus-enforcement bundles or whole philosophical traditions.",
    )


def validate_claims(decision, evidence_ids):
    claims = decision.get("claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= 20:
        raise ValueError("Decision needs 1..20 claim records")
    ids = set()
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {
            "id",
            "quote",
            "kind",
            "scope",
            "epistemic_status",
            "assumptions",
            "evidence_ids",
        }:
            raise ValueError("Invalid claim fields")
        if (
            not isinstance(claim["id"], str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", claim["id"]) is None
            or claim["id"] in set(evidence_ids) | {"campaign_notes"}
            or claim["id"] in ids
        ):
            raise ValueError("Claim IDs must be unique")
        ids.add(claim["id"])
        if (
            not isinstance(claim["quote"], str)
            or not claim["quote"].strip()
            or not any(
                claim["quote"] in decision[k]
                for k in ("reason", "explanation", "limitations")
            )
        ):
            raise ValueError("Claim quote must exactly match scientific decision text")
        if (
            claim["kind"]
            not in ("causal", "descriptive", "predictive", "measurement", "decision")
            or claim["scope"] not in ("query_settings", "transfer", "full_law")
            or claim["epistemic_status"]
            not in ("identified", "assumption_dependent", "unresolved")
        ):
            raise ValueError("Unknown claim kind, scope or epistemic status")
        if (
            not isinstance(claim["assumptions"], str)
            or not isinstance(claim["evidence_ids"], list)
            or any(not isinstance(x, str) for x in claim["evidence_ids"])
            or len(set(claim["evidence_ids"])) != len(claim["evidence_ids"])
            or not set(claim["evidence_ids"]) <= set(evidence_ids)
        ):
            raise ValueError("Invalid claim assumptions or evidence references")
    return claims


def fidelity_audit(session):
    """Uniform observable indicators for ALL arms. Arithmetic is not sincerity."""
    with session.transaction() as (c, state):
        events = session._events(c)
    rows = []
    for event in events:
        q = event["payload"]
        if q["type"] != "query":
            continue
        diagnostics = {}
        for policy in POLICIES[1:]:
            try:
                contract = validate_policy_plan(
                    resolve_method_stack(policy).to_dict(), q["plan"]
                )
                from sciquest.validation import COMPARATORS

                outcomes = {
                    p["id"]: "supported"
                    if COMPARATORS[p["operator"]](
                        q["observation"]["value"], p["threshold"]
                    )
                    else "contradicted"
                    for p in q["plan"]["predictions"]
                }
                diagnostics[policy] = dict(
                    record_present_and_valid=True,
                    evidence=policy_evidence(
                        dict(plan=q["plan"], policy_contract=contract), outcomes
                    ),
                )
            except (ValueError, KeyError, TypeError) as exc:
                diagnostics[policy] = dict(
                    record_present_and_valid=False, reason=str(exc)
                )
        rows.append(dict(observation_id=q["observation"]["id"], indicators=diagnostics))
    return dict(
        version=VERSION,
        assigned_policy=state.get("comparison", {}).get("policy"),
        queries=rows,
        scope="Same indicators for every arm, including spontaneous cross-policy behavior. Checks recorded discriminators, likelihood arithmetic and planned failure responses; does not validate scientific likelihoods, power, substantive rivals, actual follow-through or sincere philosophy. Independent masked trace assessment covers those limitations. Missing record is not proof of missing reasoning.",
    )


def optional_contract(stack, plan):
    """Same optional record validation/arithmetic in every guidance treatment."""
    contract = validate_policy_plan(stack, plan)
    requested = []
    if "discriminating_prediction_id" in plan:
        requested.append("strong_inference")
    if "belief_model" in plan:
        requested.append("bayesian")
    if "severe_test" in plan:
        requested.append("popperian_falsificationist")
    for policy in requested:
        checked = validate_policy_plan(resolve_method_stack(policy).to_dict(), plan)
        contract["requirements"].update(checked["requirements"])
    contract["enforced_plan_contracts"] = sorted(requested)
    contract["inactive_plan_extensions"] = []
    contract["scope"] = (
        "Common optional-record validation in every arm; not assignment-specific enforcement or certified scientific validity."
    )
    return contract
