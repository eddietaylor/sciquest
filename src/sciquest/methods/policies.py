"""Small explicit policy contracts; these do not encode entire philosophies."""

from __future__ import annotations

import math
from typing import Any

from sciquest.io import content_hash, finite_number
from sciquest.validation import COMPARATORS

from .schema import METHOD_PHASES

SUPPORTED_POLICIES = {"strong_inference", "bayesian", "popperian_falsificationist"}


def governed_phases(stack: dict[str, Any]) -> dict[str, str]:
    return {
        phase: stack.get("phases", {}).get(phase, stack["primary_method"])
        for phase in METHOD_PHASES
    }


def _requirements(stack: dict[str, Any]) -> dict[str, list[str]]:
    phases = governed_phases(stack)
    required = {}
    if any(
        phases[p] == "strong_inference"
        for p in ("experiment_design", "evidence_evaluation", "stress_testing")
    ):
        required["strong_inference"] = ["rivals", "discriminating_prediction_id"]
    elif phases["hypothesis_generation"] == "strong_inference":
        required["strong_inference"] = ["rivals"]
    if phases["evidence_evaluation"] == "bayesian":
        required["bayesian"] = ["rivals", "belief_model"]
    if any(
        phases[p] == "popperian_falsificationist"
        for p in ("experiment_design", "evidence_evaluation", "stress_testing")
    ):
        required["popperian_falsificationist"] = ["severe_test"]
    return required


def method_capability(method_id: str) -> dict[str, Any]:
    return {
        "implementation": "phase_scoped_plan_contract"
        if method_id in SUPPORTED_POLICIES
        else "guidance_only",
        "scope": "Philosophy-inspired instructions; structured contracts apply only in relevant assigned phases. No guarantee of philosophical fidelity, identification, information gain or test severity.",
    }


def validate_prior_chain(
    qpath, exp_id, plan, contract, file_hashes, manifest
) -> dict | None:
    """Verify optional sequential conditioning, without inferring a prior silently."""
    if "bayesian" not in contract["enforced_plan_contracts"]:
        return None
    from sciquest.state import get_record

    evidence = {
        digest
        for name, digest in file_hashes.items()
        if name.startswith("data/")
        and manifest["datasets"][int(name.split("/")[1])].get("role") == "evaluation"
    }
    chain = {"status": "unlinked_declared_prior", "evidence_hashes": sorted(evidence)}
    model = plan["belief_model"]
    source = model.get("prior_source")
    if source is None:
        return chain
    previous_id = source["experiment_id"]
    if previous_id == exp_id:
        raise ValueError("Bayesian prior_source cannot refer to the current experiment")
    registration = get_record(qpath, "registration", previous_id)
    evaluation = get_record(qpath, "evaluation", previous_id)
    completion = get_record(qpath, "completion", previous_id)
    if (
        not registration
        or not evaluation
        or not completion
        or completion.get("evaluation_hash") != content_hash(evaluation)
        or evaluation.get("registration_hash") != registration.get("registration_hash")
    ):
        raise ValueError(
            "Bayesian prior_source requires a completed authoritative evaluation"
        )
    update = evaluation.get("policy_evidence", {}).get("belief_update", {})
    posterior = update.get("posterior")
    if not isinstance(posterior, dict) or set(posterior) != set(model["prior"]):
        raise ValueError("Bayesian prior_source has no compatible posterior")
    if any(
        not math.isclose(model["prior"][key], p, rel_tol=1e-12, abs_tol=0)
        for key, p in posterior.items()
    ):
        raise ValueError("Declared priors do not match the source posterior")
    claims = lambda p: {r["id"]: r["claim"] for r in p["rivals"]}
    if claims(plan) != claims(registration["plan"]):
        raise ValueError(
            "Linked Bayesian rivals must retain their registered identities and claims"
        )
    assumptions = model.get("conditional_evidence_assumptions")
    if not isinstance(assumptions, str) or not assumptions.strip():
        raise ValueError(
            "Linked Bayesian updates require conditional_evidence_assumptions"
        )
    previous_hashes = set(
        (registration.get("prior_chain") or {}).get("evidence_hashes", [])
    )
    if not previous_hashes:
        previous_hashes = {
            digest
            for name, digest in registration["file_hashes"].items()
            if name.startswith("data/")
            and registration["data_manifest"]["datasets"][int(name.split("/")[1])].get(
                "role"
            )
            == "evaluation"
        }
    if evidence & previous_hashes:
        raise ValueError(
            "Linked Bayesian updates cannot count reused evaluation evidence again"
        )
    return {
        "status": "linked_verified",
        "source_experiment_id": previous_id,
        "source_evaluation_hash": content_hash(evaluation),
        "evidence_hashes": sorted(evidence | previous_hashes),
        "scope": "Posterior provenance and exact evidence reuse checked; conditional likelihood assumptions remain scientific responsibilities.",
    }


def policy_requirements(stack: dict[str, Any]) -> dict[str, Any]:
    requirements = _requirements(stack)
    templates = {
        "strong_inference": {
            "rivals": [
                {"id": "h0", "claim": "Null explanation", "predictions": {"p1": 0}},
                {
                    "id": "h1",
                    "claim": "Alternative explanation",
                    "predictions": {"p1": 1},
                },
            ],
            "discriminating_prediction_id": "p1",
        },
        "bayesian": {
            "rivals": [
                {"id": "h0", "claim": "Null explanation"},
                {"id": "h1", "claim": "Alternative explanation"},
            ],
            "belief_model": {
                "prediction_id": "p1",
                "prior": {"h0": 0.5, "h1": 0.5},
                "likelihood": {
                    "h0": {"supported": 0.1, "contradicted": 0.9},
                    "h1": {"supported": 0.9, "contradicted": 0.1},
                },
                "assumptions": "Replace with justified likelihood assumptions; these example probabilities are not scientific evidence.",
            },
        },
        "popperian_falsificationist": {
            "severe_test": {
                "prediction_id": "p1",
                "failure_action": "revise",
                "auxiliary_assumptions": [
                    "State observation and auxiliary assumptions"
                ],
                "diagnostic_rationale": "Explain what inadequacy the specified test can reveal.",
            }
        },
    }
    return {
        "schema_version": 2,
        "phase_governance": governed_phases(stack),
        "guidance_only_methods": sorted(
            set(governed_phases(stack).values()) - requirements.keys()
        ),
        "required_plan_extensions": {
            p: {k: templates[p][k] for k in fields}
            for p, fields in sorted(requirements.items())
        },
        "instruction": "Merge each required_plan_extensions method object directly into the root of PLAN. Method names are grouping labels, NOT plan nesting keys: use plan.rivals, plan.belief_model, plan.discriminating_prediction_id or plan.severe_test as required. Replace example content with justified domain-specific entries and actual registered prediction IDs. Templates are not evidence.",
    }


def _rivals(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rivals = plan.get("rivals")
    if not isinstance(rivals, list) or len(rivals) < 2:
        raise ValueError("Inquiry policy requires at least two explicit rivals")
    result = {}
    for rival in rivals:
        if (
            not isinstance(rival, dict)
            or not isinstance(rival.get("id"), str)
            or not rival["id"]
            or rival["id"] in result
            or not isinstance(rival.get("claim"), str)
            or not rival["claim"].strip()
        ):
            raise ValueError("Rivals require unique IDs and concrete claims")
        result[rival["id"]] = rival
    return result


def validate_policy_plan(stack: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    requirements = _requirements(stack)
    policies = set(requirements)
    prediction_ids = {p["id"] for p in plan["predictions"]}
    if "strong_inference" in policies:
        rivals = _rivals(plan)
    if "discriminating_prediction_id" in requirements.get("strong_inference", []):
        prediction_id = plan.get("discriminating_prediction_id")
        if prediction_id not in prediction_ids:
            raise ValueError(
                "Strong inference requires a registered discriminating prediction"
            )
        values = []
        for rival in rivals.values():
            values.append(
                finite_number(
                    rival.get("predictions", {}).get(prediction_id), "rival prediction"
                )
            )
        if len(set(values)) < 2:
            raise ValueError(
                "Rivals must predict different outcomes on the selected test"
            )
        prediction = next(p for p in plan["predictions"] if p["id"] == prediction_id)
        compare = COMPARATORS.get(prediction.get("operator"))
        if compare is None or "threshold" not in prediction:
            raise ValueError(
                "Discriminating predictions require an operator and threshold"
            )
        threshold = finite_number(prediction["threshold"], "discriminator threshold")
        if len({compare(v, threshold) for v in values}) < 2:
            raise ValueError(
                "Rival predictions must imply different outcomes of the registered decision rule"
            )
    if "bayesian" in policies:
        rivals = _rivals(plan)
        model = plan.get("belief_model")
        if (
            not isinstance(model, dict)
            or model.get("prediction_id") not in prediction_ids
        ):
            raise ValueError(
                "Bayesian inquiry requires a likelihood model for a registered prediction"
            )
        prior, likelihood = model.get("prior"), model.get("likelihood")
        if (
            not isinstance(prior, dict)
            or set(prior) != set(rivals)
            or not isinstance(likelihood, dict)
            or set(likelihood) != set(rivals)
        ):
            raise ValueError("Priors and likelihoods must cover every rival")
        if any(
            not 0 <= finite_number(p, "prior probability") <= 1 for p in prior.values()
        ) or not math.isclose(math.fsum(prior.values()), 1, abs_tol=1e-9):
            raise ValueError("Prior probabilities must be nonnegative and sum to one")
        for row in likelihood.values():
            if not isinstance(row, dict) or set(row) != {"supported", "contradicted"}:
                raise ValueError(
                    "Likelihoods require supported and contradicted probabilities"
                )
            if any(
                not 0 <= finite_number(p) <= 1 for p in row.values()
            ) or not math.isclose(math.fsum(row.values()), 1, abs_tol=1e-9):
                raise ValueError("Likelihood probabilities must sum to one")
        if (
            not isinstance(model.get("assumptions"), str)
            or not model["assumptions"].strip()
        ):
            raise ValueError("Likelihood assumptions must be explicit")
        source = model.get("prior_source")
        if source is not None and (
            not isinstance(source, dict)
            or set(source) != {"experiment_id"}
            or not isinstance(source["experiment_id"], str)
        ):
            raise ValueError("prior_source must identify one previous experiment_id")
    if "popperian_falsificationist" in policies:
        test = plan.get("severe_test")
        if (
            not isinstance(test, dict)
            or test.get("prediction_id") not in prediction_ids
        ):
            raise ValueError(
                "Falsification-inspired inquiry requires a registered risky prediction"
            )
        if test.get("failure_action") not in {"reject", "revise", "replicate"}:
            raise ValueError("Specify what a failed prediction changes")
        if (
            not isinstance(test.get("auxiliary_assumptions"), list)
            or not test["auxiliary_assumptions"]
            or any(
                not isinstance(a, str) or not a.strip()
                for a in test["auxiliary_assumptions"]
            )
        ):
            raise ValueError("Specify auxiliary assumptions and observation conditions")
        if (
            not isinstance(test.get("diagnostic_rationale"), str)
            or not test["diagnostic_rationale"].strip()
        ):
            raise ValueError("Specify why the test could reveal inadequacy")
    return {
        "version": 2,
        "enforced_plan_contracts": sorted(policies),
        "requirements": requirements,
        "inactive_plan_extensions": sorted(
            {
                "belief_model",
                "discriminating_prediction_id",
                "severe_test",
            }.intersection(plan)
            - {field for fields in requirements.values() for field in fields}
        ),
        "phase_governance": governed_phases(stack),
        "guidance_only_methods": sorted(
            set(governed_phases(stack).values()) - policies
        ),
        "scope": "Registered planning constraints and numerical evidence updates; not a guarantee of philosophical fidelity or test severity",
    }


def policy_evidence(
    registration: dict[str, Any], outcomes: dict[str, str]
) -> dict[str, Any]:
    plan = registration["plan"]
    result = {
        "contract": registration.get("policy_contract", {}),
        "observed_prediction_outcomes": outcomes,
    }
    contracts = registration.get("policy_contract", {}).get(
        "enforced_plan_contracts", []
    )
    discriminating = "discriminating_prediction_id" in registration.get(
        "policy_contract", {}
    ).get("requirements", {}).get("strong_inference", [])
    legacy = registration.get("policy_contract", {}).get("version", 1) == 1
    if (
        "strong_inference" in contracts
        and (discriminating or legacy)
        and plan.get("discriminating_prediction_id")
    ):
        pid = plan["discriminating_prediction_id"]
        prediction = next(p for p in plan["predictions"] if p["id"] == pid)
        compare = COMPARATORS[prediction["operator"]]
        observed = outcomes[pid]
        result["rival_assessment"] = {
            "prediction_id": pid,
            "assessments": {
                r["id"]: {
                    "predicted_rule_outcome": "supported"
                    if compare(r["predictions"][pid], prediction["threshold"])
                    else "contradicted",
                    "rule_compatibility": "inconclusive"
                    if observed == "inconclusive"
                    else "compatible"
                    if compare(r["predictions"][pid], prediction["threshold"])
                    == (observed == "supported")
                    else "incompatible",
                }
                for r in plan["rivals"]
            },
            "scope": "Compatibility with one coarse registered rule, conditional on declared rival predictions; not proof of a unique explanation.",
        }
    if "popperian_falsificationist" in contracts:
        test = plan["severe_test"]
        outcome = outcomes[test["prediction_id"]]
        result["falsification_assessment"] = {
            "prediction_id": test["prediction_id"],
            "status": {
                "supported": "survived_registered_test",
                "contradicted": "risky_prediction_failed",
                "inconclusive": "inconclusive",
            }[outcome],
            "registered_failure_action": test["failure_action"],
            "next_step": test["failure_action"]
            if outcome == "contradicted"
            else "seek_more_diagnostic_evidence",
            "auxiliary_assumptions": test["auxiliary_assumptions"],
            "scope": "The recorded response needs scientific review of auxiliaries and measurement; failure does not automatically refute a whole theory, and survival is not verification.",
        }
    model = plan.get("belief_model")
    if model and "bayesian" in registration.get("policy_contract", {}).get(
        "enforced_plan_contracts", []
    ):
        outcome = outcomes[model["prediction_id"]]
        if outcome == "inconclusive":
            result["belief_update"] = {
                "prior": model["prior"],
                "posterior": model["prior"],
                "status": "no_update_inconclusive",
                "scope": "No likelihood event observed",
                "next_step": "obtain_a_defined_likelihood_event",
                "prior_chain": registration.get(
                    "prior_chain", {"status": "unlinked_declared_prior"}
                ),
            }
            return result
        log_weights = {
            key: math.log(model["prior"][key]) + math.log(row[outcome])
            if model["prior"][key] > 0 and row[outcome] > 0
            else -math.inf
            for key, row in model["likelihood"].items()
        }
        maximum = max(log_weights.values())
        weights = {
            key: math.exp(v - maximum) if maximum != -math.inf else 0.0
            for key, v in log_weights.items()
        }
        total = math.fsum(weights.values())
        result["belief_update"] = {
            "prior": model["prior"],
            "outcome": outcome,
            "likelihood_model_hash": content_hash(model),
            "posterior": {key: value / total for key, value in weights.items()}
            if total > 0
            else None,
            "status": "updated" if total > 0 else "model_inadequate",
            "next_step": "plan_next_measurement_under_posterior"
            if total > 0
            else "review_observation_model_or_expand_rivals",
            "log_evidence_probability": maximum + math.log(total)
            if total > 0
            else None,
            "prior_chain": registration.get(
                "prior_chain", {"status": "unlinked_declared_prior"}
            ),
            "scope": "Conditional on the registered likelihood model and its assumptions",
        }
    return result
