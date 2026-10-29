"""A small mechanism-known test laboratory, not a validated industry simulator.

The environment owns truth, noise, intervention rights, budget and final draws.
Reference policies receive only observations and public experimental capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

from .io import content_hash, finite_number

SCENARIOS = (
    "elastic",
    "null",
    "confounded",
    "censored",
    "observational_only",
    "measurement_error",
    "misspecified",
)
POLICIES = ("strong_inference", "bayesian", "popperian_falsificationist")
SLOPES = (0.0, -20.0)


@dataclass(frozen=True)
class LabConfig:
    scenario: str = "elastic"
    seed: int = 0
    budget: int = 16

    def __post_init__(self):
        if self.scenario not in SCENARIOS:
            raise ValueError("Unknown laboratory scenario")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("Seed must be an unsigned 32-bit integer")
        if (
            isinstance(self.budget, bool)
            or not isinstance(self.budget, int)
            or not 1 <= self.budget <= 10000
        ):
            raise ValueError("Budget must be between 1 and 10000 observations")


class PricingLab:
    def __init__(self, config: LabConfig):
        self.__config = config
        self.__rng = random.Random(config.seed)
        self.__slope = (
            0.0
            if config.scenario == "null"
            else -60.0
            if config.scenario == "misspecified"
            else -20.0
        )
        self.__used = 0
        self.__trace: list[dict[str, Any]] = []

    def capabilities(self) -> dict[str, Any]:
        confounded = self.__config.scenario in {"confounded", "observational_only"}
        return {
            "budget": self.__config.budget,
            "remaining": self.__config.budget - self.__used,
            "interventions_available": self.__config.scenario != "observational_only",
            "observation_may_be_confounded": confounded,
            "baseline_mean": 100.0,
            "prices": [0.5, 1.0, 1.5],
            "capacities": [60.0, 150.0],
            "measurement_sigma": 10.0
            if self.__config.scenario == "measurement_error"
            else 2.0,
            "intervention_sigma": math.sqrt(404.0)
            if confounded
            else 10.0
            if self.__config.scenario == "measurement_error"
            else 2.0,
        }

    def request(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.__used >= self.__config.budget:
            raise RuntimeError("Observation budget exhausted")
        if not isinstance(action, dict) or action.get("kind") not in {
            "observe",
            "intervene",
        }:
            raise ValueError("Unknown laboratory action")
        if (
            action["kind"] == "intervene"
            and not self.capabilities()["interventions_available"]
        ):
            raise ValueError("This laboratory grants observational access only")
        price = finite_number(action.get("price", 1.0), "price")
        capacity = finite_number(action.get("capacity", 150.0), "capacity")
        if (
            price not in self.capabilities()["prices"]
            or capacity not in self.capabilities()["capacities"]
        ):
            raise ValueError("Intervention is outside the permitted action space")
        latent = self.__rng.gauss(0, 1)
        if action["kind"] == "observe":
            price = max(0.5, min(1.5, 1 + 0.2 * latent))
            capacity = 60.0 if self.__config.scenario == "censored" else 150.0
        confound = 0.0
        if self.__config.scenario == "confounded":
            confound = 20 * latent
        elif self.__config.scenario == "observational_only":
            # In the unsaturated observable range, this cancels the slope in
            # observational data. The access contract does not identify it.
            confound = 10 * latent - self.__slope * (price - 1)
        mean = min(capacity, max(0.0, 100 + self.__slope * (price - 1) + confound))
        observed = mean + self.__rng.gauss(0, self.capabilities()["measurement_sigma"])
        self.__used += 1
        observation = {
            "id": self.__used,
            "kind": action["kind"],
            "price": price,
            "capacity": capacity,
            "measured_sales": observed,
            "measurement_note": "Noisy sensor values may exceed physical sales bounds",
        }
        self.__trace.append({"action": dict(action), "observation": observation})
        return dict(observation)

    def evaluate(self, decision: dict[str, Any]) -> dict[str, Any]:
        """Final independent intervention evaluation, with a separate random stream."""
        posterior = decision.get("posterior", [0.5, 0.5])
        if (
            not isinstance(posterior, list)
            or len(posterior) != 2
            or any(finite_number(p) < 0 for p in posterior)
            or not math.isclose(sum(posterior), 1, abs_tol=1e-9)
        ):
            raise ValueError("Invalid final predictive distribution")
        rng = random.Random(self.__config.seed ^ 0x6F12A9D3)
        errors = []
        for _ in range(100):
            price = rng.uniform(0.5, 1.5)
            actual = 100 + self.__slope * (price - 1)
            predicted = math.fsum(
                p * (100 + s * (price - 1)) for p, s in zip(posterior, SLOPES)
            )
            errors.append((predicted - actual) ** 2)
        status = decision.get("status")
        identifiable = self.__config.scenario != "observational_only"
        model_contains_truth = self.__slope in SLOPES
        best = max(range(2), key=posterior.__getitem__)
        return {
            "intervention_prediction_rmse": math.sqrt(math.fsum(errors) / len(errors)),
            "correct_identifiability_abstention": not identifiable
            and status == "not_identifiable",
            "correct_model_inadequacy_abstention": not model_contains_truth
            and status == "model_inadequate",
            "mechanism_recovered": identifiable
            and model_contains_truth
            and status == "selected"
            and SLOPES[best] == self.__slope,
            "observations_used": self.__used,
            "observation_budget": self.__config.budget,
            "truth_disclosed_after_campaign": {
                "slope": self.__slope,
                "scenario": self.__config.scenario,
            },
            "scope": "Known toy mechanisms and conditional finite-model inference; no real-market validation or universal error-control claim",
        }


def _means(price: float, capacity: float) -> list[float]:
    return [min(capacity, max(0, 100 + s * (price - 1))) for s in SLOPES]


def _gaussian_information(
    means: list[float], weights: list[float], sigma: float
) -> float:
    # Deterministic quadrature of mutual information for a finite Gaussian mixture.
    low, high = min(means) - 8 * sigma, max(means) + 8 * sigma
    width = (high - low) / 160
    total = 0.0
    for step in range(160):
        value = low + (step + 0.5) * width
        densities = [
            math.exp(-0.5 * ((value - mu) / sigma) ** 2)
            / (sigma * math.sqrt(2 * math.pi))
            for mu in means
        ]
        mixture = math.fsum(p * d for p, d in zip(weights, densities))
        if mixture > 0:
            total += width * math.fsum(
                p * d * math.log(d / mixture)
                for p, d in zip(weights, densities)
                if p > 0 and d > 0
            )
    return max(0.0, total)


def reference_campaign(
    config: LabConfig, policy: str = "strong_inference"
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError("Unknown inquiry policy")
    lab = PricingLab(config)
    public = lab.capabilities()
    posterior = [0.5, 0.5]
    history = []
    if not public["interventions_available"]:
        observation = lab.request({"kind": "observe"})
        decision = {
            "status": "not_identifiable",
            "reason": "The access contract permits observational confounding and grants no identifying intervention or instrument.",
            "posterior": posterior,
        }
        history.append(
            {
                "observation": observation,
                "inference": "No causal update from unidentified observational association",
            }
        )
    else:
        decision = None
        for index in range(config.budget):
            candidates = []
            for price in public["prices"]:
                for capacity in public["capacities"]:
                    means = _means(price, capacity)
                    if policy == "bayesian":
                        score = _gaussian_information(
                            means, posterior, public["intervention_sigma"]
                        )
                        rationale = "Expected information under the stated finite Gaussian likelihood"
                    elif policy == "strong_inference":
                        score = max(means) - min(means)
                        rationale = (
                            "Separate explicit no-effect and price-effect rivals"
                        )
                    else:
                        score = abs(means[0] - means[1]) / public["intervention_sigma"]
                        rationale = "Risk a prediction against the strongest stated alternative under the observation noise model"
                    # The falsification policy alternates the sign of the risky
                    # price perturbation, checking both directions of the claim.
                    tie = (
                        price
                        if policy == "popperian_falsificationist" and index % 2 == 0
                        else -price
                    )
                    candidates.append((score, tie, price, capacity, means, rationale))
            _, _, price, capacity, means, rationale = max(candidates)
            observation = lab.request(
                {"kind": "intervene", "price": price, "capacity": capacity}
            )
            value, sigma = observation["measured_sales"], public["intervention_sigma"]
            prior = posterior[:]
            log_weights = [
                math.log(max(p, 1e-300)) - 0.5 * ((value - mu) / sigma) ** 2
                for p, mu in zip(prior, means)
            ]
            offset = max(log_weights)
            weights = [math.exp(w - offset) for w in log_weights]
            posterior = [w / math.fsum(weights) for w in weights]
            inadequate = min(abs(value - mu) / sigma for mu in means) > 4
            history.append(
                {
                    "action": {
                        "kind": "intervene",
                        "price": price,
                        "capacity": capacity,
                    },
                    "rationale": rationale,
                    "rival_predictions": means,
                    "prior": prior,
                    "observation": observation,
                    "posterior": posterior[:],
                    "model_check_failed": inadequate,
                }
            )
            if inadequate:
                decision = {
                    "status": "model_inadequate",
                    "reason": "All candidate models fail the registered four-sigma diagnostic; expand the class before claiming a mechanism.",
                    "posterior": posterior,
                }
                break
            if index >= 3 and max(posterior) > 0.99:
                decision = {
                    "status": "selected",
                    "reason": "Posterior concentration criterion met after at least four interventions.",
                    "posterior": posterior,
                }
                break
        if decision is None:
            decision = {
                "status": "inconclusive",
                "reason": "Observation budget exhausted before the stopping criterion.",
                "posterior": posterior,
            }
    result = {
        "schema_version": 1,
        "environment_version": "pricing-fixture-v1",
        "config": {
            "scenario": config.scenario,
            "seed": config.seed,
            "budget": config.budget,
        },
        "policy": policy,
        "decision": decision,
        "trace": history,
        "evaluation": lab.evaluate(decision),
        "limitations": [
            "Two explicit candidate slopes with a known intercept and observation model.",
            "Reference executable policies, not evidence about LLM agents or complete philosophical traditions.",
            "Policies can select the same experiment; no performance advantage is assumed.",
        ],
    }
    result["campaign_hash"] = content_hash(result)
    return result
