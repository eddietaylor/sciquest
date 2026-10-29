"""Supervisor-only decision, access and budget witnesses. Never stage for a scientist."""

from __future__ import annotations

import ast
import itertools
import math
from collections import defaultdict

from sciquest.io import content_hash
from .designed import P_GRID, X_GRID, Z_GRID
from .designed_v2 import validate_spec, evaluation_rows, latent

BUDGET = 24
MAX_QUERIES = 12


def invariant_rival(spec):
    class Reference(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Constant(1), node) if node.id == "p" else node

    tree = Reference().visit(ast.parse(spec["expression"], mode="eval"))
    return {**spec, "expression": ast.unparse(ast.fix_missing_locations(tree))}


def responsive_rival(spec):
    return {**spec, "expression": f"({spec['expression']})+100*(1-p)"}


def surface(spec):
    return evaluation_rows(
        dict(spec=spec, config=dict(seed=0, price_limit=spec["price_limit"]))
    )


def _groups(rows):
    result = defaultdict(list)
    for row in rows:
        if row["decision_eligible"]:
            result[row["state_id"]].append(row)
    return result


def _best(rows):
    return max(rows, key=lambda r: (r["expected_revenue"], -r["price"]))


def decision_check(spec):
    groups = _groups(surface(spec))
    states = []
    for identity, rows in groups.items():
        feasible = [r for r in rows if r["price"] <= r["price_limit"]]
        oracle = _best(feasible)
        free = _best(rows)
        highest = max(feasible, key=lambda r: r["price"])
        states.append(
            dict(
                state_id=identity,
                context=oracle["context"],
                capacity=oracle["capacity"],
                optimal_feasible_price=oracle["price"],
                unconstrained_optimal_price=free["price"],
                highest_feasible_price_regret=oracle["expected_revenue"]
                - highest["expected_revenue"],
                binding_limit_revenue_gap=free["expected_revenue"]
                - oracle["expected_revenue"],
            )
        )
    feasible_prices = sorted(
        {
            r["price"]
            for rows in groups.values()
            for r in rows
            if r["price"] <= r["price_limit"]
        }
    )

    def fixed_regret(price):
        return math.fsum(
            _best([r for r in rows if r["price"] <= r["price_limit"]])[
                "expected_revenue"
            ]
            - next(r["expected_revenue"] for r in rows if r["price"] == price)
            for rows in groups.values()
        ) / len(groups)

    flat = spec["stratum"] == "flat"
    result = dict(
        states=states,
        decision_states=len(states),
        distinct_optimal_feasible_prices=len(
            {s["optimal_feasible_price"] for s in states}
        ),
        highest_feasible_price_mean_regret=math.fsum(
            s["highest_feasible_price_regret"] for s in states
        )
        / len(states),
        best_constant_feasible_price_mean_regret=min(
            fixed_regret(p) for p in feasible_prices
        ),
        informative_states=sum(s["highest_feasible_price_regret"] >= 2 for s in states),
        binding_limit_states=sum(s["binding_limit_revenue_gap"] >= 1 for s in states),
        invariant_control_exemption=flat,
    )
    if not flat and not (
        result["distinct_optimal_feasible_prices"] >= 2
        and result["highest_feasible_price_mean_regret"] >= 2
        and result["best_constant_feasible_price_mean_regret"] >= 1
        and result["informative_states"] >= 2
    ):
        raise ValueError(
            "Decision gate: need varied useful feasible prices, highest-price mean regret>=2, best constant regret>=1, and two informative states; got "
            + str({k: v for k, v in result.items() if k != "states"})
        )
    if result["binding_limit_states"] < 2:
        raise ValueError(
            "Constraint gate: need >=2 states with a binding price limit and revenue gap>=1"
        )
    return result


def pairwise_decision_risk(invariant, responsive):
    left, right = _groups(surface(invariant)), _groups(surface(responsive))
    before = error_left = error_right = 0.0
    different = 0
    for key in left:
        a = {
            r["price"]: r["expected_revenue"]
            for r in left[key]
            if r["price"] <= r["price_limit"]
        }
        b = {
            r["price"]: r["expected_revenue"]
            for r in right[key]
            if r["price"] <= r["price_limit"]
        }
        pa = max(a, key=lambda p: (a[p], -p))
        pb = max(b, key=lambda p: (b[p], -p))
        before += 0.5 * (a[pa] + b[pb]) - max(0.5 * (a[p] + b[p]) for p in a)
        error_left += a[pa] - a[pb]
        error_right += b[pb] - b[pa]
        different += abs(pa - pb) > 1e-9
    n = len(left)
    result = dict(
        no_data_state_aware_bayes_regret=before / n,
        invariant_regret_if_choose_responsive=error_left / n,
        responsive_regret_if_choose_invariant=error_right / n,
        differing_optimal_policy_states=different,
        prior=dict(invariant=0.5, responsive=0.5),
        scope="Private constructive two-world witness, not an exhaustive model class or an experimentalist inference engine. No-data baseline may condition on the public decision state.",
    )
    if before / n < 1 or different < 2:
        raise ValueError(
            "Information-value gate: private same-apparatus rivals must imply distinct feasible policies and no-data pair regret>=1"
        )
    return result


def _averages(values, n):
    means = []
    weights = []
    for a in range(n + 1):
        for b in range(n - a + 1):
            c = n - a - b
            means.append((a * values[0] + b * values[1] + c * values[2]) / n)
            weights.append(
                math.factorial(n)
                / (math.factorial(a) * math.factorial(b) * math.factorial(c))
                * 3 ** (-n)
            )
    return means, weights


def _contrast_cdf(spec, p, q, x, n):
    import numpy as np
    from scipy.special import ndtr

    low = [min(180, latent(spec, p, x, z)) for z in Z_GRID]
    high = [min(180, latent(spec, q, x, z)) for z in Z_GRID]
    ma, wa = _averages(low, n)
    mb, wb = _averages(high, n)
    centers = np.subtract.outer(ma, mb).ravel()
    weights = np.multiply.outer(wa, wb).ravel()
    sigma = spec["sensor_sigma"] * math.sqrt(2 / n)
    # A shared stable additive sensor bias cancels in the contrast.
    return lambda threshold: float(weights @ ndtr((threshold - centers) / sigma))


def finite_budget_witness(invariant, responsive, risk):
    from scipy.optimize import brentq

    n = 5
    candidates = []
    for x in (-1, 0, 1):
        for p, q in itertools.combinations((0.5, 0.75, 1, 1.25, 1.5), 2):
            null = _contrast_cdf(invariant, p, q, x, n)
            alternative = _contrast_cdf(responsive, p, q, x, n)
            threshold = float(brentq(lambda t: null(t) - 0.95, -1000, 1000, xtol=1e-10))
            type_i = 1 - null(threshold)
            power = 1 - alternative(threshold)
            after = 0.5 * (
                type_i * risk["invariant_regret_if_choose_responsive"]
                + (1 - power) * risk["responsive_regret_if_choose_invariant"]
            )
            candidates.append(
                dict(
                    context=x,
                    low_price=p,
                    high_price=q,
                    capacity=180,
                    n_per_cell=n,
                    threshold=threshold,
                    type_i=type_i,
                    power=power,
                    expected_pair_regret_after=after,
                )
            )
    best = min(candidates, key=lambda c: c["expected_pair_regret_after"])
    cost = 2 * n * 2 + 2
    queries = 2 * n + 2
    fraction = (
        best["expected_pair_regret_after"] / risk["no_data_state_aware_bayes_regret"]
    )
    if (
        best["power"] < 0.8 - 1e-10
        or fraction > 0.6 + 1e-10
        or cost > BUDGET
        or queries > MAX_QUERIES
    ):
        raise ValueError(
            "Budget gate: no private pair-discriminating experiment with power>=.8 and >=40% regret reduction in 24cost/12queries"
        )
    actions = [
        dict(kind="calibrate", reference=0, precision="standard"),
        dict(kind="calibrate", reference=120, precision="standard"),
    ]
    actions += [
        dict(
            kind="intervene",
            price=p,
            context=best["context"],
            capacity=180,
            precision="standard",
        )
        for p in (best["low_price"], best["high_price"])
        for _ in range(n)
    ]
    return dict(
        **best,
        actions=actions,
        cost=cost,
        queries=queries,
        regret_fraction_remaining=fraction,
        interpretation="Existence of an informative experiment for this private pair, conditional on exact finite latent-mixture and stable additive Gaussian measurement assumptions. Not proof that an agent finds it, all rivals are resolved, or the full law is learnable. Calibration readings are charged and do not enter the price-contrast statistic.",
    )


def observational_equivalence(spec, twin):
    # Equality is pointwise in latent unit, so it also preserves joint sales/audit
    # measurements, random replications and any adaptive sequence of allowed actions.
    errors = [
        abs(latent(spec, 1 + z / 4, x, z) - latent(twin, 1 + z / 4, x, z))
        for x, z in itertools.product((-1, 0, 1), Z_GRID)
    ]
    assert (
        spec["sensor_bias"] == twin["sensor_bias"]
        and spec["sensor_sigma"] == twin["sensor_sigma"]
        and spec["precise_sigma"] == twin["precise_sigma"]
    )
    maximum = max(errors)
    if maximum > 1e-9:
        raise ValueError("Historical/audit response-kernel equivalence failed")
    difference = max(
        abs(latent(spec, p, x, z) - latent(twin, p, x, z))
        for p, x, z in itertools.product(P_GRID, X_GRID, Z_GRID)
    )
    if difference < 8:
        raise ValueError(
            "Equivalent worlds must have materially different causal responses"
        )
    return dict(
        proved_on_available_contexts=True,
        pointwise_historical_demand_error=maximum,
        counterfactual_max_gap=difference,
        available_actions=["observe", "replicate", "measure", "calibrate"],
        argument="Historical price is 1+z/4. The adjustment makes demand g(1,x,z)+strength*z in both worlds. Sensor/calibration laws coincide. Same-unit latent demand and price coincide, preserving paired sale/audit kernels. Fresh replications draw the same latent distribution. Equality of every available response kernel preserves all adaptive transcripts for any budget.",
        causal_response_identified=False,
        finite_budget_witness=None,
    )


def off_grid_alias(spec):
    # A bounded perturbation that is exactly zero at every possible intervention
    # and historical query price, also at p=1, preserves all available transcripts.
    query = (0.5, 0.75, 1, 1.25, 1.5)

    def h(p):
        return math.prod(p - q for q in query) ** 2

    maximum = max(h(p) for p in P_GRID)
    gaps = []
    for p, x, z in itertools.product(
        (0.4, 0.625, 0.875, 1.125, 1.375, 1.6), X_GRID, Z_GRID
    ):
        # Define the alternative on realized nonnegative demand, not a new private
        # generator admission. It is compatible with the public apparatus.
        gaps.append(5 * h(p) / maximum)
    assert max(h(q) for q in query) == 0 and max(gaps) > 0
    return dict(
        query_prices=list(query),
        perturbation="Add 5*H(price)/max_domain_H to nonnegative latent demand, H(price)=product((price-q)^2 over query prices q). Leave price assignment, instruments and capacity capping unchanged.",
        maximum_evaluation_demand_gap=max(gaps),
        all_query_response_kernels_unchanged=True,
        scope="Constructive nonidentification over mechanisms compatible with the public apparatus, not membership in the privately filtered designer grammar. Exact held-out surfaces and extrapolation need additional smoothness/shape assumptions. The private grammar is never given to the scientist.",
    )


def admission_report(spec, seed):
    from .world import draw, normalize_action, public_contract, DemandConfig
    from .designed_v2 import draw as physical_draw

    spec = validate_spec(spec)
    decision = decision_check(spec)
    invariant = spec if spec["stratum"] == "flat" else invariant_rival(spec)
    responsive = responsive_rival(spec) if spec["stratum"] == "flat" else spec
    # Rivals are possible under public apparatus assumptions, not required to
    # pass a private stratum-specific admission gate (an invariant rival is valid).
    if any(
        not 0 <= latent(r, p, x, z) <= 300
        for r in (invariant, responsive)
        for p, x, z in itertools.product(P_GRID, X_GRID, Z_GRID)
    ):
        raise ValueError("Private witness rival outside physical design bounds")
    risk = pairwise_decision_risk(invariant, responsive)
    unavailable = spec["stratum"] == "observational_only"
    equivalence = observational_equivalence(spec, invariant) if unavailable else None
    witness = (
        None if unavailable else finite_budget_witness(invariant, responsive, risk)
    )
    cfg = DemandConfig(
        scenario="observational_only" if unavailable else "clean",
        seed=seed,
        budget=BUDGET,
        price_limit=spec["price_limit"],
        world_spec=spec,
    )
    world = dict(spec=spec, config=dict(seed=seed, price_limit=spec["price_limit"]))
    public = public_contract(cfg)
    if witness:
        history = []
        spent = 0
        for index, action in enumerate(witness["actions"], 1):
            design, cost = normalize_action(action, public, history)
            obs, private = draw(world, design, index, history)
            again = physical_draw(world, design, index, history)
            if (obs, private) != again:
                raise ValueError("Action execution replay mismatch")
            history.append(dict(design=design, observation=obs, private=private))
            spent += cost
        if spent != witness["cost"]:
            raise ValueError("Witness action cost mismatch")
    moments = []
    for p, x, cap in ((0.5, -1, 60), (1, 0, 180), (1.5, 1, 180)):
        values = [min(cap, latent(spec, p, x, z)) for z in Z_GRID]
        mean = math.fsum(values) / 3 + spec["sensor_bias"]
        variance = (
            math.fsum((v - (mean - spec["sensor_bias"])) ** 2 for v in values) / 3
            + spec["sensor_sigma"] ** 2
        )
        sample = [
            physical_draw(
                world,
                dict(
                    kind="intervene",
                    price=p,
                    context=x,
                    capacity=cap,
                    precision="standard",
                ),
                10000 + j,
                [],
            )[0]["value"]
            for j in range(1600)
        ]
        error = abs(math.fsum(sample) / len(sample) - mean)
        if error > 7 * math.sqrt(variance / len(sample)) + 1e-10:
            raise ValueError("Sample moment check failed")
        observed_second = math.fsum((y - mean) ** 2 for y in sample) / len(sample)
        unit_mean = mean - spec["sensor_bias"]
        unit_variance = variance - spec["sensor_sigma"] ** 2
        fourth = (
            math.fsum((v - unit_mean) ** 4 for v in values) / 3
            + 6 * unit_variance * spec["sensor_sigma"] ** 2
            + 3 * spec["sensor_sigma"] ** 4
        )
        second_tolerance = (
            7 * math.sqrt(max(0, fourth - variance**2) / len(sample)) + 1e-9
        )
        if abs(observed_second - variance) > second_tolerance:
            raise ValueError("Actual simulator second-moment check failed")
        moments.append(
            dict(
                price=p,
                context=x,
                capacity=cap,
                mean_error=error,
                second_moment_error=abs(observed_second - variance),
                passed=True,
            )
        )
    calibration = []
    for precision in ("standard", "precise"):
        sigma = (
            spec["sensor_sigma"] if precision == "standard" else spec["precise_sigma"]
        )
        values = [
            physical_draw(
                world,
                dict(kind="calibrate", reference=60, precision=precision),
                20000 + j,
                [],
            )[0]["value"]
            for j in range(1600)
        ]
        mean = 60 + spec["sensor_bias"]
        mean_error = abs(math.fsum(values) / len(values) - mean)
        second_error = abs(
            math.fsum((v - mean) ** 2 for v in values) / len(values) - sigma**2
        )
        if mean_error > 7 * sigma / math.sqrt(
            len(values)
        ) or second_error > 7 * sigma**2 * math.sqrt(2 / len(values)):
            raise ValueError("Actual calibration mean/variance check failed")
        calibration.append(
            dict(
                precision=precision,
                mean_error=mean_error,
                second_moment_error=second_error,
                passed=True,
            )
        )
    # Directly verify that an audit retains the original latent unit; no second
    # independent demand is drawn. Sampling checks above do not replace this.
    origin, hidden = physical_draw(
        world, dict(kind="observe", context=0, precision="standard"), 1, []
    )
    audit, audited = physical_draw(
        world,
        dict(kind="measure", observation_id=origin["id"]),
        2,
        [dict(observation=origin, private=hidden)],
    )
    if audited != hidden or audit["paired_with"] != origin["id"]:
        raise ValueError("Same-unit audit identity check failed")
    return dict(
        version="private-v2-admission-v1",
        validated=True,
        spec_hash=content_hash(spec),
        declared_budget=dict(cost=BUDGET, queries=MAX_QUERIES),
        decision=decision,
        pair_risk=risk,
        private_rivals=dict(invariant=invariant, responsive=responsive),
        useful_experiment=witness,
        observational_equivalence=equivalence,
        claim_ledger=[
            dict(
                claim="Mean response at a queried intervention setting",
                population_identification="available_with_repeated_controls_and_measurement_calibration"
                if not unavailable
                else "not_identified",
                assumptions="Stable fresh-unit intervention assignment; additive stable measurement bias estimable from known references; paired demand audits for demand hidden by capacity.",
                finite_budget="Local sale-response pair witness only; no full-surface or unique-law guarantee.",
            ),
            dict(
                claim="Instrument mean bias and dispersion",
                population_identification="available_from_repeated_known_reference_readings",
                assumptions="Stable additive measurement law within precision; no arbitrary reference-dependent bias.",
                finite_budget="Only two reference readings in the private price-test schedule; no guarantee of precise variance estimation.",
            ),
            dict(
                claim="Causal demand mechanism from historical observations alone",
                population_identification="not_identified"
                if unavailable
                else "historical_data_alone_not_certified",
                evidence="Exact observational-equivalence proof"
                if unavailable
                else "Interventions are available; that does not certify every causal claim.",
            ),
            dict(
                claim="Exact response at unqueriable feature prices or complete mechanism",
                population_identification="not_nonparametrically_identified",
                evidence=off_grid_alias(spec),
            ),
        ],
        moment_checks=moments,
        calibration_checks=calibration,
        paired_audit_identity_verified=True,
        scope="Private finite-domain, decision, measurement, exact response-kernel and two-rival budget witnesses. Not industrial validation, a proof of arbitrary-law recovery, or model-agent performance. Never send this report, its laws, rivals, oracle prices or schedules to an experimentalist.",
    )
