import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from sciquest.core import create_quest, reserve_experiment
from sciquest.io import contained_path, experiment_dir, read_yaml, write_yaml
from sciquest.methods.ledger import classify_result_claim
from sciquest.state import (
    acquire_lock,
    get_record,
    read_state,
    record,
    recover,
    release_lock,
)
from sciquest.validation import (
    compute_aggregate_score,
    validate_experiment,
    validate_suite,
)


@pytest.fixture
def quest(tmp_path):
    return create_quest(
        tmp_path,
        {
            "hero_statement": "Evaluate competing explanations",
            "initial_hypothesis": "Prices affect demand",
        },
        slug="demo",
    )


def test_omission_cannot_raise_score():
    suite = {"metrics": [{"name": "good"}, {"name": "bad"}]}
    assert (
        compute_aggregate_score(suite, {"good": 1, "bad": 0})["aggregate_score"] == 0.5
    )
    with pytest.raises(ValueError, match="Missing required"):
        compute_aggregate_score(suite, {"good": 1})


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf"), True, False, "0.5", None, [], 10**1000],
)
def test_non_numeric_or_nonfinite_metrics_are_rejected(value):
    with pytest.raises(ValueError):
        compute_aggregate_score({"metrics": [{"name": "m"}]}, {"m": value})


@pytest.mark.parametrize(
    "metric",
    [
        {"weight": -1},
        {"weight": 0},
        {"weight": float("nan")},
        {"direction": "maximise"},
        {"normalization": "bogus"},
        {"normalization": {"type": "minmax", "min": 1, "max": 1}},
        {"normalization": {"type": "minmax", "min": 2, "max": 1}},
        {"direction": "target", "normalization": "identity"},
        {
            "direction": "target",
            "normalization": {"type": "target", "target": 0, "tolerance": 0},
        },
    ],
)
def test_bad_metric_configuration_fails_closed(metric):
    with pytest.raises(ValueError):
        validate_suite({"metrics": [{"name": "m", **metric}]})


def test_suite_and_results_are_not_mutated_and_physical_bounds_are_enforced():
    suite = {"metrics": [{"name": "m", "bounds": {"min": 0, "max": 1}}]}
    before = json.dumps(suite)
    compute_aggregate_score(suite, {"m": 0.5})
    assert json.dumps(suite) == before
    with pytest.raises(ValueError, match="physical bounds"):
        compute_aggregate_score(suite, {"m": 1.01})
    with pytest.raises(ValueError, match="unique"):
        validate_suite({"metrics": [{"name": "m"}, {"name": "m"}]})


def test_concurrent_locks_have_exactly_one_owner(quest):
    def attempt(_):
        try:
            return acquire_lock(quest)
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        winners = [result for result in pool.map(attempt, range(24)) if result]
    assert len(winners) == 1
    assert read_state(quest)["lock_id"] == winners[0]
    with pytest.raises(RuntimeError, match="Only the lock owner"):
        release_lock(quest, "wrong-owner")
    assert read_state(quest)["quest_status"] == "running"
    release_lock(quest, winners[0])


def test_yaml_edits_cannot_override_authoritative_lock(quest):
    lock = acquire_lock(quest)
    write_yaml(quest / "state.yaml", {"quest_status": "idle", "lock_id": None})
    with pytest.raises(RuntimeError, match="locked"):
        acquire_lock(quest)
    assert read_state(quest)["lock_id"] == lock
    release_lock(quest, lock)


def test_live_owner_cannot_be_recovered(quest):
    lock = acquire_lock(quest)
    with pytest.raises(RuntimeError, match="still alive"):
        recover(quest, lock)
    release_lock(quest, lock, "failed")
    with pytest.raises(RuntimeError, match="recover"):
        acquire_lock(quest)
    assert recover(quest)["quest_status"] == "idle"


def test_reservations_are_never_reused_even_if_directory_removed(quest):
    lock = acquire_lock(quest)
    first, path = reserve_experiment(quest, lock)
    path.rmdir()
    second, _ = reserve_experiment(quest, lock)
    assert (first, second) == ("exp_001", "exp_002")
    assert get_record(quest, "reservation", first)["run_id"] == lock
    release_lock(quest, lock)


def test_records_are_insert_only(quest):
    record(quest, "evidence", "e1", {"result": 1})
    with pytest.raises(FileExistsError):
        record(quest, "evidence", "e1", {"result": 2})
    assert get_record(quest, "evidence", "e1") == {"result": 1}


def test_scoring_missing_results_does_not_create_evidence(quest):
    exp = quest / "experiments/exp_001"
    exp.mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        validate_experiment(quest, "exp_001")
    assert not (exp / "validation_results.yaml").exists()


@pytest.mark.parametrize(
    "name", ["../outside", "/tmp/outside", "exp_001/../../outside", "exp_bad", ""]
)
def test_experiment_traversal_is_rejected(quest, name):
    with pytest.raises(ValueError):
        experiment_dir(quest, name)


def test_symlink_escape_and_exclusive_writes(quest, tmp_path):
    (quest / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        contained_path(quest, "escape/file")
    target = quest / "record.yaml"
    write_yaml(target, {"before": 1}, exclusive=True)
    with pytest.raises(FileExistsError):
        write_yaml(target, {"after": 2}, exclusive=True)
    assert read_yaml(target) == {"before": 1}


def test_self_reported_prediction_matching_is_not_confirmation():
    claim = {"claim_type": "confirmatory", "matched_prediction": True}
    assert classify_result_claim({}, claim) == "unverified_claim"
    registration = {
        "registration_status": "registered",
        "registration_hash": "abc",
        "plan": {"confirmatory_status": "confirmatory", "predictions": [{"id": "p1"}]},
    }
    claim["prediction_id"] = "p1"
    assert classify_result_claim(registration, claim) == "unverified_claim"
    # A contrary outcome is still a valid confirmatory test result.
    evaluation = {
        "registration_hash": "abc",
        "prediction_outcomes": {"p1": "contradicted"},
        "evidence_source": "supervisor_computed",
        "isolated_execution": True,
        "confirmatory_status": "confirmatory",
    }
    assert (
        classify_result_claim(registration, claim, evaluation) == "confirmatory_result"
    )


def test_legacy_state_is_visible_but_not_promoted_to_verified_completion(quest):
    old = {
        "quest_status": "idle",
        "lock_id": None,
        "last_experiment": "exp_009",
        "last_score": 0.9,
        "best_score": 0.9,
    }
    write_yaml(quest / "state.yaml", old)
    state = read_state(quest)
    assert state["last_experiment"] is None
    assert state["legacy_last_experiment"] == "exp_009"
    assert state["legacy_best_score"] == 0.9
    assert read_yaml(quest / "state.yaml") == old
