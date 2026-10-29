"""Adversarial and randomized checks against independent invariants."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
import random
import sys
import time

import pytest

from sciquest.core import create_quest
from sciquest.execution import run_command
from sciquest.io import read_yaml, write_yaml
from sciquest.state import acquire_lock, read_state, recover
from sciquest.validation import compute_aggregate_score, validate_suite


def _process_attempt(path):
    try:
        return acquire_lock(Path(path))
    except RuntimeError:
        return None


def test_process_contention_and_dead_owner_recovery(tmp_path):
    q = create_quest(tmp_path, {"hero_statement": "Contention"}, slug="contention")
    with ProcessPoolExecutor(max_workers=8) as pool:
        winners = [
            result for result in pool.map(_process_attempt, [str(q)] * 32) if result
        ]
    assert len(winners) == 1
    assert read_state(q)["lock_id"] == winners[0]
    assert recover(q, winners[0])["quest_status"] == "idle"


def test_atomic_projection_readers_never_observe_truncated_yaml(tmp_path):
    path = tmp_path / "atomic.yaml"
    write_yaml(path, {"sequence": 0, "data": "x" * 5000})

    def writer(index):
        for n in range(20):
            write_yaml(path, {"sequence": index * 20 + n, "data": "x" * 5000})

    def reader(_):
        for n in range(300):
            value = read_yaml(path)
            assert isinstance(value["sequence"], int)
            assert len(value["data"]) == 5000

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(writer, i) for i in range(3)] + [
            pool.submit(reader, i) for i in range(5)
        ]
        for future in futures:
            future.result()


def test_randomized_scores_agree_with_rational_oracle():
    rng = random.Random(20261021)
    for _ in range(1000):
        metrics, results, numerator, denominator = [], {}, Fraction(0), 0
        for index in range(rng.randint(1, 12)):
            lo, width, raw, weight = (
                rng.randint(-100, 100),
                rng.randint(1, 100),
                rng.randint(-200, 200),
                rng.randint(1, 100),
            )
            direction = rng.choice(["maximize", "minimize"])
            normalized = Fraction(raw - lo, width)
            if direction == "minimize":
                normalized = 1 - normalized
            normalized = max(Fraction(0), min(Fraction(1), normalized))
            numerator += weight * normalized
            denominator += weight
            name = f"m{index}"
            metrics.append(
                {
                    "name": name,
                    "direction": direction,
                    "weight": weight,
                    "normalization": {"type": "minmax", "min": lo, "max": lo + width},
                }
            )
            results[name] = raw
        actual = compute_aggregate_score({"metrics": metrics}, results)[
            "aggregate_score"
        ]
        assert 0 <= actual <= 1
        assert actual == pytest.approx(float(numerator / denominator), abs=1e-14)


def test_overflow_and_duplicate_yaml_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        validate_suite(
            {
                "metrics": [
                    {"name": "a", "weight": 1e308},
                    {"name": "b", "weight": 1e308},
                ]
            }
        )
    with pytest.raises(ValueError, match="overflowed"):
        validate_suite(
            {
                "metrics": [
                    {
                        "name": "a",
                        "normalization": {
                            "type": "minmax",
                            "min": -1e308,
                            "max": 1e308,
                        },
                    }
                ]
            }
        )
    path = tmp_path / "duplicate.yaml"
    path.write_text("metrics: []\nmetrics: [{name: m}]\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_yaml(path)


def test_worker_timeout_is_bounded_and_notebook_environment_excludes_secrets(
    tmp_path, monkeypatch
):
    q, workspace = tmp_path / "quest", tmp_path / "workspace"
    q.mkdir()
    workspace.mkdir()
    monkeypatch.setenv("SCIQUEST_TEST_SECRET", "should-not-enter-notebook")
    result = run_command(
        [
            sys.executable,
            "-c",
            "import os; assert 'SCIQUEST_TEST_SECRET' not in os.environ",
        ],
        workspace,
        tmp_path / "env.log",
        hidden_root=q,
        timeout=10,
    )
    assert result["status"] == "succeeded"
    started = time.monotonic()
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        workspace,
        tmp_path / "timeout.log",
        hidden_root=q,
        timeout=0.5,
    )
    assert result["status"] == "timed_out"
    assert time.monotonic() - started < 5


def test_worker_cannot_modify_host_files_or_read_hidden_authority(tmp_path):
    q, workspace = tmp_path / "quest", tmp_path / "workspace"
    q.mkdir()
    workspace.mkdir()
    secret = q / "secret.txt"
    secret.write_text("private evidence")
    sentinel = tmp_path / "host-sentinel"
    sentinel.write_text("preserve")
    code = f"""from pathlib import Path
assert not Path({str(secret)!r}).exists()
try:
    Path({str(sentinel)!r}).write_text('modified')
except OSError:
    pass
else:
    # A private /tmp copy is writable, but the host sentinel must remain intact.
    assert Path({str(sentinel)!r}).read_text() == 'modified'
"""
    result = run_command(
        [sys.executable, "-c", code],
        workspace,
        tmp_path / "isolation.log",
        hidden_root=q,
        timeout=10,
    )
    assert result["status"] == "succeeded"
    assert sentinel.read_text() == "preserve"


def test_injected_runner_cannot_count_an_old_or_absent_experiment(tmp_path):
    from sciquest.loop import run_iteration_loop

    q = create_quest(
        tmp_path, {"hero_statement": "No invented completion"}, slug="noop"
    )
    result = run_iteration_loop(q, iteration_runner=lambda *args, **kwargs: "exp_001")
    assert result["failed"]
    assert result["completed_iterations"] == 0
