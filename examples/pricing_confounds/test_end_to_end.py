"""Real installed-wheel integration, including isolated notebook execution.

Run with the wheel's interpreter, NOT an editable/source-tree installation.
There is no backend mock, model API, network request, isolation skip or fallback.
"""
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import nbformat
import pytest

import sciquest
from sciquest.io import read_yaml
from sciquest.logic import run_logic_check
from sciquest.state import get_record

HERE = Path(__file__).resolve().parent


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_installed_wheel_reproduces_scientific_workflow(tmp_path):
    package = Path(sciquest.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), f"Requires an installed wheel, got {package}"
    expected = json.loads((HERE / "expected.json").read_text())
    produced = []
    for index in range(2):
        output = tmp_path / f"run-{index}"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        subprocess.run([sys.executable, str(HERE / "run.py"), "--output", str(output)],
                       cwd=tmp_path, env=env, check=True, capture_output=True, text=True, timeout=240)
        results = json.loads((output / "results.json").read_text())
        assert results["external_agent_calls"] == 0
        assert results["genuine_external_agent_reasoning_demonstrated"] is False
        barrier = json.loads((output / "before_execution.json").read_text())
        assert barrier["executions_started"] == 0
        assert set(barrier["registrations"]) == {"observational", "randomized"}
        first_execution_time = min(get_record(output / "quests" / arm, "execution", "exp_001")["started_at"]
                                   for arm in expected)
        manifest = json.loads((output / "manifest.json").read_text())
        for rel, digest in manifest["sha256"].items():
            assert hashlib.sha256((output / rel).read_bytes()).hexdigest() == digest, rel
        for entry in results["experiments"]:
            arm = entry["arm"]
            quest = output / "quests" / arm
            exp = quest / "experiments/exp_001"
            registration = get_record(quest, "registration", "exp_001")
            assert registration["registration_hash"] == barrier["registrations"][arm]
            assert registration["created_at"] <= first_execution_time
            assert registration["plan"]["scientist_information_mode"] == "black_box_v2"
            assert len(registration["generation_replays"]) == 2
            assert all(r["execution"]["isolated"] for r in registration["generation_replays"].values())
            evaluation = get_record(quest, "evaluation", "exp_001")
            assert evaluation == read_yaml(exp / "validation_results.yaml")
            assert evaluation["evidence_source"] == "supervisor_computed"
            execution = get_record(quest, "execution", "exp_001")
            assert execution["isolated"] is True
            assert execution["blind_filesystem"] is True
            assert execution["network"] is False
            assert get_record(quest, "completion", "exp_001")
            check = run_logic_check(quest, "exp_001")
            assert check["recorded_integrity_verified"] and not check["scientific_verified"]
            predictions = {r["id"]: r for r in rows(exp / "worker_artifacts/predictions.csv")}
            training = rows(quest / "data/raw/train.csv")
            assert len(training) == len({r["id"] for r in training}) == 8
            assert {r["id"] for r in training}.isdisjoint(predictions)
            cells = Counter((float(r["price"]), (float(r["demand"]) - 100 + 2 * float(r["price"])) / 40)
                            for r in training)
            assert cells == ({(30, 0): 4, (40, 1): 4} if arm == "observational" else
                             {(30, 0): 2, (30, 1): 2, (40, 0): 2, (40, 1): 2})
            truth = rows(quest / "data/raw/eval.csv")
            assert set(predictions) == {"evaluation_30", "evaluation_40"}
            # Independent oracle arithmetic; do not call the demo generator or
            # SciQuest's evaluator to compute these expected values.
            errors = []
            values = []
            for row in truth:
                price = float(row["price"])
                assert float(row["mean_demand"]) == 120 - 2 * price
                assert float(row["expected_revenue"]) == price * (120 - 2 * price)
                pred = predictions[row["id"]]
                mean = float(pred["predicted_mean_demand"])
                revenue = float(pred["predicted_revenue"])
                assert revenue == pytest.approx(price * mean, rel=0, abs=1e-10)
                errors.append((mean - (120 - 2 * price)) ** 2)
                values.append((revenue, -price, price * (120 - 2 * price)))
                assert float(pred["fitted_intercept"]) == pytest.approx(expected[arm]["fitted_intercept"], rel=0, abs=1e-10)
                assert float(pred["fitted_slope"]) == pytest.approx(expected[arm]["fitted_slope"], rel=0, abs=1e-10)
            independent = dict(mean_demand_rmse=math.sqrt(sum(errors) / len(errors)),
                               policy_value=max(values)[2], regret=1800 - max(values)[2])
            assert entry["metrics"] == pytest.approx(independent, rel=0, abs=1e-10)
            for name, value in independent.items():
                assert value == pytest.approx(expected[arm][name], rel=0, abs=1e-10)
            assert entry["selected_price"] == expected[arm]["selected_price"]
            outcomes = {"mean_recovery": "contradicted", "optimal_listed_price": "contradicted"} if arm == "observational" else {"mean_recovery": "supported", "optimal_listed_price": "supported"}
            assert evaluation["prediction_outcomes"] == outcomes
            notebook = nbformat.read(exp / "executed_notebook.ipynb", as_version=4)
            assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
            assert "owner files and supervisor package excluded" in json.dumps(notebook)
            assert "Supervisor evaluation recorded" in (quest / "journal.md").read_text()
            assert "<!-- sciquest-evaluation:exp_001 -->" in (quest / "journal.md").read_text()
            assert (exp / "artifacts/plots/comparison.svg").stat().st_size > 1000
            dashboard = (quest / "reports/dashboard/index.html").read_text()
            assert "Result Graphs" in dashboard and "<svg" in dashboard
            assert "mean demand rmse" in dashboard and "Supervisor computed" in dashboard
            assert "fonts.googleapis.com" not in dashboard
            assert "Generic glossary" in dashboard
            assert "Ridge regression is the actual prediction model" not in dashboard
            assert "The model predicts the logarithm of demand" not in dashboard
            assert "Template suggestion; not an agent decision" in dashboard
            assert (quest / "reports/dashboard/assets/mathjax/tex-mml-chtml.js").is_file()
            assert (output / "comparison.png").read_bytes().startswith(b"\x89PNG")
            assert "Metrics below are read from actual SciQuest" in (output / "report.md").read_text()
        produced.append(output)
    # Reproducibility contract: exact scientific JSON/data/predictions. Whole
    # directories contain timestamps, absolute paths and process/runtime evidence.
    assert (produced[0] / "results.json").read_bytes() == (produced[1] / "results.json").read_bytes()
    for arm in expected:
        for rel in ["data/raw/train.csv", "data/raw/eval.csv", "experiments/exp_001/worker_artifacts/predictions.csv"]:
            assert (produced[0] / "quests" / arm / rel).read_bytes() == (produced[1] / "quests" / arm / rel).read_bytes()
    attempt = subprocess.run([sys.executable, str(HERE / "run.py"), "--output", str(produced[0])],
                             cwd=tmp_path, capture_output=True, text=True)
    assert attempt.returncode != 0  # previous evidence cannot be overwritten


def test_analysis_responds_to_changed_training_instead_of_expected_coefficients(tmp_path):
    (tmp_path / "data/0").mkdir(parents=True)
    (tmp_path / "data/1").mkdir(parents=True)
    (tmp_path / "data/0/train.csv").write_text("id,price,demand\na,30,135\nb,40,130\n")
    (tmp_path / "data/1/features.csv").write_text("id,population,price\ntest,balanced_population,35\n")
    subprocess.run([sys.executable, str(HERE / "analysis.py")], cwd=tmp_path, check=True, capture_output=True)
    prediction = rows(tmp_path / "predictions.csv")[0]
    assert float(prediction["fitted_slope"]) == -0.5
    assert float(prediction["fitted_intercept"]) == 150
    assert float(prediction["predicted_mean_demand"]) == 132.5
    assert float(prediction["predicted_revenue"]) == 35 * 132.5
