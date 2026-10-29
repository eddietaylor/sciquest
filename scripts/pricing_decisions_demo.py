"""Two inspectable pricing-policy trials with known analytic values.

Run with SciQuest's Python: scripts/pricing_decisions_demo.py --root NEW_DIRECTORY
This illustrates evaluation semantics, not autonomous learning or discovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def write_data(path: Path, split: str, seed: int) -> None:
    if seed != 0:
        raise ValueError("This analytic fixture uses deterministic seed 0")
    rows = (
        [("past", 45, 4500, 0)]
        if split == "train"
        else [
            ("a", 50, 5000, 0),
            ("a", 60, 5400, 0),
            ("b", 50, 3000, 0),
            ("b", 60, 3600, 1),
        ]
    )
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "state", "price", "expected_revenue", "violation"])
        for i, (state, price, value, violation) in enumerate(rows):
            w.writerow([f"{split}_{i}", state, price, value, violation])


def run(root: Path) -> None:
    import nbformat

    from sciquest.core import create_quest, run_next
    from sciquest.dashboard import build_dashboard
    from sciquest.io import write_yaml
    from sciquest.logic import run_logic_check
    from sciquest.methods.ledger import preregister_experiment
    from sciquest.workflow import execute_registered

    prepared = []
    for mode in ("conservative", "aggressive"):
        q = create_quest(
            root.resolve(),
            {
                "hero_statement": "Evaluate a pricing decision rule",
                "initial_hypothesis": "The policy meets value and constraint requirements.",
            },
            slug=f"pricing-decisions-{mode}",
        )
        shutil.copyfile(__file__, q / "data/generator.py")
        datasets = []
        for split, filename in [("train", "train.csv"), ("eval", "eval.csv")]:
            write_data(q / "data/raw" / filename, split, 0)
            d = {
                "path": f"data/raw/{filename}",
                "role": "training" if split == "train" else "evaluation",
                "provenance": "Synthetic analytic pricing fixture with explicit expected intervention revenues and a fictional operating constraint.",
                "generator": "data/generator.py",
                "generator_args": [
                    "--generate",
                    split,
                    "--seed",
                    "0",
                    "--output",
                    "output.csv",
                ],
                "generated_output": "output.csv",
                "seed": 0,
            }
            if split == "eval":
                d["input_columns"] = ["id", "state", "price"]
            datasets.append(d)
        write_yaml(q / "data_manifest.yaml", {"status": "ready", "datasets": datasets})
        cfg = {
            "truth": "data/1/eval.csv",
            "target": "expected_revenue",
            "prediction": "predicted_revenue",
            "group_by": ["state"],
            "action": "price",
            "objective": "maximize",
            "tie_break": "lowest_action",
        }
        metrics = [
            {
                "name": "value",
                "role": "primary",
                "direction": "maximize",
                "normalization": {"type": "minmax", "min": 0, "max": 6000},
                "computation": dict(
                    **cfg,
                    kind="policy_value",
                    value_semantics="expected_interventional",
                ),
            },
            {
                "name": "regret",
                "role": "diagnostic",
                "computation": dict(
                    **cfg, kind="regret", value_semantics="expected_interventional"
                ),
            },
            {
                "name": "violations",
                "role": "constraint",
                "requirement": {"operator": "<=", "threshold": 0},
                "computation": {
                    **cfg,
                    "kind": "constraint_violation_rate",
                    "target": "violation",
                },
            },
            {
                "name": "constant_rank",
                "role": "diagnostic",
                "computation": {
                    "kind": "rank_correlation",
                    "truth": "data/1/eval.csv",
                    "target": "expected_revenue",
                    "prediction": "constant",
                    "group_by": ["state"],
                    "order_by": "price",
                },
            },
        ]
        write_yaml(q / "validation.yaml", {"status": "ready", "metrics": metrics})
        expid = run_next(q, agent_stub=True)
        exp = q / "experiments" / expid
        write_yaml(
            exp / "experiment_plan.yaml",
            {
                "question": "How does this fixed pricing rule perform on the listed action grid?",
                "hypothesis": "Mean expected revenue is at least 4000 and no selected action violates the registered limit.",
                "scope": "Two analytic states with two prices each; illustrative software trial only.",
                "assumptions": "Owner-supplied expected revenues and binary violation labels. Regret oracle includes all listed actions, including prohibited ones.",
                "data_access": "Training fixture and evaluation features; expected evaluation revenues and violations hidden.",
                "stopping_rule": "One registered evaluation per fixed rule; no adaptive tuning.",
                "action": "hypothesis_test",
                "confirmatory_status": "exploratory",
                "predictions": [
                    {
                        "id": "value",
                        "metric": "value",
                        "operator": ">=",
                        "threshold": 4000,
                    },
                    {
                        "id": "safe",
                        "metric": "violations",
                        "operator": "<=",
                        "threshold": 0,
                    },
                ],
            },
        )
        code = f"""import csv
rows=list(csv.DictReader(open('data/1/features.csv')))
assert 'expected_revenue' not in rows[0] and 'violation' not in rows[0]
mode={mode!r}
with open('predictions.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['id','predicted_revenue','constant']);w.writeheader()
    for r in rows:
        price=float(r['price'])
        score=10000-price if mode=='conservative' else price*100
        w.writerow(dict(id=r['id'],predicted_revenue=score,constant=1))
"""
        nbformat.write(
            nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]),
            exp / "notebook.ipynb",
        )
        preregister_experiment(q, expid)
        prepared.append((mode, q, expid))
    summaries = []
    for mode, q, expid in prepared:
        result = execute_registered(q, expid, timeout=60)
        check = run_logic_check(q, expid)
        if not check["passed"]:
            raise RuntimeError(check)
        dashboard = build_dashboard(q)
        summary = {
            "mode": mode,
            "quest": str(q),
            "dashboard": str(dashboard),
            "metrics": result["metrics"],
            "undefined_metrics": result["undefined_metrics"],
            "status": result["validation_status"],
            "aggregate_score": result["aggregate_score"],
            "constraint_outcomes": result["constraint_outcomes"],
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (root / "pricing_decisions_results.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--generate", choices=["train", "eval"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.generate:
        if args.output is None:
            parser.error("--generate requires --output")
        write_data(args.output, args.generate, args.seed)
    elif args.root is None:
        parser.error("--root is required for the demo")
    else:
        run(args.root)
