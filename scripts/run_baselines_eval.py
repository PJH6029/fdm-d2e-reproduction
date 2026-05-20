#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.eval.action_metrics import compute_metrics
from fdm_d2e.eval.baselines import build_baseline_predictions, write_baseline_predictions
from fdm_d2e.eval.statistics import compare_systems
from fdm_d2e.io_utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline policies and cluster-bootstrap statistical comparisons.")
    parser.add_argument("--train-records", default="outputs/data/real_sample/Apex_Legends/0805_01/train.jsonl")
    parser.add_argument("--ground-truth", default="outputs/data/real_sample/Apex_Legends/0805_01/heldout.jsonl")
    parser.add_argument("--endpoints", default="configs/eval/primary_endpoints.yaml")
    parser.add_argument("--output-dir", default="outputs/eval/baselines")
    parser.add_argument("--summary-out", default="artifacts/eval/baseline_stat_eval_sample.json")
    parser.add_argument("--baselines", nargs="+", default=["noop", "global_majority", "game_majority", "last_seen_train"])
    args = parser.parse_args()

    train = read_jsonl(args.train_records)
    ground_truth = read_jsonl(args.ground_truth)
    predictions = build_baseline_predictions(train, ground_truth, baseline_names=args.baselines)
    prediction_paths = write_baseline_predictions(predictions, args.output_dir)
    metrics_by_name = {name: compute_metrics(rows, ground_truth) for name, rows in predictions.items()}
    comparison = compare_systems(predictions, ground_truth, load_config(args.endpoints))
    summary = {
        "schema": "baseline_stat_eval_sample.v1",
        "train_records": args.train_records,
        "ground_truth": args.ground_truth,
        "endpoints_config": args.endpoints,
        "prediction_paths": prediction_paths,
        "metrics": metrics_by_name,
        "statistical_comparison": comparison,
    }
    write_json(Path(args.output_dir) / "metrics_by_baseline.json", metrics_by_name)
    write_json(Path(args.output_dir) / "statistical_comparison.json", comparison)
    write_json(args.summary_out, summary)
    print(
        "baseline/stat eval: "
        f"baselines={','.join(predictions)} examples={len(ground_truth)} "
        f"comparisons={len(comparison['comparisons'])} summary={args.summary_out}"
    )


if __name__ == "__main__":
    main()
