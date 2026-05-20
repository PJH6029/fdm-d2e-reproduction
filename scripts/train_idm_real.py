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
from fdm_d2e.training.neural_idm import train_idm_variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny neural IDM variants on decoded real D2E records.")
    parser.add_argument("--config", default="configs/model/idm_real_sample.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train_records = read_jsonl(config["train_records"])
    target_records = read_jsonl(config["target_records"])
    output_dir = Path(config.get("output_dir", "outputs/idm_real"))
    threshold = float(config.get("confidence_threshold", 0.25))
    results = {}
    predictions_by_name = build_baseline_predictions(train_records, target_records)
    write_baseline_predictions(predictions_by_name, output_dir / "baselines")
    for variant in config.get("variants", []):
        name = variant["model_name"]
        result = train_idm_variant(
            train_records,
            target_records,
            model_name=name,
            hidden_dim=int(variant.get("hidden_dim", 8)),
            epochs=int(variant.get("epochs", 500)),
            lr=float(variant.get("lr", 0.02)),
            seed=int(variant.get("seed", 0)),
            confidence_threshold=threshold,
            output_dir=output_dir,
        )
        metrics = compute_metrics(result["predictions"], target_records)
        metrics_path = output_dir / name / "metrics.json"
        write_json(metrics_path, metrics)
        predictions_by_name[name] = result["predictions"]
        results[name] = {"metadata": result["metadata"], "metrics": metrics}
    endpoints = load_config(config.get("endpoints", "configs/eval/primary_endpoints.yaml"))
    comparison = compare_systems(predictions_by_name, target_records, endpoints)
    summary = {
        "schema": "idm_real_train_summary.v1",
        "train_records": config["train_records"],
        "target_records": config["target_records"],
        "endpoints": config.get("endpoints", "configs/eval/primary_endpoints.yaml"),
        "variants": results,
        "baseline_metrics": {name: compute_metrics(rows, target_records) for name, rows in predictions_by_name.items() if name not in results},
        "statistical_comparison": comparison,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(config.get("summary_out", "artifacts/idm/idm_real_sample_summary.json"), summary)
    best = max(results, key=lambda name: results[name]["metrics"].get("mouse_move", {}).get("pearson") or -999)
    print(
        "trained real IDM variants: "
        f"variants={','.join(results)} target={len(target_records)} best_mouse_pearson={best}:"
        f"{results[best]['metrics'].get('mouse_move', {}).get('pearson')}"
    )


if __name__ == "__main__":
    main()
