#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json
from fdm_d2e.training.knn_fdm import train_knn_fdm


def _floats(csv: str) -> list[float]:
    return [float(part) for part in csv.split(",") if part.strip()]


def _ints(csv: str) -> list[int]:
    return [int(float(part)) for part in csv.split(",") if part.strip()]


def _strings(csv: str) -> list[str]:
    return [part.strip() for part in csv.split(",") if part.strip()]


def _model_rows(summary: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    return [row for row in summary.get("statistical_comparison", {}).get("comparisons", []) if row.get("model") == model_name]


def _score(row: dict[str, Any]) -> tuple[Any, ...]:
    comps = {item["endpoint"]: item for item in row.get("comparisons", [])}
    rejects = sum(1 for item in comps.values() if item.get("reject_holm_0_05"))
    button = row["metrics"].get("mouse_button", {})
    mouse = row["metrics"].get("mouse_move", {})
    keyboard = row["metrics"].get("keyboard", {})
    return (
        -rejects,
        comps.get("mouse_move_scale_ratio_distance", {}).get("p_adjusted_holm") or 9.0,
        comps.get("mouse_button_accuracy", {}).get("p_adjusted_holm") or 9.0,
        comps.get("mouse_move_pearson", {}).get("p_adjusted_holm") or 9.0,
        comps.get("keyboard_accuracy", {}).get("p_adjusted_holm") or 9.0,
        -(button.get("f1") or 0.0),
        button.get("no_button_false_positive_rate") or 1.0,
        -(mouse.get("pearson") or -9.0),
        -(keyboard.get("accuracy") or 0.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep nonparametric KNN FDM variants.")
    parser.add_argument("--config", default="configs/model/fdm_knn_shooter64_surface_motion.yaml")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--work-dir", default="outputs/fdm_knn_sweeps")
    parser.add_argument("--feature-modes", default="")
    parser.add_argument("--ks", default="")
    parser.add_argument("--distance-temperatures", default="")
    parser.add_argument("--keyboard-thresholds", default="")
    parser.add_argument("--button-thresholds", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    base = load_config(args.config)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) / Path(args.config).stem
    feature_modes = _strings(args.feature_modes) or [str(base.get("feature_mode", "summary_grid8_shift_surface_time"))]
    ks = _ints(args.ks) or [int(base.get("k", 3))]
    temperatures = _floats(args.distance_temperatures) or [float(base.get("distance_temperature", 0.1))]
    keyboard_thresholds = _floats(args.keyboard_thresholds) or [float(base.get("keyboard_vote_threshold", 0.5))]
    button_thresholds = _floats(args.button_thresholds) or [float(base.get("button_vote_threshold", 0.5))]
    rows: list[dict[str, Any]] = []
    for idx, (feature_mode, k, temperature, keyboard_threshold, button_threshold) in enumerate(
        itertools.product(feature_modes, ks, temperatures, keyboard_thresholds, button_thresholds), start=1
    ):
        if args.max_runs is not None and idx > args.max_runs:
            break
        variant = f"fm{feature_mode}_k{k}_dt{temperature:g}_kt{keyboard_threshold:g}_bt{button_threshold:g}"
        cfg = copy.deepcopy(base)
        model_name = f"{base.get('model_name', 'knn_fdm')}_{variant}"
        cfg.update({
            "model_name": model_name,
            "feature_mode": feature_mode,
            "k": k,
            "distance_temperature": temperature,
            "keyboard_vote_threshold": keyboard_threshold,
            "button_vote_threshold": button_threshold,
            "output_dir": str(work_dir / variant),
        })
        if args.batch_size is not None:
            cfg["batch_size"] = args.batch_size
        summary = train_knn_fdm(cfg)
        row = {
            "variant": variant,
            "model_name": model_name,
            "output_dir": cfg["output_dir"],
            "config": {key: cfg[key] for key in ["feature_mode", "k", "distance_temperature", "keyboard_vote_threshold", "button_vote_threshold", "batch_size"] if key in cfg},
            "metrics": summary["metrics"],
            "checkpoint": summary["checkpoint"],
            "comparisons": _model_rows(summary, model_name),
        }
        rows.append(row)
        ranked = sorted(rows, key=_score)
        payload = {"schema": "fdm_knn_sweep.v1", "base_config": args.config, "num_runs": len(rows), "rows": rows, "top_variants": ranked[: min(10, len(ranked))]}
        write_json(output_path, payload)
        print(json.dumps({"run": idx, "variant": variant, "metrics": row["metrics"], "rejects": [item["endpoint"] for item in row["comparisons"] if item.get("reject_holm_0_05")]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
