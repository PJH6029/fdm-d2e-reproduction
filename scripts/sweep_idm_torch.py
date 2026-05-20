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
from fdm_d2e.training.torch_idm import train_torch_idm


def _floats(csv: str) -> list[float]:
    return [float(part) for part in csv.split(",") if part.strip()]


def _model_rows(summary: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    comparison = summary.get("statistical_comparison") or {}
    return [row for row in comparison.get("comparisons", []) if row.get("model") == model_name]


def _score(row: dict[str, Any]) -> tuple[Any, ...]:
    comparisons = {item["endpoint"]: item for item in row["comparisons"]}
    reject_count = sum(1 for item in comparisons.values() if item.get("reject_holm_0_05"))
    mouse_p = comparisons.get("mouse_move_pearson", {}).get("p_adjusted_holm")
    scale_p = comparisons.get("mouse_move_scale_ratio_distance", {}).get("p_adjusted_holm")
    keyboard_p = comparisons.get("keyboard_accuracy", {}).get("p_adjusted_holm")
    metrics = row["metrics"]
    return (
        -reject_count,
        mouse_p if mouse_p is not None else 9.0,
        scale_p if scale_p is not None else 9.0,
        keyboard_p if keyboard_p is not None else 9.0,
        -(metrics.get("mouse_move", {}).get("pearson") or -9.0),
        -(metrics.get("keyboard", {}).get("accuracy") or 0.0),
        -(metrics.get("mouse_button", {}).get("accuracy") or 0.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible Torch IDM hyperparameter sweeps.")
    parser.add_argument("--config", default="configs/model/idm_torch_apex8.yaml")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--work-dir", default="outputs/idm_torch_sweeps")
    parser.add_argument("--loss-weights", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--poscaps", default="1,5,20")
    parser.add_argument("--thresholds", default="0.25,0.45")
    parser.add_argument("--hidden-dims", default="")
    parser.add_argument("--depths", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    base = load_config(args.config)
    if args.epochs is not None:
        base["epochs"] = args.epochs
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) / Path(args.config).stem
    rows: list[dict[str, Any]] = []
    hidden_dims = [int(value) for value in _floats(args.hidden_dims)] or [int(base.get("hidden_dim", 128))]
    depths = [int(value) for value in _floats(args.depths)] or [int(base.get("depth", 3))]
    grid = itertools.product(_floats(args.loss_weights), _floats(args.poscaps), _floats(args.thresholds), hidden_dims, depths)
    for idx, (loss_weight, poscap, threshold, hidden_dim, depth) in enumerate(grid, start=1):
        if args.max_runs is not None and idx > args.max_runs:
            break
        cfg = copy.deepcopy(base)
        variant = f"lw{loss_weight:g}_pc{poscap:g}_th{threshold:g}_h{hidden_dim}_d{depth}"
        model_name = f"{base.get('model_name', 'torch_mlp_idm')}_{variant}"
        cfg.update(
            {
                "model_name": model_name,
                "categorical_loss_weight": loss_weight,
                "categorical_pos_weight_cap": poscap,
                "category_threshold": threshold,
                "hidden_dim": hidden_dim,
                "depth": depth,
                "output_dir": str(work_dir / variant),
                "summary_out": str(work_dir / variant / "summary.json"),
            }
        )
        summary = train_torch_idm(cfg)
        row = {
            "variant": variant,
            "model_name": model_name,
            "config": {
                "categorical_loss_weight": loss_weight,
                "categorical_pos_weight_cap": poscap,
                "category_threshold": threshold,
                "hidden_dim": hidden_dim,
                "depth": depth,
                "epochs": cfg.get("epochs"),
                "feature_mode": cfg.get("feature_mode", "summary"),
                "train_records": cfg.get("train_records"),
                "target_records": cfg.get("target_records"),
            },
            "metrics": summary["metrics"],
            "metadata": summary["metadata"],
            "comparisons": _model_rows(summary, model_name),
        }
        rows.append(row)
        ranked = sorted(rows, key=_score)
        payload = {
            "schema": "idm_torch_sweep.v1",
            "base_config": args.config,
            "num_runs": len(rows),
            "rows": rows,
            "top_variants": ranked[: min(10, len(ranked))],
        }
        write_json(output_path, payload)
        print(
            json.dumps(
                {
                    "run": idx,
                    "variant": variant,
                    "metrics": row["metrics"],
                    "rejects": [item["endpoint"] for item in row["comparisons"] if item.get("reject_holm_0_05")],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
