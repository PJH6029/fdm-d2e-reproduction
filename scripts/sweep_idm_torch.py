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


def _strings(csv: str) -> list[str]:
    return [part.strip() for part in csv.split(",") if part.strip()]


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
    parser.add_argument("--category-losses", default="")
    parser.add_argument("--focal-gammas", default="")
    parser.add_argument("--calibration-modes", default="")
    parser.add_argument("--calibration-betas", default="")
    parser.add_argument("--calibration-fractions", default="")
    parser.add_argument("--button-head-modes", default="")
    parser.add_argument("--button-thresholds", default="")
    parser.add_argument("--button-threshold-modes", default="")
    parser.add_argument("--button-loss-weights", default="")
    parser.add_argument("--button-class-weight-caps", default="")
    parser.add_argument("--button-no-button-weights", default="")
    parser.add_argument("--button-positive-weights", default="")
    parser.add_argument("--mouse-head-modes", default="")
    parser.add_argument("--mouse-axis-loss-weights", default="")
    parser.add_argument("--mouse-regression-loss-weights", default="")
    parser.add_argument("--mouse-axis-class-weight-caps", default="")
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
    category_losses = _strings(args.category_losses) or [str(base.get("categorical_loss", "bce"))]
    focal_gammas = _floats(args.focal_gammas) or [float(base.get("focal_gamma", 2.0))]
    calibration_modes = _strings(args.calibration_modes) or [str(base.get("category_threshold_mode", "global"))]
    calibration_betas = _floats(args.calibration_betas) or [float(base.get("category_calibration_beta", 1.0))]
    calibration_fractions = _floats(args.calibration_fractions) or [float(base.get("category_calibration_fraction", 0.0))]
    button_head_modes = _strings(args.button_head_modes) or [str(base.get("button_head_mode", "multilabel"))]
    button_thresholds = _floats(args.button_thresholds) or [float(base.get("button_softmax_threshold", 0.5))]
    button_threshold_modes = _strings(args.button_threshold_modes) or [str(base.get("button_softmax_threshold_mode", "global"))]
    button_loss_weights = _floats(args.button_loss_weights) or [float(base.get("button_softmax_loss_weight", 1.0))]
    button_class_weight_caps = _floats(args.button_class_weight_caps) or [float(base.get("button_softmax_class_weight_cap", 20.0))]
    button_no_button_weights = _floats(args.button_no_button_weights) or [float(base.get("button_softmax_no_button_weight", 1.0))]
    button_positive_weights = _floats(args.button_positive_weights) or [float(base.get("button_softmax_positive_weight", 1.0))]
    mouse_head_modes = _strings(args.mouse_head_modes) or [str(base.get("mouse_head_mode", "regression"))]
    mouse_axis_loss_weights = _floats(args.mouse_axis_loss_weights) or [float(base.get("mouse_axis_loss_weight", 1.0))]
    mouse_regression_loss_weights = _floats(args.mouse_regression_loss_weights) or [float(base.get("mouse_regression_loss_weight", 1.0))]
    mouse_axis_class_weight_caps = _floats(args.mouse_axis_class_weight_caps) or [float(base.get("mouse_axis_class_weight_cap", 20.0))]
    grid = itertools.product(
        _floats(args.loss_weights),
        _floats(args.poscaps),
        _floats(args.thresholds),
        hidden_dims,
        depths,
        category_losses,
        focal_gammas,
        calibration_modes,
        calibration_betas,
        calibration_fractions,
        button_head_modes,
        button_thresholds,
        button_threshold_modes,
        button_loss_weights,
        button_class_weight_caps,
        button_no_button_weights,
        button_positive_weights,
        mouse_head_modes,
        mouse_axis_loss_weights,
        mouse_regression_loss_weights,
        mouse_axis_class_weight_caps,
    )
    for idx, (
        loss_weight,
        poscap,
        threshold,
        hidden_dim,
        depth,
        loss_mode,
        focal_gamma,
        calibration_mode,
        calibration_beta,
        calibration_fraction,
        button_head_mode,
        button_threshold,
        button_threshold_mode,
        button_loss_weight,
        button_class_weight_cap,
        button_no_button_weight,
        button_positive_weight,
        mouse_head_mode,
        mouse_axis_loss_weight,
        mouse_regression_loss_weight,
        mouse_axis_class_weight_cap,
    ) in enumerate(grid, start=1):
        if args.max_runs is not None and idx > args.max_runs:
            break
        cfg = copy.deepcopy(base)
        variant = (
            f"lw{loss_weight:g}_pc{poscap:g}_th{threshold:g}_h{hidden_dim}_d{depth}"
            f"_loss{loss_mode}_fg{focal_gamma:g}_cal{calibration_mode}_cb{calibration_beta:g}_cf{calibration_fraction:g}"
            f"_bh{button_head_mode}_bth{button_threshold:g}_btm{button_threshold_mode}"
            f"_blw{button_loss_weight:g}_bcw{button_class_weight_cap:g}_bnw{button_no_button_weight:g}_bpw{button_positive_weight:g}"
            f"_mh{mouse_head_mode}_malw{mouse_axis_loss_weight:g}_mrlw{mouse_regression_loss_weight:g}_macw{mouse_axis_class_weight_cap:g}"
        )
        model_name = f"{base.get('model_name', 'torch_mlp_idm')}_{variant}"
        cfg.update(
            {
                "model_name": model_name,
                "categorical_loss_weight": loss_weight,
                "categorical_pos_weight_cap": poscap,
                "category_threshold": threshold,
                "hidden_dim": hidden_dim,
                "depth": depth,
                "categorical_loss": loss_mode,
                "focal_gamma": focal_gamma,
                "category_threshold_mode": calibration_mode,
                "category_calibration_beta": calibration_beta,
                "category_calibration_fraction": calibration_fraction,
                "button_head_mode": button_head_mode,
                "button_softmax_threshold": button_threshold,
                "button_softmax_threshold_mode": button_threshold_mode,
                "button_softmax_loss_weight": button_loss_weight,
                "button_softmax_class_weight_cap": button_class_weight_cap,
                "button_softmax_no_button_weight": button_no_button_weight,
                "button_softmax_positive_weight": button_positive_weight,
                "mouse_head_mode": mouse_head_mode,
                "mouse_axis_loss_weight": mouse_axis_loss_weight,
                "mouse_regression_loss_weight": mouse_regression_loss_weight,
                "mouse_axis_class_weight_cap": mouse_axis_class_weight_cap,
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
                "categorical_loss": loss_mode,
                "focal_gamma": focal_gamma,
                "category_threshold_mode": calibration_mode,
                "category_calibration_beta": calibration_beta,
                "category_calibration_fraction": calibration_fraction,
                "button_head_mode": button_head_mode,
                "button_softmax_threshold": button_threshold,
                "button_softmax_threshold_mode": button_threshold_mode,
                "button_softmax_loss_weight": button_loss_weight,
                "button_softmax_class_weight_cap": button_class_weight_cap,
                "button_softmax_no_button_weight": button_no_button_weight,
                "button_softmax_positive_weight": button_positive_weight,
                "mouse_head_mode": mouse_head_mode,
                "mouse_axis_loss_weight": mouse_axis_loss_weight,
                "mouse_regression_loss_weight": mouse_regression_loss_weight,
                "mouse_axis_class_weight_cap": mouse_axis_class_weight_cap,
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
