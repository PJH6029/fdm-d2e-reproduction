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
from fdm_d2e.training.train_fdm import train_fdm_real


def _floats(csv: str) -> list[float]:
    return [float(part) for part in csv.split(",") if part.strip()]


def _ints(csv: str) -> list[int]:
    return [int(float(part)) for part in csv.split(",") if part.strip()]


def _strings(csv: str) -> list[str]:
    return [part.strip() for part in csv.split(",") if part.strip()]


def _model_rows(summary: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    comparison = summary.get("statistical_comparison") or {}
    return [row for row in comparison.get("comparisons", []) if row.get("model") == model_name]


def _comparison_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["endpoint"]: item for item in row.get("comparisons", [])}


def _score(row: dict[str, Any]) -> tuple[Any, ...]:
    comparisons = _comparison_map(row)
    reject_count = sum(1 for item in comparisons.values() if item.get("reject_holm_0_05"))
    metrics = row["metrics"]
    button = metrics.get("mouse_button", {})
    mouse = metrics.get("mouse_move", {})
    keyboard = metrics.get("keyboard", {})
    return (
        -reject_count,
        comparisons.get("mouse_button_accuracy", {}).get("p_adjusted_holm") or 9.0,
        comparisons.get("mouse_move_pearson", {}).get("p_adjusted_holm") or 9.0,
        comparisons.get("keyboard_accuracy", {}).get("p_adjusted_holm") or 9.0,
        -(button.get("f1") or 0.0),
        -(button.get("accuracy") or 0.0),
        button.get("no_button_false_positive_rate") or 1.0,
        -(mouse.get("pearson") or -9.0),
        -(keyboard.get("accuracy") or 0.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible real-D2E FDM sweeps from IDM pseudo-labels.")
    parser.add_argument("--config", default="configs/model/fdm_shooter64_surface_motion_fulltrain.yaml")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--work-dir", default="outputs/fdm_real_sweeps")
    parser.add_argument("--hidden-dims", default="")
    parser.add_argument("--depths", default="")
    parser.add_argument("--epochs", default="")
    parser.add_argument("--category-loss-weights", default="")
    parser.add_argument("--category-thresholds", default="")
    parser.add_argument("--category-calibration-modes", default="")
    parser.add_argument("--category-calibration-betas", default="")
    parser.add_argument("--category-calibration-fractions", default="")
    parser.add_argument("--button-thresholds", default="")
    parser.add_argument("--button-threshold-modes", default="")
    parser.add_argument("--button-calibration-betas", default="")
    parser.add_argument("--button-loss-weights", default="")
    parser.add_argument("--button-class-weight-caps", default="")
    parser.add_argument("--button-no-button-weights", default="")
    parser.add_argument("--button-positive-weights", default="")
    parser.add_argument("--action-history-lens", default="")
    parser.add_argument("--mouse-axis-loss-weights", default="")
    parser.add_argument("--mouse-regression-loss-weights", default="")
    parser.add_argument("--mouse-axis-decode-modes", default="")
    parser.add_argument("--mouse-axis-temperatures", default="")
    parser.add_argument("--mouse-output-gain-modes", default="")
    parser.add_argument("--mouse-output-gains", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    base = load_config(args.config)
    torch_base = dict(base.get("torch_idm_config", {}))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) / Path(args.config).stem

    hidden_dims = _ints(args.hidden_dims) or [int(torch_base.get("hidden_dim", 128))]
    depths = _ints(args.depths) or [int(torch_base.get("depth", 3))]
    epochs = _ints(args.epochs) or [int(torch_base.get("epochs", 80))]
    category_loss_weights = _floats(args.category_loss_weights) or [float(torch_base.get("categorical_loss_weight", 1.0))]
    category_thresholds = _floats(args.category_thresholds) or [float(torch_base.get("category_threshold", 0.35))]
    category_calibration_modes = _strings(args.category_calibration_modes) or [str(torch_base.get("category_threshold_mode", "global"))]
    category_calibration_betas = _floats(args.category_calibration_betas) or [float(torch_base.get("category_calibration_beta", 1.0))]
    category_calibration_fractions = _floats(args.category_calibration_fractions) or [float(torch_base.get("category_calibration_fraction", 0.0))]
    button_thresholds = _floats(args.button_thresholds) or [float(torch_base.get("button_softmax_threshold", 0.5))]
    button_threshold_modes = _strings(args.button_threshold_modes) or [str(torch_base.get("button_softmax_threshold_mode", "global"))]
    button_calibration_betas = _floats(args.button_calibration_betas) or [float(torch_base.get("button_softmax_calibration_beta", torch_base.get("category_calibration_beta", 0.5)))]
    button_loss_weights = _floats(args.button_loss_weights) or [float(torch_base.get("button_softmax_loss_weight", 1.0))]
    button_class_weight_caps = _floats(args.button_class_weight_caps) or [float(torch_base.get("button_softmax_class_weight_cap", 20.0))]
    button_no_button_weights = _floats(args.button_no_button_weights) or [float(torch_base.get("button_softmax_no_button_weight", 1.0))]
    button_positive_weights = _floats(args.button_positive_weights) or [float(torch_base.get("button_softmax_positive_weight", 1.0))]
    action_history_lens = _ints(args.action_history_lens) or [int(torch_base.get("action_history_len", 0))]
    mouse_axis_loss_weights = _floats(args.mouse_axis_loss_weights) or [float(torch_base.get("mouse_axis_loss_weight", 1.0))]
    mouse_regression_loss_weights = _floats(args.mouse_regression_loss_weights) or [float(torch_base.get("mouse_regression_loss_weight", 1.0))]
    mouse_axis_decode_modes = _strings(args.mouse_axis_decode_modes) or [str(torch_base.get("mouse_axis_decode_mode", "argmax"))]
    mouse_axis_temperatures = _floats(args.mouse_axis_temperatures) or [float(torch_base.get("mouse_axis_temperature", 1.0))]
    mouse_output_gain_modes = _strings(args.mouse_output_gain_modes) or [str(torch_base.get("mouse_output_gain_mode", "fixed"))]
    mouse_output_gains = _floats(args.mouse_output_gains) or [float(torch_base.get("mouse_output_gain", 1.0))]
    seeds = _ints(args.seeds) or [int(torch_base.get("seed", 0))]

    grid = itertools.product(
        hidden_dims,
        depths,
        epochs,
        category_loss_weights,
        category_thresholds,
        category_calibration_modes,
        category_calibration_betas,
        category_calibration_fractions,
        button_thresholds,
        button_threshold_modes,
        button_calibration_betas,
        button_loss_weights,
        button_class_weight_caps,
        button_no_button_weights,
        button_positive_weights,
        action_history_lens,
        mouse_axis_loss_weights,
        mouse_regression_loss_weights,
        mouse_axis_decode_modes,
        mouse_axis_temperatures,
        mouse_output_gain_modes,
        mouse_output_gains,
        seeds,
    )
    rows: list[dict[str, Any]] = []
    for idx, values in enumerate(grid, start=1):
        if args.max_runs is not None and idx > args.max_runs:
            break
        (
            hidden_dim,
            depth,
            epoch_count,
            category_loss_weight,
            category_threshold,
            category_calibration_mode,
            category_calibration_beta,
            category_calibration_fraction,
            button_threshold,
            button_threshold_mode,
            button_calibration_beta,
            button_loss_weight,
            button_class_weight_cap,
            button_no_button_weight,
            button_positive_weight,
            action_history_len,
            mouse_axis_loss_weight,
            mouse_regression_loss_weight,
            mouse_axis_decode_mode,
            mouse_axis_temperature,
            mouse_output_gain_mode,
            mouse_output_gain,
            seed,
        ) = values
        variant = (
            f"h{hidden_dim}_d{depth}_e{epoch_count}_clw{category_loss_weight:g}_cth{category_threshold:g}"
            f"_ccm{category_calibration_mode}_ccb{category_calibration_beta:g}_ccf{category_calibration_fraction:g}"
            f"_bth{button_threshold:g}_btm{button_threshold_mode}_bcb{button_calibration_beta:g}"
            f"_blw{button_loss_weight:g}_bcw{button_class_weight_cap:g}_bnw{button_no_button_weight:g}"
            f"_bpw{button_positive_weight:g}_hist{action_history_len}_malw{mouse_axis_loss_weight:g}"
            f"_mrlw{mouse_regression_loss_weight:g}_mad{mouse_axis_decode_mode}"
            f"_mat{mouse_axis_temperature:g}_mogm{mouse_output_gain_mode}"
            f"_mog{mouse_output_gain:g}_seed{seed}"
        )
        cfg = copy.deepcopy(base)
        model_name = f"{base.get('model_name', 'torch_fdm_real')}_{variant}"
        torch_cfg = dict(cfg.get("torch_idm_config", {}))
        torch_cfg.update(
            {
                "hidden_dim": hidden_dim,
                "depth": depth,
                "epochs": epoch_count,
                "categorical_loss_weight": category_loss_weight,
                "category_threshold": category_threshold,
                "category_threshold_mode": category_calibration_mode,
                "category_calibration_beta": category_calibration_beta,
                "category_calibration_fraction": category_calibration_fraction,
                "button_softmax_threshold": button_threshold,
                "button_softmax_threshold_mode": button_threshold_mode,
                "button_softmax_calibration_beta": button_calibration_beta,
                "button_softmax_loss_weight": button_loss_weight,
                "button_softmax_class_weight_cap": button_class_weight_cap,
                "button_softmax_no_button_weight": button_no_button_weight,
                "button_softmax_positive_weight": button_positive_weight,
                "action_history_len": action_history_len,
                "mouse_axis_loss_weight": mouse_axis_loss_weight,
                "mouse_regression_loss_weight": mouse_regression_loss_weight,
                "mouse_axis_decode_mode": mouse_axis_decode_mode,
                "mouse_axis_temperature": mouse_axis_temperature,
                "mouse_output_gain_mode": mouse_output_gain_mode,
                "mouse_output_gain": mouse_output_gain,
                "seed": seed,
            }
        )
        cfg.update(
            {
                "model_name": model_name,
                "output_dir": str(work_dir / variant),
                "torch_idm_config": torch_cfg,
            }
        )
        summary = train_fdm_real(cfg)
        row = {
            "variant": variant,
            "model_name": model_name,
            "output_dir": cfg["output_dir"],
            "config": {
                "hidden_dim": hidden_dim,
                "depth": depth,
                "epochs": epoch_count,
                "categorical_loss_weight": category_loss_weight,
                "category_threshold": category_threshold,
                "category_threshold_mode": category_calibration_mode,
                "category_calibration_beta": category_calibration_beta,
                "category_calibration_fraction": category_calibration_fraction,
                "button_softmax_threshold": button_threshold,
                "button_softmax_threshold_mode": button_threshold_mode,
                "button_softmax_calibration_beta": button_calibration_beta,
                "button_softmax_loss_weight": button_loss_weight,
                "button_softmax_class_weight_cap": button_class_weight_cap,
                "button_softmax_no_button_weight": button_no_button_weight,
                "button_softmax_positive_weight": button_positive_weight,
                "action_history_len": action_history_len,
                "mouse_axis_loss_weight": mouse_axis_loss_weight,
                "mouse_regression_loss_weight": mouse_regression_loss_weight,
                "mouse_axis_decode_mode": mouse_axis_decode_mode,
                "mouse_axis_temperature": mouse_axis_temperature,
                "mouse_output_gain_mode": mouse_output_gain_mode,
                "mouse_output_gain": mouse_output_gain,
                "seed": seed,
            },
            "metrics": summary["metrics"],
            "metadata": summary["checkpoint"]["torch_checkpoint_metadata"],
            "checkpoint": summary["checkpoint"],
            "comparisons": _model_rows(summary, model_name),
        }
        rows.append(row)
        ranked = sorted(rows, key=_score)
        payload = {
            "schema": "fdm_real_sweep.v1",
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
