#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.torch_idm import torch_available
from fdm_d2e.training.video_idm import predict_video_idm_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Run target prediction from a trained raw frame-pair video IDM checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-target-examples", type=int)
    parser.add_argument("--recalibrate-from-train-cache", action="store_true")
    parser.add_argument("--keyboard-softmax-threshold", type=float)
    parser.add_argument("--keyboard-softmax-calibration-max-no-key-fpr", type=float)
    parser.add_argument("--button-softmax-calibration-max-no-button-fpr", type=float)
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    if not torch_available():
        msg = "torch unavailable; run `uv sync --extra train` or execute inside the MLXP training image"
        if args.require_torch:
            raise SystemExit(msg)
        print(msg)
        return 0
    config = load_config(args.config)
    if args.checkpoint_path:
        config["checkpoint_path"] = args.checkpoint_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_target_examples is not None:
        config["max_target_examples"] = args.max_target_examples
    if args.recalibrate_from_train_cache:
        config["recalibrate_from_train_cache"] = True
    if args.keyboard_softmax_threshold is not None:
        config["keyboard_softmax_threshold"] = args.keyboard_softmax_threshold
    if args.keyboard_softmax_calibration_max_no_key_fpr is not None:
        config["keyboard_softmax_calibration_max_no_key_fpr"] = args.keyboard_softmax_calibration_max_no_key_fpr
    if args.button_softmax_calibration_max_no_button_fpr is not None:
        config["button_softmax_calibration_max_no_button_fpr"] = args.button_softmax_calibration_max_no_button_fpr
    summary = predict_video_idm_checkpoint(config)
    print(
        "predicted video IDM: "
        f"checkpoint={summary['checkpoint_path']} target={summary['target_records']} "
        f"predictions={summary['prediction']['predictions_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
