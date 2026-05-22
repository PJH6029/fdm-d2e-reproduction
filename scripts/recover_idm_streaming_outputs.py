#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.streaming_idm import recover_streaming_idm_outputs_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover streaming IDM metrics/metadata/summary from an existing checkpoint "
            "without retraining. Useful when full-corpus target prediction was interrupted."
        )
    )
    parser.add_argument("--config", default="configs/model/idm_streaming_d2e_full_compact_accel64.yaml")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--no-resume-predictions", action="store_true")
    parser.add_argument(
        "--prediction-workers",
        type=int,
        default=None,
        help="Shard-parallel checkpoint prediction workers for recovery; use 4 on 4xH200 full-corpus G003.",
    )
    parser.add_argument(
        "--prediction-cuda-devices",
        default=None,
        help=(
            "Comma-separated CUDA devices to round-robin across prediction workers. "
            "Defaults to CUDA_VISIBLE_DEVICES when set."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override target prediction batch size during checkpoint recovery.",
    )
    parser.add_argument(
        "--no-pseudolabel-validation",
        action="store_true",
        help=(
            "Skip per-row pseudolabel JSON-schema validation during recovery. "
            "Use only when the fixed generator path is covered by tests/audits and full-corpus throughput matters."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("config_path", args.config)
    output_dir = args.output_dir or config.get("output_dir")
    if output_dir:
        config["output_dir"] = output_dir
    config["checkpoint_path"] = args.checkpoint_path or str(Path(config.get("output_dir", output_dir or ".")) / "checkpoint.pt")
    if args.summary_out:
        config["summary_out"] = args.summary_out
    if args.force_cpu:
        config["force_cpu"] = True
    config["resume_predictions"] = not args.no_resume_predictions
    if args.prediction_workers is not None:
        config["prediction_workers"] = int(args.prediction_workers)
    if args.prediction_cuda_devices:
        config["prediction_cuda_devices"] = [device.strip() for device in args.prediction_cuda_devices.split(",") if device.strip()]
    if args.eval_batch_size is not None:
        config["eval_batch_size"] = int(args.eval_batch_size)
    if args.no_pseudolabel_validation:
        config["validate_pseudolabels"] = False

    summary = recover_streaming_idm_outputs_from_checkpoint(config)
    print(
        "streaming idm checkpoint recovery complete: "
        f"status={summary['status']} target_records={summary['target_records']} "
        f"metadata={summary['metadata_path']}"
    )


if __name__ == "__main__":
    main()
