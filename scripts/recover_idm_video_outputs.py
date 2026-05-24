#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import ensure_dir, read_json, write_json
from fdm_d2e.schema import validate_named
from fdm_d2e.training.torch_idm import require_torch, torch_available
from fdm_d2e.training.video_idm import predict_video_idm_checkpoint


def _read_dict(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def recover_video_idm_outputs(config: dict) -> dict:
    torch = require_torch()
    checkpoint_value = config.get("checkpoint_path") or config.get("checkpoint")
    if not checkpoint_value:
        raise ValueError("recover_video_idm_outputs requires checkpoint_path")
    checkpoint_path = Path(checkpoint_value)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = dict(checkpoint.get("model_config", {}))
    recovery_config = dict(checkpoint_config)
    recovery_config.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_dir(recovery_config.get("output_dir", checkpoint_path.parent))
    prediction_summary_out = recovery_config.get("prediction_summary_out")
    if not prediction_summary_out:
        recovery_config["prediction_summary_out"] = str(Path(output_dir) / "prediction_summary.json")

    prediction_summary = predict_video_idm_checkpoint(recovery_config)
    prediction = dict(prediction_summary["prediction"])
    stats = dict(checkpoint["stats"])
    metadata_path = Path(output_dir) / "checkpoint_metadata.json"
    train_summary_path = Path(recovery_config.get("summary_out", Path(output_dir) / "summary.json"))
    train_summary = _read_dict(train_summary_path)
    old_metadata = _read_dict(metadata_path)
    distributed = old_metadata.get("distributed") or train_summary.get("distributed") or {}
    metadata = {
        **old_metadata,
        "schema": "idm_checkpoint_metadata.v1",
        "model": str(recovery_config.get("model_name", checkpoint_config.get("model_name", "video_pair_idm"))),
        "dataset_fingerprint": str(stats["dataset_fingerprint"]),
        "train_records": int(stats["num_examples"]),
        "target_records": int(prediction["target_records"]),
        "pseudo_label_path": prediction["pseudo_label_path"],
        "filtered_pseudo_label_path": prediction["pseudo_label_path"],
        "checkpoint_path": str(checkpoint_path),
        "train_state_path": str(recovery_config.get("train_state_path", Path(output_dir) / "train_state.pt")),
        "metrics_path": prediction["metrics_path"],
        "calibration": dict(checkpoint.get("calibration", {})),
        "distributed": distributed,
    }
    validate_named(metadata, "idm_checkpoint_metadata.schema.json")
    write_json(metadata_path, metadata)

    if not train_summary:
        train_summary = {
            "schema": "video_idm_train_summary.v1",
            "model_name": metadata["model"],
            "checkpoint_path": str(checkpoint_path),
            "train_state_path": metadata["train_state_path"],
            "stats_path": str(recovery_config.get("stats_path", Path(output_dir) / "video_idm_stats.json")),
            "train_history_path": str(Path(output_dir) / "train_history.json"),
            "convergence_report_path": str(Path(output_dir) / "convergence_report.json"),
            "distributed": distributed,
        }
    train_summary["metadata"] = metadata
    train_summary["prediction"] = prediction
    train_summary["metrics"] = prediction.get("metrics")
    train_summary["label_quality_report"] = prediction.get("label_quality_report")
    train_summary["statistical_comparison"] = prediction.get("statistical_comparison")
    train_summary["recovered_from_checkpoint"] = True
    train_summary["recovery_prediction_summary_path"] = recovery_config["prediction_summary_out"]
    write_json(train_summary_path, train_summary)

    return {
        "schema": "video_idm_recovery_summary.v1",
        "status": "pass",
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "checkpoint_metadata_path": str(metadata_path),
        "train_summary_path": str(train_summary_path),
        "prediction_summary_path": str(recovery_config["prediction_summary_out"]),
        "target_records": int(prediction["target_records"]),
        "predictions_path": prediction["predictions_path"],
        "metrics_path": prediction["metrics_path"],
        "parallel_prediction": dict(prediction.get("prediction_parallel", {})),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover video IDM metadata/summary after checkpoint-only training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--prediction-workers", type=int)
    parser.add_argument("--prediction-parts-dir")
    parser.add_argument("--prediction-cuda-devices")
    parser.add_argument("--prediction-summary-out")
    parser.add_argument("--summary-out")
    parser.add_argument("--max-target-examples", type=int)
    parser.add_argument("--force-cpu", action="store_true")
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
    if args.prediction_workers is not None:
        config["prediction_workers"] = args.prediction_workers
    if args.prediction_parts_dir:
        config["prediction_parts_dir"] = args.prediction_parts_dir
    if args.prediction_cuda_devices:
        config["prediction_cuda_devices"] = [item.strip() for item in args.prediction_cuda_devices.split(",") if item.strip()]
    if args.prediction_summary_out:
        config["prediction_summary_out"] = args.prediction_summary_out
    if args.summary_out:
        config["summary_out"] = args.summary_out
    if args.max_target_examples is not None:
        config["max_target_examples"] = args.max_target_examples
    if args.force_cpu:
        config["force_cpu"] = True
    summary = recover_video_idm_outputs(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
