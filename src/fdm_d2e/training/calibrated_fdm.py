from __future__ import annotations

from pathlib import Path
from typing import Any

from fdm_d2e.config import load_config
from fdm_d2e.eval.action_metrics import compute_metrics
from fdm_d2e.eval.baselines import build_baseline_predictions
from fdm_d2e.eval.statistics import compare_systems
from fdm_d2e.io_utils import read_jsonl, sha256_file, stable_hash_json, write_json, write_jsonl
from fdm_d2e.schema import validate_named
from fdm_d2e.tokenization.actions import token_to_delta_class
from fdm_d2e.training.neural_idm import target_mouse_delta, tokens_from_delta


def _non_mouse_motion_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if not token.startswith(("MOUSE_DX_", "MOUSE_DY_"))]


def _abs_axis_mean(deltas: list[tuple[float, float]]) -> float | None:
    values = [abs(float(value)) for pair in deltas for value in pair]
    return sum(values) / len(values) if values else None


def _recording_id(row: dict[str, Any]) -> str:
    return str(row.get("recording_id") or str(row.get("sequence_id", "")).split("#", 1)[0])


def _game_id(row: dict[str, Any]) -> str:
    return str(row.get("game", "unknown"))


def _prediction_mouse_delta(row: dict[str, Any]) -> tuple[float, float]:
    tokens = list(row.get("predicted_tokens", []))
    dxs = [float(value) for token in tokens if token.startswith("MOUSE_DX_") and (value := token_to_delta_class(token)) is not None]
    dys = [float(value) for token in tokens if token.startswith("MOUSE_DY_") and (value := token_to_delta_class(token)) is not None]
    return (sum(dxs) / len(dxs) if dxs else 0.0, sum(dys) / len(dys) if dys else 0.0)


def _mean_abs_by_key(rows: list[dict[str, Any]], key_fn, *, source: str) -> dict[str, float]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        delta = target_mouse_delta(row) if source == "ground_truth" else _prediction_mouse_delta(row)
        grouped.setdefault(key_fn(row), []).append(delta)
    means: dict[str, float] = {}
    for key, deltas in grouped.items():
        value = _abs_axis_mean(deltas)
        if value is not None and value > 0:
            means[key] = value
    return means


def _scale_predictions(
    predictions: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    *,
    min_gain: float,
    max_gain: float,
    calibration_predictions: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if min_gain <= 0 or max_gain <= 0 or min_gain > max_gain:
        raise ValueError("min_gain/max_gain must be positive and ordered")
    prediction_reference_rows = calibration_predictions if calibration_predictions is not None else predictions
    train_by_recording = _mean_abs_by_key(train_records, _recording_id, source="ground_truth")
    train_by_game = _mean_abs_by_key(train_records, _game_id, source="ground_truth")
    pred_by_recording = _mean_abs_by_key(prediction_reference_rows, _recording_id, source="prediction")
    global_train = _abs_axis_mean([target_mouse_delta(row) for row in train_records]) or 1.0
    global_pred = _abs_axis_mean([_prediction_mouse_delta(row) for row in prediction_reference_rows]) or 1.0
    global_gain = min(max_gain, max(min_gain, global_train / max(global_pred, 1e-9)))
    gains: dict[str, dict[str, Any]] = {}

    calibrated: list[dict[str, Any]] = []
    for row in predictions:
        rid = _recording_id(row)
        game = _game_id(row)
        target_abs = train_by_recording.get(rid) or train_by_game.get(game) or global_train
        pred_abs = pred_by_recording.get(rid) or global_pred
        raw_gain = target_abs / max(pred_abs, 1e-9)
        gain = min(max_gain, max(min_gain, raw_gain))
        gains.setdefault(
            rid,
            {
                "recording_id": rid,
                "game": game,
                "target_train_abs_mean": target_abs,
                "prediction_abs_mean": pred_abs,
                "raw_gain": raw_gain,
                "gain": gain,
                "source": "recording" if rid in train_by_recording and rid in pred_by_recording else ("game" if game in train_by_game else "global"),
            },
        )
        dx, dy = _prediction_mouse_delta(row)
        tokens = tokens_from_delta(dx * gain, dy * gain) + _non_mouse_motion_tokens(list(row.get("predicted_tokens", [])))
        out = dict(row)
        out["predicted_tokens"] = tokens or ["NOOP"]
        calibrated.append(out)

    diagnostics = {
        "mode": "recording_train_abs_ratio",
        "min_gain": min_gain,
        "max_gain": max_gain,
        "global_train_abs_mean": global_train,
        "global_prediction_abs_mean": global_pred,
        "global_gain": global_gain,
        "prediction_reference": "calibration_predictions" if calibration_predictions is not None else "target_predictions",
        "recording_gains": sorted(gains.values(), key=lambda item: item["recording_id"]),
    }
    return calibrated, diagnostics


def calibrate_fdm_predictions(config: dict[str, Any]) -> dict[str, Any]:
    """Post-calibrate a trained FDM decoder's mouse scale without target labels.

    ``train_records_path`` remains the FDM training/baseline record stream for
    backwards compatibility.  Serious D2E runs may additionally pass
    ``calibration_records_path`` when the scale calibration signal intentionally
    differs from the IDM pseudo-labels consumed by the FDM (for example, real
    D2E train-split labels).  ``baseline_train_records_path`` can also be set
    explicitly so endpoint references stay tied to the model's training signal
    while the calibration metadata records the separate train-only scale signal.
    """

    source_predictions_path = Path(config["source_predictions_path"])
    train_records_path = Path(config["train_records_path"])
    calibration_records_path = Path(config.get("calibration_records_path", train_records_path))
    calibration_predictions_path = Path(config["calibration_predictions_path"]) if config.get("calibration_predictions_path") else None
    baseline_train_records_path = Path(config.get("baseline_train_records_path", train_records_path))
    target_records_path = Path(config["target_records_path"])
    labels_path = Path(config["labels_path"])
    endpoints_path = str(config.get("endpoints", "configs/eval/primary_endpoints.yaml"))
    output_dir = Path(config.get("output_dir", "outputs/fdm_recording_scale_calibrated"))
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_jsonl(source_predictions_path)
    train_records = read_jsonl(train_records_path)
    calibration_records = read_jsonl(calibration_records_path)
    calibration_predictions = read_jsonl(calibration_predictions_path) if calibration_predictions_path is not None else None
    baseline_train_records = read_jsonl(baseline_train_records_path)
    target_records = read_jsonl(target_records_path)
    calibrated, diagnostics = _scale_predictions(
        predictions,
        calibration_records,
        min_gain=float(config.get("min_gain", 0.25)),
        max_gain=float(config.get("max_gain", 4.0)),
        calibration_predictions=calibration_predictions,
    )
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, calibrated)
    metrics = compute_metrics(calibrated, target_records)
    write_json(output_dir / "metrics.json", metrics)
    predictions_by_name = build_baseline_predictions(baseline_train_records, target_records)
    model_name = str(config.get("model_name", "fdm_recording_scale_calibrated"))
    predictions_by_name[model_name] = calibrated
    stat = compare_systems(predictions_by_name, target_records, load_config(endpoints_path))
    write_json(output_dir / "statistical_comparison.json", stat)
    label_hash = sha256_file(labels_path)
    checkpoint = {
        "schema": "fdm_checkpoint_metadata.v1",
        "model": model_name,
        "label_source": "idm_pseudolabel",
        "source_label_artifact": str(labels_path),
        "source_label_sha256": label_hash,
        "predictions_path": str(predictions_path),
        "num_training_examples": len(train_records),
        "oracle_ground_truth_control": False,
        "source_predictions_path": str(source_predictions_path),
        "train_records_path": str(train_records_path),
        "calibration_records_path": str(calibration_records_path),
        "calibration_predictions_path": str(calibration_predictions_path) if calibration_predictions_path is not None else "",
        "baseline_train_records_path": str(baseline_train_records_path),
        "target_records_path": str(target_records_path),
        "target_examples": len(target_records),
        "num_calibration_examples": len(calibration_records),
        "num_calibration_prediction_examples": len(calibration_predictions or []),
        "num_baseline_train_examples": len(baseline_train_records),
        "calibration_label_source": str(config.get("calibration_label_source", "train_records_ground_truth_tokens")),
        "calibration_uses_target_ground_truth": False,
        "calibration_uses_target_prediction_distribution": calibration_predictions_path is None,
        "calibration": diagnostics,
        "dataset_fingerprint": stable_hash_json(
            {
                "source_predictions_path": str(source_predictions_path),
                "source_predictions_sha256": sha256_file(source_predictions_path),
                "labels_sha256": label_hash,
                "train_sequence_ids": [row["sequence_id"] for row in train_records],
                "calibration_records_path": str(calibration_records_path),
                "calibration_records_sha256": sha256_file(calibration_records_path),
                "calibration_predictions_path": str(calibration_predictions_path) if calibration_predictions_path is not None else "",
                "calibration_predictions_sha256": sha256_file(calibration_predictions_path) if calibration_predictions_path is not None else "",
                "calibration_sequence_ids": [row["sequence_id"] for row in calibration_records],
                "calibration_prediction_sequence_ids": [row["sequence_id"] for row in calibration_predictions or []],
                "baseline_train_records_path": str(baseline_train_records_path),
                "baseline_train_records_sha256": sha256_file(baseline_train_records_path),
                "baseline_train_sequence_ids": [row["sequence_id"] for row in baseline_train_records],
                "target_sequence_ids": [row["sequence_id"] for row in target_records],
                "config": {key: value for key, value in config.items() if key != "output_dir"},
            }
        ),
        "statistical_comparison_path": str(output_dir / "statistical_comparison.json"),
        "summary_path": str(output_dir / "summary.json"),
    }
    validate_named(checkpoint, "fdm_checkpoint_metadata.schema.json")
    write_json(output_dir / "checkpoint_metadata.json", checkpoint)
    summary = {
        "schema": "fdm_recording_scale_calibration_summary.v1",
        "checkpoint": checkpoint,
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "statistical_comparison": stat,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
