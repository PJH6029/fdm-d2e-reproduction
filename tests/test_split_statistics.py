from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json, write_jsonl
from fdm_d2e.eval.split_statistics import write_split_statistical_comparisons


def _record(idx: int, split_tag: str) -> dict:
    button = "MOUSE_LEFT_DOWN" if idx % 2 else "MOUSE_LEFT_UP"
    return {
        "schema": "d2e_window_record.v1",
        "sequence_id": f"rec_{split_tag}#{idx:04d}",
        "recording_id": f"rec_{split_tag}",
        "cross_resolution_key": f"Game/rec_{split_tag}",
        "game": "Game",
        "source_id": "d2e_480p",
        "resolution_tier": "480p",
        "split": "eval",
        "eval_split_tags": [split_tag],
        "timestamp_ns": idx,
        "bin_index": idx,
        "frame": {"path": "", "index": idx, "features": [0.1, 0.2], "grid8": [0.1] * 192},
        "ground_truth_tokens": ["KEY_PRESS_87", button, "MOUSE_DX_P1", "MOUSE_DY_Z0"],
    }


def test_write_split_statistical_comparisons_outputs_required_split_files(tmp_path: Path):
    splits = ["temporal", "heldout_recording", "heldout_game"]
    train = [_record(i, "temporal") for i in range(3)]
    target = [_record(i, split) for split in splits for i in range(3)]
    preds = [{"sequence_id": row["sequence_id"], "predicted_tokens": row["ground_truth_tokens"]} for row in target]
    endpoints = {
        "schema": "primary_endpoints.v1",
        "cluster_key": "recording_id",
        "bootstrap": {"n_resamples": 20, "confidence": 0.95, "seed": 1},
        "correction": "holm_bonferroni",
        "reference_baseline": "noop",
        "endpoints": [
            {"name": "keyboard_accuracy", "metric_path": ["keyboard", "accuracy"], "direction": "higher"},
            {"name": "mouse_button_f1", "metric_path": ["mouse_button", "f1"], "direction": "higher"},
            {"name": "no_button_false_positive_rate", "metric_path": ["mouse_button", "no_button_false_positive_rate"], "direction": "lower"},
        ],
    }
    write_jsonl(tmp_path / "train.jsonl", train)
    write_jsonl(tmp_path / "target.jsonl", target)
    write_jsonl(tmp_path / "predictions.jsonl", preds)
    write_json(tmp_path / "endpoints.json", endpoints)
    config = {
        "model_name": "tiny_model",
        "predictions_path": "predictions.jsonl",
        "ground_truth_path": "target.jsonl",
        "train_records_path": "train.jsonl",
        "output_dir": "split_stats",
        "summary_out": "summary.json",
        "endpoints": "endpoints.json",
        "baseline_names": ["noop", "global_majority", "last_seen_train"],
        "split_tags": splits,
    }
    summary = write_split_statistical_comparisons(config, root=tmp_path)
    assert summary["status"] == "pass"
    assert len(summary["outputs"]) == 3
    for split in splits:
        payload = json.loads((tmp_path / "split_stats" / f"split_{split}_statistical_comparison.json").read_text())
        assert payload["split"] == split
        assert payload["ground_truth_records"] == 3
        assert payload["model_prediction_records"] == 3
        assert {row["endpoint"] for row in payload["comparisons"]} == {"keyboard_accuracy", "mouse_button_f1", "no_button_false_positive_rate"}
        assert all(row.get("candidate_value") is not None for row in payload["comparisons"] if row["endpoint"] != "no_button_false_positive_rate")


def test_write_split_statistical_comparisons_streams_with_precomputed_train_stats(tmp_path: Path):
    splits = ["temporal", "heldout_recording", "heldout_game"]
    target = [_record(i, split) for split in splits for i in range(3)]
    preds = [{"sequence_id": row["sequence_id"], "predicted_tokens": row["ground_truth_tokens"]} for row in target]
    endpoints = {
        "schema": "primary_endpoints.v1",
        "cluster_key": "recording_id",
        "bootstrap": {"n_resamples": 20, "confidence": 0.95, "seed": 1},
        "correction": "holm_bonferroni",
        "reference_baseline": "noop",
        "endpoints": [
            {"name": "keyboard_accuracy", "metric_path": ["keyboard", "accuracy"], "direction": "higher"},
            {"name": "mouse_button_f1", "metric_path": ["mouse_button", "f1"], "direction": "higher"},
        ],
    }
    write_jsonl(tmp_path / "target.jsonl", target)
    write_jsonl(tmp_path / "predictions.jsonl", preds)
    write_json(tmp_path / "endpoints.json", endpoints)
    write_json(
        tmp_path / "streaming_stats.json",
        {
            "schema": "streaming_idm_stats.v1",
            "global_majority_tokens": ["NOOP"],
            "last_tokens_by_recording": {},
            "last_tokens_by_game": {},
        },
    )
    config = {
        "model_name": "tiny_model",
        "predictions_path": "predictions.jsonl",
        "ground_truth_path": "target.jsonl",
        "streaming": True,
        "train_stats_path": "streaming_stats.json",
        "output_dir": "split_stats_streaming",
        "summary_out": "summary_streaming.json",
        "endpoints": "endpoints.json",
        "baseline_names": ["noop", "global_majority", "last_seen_train"],
        "split_tags": splits,
    }

    summary = write_split_statistical_comparisons(config, root=tmp_path)

    assert summary["status"] == "pass"
    assert len(summary["outputs"]) == 3
    for split in splits:
        payload = json.loads((tmp_path / "split_stats_streaming" / f"split_{split}_statistical_comparison.json").read_text())
        assert payload["split"] == split
        assert payload["ground_truth_records"] == 3
        assert payload["model_prediction_records"] == 3
        assert {row["endpoint"] for row in payload["comparisons"]} == {"keyboard_accuracy", "mouse_button_f1"}


def test_streaming_train_stats_falls_back_to_train_records_for_last_seen(tmp_path: Path):
    splits = ["temporal", "heldout_recording", "heldout_game"]
    train_dir = tmp_path / "train_shards"
    train_dir.mkdir()
    train = [_record(99, split) for split in splits]
    target = [_record(i, split) for split in splits for i in range(3)]
    preds = [{"sequence_id": row["sequence_id"], "predicted_tokens": row["ground_truth_tokens"]} for row in target]
    endpoints = {
        "schema": "primary_endpoints.v1",
        "cluster_key": "recording_id",
        "bootstrap": {"n_resamples": 20, "confidence": 0.95, "seed": 1},
        "correction": "holm_bonferroni",
        "reference_baseline": "noop",
        "endpoints": [
            {
                "name": "keyboard_accuracy",
                "reference_baseline": "last_seen_train",
                "metric_path": ["keyboard", "accuracy"],
                "direction": "higher",
            },
        ],
    }
    write_jsonl(train_dir / "part_0.jsonl", train)
    write_jsonl(tmp_path / "target.jsonl", target)
    write_jsonl(tmp_path / "predictions.jsonl", preds)
    write_json(tmp_path / "endpoints.json", endpoints)
    write_json(
        tmp_path / "streaming_stats.json",
        {
            "schema": "streaming_idm_stats.v1",
            "global_majority_tokens": ["NOOP"],
        },
    )
    config = {
        "model_name": "tiny_model",
        "predictions_path": "predictions.jsonl",
        "ground_truth_path": "target.jsonl",
        "streaming": True,
        "train_stats_path": "streaming_stats.json",
        "train_records_glob": "train_shards/*.jsonl",
        "output_dir": "split_stats_last_seen_fallback",
        "summary_out": "summary_last_seen_fallback.json",
        "endpoints": "endpoints.json",
        "baseline_names": ["noop", "global_majority", "last_seen_train"],
        "split_tags": splits,
    }

    summary = write_split_statistical_comparisons(config, root=tmp_path)

    assert summary["status"] == "pass"
    payload = json.loads((tmp_path / "split_stats_last_seen_fallback" / "split_temporal_statistical_comparison.json").read_text())
    model_row = next(row for row in payload["comparisons"] if row["model"] == "tiny_model")
    assert model_row["reference"] == "last_seen_train"
    assert model_row["baseline_value"] == 1.0


def test_write_split_statistical_comparisons_streams_with_ground_truth_glob(tmp_path: Path):
    splits = ["temporal", "heldout_recording", "heldout_game"]
    target = [_record(i, split) for split in splits for i in range(3)]
    preds = [{"sequence_id": row["sequence_id"], "predicted_tokens": row["ground_truth_tokens"]} for row in target]
    endpoints = {
        "schema": "primary_endpoints.v1",
        "cluster_key": "recording_id",
        "bootstrap": {"n_resamples": 20, "confidence": 0.95, "seed": 1},
        "correction": "holm_bonferroni",
        "reference_baseline": "noop",
        "endpoints": [
            {"name": "keyboard_accuracy", "metric_path": ["keyboard", "accuracy"], "direction": "higher"},
        ],
    }
    shard_dir = tmp_path / "target_shards"
    shard_dir.mkdir()
    write_jsonl(shard_dir / "part_0.jsonl", target[:4])
    write_jsonl(shard_dir / "part_1.jsonl", target[4:])
    write_jsonl(tmp_path / "predictions.jsonl", preds)
    write_json(tmp_path / "endpoints.json", endpoints)
    write_json(
        tmp_path / "streaming_stats.json",
        {
            "schema": "streaming_idm_stats.v1",
            "global_majority_tokens": ["NOOP"],
            "last_tokens_by_recording": {},
            "last_tokens_by_game": {},
        },
    )
    config = {
        "model_name": "tiny_model",
        "predictions_path": "predictions.jsonl",
        "ground_truth_glob": "target_shards/*.jsonl",
        "streaming": True,
        "train_stats_path": "streaming_stats.json",
        "output_dir": "split_stats_streaming_glob",
        "summary_out": "summary_streaming_glob.json",
        "endpoints": "endpoints.json",
        "baseline_names": ["noop", "global_majority"],
        "split_tags": splits,
    }

    summary = write_split_statistical_comparisons(config, root=tmp_path)

    assert summary["status"] == "pass"
    for split in splits:
        payload = json.loads((tmp_path / "split_stats_streaming_glob" / f"split_{split}_statistical_comparison.json").read_text())
        assert payload["ground_truth_records"] == 3
        assert payload["model_prediction_records"] == 3


def test_write_split_statistical_comparisons_streams_parallel_prediction_parts(tmp_path: Path):
    splits = ["temporal", "heldout_recording", "heldout_game"]
    target = [_record(i, split) for split in splits for i in range(3)]
    preds = [{"sequence_id": row["sequence_id"], "predicted_tokens": row["ground_truth_tokens"]} for row in target]
    endpoints = {
        "schema": "primary_endpoints.v1",
        "cluster_key": "recording_id",
        "bootstrap": {"n_resamples": 20, "confidence": 0.95, "seed": 1},
        "correction": "holm_bonferroni",
        "reference_baseline": "noop",
        "endpoints": [
            {"name": "keyboard_accuracy", "metric_path": ["keyboard", "accuracy"], "direction": "higher"},
        ],
    }
    shard_dir = tmp_path / "target_shards"
    pred_dir = tmp_path / "prediction_parts"
    shard_dir.mkdir()
    (pred_dir / "part_000").mkdir(parents=True)
    (pred_dir / "part_001").mkdir(parents=True)
    write_jsonl(shard_dir / "part_0.jsonl", target[:4])
    write_jsonl(shard_dir / "part_1.jsonl", target[4:])
    write_jsonl(pred_dir / "part_000" / "predictions.jsonl", preds[:4])
    write_jsonl(pred_dir / "part_001" / "predictions.jsonl", preds[4:])
    write_json(tmp_path / "endpoints.json", endpoints)
    write_json(
        tmp_path / "streaming_stats.json",
        {
            "schema": "streaming_idm_stats.v1",
            "global_majority_tokens": ["NOOP"],
            "last_tokens_by_recording": {},
            "last_tokens_by_game": {},
        },
    )
    write_json(
        tmp_path / "prediction_summary.json",
        {
            "schema": "video_idm_prediction_summary.v1",
            "prediction": {
                "prediction_parallel": {
                    "parts": [
                        {
                            "part_index": 0,
                            "predictions_path": "prediction_parts/part_000/predictions.jsonl",
                            "record_paths": ["target_shards/part_0.jsonl"],
                        },
                        {
                            "part_index": 1,
                            "predictions_path": "prediction_parts/part_001/predictions.jsonl",
                            "record_paths": ["target_shards/part_1.jsonl"],
                        },
                    ],
                },
            },
        },
    )
    config = {
        "model_name": "tiny_model",
        "predictions_path": "unused_merged_predictions.jsonl",
        "prediction_summary_path": "prediction_summary.json",
        "ground_truth_glob": "target_shards/*.jsonl",
        "streaming": True,
        "split_workers": 2,
        "train_stats_path": "streaming_stats.json",
        "output_dir": "split_stats_parallel_parts",
        "summary_out": "summary_parallel_parts.json",
        "endpoints": "endpoints.json",
        "baseline_names": ["noop", "global_majority"],
        "split_tags": splits,
    }

    summary = write_split_statistical_comparisons(config, root=tmp_path)

    assert summary["status"] == "pass"
    for split in splits:
        payload = json.loads((tmp_path / "split_stats_parallel_parts" / f"split_{split}_statistical_comparison.json").read_text())
        assert payload["ground_truth_records"] == 3
        assert payload["model_prediction_records"] == 3
