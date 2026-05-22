from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdm_d2e.io_utils import read_jsonl
from fdm_d2e.training.streaming_fdm import materialize_fdm_streaming_splits, train_streaming_fdm
from fdm_d2e.training.torch_idm import torch_available


def _record(idx: int, recording_id: str = "d2e_480p:Apex/rec") -> dict:
    token = "MOUSE_DX_P1" if idx % 2 else "MOUSE_DX_N1"
    return {
        "schema": "d2e_window_record.v1",
        "sequence_id": f"{recording_id}#%06d" % idx,
        "recording_id": recording_id,
        "cross_resolution_key": "Apex/rec",
        "game": "Apex",
        "source_id": "d2e_480p",
        "resolution_tier": "480p",
        "split": "eval",
        "timestamp_ns": idx,
        "bin_index": idx,
        "frame": {
            "path": "",
            "index": idx,
            "features": [0.1 * idx, 0.2, 0.3, 0.4, 0.5],
            "grid8": [0.01 * (idx + 1)] * (8 * 8 * 3),
            "luma16": [0.02 * (idx + 1)] * (16 * 16),
        },
        "next_frame_features": [0.1 * (idx + 1), 0.2, 0.3, 0.4, 0.5],
        "frame_delta_features": [0.1, 0.0, 0.0, 0.0, 0.0],
        "next_frame_grid8": [0.01 * (idx + 2)] * (8 * 8 * 3),
        "next_frame_luma16": [0.02 * (idx + 2)] * (16 * 16),
        "events": [],
        "ground_truth_tokens": [token, "MOUSE_DY_Z0", "KEY_PRESS_87"],
        "source": "test",
        "eval_split_tags": ["temporal"],
    }


def _label(row: dict, idx: int) -> dict:
    token = "MOUSE_DX_P2" if idx % 2 else "MOUSE_DX_N2"
    return {
        "schema": "idm_pseudolabel.v1",
        "sequence_id": row["sequence_id"],
        "timestamp_ns": row["timestamp_ns"],
        "predicted_tokens": [token, "MOUSE_DY_Z0"],
        "label_source": "idm_generated",
        "confidence": 0.8,
        "model": "tiny_idm",
        "training_split_hash": "abc",
        "input_window": {"frame_ref": "", "frame_index": idx},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_streaming_fdm_materializes_pseudolabel_train_and_ground_truth_eval(tmp_path: Path):
    records = [_record(idx) for idx in range(6)]
    labels = [_label(row, idx) for idx, row in enumerate(records)]
    records_path = tmp_path / "records.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    out = tmp_path / "fdm"
    _write_jsonl(records_path, records)
    _write_jsonl(labels_path, labels)

    summary = materialize_fdm_streaming_splits(
        {
            "records_path": str(records_path),
            "labels_path": str(labels_path),
            "output_dir": str(out),
            "fdm_train_fraction": 0.5,
        }
    )

    train_rows = read_jsonl(summary["train_records_path"])
    target_rows = read_jsonl(summary["target_records_path"])
    assert summary["counts"]["pairs"] == 6
    assert summary["counts"]["train"] == 3
    assert summary["counts"]["target"] == 3
    assert summary["prior_action_context"]["train_source"] == "idm_pseudolabel_previous_teacher_forced"
    assert summary["prior_action_context"]["target_source"] == "d2e_ground_truth_previous_teacher_forced"
    assert train_rows[0]["ground_truth_tokens"] == labels[0]["predicted_tokens"]
    assert train_rows[0]["label_source"] == "idm_pseudolabel_for_fdm"
    assert train_rows[0]["prior_action_tokens"] == ["NOOP"]
    assert train_rows[1]["prior_action_tokens"] == labels[0]["predicted_tokens"]
    assert train_rows[1]["prior_action_source"] == "idm_pseudolabel_previous_teacher_forced"
    assert target_rows[0]["ground_truth_tokens"] == records[3]["ground_truth_tokens"]
    assert target_rows[0]["prior_action_source"] == "d2e_ground_truth_previous_teacher_forced"


def test_streaming_fdm_explicit_target_preserves_heldout_eval_namespace(tmp_path: Path):
    train_records = [_record(idx, "d2e_480p:Apex/train_rec") for idx in range(4)]
    target_records = [_record(idx + 10, "d2e_original:Celeste/heldout_rec") for idx in range(3)]
    for row in target_records:
        row["source_id"] = "d2e_original"
        row["resolution_tier"] = "original"
        row["split"] = "heldout_game"
        row["eval_split_tags"] = ["heldout_game"]
    labels = [_label(row, idx) for idx, row in enumerate(train_records)]
    records_path = tmp_path / "train_core.jsonl"
    labels_path = tmp_path / "train_core_labels.jsonl"
    target_path = tmp_path / "target_all_eval.jsonl"
    out = tmp_path / "fdm"
    _write_jsonl(records_path, train_records)
    _write_jsonl(labels_path, labels)
    _write_jsonl(target_path, target_records)

    summary = materialize_fdm_streaming_splits(
        {
            "records_path": str(records_path),
            "labels_path": str(labels_path),
            "target_records_path": str(target_path),
            "output_dir": str(out),
            "fdm_train_fraction": 0.5,
        }
    )

    train_rows = read_jsonl(summary["train_records_path"])
    target_rows = read_jsonl(summary["target_records_path"])
    assert summary["counts"]["mode"] == "explicit_target"
    assert summary["counts"]["train"] == 4
    assert summary["counts"]["target"] == 3
    assert summary["records_path"] == str(records_path)
    assert summary["target_records_source_path"] == str(target_path)
    assert {row["source_id"] for row in train_rows} == {"d2e_480p"}
    assert {row["source_id"] for row in target_rows} == {"d2e_original"}
    assert summary["counts"]["target_eval_split_tags"] == {"heldout_game": 3}
    assert train_rows[0]["prior_action_tokens"] == ["NOOP"]
    assert train_rows[1]["prior_action_tokens"] == labels[0]["predicted_tokens"]
    assert target_rows[0]["prior_action_tokens"] == ["NOOP"]
    assert target_rows[1]["prior_action_tokens"] == target_records[0]["ground_truth_tokens"]


def test_streaming_fdm_materializes_sharded_training_and_target_outputs(tmp_path: Path):
    train_records = [_record(idx, "d2e_480p:Apex/train_rec") for idx in range(6)]
    target_records = [_record(idx + 10, "d2e_original:Celeste/heldout_rec") for idx in range(4)]
    labels = [_label(row, idx) for idx, row in enumerate(train_records)]
    records_path = tmp_path / "train_core.jsonl"
    labels_path = tmp_path / "train_core_labels.jsonl"
    target_path = tmp_path / "target_all_eval.jsonl"
    out = tmp_path / "fdm"
    _write_jsonl(records_path, train_records)
    _write_jsonl(labels_path, labels)
    _write_jsonl(target_path, target_records)

    summary = materialize_fdm_streaming_splits(
        {
            "records_path": str(records_path),
            "labels_path": str(labels_path),
            "target_records_path": str(target_path),
            "output_dir": str(out),
            "num_output_shards": 3,
        }
    )

    train_shards = [Path(path) for path in summary["train_record_paths"]]
    target_shards = [Path(path) for path in summary["target_record_paths"]]
    assert summary["output_shards"]["enabled"] is True
    assert summary["output_shards"]["num_shards"] == 3
    assert summary["train_records_glob"].endswith("fdm_train_shards/shard_*.jsonl")
    assert summary["target_records_glob"].endswith("fdm_target_shards/shard_*.jsonl")
    assert all(path.exists() for path in train_shards)
    assert all(path.exists() for path in target_shards)
    assert sum(len(read_jsonl(path)) for path in train_shards) == summary["counts"]["train"] == 6
    assert sum(len(read_jsonl(path)) for path in target_shards) == summary["counts"]["target"] == 4
    assert read_jsonl(train_shards[0])[0]["sequence_id"] == train_records[0]["sequence_id"]
    assert read_jsonl(train_shards[1])[0]["sequence_id"] == train_records[1]["sequence_id"]
    assert len(read_jsonl(summary["train_records_path"])) == 6


def test_streaming_fdm_rejects_misaligned_labels(tmp_path: Path):
    records = [_record(idx) for idx in range(2)]
    labels = [_label(row, idx) for idx, row in enumerate(records)]
    labels[1]["sequence_id"] = "wrong#id"
    records_path = tmp_path / "records.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(records_path, records)
    _write_jsonl(labels_path, labels)

    with pytest.raises(ValueError, match="sequence_id mismatch"):
        materialize_fdm_streaming_splits(
            {
                "records_path": str(records_path),
                "labels_path": str(labels_path),
                "output_dir": str(tmp_path / "fdm"),
            }
        )


def test_streaming_fdm_trains_tiny_checkpoint(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    records = [_record(idx) for idx in range(8)]
    labels = [_label(row, idx) for idx, row in enumerate(records)]
    records_path = tmp_path / "records.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    endpoints = tmp_path / "endpoints.json"
    out = tmp_path / "fdm"
    _write_jsonl(records_path, records)
    _write_jsonl(labels_path, labels)
    endpoints.write_text(
        json.dumps(
            {
                "schema": "primary_endpoints.v1",
                "reference_baseline": "noop",
                "endpoints": [],
            }
        )
    )

    summary = train_streaming_fdm(
        {
            "model_name": "tiny_streaming_fdm",
            "records_path": str(records_path),
            "labels_path": str(labels_path),
            "output_dir": str(out),
            "endpoints": str(endpoints),
            "config_path": "test_fdm_inline_config",
            "source_namespace": "unit_d2e_fdm",
            "fdm_train_fraction": 0.75,
            "num_output_shards": 2,
            "torch_idm_config": {
                "feature_mode": "summary_causal_compact_grid8_time_prior_action",
                "hidden_dim": 8,
                "depth": 1,
                "epochs": 1,
                "eval_interval_epochs": 1,
                "batch_size": 4,
                "categorical_min_count": 1,
                "mouse_head_mode": "axis_softmax",
                "force_cpu": True,
                "seed": 9,
            },
        }
    )

    checkpoint = summary["checkpoint"]
    assert checkpoint["label_source"] == "idm_pseudolabel"
    assert checkpoint["oracle_ground_truth_control"] is False
    assert checkpoint["config_fingerprint"]
    assert checkpoint["config_path"] == "test_fdm_inline_config"
    assert checkpoint["source_namespace"] == "unit_d2e_fdm"
    assert checkpoint["source_ids"] == ["d2e_480p"]
    assert checkpoint["resolution_tiers"] == ["480p"]
    assert checkpoint["target_source_ids"] == ["d2e_480p"]
    assert checkpoint["target_resolution_tiers"] == ["480p"]
    assert checkpoint["split_names"] == ["eval"]
    assert checkpoint["target_split_names"] == ["eval"]
    assert checkpoint["target_games"] == ["Apex"]
    assert checkpoint["target_eval_split_tags"] == ["temporal"]
    assert checkpoint["torch_checkpoint_metadata"]["feature_mode"] == "summary_causal_compact_grid8_time_prior_action"
    assert Path(checkpoint["resolved_config_path"]).exists()
    assert checkpoint["num_training_examples"] == 6
    assert checkpoint["target_examples"] == 2
    assert len(checkpoint["train_record_paths"]) == 2
    assert len(checkpoint["target_record_paths"]) == 2
    assert checkpoint["train_records_glob"].endswith("fdm_train_shards/shard_*.jsonl")
    assert checkpoint["target_records_glob"].endswith("fdm_target_shards/shard_*.jsonl")
    assert summary["split_summary"]["output_shards"]["enabled"] is True
    assert checkpoint["convergence_report_path"]
    assert summary["convergence_report"]["num_validation_checkpoints"] == 1
    assert Path(checkpoint["predictions_path"]).exists()
    assert Path(checkpoint["train_records_path"]).exists()
    assert summary["statistical_comparison"]["schema"] == "stat_comparison.v1"
