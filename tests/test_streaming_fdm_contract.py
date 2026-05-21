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
    assert train_rows[0]["ground_truth_tokens"] == labels[0]["predicted_tokens"]
    assert train_rows[0]["label_source"] == "idm_pseudolabel_for_fdm"
    assert target_rows[0]["ground_truth_tokens"] == records[3]["ground_truth_tokens"]


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
            "fdm_train_fraction": 0.75,
            "torch_idm_config": {
                "feature_mode": "summary_compact_grid8_shift_surface_time",
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
    assert checkpoint["num_training_examples"] == 6
    assert checkpoint["target_examples"] == 2
    assert checkpoint["convergence_report_path"]
    assert summary["convergence_report"]["num_validation_checkpoints"] == 1
    assert Path(checkpoint["predictions_path"]).exists()
    assert Path(checkpoint["train_records_path"]).exists()
    assert summary["statistical_comparison"]["schema"] == "stat_comparison.v1"
