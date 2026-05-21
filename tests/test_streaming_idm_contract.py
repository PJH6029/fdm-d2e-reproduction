from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdm_d2e.training.streaming_idm import train_streaming_idm
from fdm_d2e.training.torch_idm import torch_available


def _record(idx: int, split: str) -> dict:
    token = "MOUSE_DX_P1" if idx % 2 else "MOUSE_DX_N1"
    return {
        "schema": "d2e_window_record.v1",
        "sequence_id": f"d2e_480p:Apex/rec#%06d" % idx,
        "recording_id": "d2e_480p:Apex/rec",
        "cross_resolution_key": "Apex/rec",
        "game": "Apex",
        "split": split,
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
        "ground_truth_tokens": [token, "MOUSE_DY_Z0", "KEY_PRESS_87"] if idx % 3 == 0 else [token, "MOUSE_DY_Z0"],
        "source": "test",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_streaming_idm_trains_tiny_compact_feature_checkpoint(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    _write_jsonl(target_path, [_record(idx + 8, "eval") for idx in range(4)])

    summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "idm"),
            "summary_out": str(tmp_path / "summary.json"),
            "endpoints": "configs/eval/primary_endpoints.yaml",
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "eval_interval_epochs": 1,
            "batch_size": 4,
            "categorical_min_count": 1,
            "mouse_head_mode": "axis_softmax",
            "seed": 7,
            "force_cpu": True,
        }
    )

    assert summary["metadata"]["train_records"] == 8
    assert summary["metadata"]["target_records"] == 4
    assert Path(summary["metadata"]["checkpoint_path"]).exists()
    assert Path(summary["metadata"]["pseudo_label_path"]).exists()
    assert Path(summary["metadata"]["label_quality_report_path"]).exists()
    assert Path(summary["metadata"]["convergence_report_path"]).exists()
    assert summary["convergence_report"]["num_validation_checkpoints"] == 1
    assert summary["convergence_report"]["history"][0]["validation_score"]["mode"] == "composite_primary"
    assert summary["label_quality_report"]["baseline_metrics"]["noop"]["num_examples"] == 4
    assert "game:Apex" in summary["label_quality_report"]["groups_by_model"]["tiny_streaming_idm"]
    assert Path(summary["metadata"]["statistical_comparison_path"]).exists()
    assert summary["statistical_comparison"]["schema"] == "stat_comparison.v1"
