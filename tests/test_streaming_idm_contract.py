from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdm_d2e.training.streaming_idm import _distributed_runtime, predict_streaming_idm_checkpoint, scan_streaming_idm_stats, train_streaming_idm
from fdm_d2e.training.torch_idm import torch_available


def _record(idx: int, split: str) -> dict:
    token = "MOUSE_DX_P1" if idx % 2 else "MOUSE_DX_N1"
    return {
        "schema": "d2e_window_record.v1",
        "sequence_id": f"d2e_480p:Apex/rec#%06d" % idx,
        "recording_id": "d2e_480p:Apex/rec",
        "cross_resolution_key": "Apex/rec",
        "game": "Apex",
        "source_id": "d2e_480p",
        "resolution_tier": "480p",
        "split": split,
        "eval_split_tags": ["temporal"] if split == "eval" else [],
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
            "config_path": "test_inline_config",
            "source_namespace": "unit_d2e_stream",
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
    assert summary["metadata"]["config_fingerprint"]
    assert summary["metadata"]["config_path"] == "test_inline_config"
    assert summary["metadata"]["source_namespace"] == "unit_d2e_stream"
    assert summary["metadata"]["source_ids"] == ["d2e_480p"]
    assert summary["metadata"]["resolution_tiers"] == ["480p"]
    assert summary["metadata"]["target_source_ids"] == ["d2e_480p"]
    assert summary["metadata"]["target_resolution_tiers"] == ["480p"]
    assert summary["metadata"]["split_names"] == ["train_core"]
    assert summary["metadata"]["target_eval_split_tags"] == ["temporal"]
    assert Path(summary["metadata"]["resolved_config_path"]).exists()
    assert Path(summary["metadata"]["train_records_path"]).exists()
    assert Path(summary["metadata"]["target_records_path"]).exists()
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


def test_streaming_idm_predicts_train_core_pseudolabels_without_retraining(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    fdm_train_path = tmp_path / "fdm_train_core.jsonl"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    _write_jsonl(target_path, [_record(idx + 8, "eval") for idx in range(4)])
    _write_jsonl(fdm_train_path, [_record(idx + 20, "train_core") for idx in range(5)])
    idm_out = tmp_path / "idm"

    train_summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm_predict_only",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(idm_out),
            "summary_out": str(tmp_path / "summary.json"),
            "config_path": "test_inline_config",
            "source_namespace": "unit_d2e_stream",
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "eval_interval_epochs": 1,
            "batch_size": 4,
            "categorical_min_count": 1,
            "mouse_head_mode": "axis_softmax",
            "seed": 11,
            "force_cpu": True,
        }
    )

    pred_summary = predict_streaming_idm_checkpoint(
        {
            "checkpoint_path": train_summary["metadata"]["checkpoint_path"],
            "checkpoint_metadata_path": str(idm_out / "checkpoint_metadata.json"),
            "records_path": str(fdm_train_path),
            "output_dir": str(tmp_path / "fdm_train_core_pseudolabels"),
            "summary_out": str(tmp_path / "fdm_train_core_summary.json"),
            "force_cpu": True,
            "eval_batch_size": 2,
        }
    )

    assert pred_summary["schema"] == "streaming_idm_predict_summary.v1"
    assert pred_summary["records"] == 5
    assert pred_summary["source_checkpoint_artifact"]["exists"] is True
    assert pred_summary["source_checkpoint_metadata"]["exists"] is True
    assert Path(pred_summary["pseudo_label_path"]).exists()
    assert Path(pred_summary["predictions_path"]).exists()
    assert Path(tmp_path / "fdm_train_core_summary.json").exists()


def test_distributed_runtime_passes_configured_timeout(monkeypatch):
    calls = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def set_device(local_rank):
            calls["set_device"] = local_rank

    class FakeDistributed:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def init_process_group(**kwargs):
            calls["init_kwargs"] = kwargs

    class FakeTorch:
        cuda = FakeCuda()
        distributed = FakeDistributed()

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")

    dist = _distributed_runtime(FakeTorch(), {"distributed_timeout_seconds": 21600})

    assert calls["set_device"] == 2
    assert calls["init_kwargs"]["backend"] == "nccl"
    assert calls["init_kwargs"]["timeout"].total_seconds() == 21600
    assert dist["timeout_seconds"] == 21600


def test_streaming_idm_stats_can_scan_multiple_record_files(tmp_path: Path):
    shard_a = tmp_path / "shard_a.jsonl"
    shard_b = tmp_path / "shard_b.jsonl"
    _write_jsonl(shard_a, [_record(idx, "train_core") for idx in range(3)])
    _write_jsonl(shard_b, [_record(idx + 3, "train_core") for idx in range(3)])

    stats = scan_streaming_idm_stats(
        [shard_a, shard_b],
        feature_mode="summary_compact_grid8_shift_surface_time",
        categorical_min_count=1,
        num_workers=1,
    )

    assert stats["num_examples"] == 6
    assert stats["input_dim"] > 0
    assert stats["source_ids"] == ["d2e_480p"]
    assert "KEY_PRESS_87" in stats["category_vocab"]
    assert stats["last_tokens_by_recording"]["d2e_480p:Apex/rec"]
