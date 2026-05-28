from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import read_json
from fdm_d2e.training.masked_diffusion_idm_trainer import torch_available, train_masked_diffusion_idm, video_feature_vector


def _row(idx: int, *, split: str) -> dict:
    return {
        "sequence_id": f"unit#{idx:03d}",
        "split": split,
        "eval_split_tags": ["temporal"] if split != "train_core" else [],
        "frame": {"features": [idx / 10, 1.0], "width": 854, "height": 480},
        "next_frame_features": [idx / 10 + 0.1, 0.5],
        "frame_delta_features": [0.1, -0.1],
        "ground_truth_tokens": ["KEY_PRESS_W", "MOUSE_DX_P1", "MOUSE_DY_Z0"] if idx % 2 == 0 else [],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_video_feature_vector_uses_configured_paths_and_padding():
    row = _row(1, split="train_core")
    features = video_feature_vector(row, feature_paths=["frame.features", "next_frame_features"], dim=6)
    assert features[:4] == [0.1, 1.0, 0.2, 0.5]
    assert features[4:] == [0.0, 0.0]


def test_video_feature_vector_flattens_luma_window_tokens():
    row = _row(2, split="train_core")
    row["compact_luma_window"] = [[0.1, 0.2], [0.3, 0.4]]
    row["compact_luma_window_mask"] = [1, 0]
    row["frame"]["stats"] = {"b": 0.6, "a": 0.5}
    features = video_feature_vector(
        row,
        feature_paths=["compact_luma_window", "compact_luma_window_mask", "frame.stats"],
        dim=10,
    )
    assert features == [0.1, 0.2, 0.3, 0.4, 1.0, 0.0, 0.5, 0.6, 0.0, 0.0]


def test_train_masked_diffusion_idm_tiny_smoke(tmp_path: Path):
    if not torch_available():
        return
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    _write_jsonl(train_path, [_row(i, split="train_core") for i in range(6)])
    _write_jsonl(target_path, [_row(i, split="eval") for i in range(6, 9)])
    summary = train_masked_diffusion_idm(
        {
            "model_name": "unit_masked_diffusion_idm",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "out"),
            "summary_out": str(tmp_path / "summary.json"),
            "max_train_rows": 6,
            "max_target_rows": 3,
            "max_action_tokens_per_bin": 4,
            "video_feature_paths": ["frame.features", "next_frame_features", "frame_delta_features"],
            "video_feature_dim": 6,
            "mask_probability": 0.75,
            "random_token_probability": 0.0,
            "diffusion_steps": 4,
            "hidden_dim": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "force_cpu": True,
            "seed": 11,
            "noop_loss_weight": 0.2,
            "keyboard_loss_weight": 2.0,
        }
    )
    assert summary["status"] == "pass"
    assert summary["loss_weights"]["noop_loss_weight"] == 0.2
    assert summary["loss_weights"]["keyboard_loss_weight"] == 2.0
    assert summary["train_rows"] == 6
    assert summary["target_rows"] == 3
    assert Path(summary["checkpoint_path"]).exists()
    assert Path(summary["predictions_path"]).exists()
    assert Path(summary["metrics_path"]).exists()
    assert len(Path(summary["predictions_path"]).read_text(encoding="utf-8").strip().splitlines()) == 3
    metrics = read_json(summary["metrics_path"])
    assert metrics["status"] == "pass"
    assert metrics["alignment"]["rows_seen"] == 3
    assert "masked-diffusion IDM" in summary["recipe_alignment"]


def test_train_factorized_masked_diffusion_idm_tiny_smoke(tmp_path: Path):
    if not torch_available():
        return
    train_path = tmp_path / "train_factorized.jsonl"
    target_path = tmp_path / "target_factorized.jsonl"
    rows = [_row(i, split="train_core") for i in range(8)]
    rows[1]["ground_truth_tokens"] = ["KEY_RELEASE_W", "MOUSE_LEFT_DOWN", "MOUSE_DX_N1", "MOUSE_DY_P1"]
    rows[3]["ground_truth_tokens"] = ["KEY_PRESS_A", "MOUSE_LEFT_UP", "MOUSE_DX_P1", "MOUSE_DY_N1"]
    _write_jsonl(train_path, rows)
    _write_jsonl(target_path, [_row(i, split="eval") for i in range(8, 11)])
    summary = train_masked_diffusion_idm(
        {
            "model_name": "unit_factorized_masked_diffusion_idm",
            "factorized_action_tokens": True,
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "out_factorized"),
            "summary_out": str(tmp_path / "summary_factorized.json"),
            "max_train_rows": 8,
            "max_target_rows": 3,
            "video_feature_paths": ["frame.features", "next_frame_features", "frame_delta_features"],
            "video_feature_dim": 6,
            "hidden_dim": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "force_cpu": True,
            "key_threshold": 0.99,
            "button_threshold": 0.99,
            "calibrate_thresholds": True,
            "factorized_calibration_fraction": 0.25,
            "factorized_calibration_max_rows": 2,
            "threshold_candidates": [0.25, 0.5, 0.75],
            "calibrate_per_token_thresholds": True,
        }
    )
    assert summary["schema"] == "factorized_masked_diffusion_idm_train_summary.v1"
    assert summary["status"] == "pass"
    assert summary["key_vocab_size"] >= 2
    assert summary["button_vocab_size"] >= 2
    assert summary["factorization"]["mouse_axis_bins"] == 49
    assert summary["threshold_calibration"]["status"] == "pass"
    assert summary["threshold_calibration"]["selected"]["key_threshold"] in {0.25, 0.5, 0.75}
    assert summary["threshold_calibration"]["per_token"]["status"] == "pass"
    assert summary["factorization"]["key_token_threshold_count"] == summary["key_vocab_size"]
    assert Path(summary["checkpoint_path"]).exists()
    metrics = read_json(summary["metrics_path"])
    assert metrics["status"] == "pass"
    assert metrics["alignment"]["rows_seen"] == 3
    assert "typed masked action-token planes" in summary["recipe_alignment"]


def test_train_factorized_masked_diffusion_idm_luma_cnn_tiny_smoke(tmp_path: Path):
    if not torch_available():
        return
    train_path = tmp_path / "train_luma_factorized.jsonl"
    target_path = tmp_path / "target_luma_factorized.jsonl"
    train_rows = []
    for i in range(8):
        row = _row(i, split="train_core")
        row["compact_luma_window"] = [[float(i + j + k) / 10.0 for k in range(4)] for j in range(2)]
        row["compact_luma_window_mask"] = [1.0, 1.0]
        if i in {1, 3}:
            row["ground_truth_tokens"] = ["MOUSE_LEFT_DOWN", "KEY_PRESS_A", "MOUSE_DX_P1", "MOUSE_DY_Z0"]
        train_rows.append(row)
    target_rows = []
    for i in range(8, 11):
        row = _row(i, split="eval")
        row["compact_luma_window"] = [[float(i + j + k) / 10.0 for k in range(4)] for j in range(2)]
        row["compact_luma_window_mask"] = [1.0, 1.0]
        target_rows.append(row)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(target_path, target_rows)
    summary = train_masked_diffusion_idm(
        {
            "model_name": "unit_factorized_masked_diffusion_idm_luma_cnn",
            "factorized_action_tokens": True,
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "out_luma_factorized"),
            "summary_out": str(tmp_path / "summary_luma_factorized.json"),
            "max_train_rows": 8,
            "max_target_rows": 3,
            "video_feature_paths": ["compact_luma_window", "compact_luma_window_mask", "frame.features"],
            "video_feature_dim": 12,
            "video_encoder_arch": "compact_luma_window_cnn",
            "luma_window_frames": 2,
            "luma_window_size": 2,
            "luma_encoder_channels": 4,
            "luma_encoder_pool_hw": 1,
            "luma_aux_hidden_dim": 4,
            "hidden_dim": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "force_cpu": True,
            "key_threshold": 0.99,
            "button_threshold": 0.99,
            "calibrate_thresholds": True,
            "factorized_calibration_fraction": 0.25,
            "factorized_calibration_max_rows": 2,
            "threshold_candidates": [0.25, 0.5, 0.75],
            "calibrate_per_token_thresholds": True,
        }
    )
    assert summary["status"] == "pass"
    assert summary["threshold_calibration"]["status"] == "pass"
    assert Path(summary["checkpoint_path"]).exists()
    assert read_json(summary["metrics_path"])["alignment"]["rows_seen"] == 3
