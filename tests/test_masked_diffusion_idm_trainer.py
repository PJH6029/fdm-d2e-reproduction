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
        }
    )
    assert summary["status"] == "pass"
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
