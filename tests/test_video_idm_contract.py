from __future__ import annotations

import json
from pathlib import Path

from fdm_d2e.training.video_idm import _VideoFrameStream, precompute_video_idm_cache, predict_video_idm_checkpoint, train_video_idm
from fdm_d2e.training.torch_idm import torch_available


def _write_ppm(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = height = 8
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend([(value + x) % 256, (value + y) % 256, value % 256])
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(pixels))


def _record(idx: int, *, split: str, frame_dir: Path) -> dict:
    tokens = ["MOUSE_DX_P1", "MOUSE_DY_Z0"]
    if idx % 2 == 0:
        tokens.append("KEY_PRESS_87")
    if idx % 3 == 0:
        tokens.append("MOUSE_LEFT_DOWN")
    return {
        "schema": "d2e_window_record.v1",
        "sequence_id": f"Apex/test#{idx:06d}",
        "recording_id": "Apex/test",
        "cross_resolution_key": "Apex/test",
        "game": "Apex",
        "source_id": "unit",
        "resolution_tier": "480p",
        "split": split,
        "eval_split_tags": ["temporal"] if split == "eval" else [],
        "timestamp_ns": idx * 50_000_000,
        "bin_index": idx,
        "frame": {
            "path": str(frame_dir / f"frame_{idx + 1:06d}.ppm"),
            "index": idx,
            "features": [],
        },
        "events": [],
        "ground_truth_tokens": tokens,
        "source": "unit",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_video_frame_stream_reuses_next_frame_without_restart():
    stream = _VideoFrameStream("unused.mkv", image_size=1, fps=20)
    calls: list[int] = []

    def fake_read_next() -> bytes:
        frame_index = stream.current_index
        calls.append(frame_index)
        frame = bytes([frame_index])
        stream.current_index += 1
        stream.last_frame = frame
        stream.cache[frame_index] = frame
        return frame

    stream._read_next = fake_read_next  # type: ignore[method-assign]
    assert stream.get(0) == b"\x00"
    assert stream.get(1) == b"\x01"
    assert stream.get(1) == b"\x01"
    assert calls == [0, 1]


def test_video_idm_precompute_and_train_from_precomputed_cache(tmp_path: Path):
    if not torch_available():
        return
    frame_dir = tmp_path / "frames"
    for idx in range(1, 10):
        _write_ppm(frame_dir / f"frame_{idx:06d}.ppm", value=20 + idx)
    train_rows = [_record(idx, split="train_core", frame_dir=frame_dir) for idx in range(6)]
    target_rows = [_record(idx, split="eval", frame_dir=frame_dir) for idx in range(6, 8)]
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(target_path, target_rows)

    config = {
        "model_name": "unit_video_pair_idm",
        "train_records": str(train_path),
        "target_records": str(target_path),
        "output_dir": str(tmp_path / "out"),
        "summary_out": str(tmp_path / "summary.json"),
        "stats_path": str(tmp_path / "out" / "video_idm_stats.json"),
        "cache_summary_out": str(tmp_path / "cache_summary.json"),
        "video_cache_dir": str(tmp_path / "cache"),
        "video_image_size": 8,
        "video_frame_fps": 20,
        "video_cache_chunk_size": 2,
        "video_cache_num_workers": 1,
        "video_input_mode": "pair_delta_abs",
        "missing_frame_policy": "zero",
        "force_cpu": True,
        "seed": 7,
        "video_conv_channels": [4, 8],
        "hidden_dim": 16,
        "depth": 1,
        "epochs": 1,
        "batch_size": 2,
        "eval_batch_size": 2,
        "categorical_min_count": 1,
        "button_head_mode": "softmax",
        "mouse_target_mode": "sum",
        "mouse_head_mode": "regression",
        "mouse_emit_mode": "decompose",
        "category_calibration_max_examples": 4,
        "category_calibration_batch_size": 2,
        "require_precomputed_video_cache": True,
    }

    cache_summary = precompute_video_idm_cache(config)
    assert cache_summary["status"] == "pass"
    assert cache_summary["train_cache"]["rows"] == len(train_rows)
    assert cache_summary["target_cache"]["rows"] == len(target_rows)

    summary = train_video_idm(config)
    assert summary["metadata"]["train_records"] == len(train_rows)
    assert summary["metadata"]["target_records"] == len(target_rows)
    assert Path(summary["metadata"]["checkpoint_path"]).exists()
    assert Path(summary["prediction"]["predictions_path"]).exists()
    predictions = Path(summary["prediction"]["predictions_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(predictions) == len(target_rows)
    first = json.loads(predictions[0])
    assert first["sequence_id"] == target_rows[0]["sequence_id"]
    assert isinstance(first["predicted_tokens"], list)

    predict_dir = tmp_path / "predict_again"
    predict_summary = predict_video_idm_checkpoint(
        {
            **config,
            "checkpoint_path": summary["metadata"]["checkpoint_path"],
            "output_dir": str(predict_dir),
            "summary_out": str(tmp_path / "predict_summary.json"),
            "max_target_examples": 1,
        }
    )
    assert predict_summary["target_records"] == 1
    assert Path(predict_summary["prediction"]["predictions_path"]).exists()
