from __future__ import annotations

import json
from pathlib import Path

from scripts.migrate_video_idm_cache_keysoftmax import migrate_cache

from fdm_d2e.training.video_idm import (
    load_video_idm_cache_manifests,
    _VideoFrameStream,
    _select_button_softmax_threshold,
    precompute_video_idm_cache,
    scan_video_idm_stats,
    predict_video_idm_checkpoint,
    train_video_idm,
)
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


def test_button_softmax_calibration_can_enforce_no_button_fpr_cap():
    counts = {
        0.3: {
            "tp": 9,
            "fp": 2,
            "fn": 1,
            "predicted_positive": 11,
            "no_button_examples": 100,
            "no_button_false_positive_examples": 7,
        },
        0.5: {
            "tp": 6,
            "fp": 2,
            "fn": 4,
            "predicted_positive": 8,
            "no_button_examples": 100,
            "no_button_false_positive_examples": 2,
        },
        0.7: {
            "tp": 2,
            "fp": 0,
            "fn": 8,
            "predicted_positive": 2,
            "no_button_examples": 100,
            "no_button_false_positive_examples": 0,
        },
    }

    unconstrained, unconstrained_stats = _select_button_softmax_threshold(
        counts,
        default_threshold=0.5,
        beta=1.0,
    )
    constrained, constrained_stats = _select_button_softmax_threshold(
        counts,
        default_threshold=0.5,
        beta=1.0,
        max_no_button_fpr=0.05,
    )

    assert unconstrained == 0.3
    assert unconstrained_stats["no_button_false_positive_rate"] == 0.07
    assert constrained == 0.5
    assert constrained_stats["constraint_satisfied"] is True
    assert constrained_stats["no_button_false_positive_rate"] == 0.02


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
        "keyboard_head_mode": "softmax",
        "keyboard_softmax_min_count": 1,
        "keyboard_softmax_calibration_max_no_key_fpr": 0.5,
        "require_precomputed_video_cache": True,
    }

    cache_summary = precompute_video_idm_cache(config)
    assert cache_summary["status"] == "pass"
    assert cache_summary["train_cache"]["rows"] == len(train_rows)
    assert cache_summary["target_cache"]["rows"] == len(target_rows)
    train_only_summary = precompute_video_idm_cache(
        {
            **config,
            "cache_summary_out": str(tmp_path / "cache_summary_train_only.json"),
            "video_cache_precompute_splits": ["train"],
        }
    )
    assert train_only_summary["requested_splits"] == ["train"]
    assert train_only_summary["train_cache"]["precomputed"] is True
    assert train_only_summary["target_cache"]["precomputed"] is False
    assert train_only_summary["target_cache"]["rows"] == len(target_rows)

    summary = train_video_idm(config)
    assert summary["metadata"]["train_records"] == len(train_rows)
    assert summary["metadata"]["target_records"] == len(target_rows)
    assert Path(summary["metadata"]["checkpoint_path"]).exists()
    assert Path(summary["metadata"]["train_state_path"]).exists()
    assert "keyboard_softmax_threshold" in summary["metadata"]["calibration"]
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

    recalibrated_summary = predict_video_idm_checkpoint(
        {
            **config,
            "checkpoint_path": summary["metadata"]["checkpoint_path"],
            "output_dir": str(tmp_path / "predict_recalibrated"),
            "max_target_examples": 1,
            "recalibrate_from_train_cache": True,
            "button_softmax_calibration_max_no_button_fpr": 0.0,
        }
    )
    assert recalibrated_summary["recalibration"]["button_softmax_threshold_diagnostics"][
        "max_no_button_false_positive_rate"
    ] == 0.0

    resumed_summary = train_video_idm({**config, "epochs": 2, "skip_prediction": True})
    assert resumed_summary["resumed_from_train_state"] is True
    assert resumed_summary["start_epoch"] == 1
    assert len(resumed_summary["metadata"]["calibration"]) > 0


def test_keysoftmax_cache_migration_reuses_frame_payloads_and_rewrites_labels(tmp_path: Path):
    if not torch_available():
        return
    import torch

    frame_dir = tmp_path / "frames"
    for idx in range(1, 12):
        _write_ppm(frame_dir / f"frame_{idx:06d}.ppm", value=40 + idx)
    train_rows = [_record(idx, split="train_core", frame_dir=frame_dir) for idx in range(8)]
    target_rows = [_record(idx, split="eval", frame_dir=frame_dir) for idx in range(8, 10)]
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(target_path, target_rows)

    source_config = {
        "model_name": "unit_video_pair_idm_multilabel",
        "train_records": str(train_path),
        "target_records": str(target_path),
        "output_dir": str(tmp_path / "source_out"),
        "summary_out": str(tmp_path / "source_summary.json"),
        "stats_path": str(tmp_path / "source_out" / "video_idm_stats.json"),
        "cache_summary_out": str(tmp_path / "source_cache_summary.json"),
        "video_cache_dir": str(tmp_path / "source_cache"),
        "video_image_size": 8,
        "video_frame_fps": 20,
        "video_cache_chunk_size": 3,
        "video_cache_num_workers": 1,
        "video_input_mode": "pair_delta_abs",
        "missing_frame_policy": "zero",
        "force_cpu": True,
        "seed": 9,
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
    target_config = {
        **source_config,
        "model_name": "unit_video_pair_idm_keysoftmax",
        "output_dir": str(tmp_path / "target_out"),
        "summary_out": str(tmp_path / "target_summary.json"),
        "stats_path": str(tmp_path / "target_out" / "video_idm_stats.json"),
        "cache_summary_out": str(tmp_path / "target_cache_summary.json"),
        "video_cache_dir": str(tmp_path / "target_cache"),
        "keyboard_head_mode": "softmax",
        "keyboard_softmax_min_count": 1,
    }

    precompute_video_idm_cache(source_config)
    summary = migrate_cache(source_config, target_config)
    assert summary["status"] == "pass"
    assert summary["train_cache"]["rows"] == len(train_rows)
    assert summary["target_cache"]["rows"] == len(target_rows)
    assert summary["target_keyboard_classes"] >= 2

    source_stats = scan_video_idm_stats([train_path], config=source_config)
    target_stats = scan_video_idm_stats([train_path], config=target_config)
    assert "KEY_PRESS_87" in source_stats["category_vocab"]
    assert "KEY_PRESS_87" not in target_stats["category_vocab"]
    assert target_stats["keyboard_head_mode"] == "softmax"

    source_manifest = load_video_idm_cache_manifests(
        [train_path], stats=source_stats, config=source_config, split_name="train"
    )[0]
    target_manifest = load_video_idm_cache_manifests(
        [train_path], stats=target_stats, config=target_config, split_name="train"
    )[0]
    source_payload = torch.load(source_manifest["chunks"][0]["path"], map_location="cpu", weights_only=False)
    target_payload = torch.load(target_manifest["chunks"][0]["path"], map_location="cpu", weights_only=False)

    assert target_payload["payload_source_path"] == source_manifest["chunks"][0]["path"]
    assert "frames" not in target_payload
    assert "aux" not in target_payload
    assert "mouse_y" not in target_payload
    assert "button_y" not in target_payload
    assert target_payload["cat_y"].shape[1] == len(target_stats["category_vocab"])
    assert target_payload["keyboard_y"].dtype == torch.long
    assert target_manifest["chunks"][0]["bytes"] < source_manifest["chunks"][0]["bytes"]

    keyboard_classes = [tuple(row) for row in target_stats["keyboard_classes"]]
    expected_first_class = keyboard_classes.index(("KEY_PRESS_87",))
    expected_second_class = keyboard_classes.index(())
    assert int(target_payload["keyboard_y"][0]) == expected_first_class
    assert int(target_payload["keyboard_y"][1]) == expected_second_class

    summary = train_video_idm(target_config)
    assert summary["metadata"]["train_records"] == len(train_rows)
    assert Path(summary["prediction"]["predictions_path"]).exists()


def test_video_idm_stack_offsets_parallel_prediction_merges_parts(tmp_path: Path):
    if not torch_available():
        return
    frame_dir = tmp_path / "frames"
    for idx in range(1, 14):
        _write_ppm(frame_dir / f"frame_{idx:06d}.ppm", value=60 + idx)
    train_rows = [_record(idx, split="train_core", frame_dir=frame_dir) for idx in range(6)]
    target_a_rows = [_record(idx, split="eval", frame_dir=frame_dir) for idx in range(6, 8)]
    target_b_rows = [_record(idx, split="eval", frame_dir=frame_dir) for idx in range(8, 10)]
    train_path = tmp_path / "train.jsonl"
    target_a_path = tmp_path / "target_a.jsonl"
    target_b_path = tmp_path / "target_b.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(target_a_path, target_a_rows)
    _write_jsonl(target_b_path, target_b_rows)

    config = {
        "model_name": "unit_video_stack_idm",
        "train_records": str(train_path),
        "target_record_paths": [str(target_a_path), str(target_b_path)],
        "output_dir": str(tmp_path / "out"),
        "summary_out": str(tmp_path / "summary.json"),
        "stats_path": str(tmp_path / "out" / "video_idm_stats.json"),
        "cache_summary_out": str(tmp_path / "cache_summary.json"),
        "video_cache_dir": str(tmp_path / "cache"),
        "video_image_size": 8,
        "video_frame_fps": 20,
        "video_frame_offsets": [0, 1, 2],
        "next_frame_offset": 2,
        "video_cache_chunk_size": 2,
        "video_cache_num_workers": 1,
        "video_input_mode": "stack_delta_abs",
        "missing_frame_policy": "zero",
        "force_cpu": True,
        "seed": 11,
        "video_conv_channels": [4, 8],
        "hidden_dim": 16,
        "depth": 1,
        "epochs": 1,
        "batch_size": 2,
        "eval_batch_size": 2,
        "categorical_min_count": 1,
        "button_head_mode": "softmax",
        "keyboard_head_mode": "softmax",
        "keyboard_softmax_min_count": 1,
        "mouse_target_mode": "sum",
        "mouse_head_mode": "regression",
        "mouse_emit_mode": "decompose",
        "category_calibration_max_examples": 4,
        "category_calibration_batch_size": 2,
        "require_precomputed_video_cache": True,
    }
    precompute_video_idm_cache(config)
    train_summary = train_video_idm({**config, "skip_prediction": True})
    assert train_summary["metadata"]["distributed"]["world_size"] == 1

    predict_summary = predict_video_idm_checkpoint(
        {
            **config,
            "checkpoint_path": train_summary["metadata"]["checkpoint_path"],
            "output_dir": str(tmp_path / "predict_parallel"),
            "prediction_workers": 2,
            "prediction_parts_dir": str(tmp_path / "predict_parallel_parts"),
        }
    )
    assert predict_summary["target_records"] == len(target_a_rows) + len(target_b_rows)
    assert predict_summary["prediction"]["prediction_parallel"]["enabled"] is True
    prediction_lines = Path(predict_summary["prediction"]["predictions_path"]).read_text(encoding="utf-8").strip().splitlines()
    predicted_ids = [json.loads(line)["sequence_id"] for line in prediction_lines]
    assert predicted_ids == [row["sequence_id"] for row in [*target_a_rows, *target_b_rows]]
