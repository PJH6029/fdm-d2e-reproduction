from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdm_d2e.data.frame_embedding_materializer import (
    FrameEmbeddingMaterializerConfig,
    materialize_frame_embedding_features,
    parse_offsets,
)
from fdm_d2e.training.neural_idm import record_features
from fdm_d2e.training.streaming_idm import scan_streaming_idm_stats


def _write_ppm(path: Path, *, value: int, size: int = 4) -> None:
    payload = bytes([value, max(0, value - 5), min(255, value + 5)] * size * size)
    path.write_bytes(f"P6\n{size} {size}\n255\n".encode("ascii") + payload)


def _compact_fields(value: float) -> dict:
    grid = [value for _ in range(8 * 8 * 3)]
    luma = [value for _ in range(16 * 16)]
    return {"grid8": grid, "luma16": luma}


def _row(frame_path: Path, next_value: float, *, idx: int) -> dict:
    frame = {"path": str(frame_path), "features": [1, 2, 3, 4, 5], **_compact_fields(0.1 + idx)}
    return {
        "sequence_id": f"seq-{idx}",
        "recording_id": "rec-a",
        "timestamp_ns": idx * 50_000_000,
        "bin_index": idx,
        "game": "test",
        "split": "target",
        "frame": frame,
        "next_frame_features": [2, 3, 4, 5, 6],
        "frame_delta_features": [1, 1, 1, 1, 1],
        "next_frame_grid8": [next_value for _ in range(8 * 8 * 3)],
        "next_frame_luma16": [next_value for _ in range(16 * 16)],
        "prior_action_tokens": ["KEY_DOWN_W"],
        "prior_key_hold_bins": {"W": 3},
        "prior_button_hold_bins": {},
        "prior_since_key_transition_bins": 3,
        "prior_since_button_transition_bins": 7,
        "previous_event_tokens": ["KEY_PRESS_W"],
        "ground_truth_tokens": ["KEY_PRESS_W", "MOUSE_DX_P1"],
        "eval_split_tags": ["temporal"],
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_parse_offsets_rejects_empty_and_duplicates() -> None:
    assert parse_offsets("0,2,-1") == (0, 2, -1)
    with pytest.raises(ValueError, match="must not be empty"):
        parse_offsets("")
    with pytest.raises(ValueError, match="unique"):
        parse_offsets("0,1,1")


def test_dummy_stat_materializer_writes_streaming_feature_overrides(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for idx, value in enumerate([20, 40, 60]):
        _write_ppm(frames / f"frame_{idx:04d}.ppm", value=value)
    rows = [
        _row(frames / "frame_0000.ppm", 0.2, idx=0),
        _row(frames / "frame_0001.ppm", 0.3, idx=1),
    ]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    output_path = tmp_path / "out.jsonl"
    summary_path = tmp_path / "summary.json"
    progress_path = tmp_path / "progress.json"

    summary = materialize_frame_embedding_features(
        FrameEmbeddingMaterializerConfig(
            input_path=input_path,
            output_path=output_path,
            summary_out=summary_path,
            backend="dummy-stat",
            frame_offsets=(0, 1),
            image_size=4,
            batch_size=1,
            progress_output=progress_path,
            progress_rows=1,
        )
    )

    assert summary["status"] == "pass"
    assert summary["rows_written"] == 2
    assert summary["feature_override_rows"] == 2
    assert summary["embedding_dim_per_frame"] == 12
    assert summary["missing_frames"] == 0
    out_rows = _read_jsonl(output_path)
    assert len(out_rows) == 2
    base_len = len(record_features(rows[0], feature_mode="summary_compact_luma16_pair_shift_time_state_duration_prior_action"))
    # two frame embeddings + one embedding delta + existing compact/state context
    assert len(out_rows[0]["__streaming_idm_features"]) == 12 * 3 + base_len
    assert out_rows[0]["ground_truth_tokens"] == rows[0]["ground_truth_tokens"]
    metadata = out_rows[0]["frame_embedding_feature_metadata"]
    assert metadata["backend"] == "dummy-stat"
    assert metadata["frame_offsets"] == [0, 1]
    assert metadata["feature_dim"] == len(out_rows[0]["__streaming_idm_features"])
    stats = scan_streaming_idm_stats(output_path, feature_mode="summary")
    assert stats["input_dim"] == len(out_rows[0]["__streaming_idm_features"])
    assert json.loads(summary_path.read_text())["claim_boundary"].startswith("Frozen frame-embedding materialization")
    assert json.loads(progress_path.read_text())["status"] == "pass"


def test_dummy_stat_materializer_detects_missing_frames_when_zero_policy(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_ppm(frames / "frame_0000.ppm", value=20)
    rows = [_row(frames / "frame_0000.ppm", 0.2, idx=0)]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(rows[0], sort_keys=True) + "\n")
    output_path = tmp_path / "out.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = materialize_frame_embedding_features(
        FrameEmbeddingMaterializerConfig(
            input_path=input_path,
            output_path=output_path,
            summary_out=summary_path,
            backend="dummy-stat",
            frame_offsets=(0, 2),
            image_size=4,
            missing_frame_policy="zero",
            include_summary_features=False,
        )
    )

    assert summary["status"] == "pass"
    assert summary["missing_frames"] == 1
    out_row = _read_jsonl(output_path)[0]
    # two embeddings + delta, no existing compact/state features
    assert len(out_row["__streaming_idm_features"]) == 12 * 3
