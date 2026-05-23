from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from synthesize_streaming_idm_stats import synthesize_streaming_idm_stats


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_synthesize_streaming_idm_stats_reuses_corpus_provenance_with_unit_normalizer(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        [
            {
                "sequence_id": "seq-1",
                "recording_id": "rec",
                "game": "Apex",
                "timestamp_ns": 1,
                "bin_index": 1,
                "frame": {
                    "features": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "grid8": [0.1] * (8 * 8 * 3),
                    "luma16": [0.2] * (16 * 16),
                },
                "next_frame_features": [0.2, 0.3, 0.4, 0.5, 0.6],
                "frame_delta_features": [0.1, 0.1, 0.1, 0.1, 0.1],
                "next_frame_grid8": [0.2] * (8 * 8 * 3),
                "next_frame_luma16": [0.3] * (16 * 16),
                "ground_truth_tokens": ["KEY_PRESS_87", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
            }
        ],
    )
    source_stats = tmp_path / "source_stats.json"
    _write_json(
        source_stats,
        {
            "schema": "streaming_idm_stats.v1",
            "train_records": ["old.jsonl"],
            "num_examples": 123,
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "input_dim": 620,
            "mean": [0.0] * 620,
            "std": [1.0] * 620,
            "category_vocab": ["KEY_PRESS_87"],
            "category_counts": {"KEY_PRESS_87": 7},
            "global_majority_tokens": ["NOOP"],
            "last_tokens_by_recording": {"rec": [1, ["KEY_PRESS_87"]]},
            "last_tokens_by_game": {"Apex": [1, ["KEY_PRESS_87"]]},
            "source_ids": ["d2e_480p"],
            "resolution_tiers": ["480p"],
            "split_names": ["train_core"],
            "eval_split_tags": [],
            "dataset_fingerprint": "old-fingerprint",
        },
    )

    stats = synthesize_streaming_idm_stats(
        {
            "train_records": str(train),
            "feature_mode": "summary_compact_luma16_pair_shift_time",
        },
        source_stats_path=source_stats,
    )

    assert stats["num_examples"] == 123
    assert stats["feature_mode"] == "summary_compact_luma16_pair_shift_time"
    assert stats["input_dim"] == 812
    assert len(stats["mean"]) == 812
    assert len(stats["std"]) == 812
    assert set(stats["mean"]) == {0.0}
    assert set(stats["std"]) == {1.0}
    assert stats["category_vocab"] == ["KEY_PRESS_87"]
    assert stats["category_counts"] == {"KEY_PRESS_87": 7}
    assert stats["stats_synthesis"]["status"] == "synthetic_unit_normalizer"
    assert stats["stats_synthesis"]["source_dataset_fingerprint"] == "old-fingerprint"
    assert stats["dataset_fingerprint"] != "old-fingerprint"


def test_synthesize_streaming_idm_stats_accounts_for_action_history_dim(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        [
            {
                "sequence_id": "seq-1",
                "recording_id": "rec",
                "game": "Apex",
                "timestamp_ns": 1,
                "bin_index": 1,
                "frame": {
                    "features": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "grid8": [0.1] * (8 * 8 * 3),
                    "luma16": [0.2] * (16 * 16),
                },
                "next_frame_features": [0.2, 0.3, 0.4, 0.5, 0.6],
                "frame_delta_features": [0.1, 0.1, 0.1, 0.1, 0.1],
                "next_frame_grid8": [0.2] * (8 * 8 * 3),
                "next_frame_luma16": [0.3] * (16 * 16),
                "ground_truth_tokens": ["KEY_PRESS_87", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
            }
        ],
    )
    source_stats = tmp_path / "source_stats.json"
    _write_json(
        source_stats,
        {
            "schema": "streaming_idm_stats.v1",
            "train_records": ["old.jsonl"],
            "num_examples": 123,
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "input_dim": 620,
            "mean": [0.0] * 620,
            "std": [1.0] * 620,
            "category_vocab": ["KEY_PRESS_87", "MOUSE_LEFT_DOWN"],
            "category_counts": {"KEY_PRESS_87": 7, "MOUSE_LEFT_DOWN": 2},
            "dataset_fingerprint": "old-fingerprint",
        },
    )

    stats = synthesize_streaming_idm_stats(
        {
            "train_records": str(train),
            "feature_mode": "summary_compact_luma16_pair_shift_time",
            "action_history_len": 4,
        },
        source_stats_path=source_stats,
    )

    expected_history_dim = (2 * 4) + (2 * 4) + 3
    assert stats["action_history_len"] == 4
    assert stats["action_history_vocab"] == ["KEY_PRESS_87", "MOUSE_LEFT_DOWN"]
    assert stats["action_history_dim"] == expected_history_dim
    assert stats["input_dim"] == 812 + expected_history_dim
    assert len(stats["mean"]) == stats["input_dim"]
    assert stats["action_history_feedback"] == "teacher_forced_train"
