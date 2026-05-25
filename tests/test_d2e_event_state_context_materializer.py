from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import read_jsonl, write_jsonl
from fdm_d2e.training.neural_idm import record_features
from materialize_d2e_event_state_context_corpus import materialize_event_state_context_corpus


def test_event_state_context_preserves_event_targets_and_carries_prior_state(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    train = input_root / "shard_00/train_core.jsonl"
    target = input_root / "shard_00/target_all_eval.jsonl"
    write_jsonl(
        train,
        [
            {"sequence_id": "rec#0", "recording_id": "rec", "ground_truth_tokens": ["KEY_PRESS_87"]},
            {"sequence_id": "rec#1", "recording_id": "rec", "ground_truth_tokens": ["MOUSE_LEFT_DOWN"]},
        ],
    )
    write_jsonl(
        target,
        [
            {"sequence_id": "rec#2", "recording_id": "rec", "ground_truth_tokens": ["KEY_RELEASE_87"]},
            {"sequence_id": "rec#3", "recording_id": "rec", "ground_truth_tokens": ["MOUSE_LEFT_UP"]},
        ],
    )

    summary = materialize_event_state_context_corpus(
        train_inputs=[train],
        target_inputs=[target],
        input_root=input_root,
        output_root=output_root,
        summary_path=tmp_path / "summary.json",
        workers=1,
    )
    train_rows = read_jsonl(output_root / "shard_00/train_core.jsonl")
    target_rows = read_jsonl(output_root / "shard_00/target_all_eval.jsonl")

    assert summary["status"] == "pass"
    assert train_rows[0]["ground_truth_tokens"] == ["KEY_PRESS_87"]
    assert train_rows[0]["prior_action_tokens"] == ["NOOP"]
    assert train_rows[1]["prior_action_tokens"] == ["KEY_DOWN_87"]
    assert target_rows[0]["ground_truth_tokens"] == ["KEY_RELEASE_87"]
    assert target_rows[0]["prior_action_tokens"] == ["KEY_DOWN_87", "MOUSE_LEFT_DOWN"]
    assert target_rows[1]["prior_action_tokens"] == ["MOUSE_LEFT_DOWN"]


def test_prior_action_luma_pair_feature_mode_adds_context_features() -> None:
    row = {
        "bin_index": 3,
        "frame": {"features": [0, 1, 2, 3, 4]},
        "next_frame_features": [1, 2, 3, 4, 5],
        "frame_delta_features": [1, 1, 1, 1, 1],
        "prior_action_tokens": ["KEY_DOWN_87", "MOUSE_LEFT_DOWN"],
    }
    base = record_features(row, feature_mode="summary_compact_luma16_pair_shift_time")
    with_prior = record_features(row, feature_mode="summary_compact_luma16_pair_shift_time_prior_action")

    assert len(with_prior) == len(base) + 38
    assert with_prior != [*base, *([0.0] * 38)]
