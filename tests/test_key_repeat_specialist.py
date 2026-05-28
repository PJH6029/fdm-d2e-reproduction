from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.key_repeat_specialist import build_key_repeat_specialist_matrix
from fdm_d2e.io_utils import write_jsonl


def test_key_repeat_specialist_can_add_held_repeat_key(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    train_rows = []
    for i in range(12):
        train_rows.append(
            {
                "sequence_id": f"train#{i}",
                "recording_id": "train",
                "prior_key_hold_bins": {"65": 10},
                "prior_since_key_transition_bins": 3,
                "ground_truth_tokens": ["KEY_PRESS_65"] if i % 2 == 0 else ["MOUSE_DX_Z0", "MOUSE_DY_Z0"],
            }
        )
    target_rows = [
        {
            "sequence_id": "target#0",
            "recording_id": "target",
            "eval_split_tags": ["temporal"],
            "prior_key_hold_bins": {"65": 10},
            "prior_since_key_transition_bins": 3,
            "ground_truth_tokens": ["KEY_PRESS_65", "MOUSE_DX_Z0", "MOUSE_DY_Z0"],
        }
    ]
    write_jsonl(train, train_rows)
    write_jsonl(target, target_rows)
    write_jsonl(base, [{"sequence_id": "target#0", "predicted_tokens": ["MOUSE_DX_Z0", "MOUSE_DY_Z0"]}])

    payload = build_key_repeat_specialist_matrix(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=12,
        max_target_rows=1,
        press_thresholds=[0.4],
        release_thresholds=[0.9],
        min_support=1,
    )

    best = payload["policies"]["prior_state_forced_press0.4_release0.9_replace_base_keys"]["summary"]
    assert best["keyboard_accuracy"] == 1.0


def test_key_repeat_specialist_base_policy_is_preserved(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    write_jsonl(train, [{"sequence_id": "train#0", "recording_id": "train", "ground_truth_tokens": ["NOOP"]}])
    write_jsonl(
        target,
        [{"sequence_id": "target#0", "recording_id": "target", "ground_truth_tokens": ["KEY_PRESS_87"], "eval_split_tags": ["temporal"]}],
    )
    write_jsonl(base, [{"sequence_id": "target#0", "predicted_tokens": ["KEY_PRESS_87"]}])

    payload = build_key_repeat_specialist_matrix(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        press_thresholds=[0.5],
        release_thresholds=[0.5],
    )

    assert payload["policies"]["base_all"]["summary"]["keyboard_accuracy"] == 1.0


def test_key_repeat_specialist_reports_base_sequence_mismatch(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    write_jsonl(train, [{"sequence_id": "train#0", "recording_id": "train", "ground_truth_tokens": ["NOOP"]}])
    write_jsonl(target, [{"sequence_id": "target#0", "recording_id": "target", "ground_truth_tokens": ["NOOP"]}])
    write_jsonl(base, [{"sequence_id": "other#0", "predicted_tokens": ["NOOP"]}])

    payload = build_key_repeat_specialist_matrix(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        press_thresholds=[0.5],
        release_thresholds=[0.5],
    )

    assert payload["alignment"]["sequence_id_mismatches"] == 1
    assert payload["alignment"]["examples"][0]["prediction_sequence_id"] == "other#0"
