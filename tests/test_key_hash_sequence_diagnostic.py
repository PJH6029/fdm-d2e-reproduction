from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.key_hash_sequence_diagnostic import build_key_hash_sequence_diagnostic, sequence_bin_index
from fdm_d2e.io_utils import write_jsonl


def test_hash_sequence_diagnostic_learns_simple_repeat_pattern(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    train_rows = []
    for idx in range(40):
        train_rows.append(
            {
                "sequence_id": f"train#{idx:06d}",
                "recording_id": "game/rec",
                "prior_key_hold_bins": {"65": 10},
                "prior_since_key_transition_bins": 2,
                "ground_truth_tokens": ["KEY_PRESS_65"] if idx % 2 == 0 else [],
            }
        )
    target_rows = [
        {"sequence_id": "target#000000", "recording_id": "game/target", "prior_key_hold_bins": {"65": 10}, "prior_since_key_transition_bins": 2, "ground_truth_tokens": ["KEY_PRESS_65"]},
        {"sequence_id": "target#000001", "recording_id": "game/target", "prior_key_hold_bins": {"65": 10}, "prior_since_key_transition_bins": 2, "ground_truth_tokens": []},
    ]
    write_jsonl(train, train_rows)
    write_jsonl(target, target_rows)
    write_jsonl(base, [{"sequence_id": row["sequence_id"], "predicted_tokens": []} for row in target_rows])

    payload = build_key_hash_sequence_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=40,
        max_target_rows=2,
        epochs=2,
        dim=4096,
        learning_rate=0.1,
        press_thresholds=[0.2],
        release_thresholds=[0.9],
    )

    assert payload["alignment"]["sequence_id_mismatches"] == 0
    assert payload["ranked_policies"][0]["keyboard_accuracy"] is not None
    assert payload["model_positive_press"] > 0


def test_hash_sequence_diagnostic_reports_mismatches(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    write_jsonl(train, [{"sequence_id": "train#0", "ground_truth_tokens": []}])
    write_jsonl(target, [{"sequence_id": "target#0", "ground_truth_tokens": []}])
    write_jsonl(base, [{"sequence_id": "other#0", "predicted_tokens": []}])

    payload = build_key_hash_sequence_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        dim=1024,
    )

    assert payload["alignment"]["sequence_id_mismatches"] == 1


def test_sequence_bin_index_parses_suffix() -> None:
    assert sequence_bin_index({"sequence_id": "rec#000007"}) == 7


def test_hash_sequence_diagnostic_accepts_visual_hash_features(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    frame = {"luma16": [0.1] * 256}
    row = {
        "sequence_id": "train#000000",
        "recording_id": "game/train",
        "frame": frame,
        "next_frame_luma16": [0.2] * 256,
        "prior_key_hold_bins": {"65": 2},
        "ground_truth_tokens": ["KEY_PRESS_65"],
    }
    write_jsonl(train, [row])
    write_jsonl(target, [{**row, "sequence_id": "target#000000", "recording_id": "game/target"}])
    write_jsonl(base, [{"sequence_id": "target#000000", "predicted_tokens": []}])

    payload = build_key_hash_sequence_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        dim=2048,
        include_visual_hash=True,
        press_thresholds=[0.1],
        release_thresholds=[0.9],
    )

    assert payload["include_visual_hash"] is True
    assert payload["alignment"]["sequence_id_mismatches"] == 0


def test_hash_sequence_diagnostic_can_predict_top_vocab_nonheld_key(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    row = {
        "sequence_id": "train#000000",
        "recording_id": "game/train",
        "prior_key_hold_bins": {},
        "previous_event_tokens": ["NOOP"],
        "prior_action_tokens": ["NOOP"],
        "ground_truth_tokens": ["KEY_PRESS_65"],
    }
    write_jsonl(train, [row])
    write_jsonl(target, [{**row, "sequence_id": "target#000000", "recording_id": "game/target"}])
    write_jsonl(base, [{"sequence_id": "target#000000", "predicted_tokens": []}])

    payload = build_key_hash_sequence_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        dim=2048,
        learning_rate=0.2,
        candidate_key_count=1,
        press_thresholds=[0.1],
        release_thresholds=[0.9],
    )

    assert payload["candidate_key_codes"] == ["65"]
    assert payload["alignment"]["sequence_id_mismatches"] == 0
    assert payload["ranked_policies"][0]["keyboard_accuracy"] == 1.0


def test_hash_sequence_diagnostic_can_emit_double_press_count(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    row = {
        "sequence_id": "train#000000",
        "recording_id": "game/train",
        "prior_key_hold_bins": {"87": 8},
        "prior_since_key_transition_bins": 4,
        "previous_event_tokens": ["KEY_PRESS_87"],
        "prior_action_tokens": ["KEY_DOWN_87"],
        "ground_truth_tokens": ["KEY_PRESS_87", "KEY_PRESS_87"],
    }
    write_jsonl(train, [row])
    write_jsonl(target, [{**row, "sequence_id": "target#000000", "recording_id": "game/target"}])
    write_jsonl(base, [{"sequence_id": "target#000000", "predicted_tokens": ["KEY_PRESS_87"]}])

    payload = build_key_hash_sequence_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        dim=2048,
        learning_rate=0.2,
        press_thresholds=[0.9],
        release_thresholds=[0.99],
        double_press_thresholds=[0.1],
    )

    assert payload["model_positive_double_press"] == 1
    assert payload["alignment"]["sequence_id_mismatches"] == 0
    assert payload["ranked_policies"][0]["policy"].startswith("press_count_union_base_keys")
    assert payload["ranked_policies"][0]["keyboard_accuracy"] == 1.0
