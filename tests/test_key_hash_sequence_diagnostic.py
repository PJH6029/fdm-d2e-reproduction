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
