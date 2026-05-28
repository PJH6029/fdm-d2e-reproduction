from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.joint_key_state_diagnostic import build_joint_key_state_diagnostic, sequence_bin_index
from fdm_d2e.io_utils import write_jsonl


def test_joint_key_state_diagnostic_preserves_repeated_key_counts(tmp_path: Path) -> None:
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
    write_jsonl(train, [row] * 4)
    target_row = {**row, "sequence_id": "target#000000", "recording_id": "game/target"}
    write_jsonl(target, [target_row])
    write_jsonl(base, [{"sequence_id": "target#000000", "predicted_tokens": ["KEY_PRESS_87"]}])

    payload = build_joint_key_state_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=4,
        max_target_rows=1,
        thresholds=[0.1],
        min_supports=[1],
    )

    assert payload["alignment"]["sequence_id_mismatches"] == 0
    assert payload["train_repeated_key_rows"] == 4
    assert payload["ranked_policies"][0]["keyboard_accuracy"] == 1.0
    assert any(item["policy"].startswith("joint_union_") for item in payload["ranked_policies"][:10])


def test_joint_key_state_diagnostic_can_drop_repeated_false_positive(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    row = {
        "sequence_id": "train#000000",
        "recording_id": "game/train",
        "prior_key_hold_bins": {"65": 3},
        "prior_since_key_transition_bins": 9,
        "previous_event_tokens": ["KEY_RELEASE_65"],
        "prior_action_tokens": ["KEY_DOWN_65"],
        "ground_truth_tokens": [],
    }
    write_jsonl(train, [row] * 5)
    target_row = {**row, "sequence_id": "target#000000", "recording_id": "game/target"}
    write_jsonl(target, [target_row])
    write_jsonl(base, [{"sequence_id": "target#000000", "predicted_tokens": ["KEY_PRESS_65"]}])

    payload = build_joint_key_state_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=5,
        max_target_rows=1,
        thresholds=[0.5],
        min_supports=[1],
    )

    base = payload["policies"]["base_all"]["summary"]["keyboard_accuracy"]
    empty_replace = [
        policy for name, policy in payload["policies"].items()
        if name.startswith("joint_replace_") and policy.get("usage", {}).get("empty_predictions") == 1
    ]
    assert base == 0.0
    assert empty_replace


def test_joint_key_state_diagnostic_reports_alignment_mismatch(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    write_jsonl(train, [{"sequence_id": "train#000000", "ground_truth_tokens": []}])
    write_jsonl(target, [{"sequence_id": "target#000000", "ground_truth_tokens": []}])
    write_jsonl(base, [{"sequence_id": "other#000000", "predicted_tokens": []}])

    payload = build_joint_key_state_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        thresholds=[0.5],
        min_supports=[1],
    )

    assert payload["alignment"]["sequence_id_mismatches"] == 1


def test_joint_key_state_diagnostic_can_limit_lookup_names(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    row = {
        "sequence_id": "train#000000",
        "recording_id": "game/train",
        "prior_key_hold_bins": {"65": 2},
        "ground_truth_tokens": ["KEY_PRESS_65"],
    }
    write_jsonl(train, [row])
    write_jsonl(target, [{**row, "sequence_id": "target#000000", "recording_id": "game/target"}])
    write_jsonl(base, [{"sequence_id": "target#000000", "predicted_tokens": []}])

    payload = build_joint_key_state_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        thresholds=[0.1],
        min_supports=[1],
        lookup_names=["held_codes_only", "chain:duration_to_codes"],
    )

    assert payload["lookup_names"] == ["held_codes_only", "chain:duration_to_codes"]
    assert "game_held_mod_since_phase" not in payload["context_names"]
    assert set(payload["context_names"]) >= {"held_codes_only", "held_mod_since_phase"}


def test_sequence_bin_index_parses_suffix() -> None:
    assert sequence_bin_index({"sequence_id": "rec#000123"}) == 123
