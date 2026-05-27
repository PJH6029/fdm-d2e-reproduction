from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_jsonl
from fdm_d2e.training.streaming_idm import (
    _ClosedLoopStateContextTracker,
    _closed_loop_state_context_enabled,
    _seed_closed_loop_state_context_from_records,
)


def test_closed_loop_state_context_overrides_leaked_target_fields() -> None:
    tracker = _ClosedLoopStateContextTracker()
    leaked = {
        "sequence_id": "rec#0",
        "recording_id": "rec",
        "prior_action_tokens": ["KEY_DOWN_999", "MOUSE_LEFT_DOWN"],
        "prior_key_hold_bins": {"999": 8},
        "prior_button_hold_bins": {"LEFT": 8},
        "prior_since_key_transition_bins": 7,
        "previous_event_tokens": ["KEY_PRESS_999"],
    }

    causal = tracker.row_with_prior_context(leaked)

    assert causal["prior_action_tokens"] == ["NOOP"]
    assert causal["prior_action_source"] == "predicted_closed_loop_before_current_event_bin"
    assert causal["prior_key_hold_bins"] == {}
    assert causal["prior_button_hold_bins"] == {}
    assert causal["prior_since_key_transition_bins"] is None
    assert causal["previous_event_tokens"] == ["NOOP"]


def test_closed_loop_state_context_updates_next_row_from_predictions() -> None:
    tracker = _ClosedLoopStateContextTracker()
    row0 = {"sequence_id": "rec#0", "recording_id": "rec"}
    tracker.observe_tokens(row0, ["KEY_PRESS_87", "MOUSE_LEFT_DOWN"])

    row1 = tracker.row_with_prior_context({"sequence_id": "rec#1", "recording_id": "rec"})
    assert row1["prior_action_tokens"] == ["KEY_DOWN_87", "MOUSE_LEFT_DOWN"]
    assert row1["prior_key_hold_bins"] == {"87": 1}
    assert row1["prior_button_hold_bins"] == {"LEFT": 1}
    assert row1["prior_since_key_transition_bins"] == 0
    assert row1["prior_since_button_transition_bins"] == 0
    assert row1["previous_event_tokens"] == ["KEY_PRESS_87", "MOUSE_LEFT_DOWN"]

    tracker.observe_tokens(row1, ["NOOP"])
    row2 = tracker.row_with_prior_context({"sequence_id": "rec#2", "recording_id": "rec"})
    assert row2["prior_action_tokens"] == ["KEY_DOWN_87", "MOUSE_LEFT_DOWN"]
    assert row2["prior_key_hold_bins"] == {"87": 2}
    assert row2["prior_button_hold_bins"] == {"LEFT": 2}
    assert row2["prior_since_key_transition_bins"] == 1
    assert row2["prior_since_button_transition_bins"] == 1
    assert row2["previous_event_tokens"] == ["NOOP"]


def test_closed_loop_state_context_seed_from_train_records(tmp_path: Path) -> None:
    train = tmp_path / "train_core.jsonl"
    write_jsonl(
        train,
        [
            {"sequence_id": "rec#0", "recording_id": "rec", "ground_truth_tokens": ["KEY_PRESS_87"]},
            {"sequence_id": "rec#1", "recording_id": "rec", "ground_truth_tokens": ["NOOP"]},
        ],
    )

    tracker = _seed_closed_loop_state_context_from_records([train])
    target0 = tracker.row_with_prior_context({"sequence_id": "rec#2", "recording_id": "rec"})

    assert tracker.seed_rows == 2
    assert target0["prior_action_tokens"] == ["KEY_DOWN_87"]
    assert target0["prior_key_hold_bins"] == {"87": 2}
    assert target0["prior_since_key_transition_bins"] == 1
    assert target0["previous_event_tokens"] == ["NOOP"]


def test_closed_loop_state_context_can_seed_first_target_prior_once() -> None:
    tracker = _ClosedLoopStateContextTracker()
    first_target = {
        "sequence_id": "rec#target0",
        "recording_id": "rec",
        "prior_action_tokens": ["KEY_DOWN_87", "MOUSE_LEFT_DOWN"],
        "prior_key_hold_bins": {"87": 5},
        "prior_button_hold_bins": {"LEFT": 3},
        "prior_since_key_transition_bins": 2,
        "prior_since_button_transition_bins": 1,
        "previous_event_tokens": ["KEY_PRESS_87"],
    }

    tracker.seed_recording_from_prior_context(first_target)
    seeded = tracker.row_with_prior_context({"sequence_id": "rec#target0", "recording_id": "rec"})

    assert tracker.target_prior_seed_recordings == 1
    assert seeded["prior_action_tokens"] == ["KEY_DOWN_87", "MOUSE_LEFT_DOWN"]
    assert seeded["prior_key_hold_bins"] == {"87": 5}
    assert seeded["prior_button_hold_bins"] == {"LEFT": 3}
    assert seeded["prior_since_key_transition_bins"] == 2
    assert seeded["prior_since_button_transition_bins"] == 1
    assert seeded["previous_event_tokens"] == ["KEY_PRESS_87"]

    tracker.observe_tokens(seeded, ["KEY_RELEASE_87"])
    tracker.seed_recording_from_prior_context({**first_target, "prior_key_hold_bins": {"999": 99}})
    after_prediction = tracker.row_with_prior_context({"sequence_id": "rec#target1", "recording_id": "rec"})

    assert tracker.target_prior_seed_recordings == 1
    assert after_prediction["prior_action_tokens"] == ["MOUSE_LEFT_DOWN"]
    assert after_prediction["prior_key_hold_bins"] == {}


def test_closed_loop_state_context_requires_prior_action_feature_mode() -> None:
    assert _closed_loop_state_context_enabled(
        {"closed_loop_state_context": True},
        {"feature_mode": "summary_compact_luma16_pair_shift_time_prior_action"},
    )

    try:
        _closed_loop_state_context_enabled(
            {"closed_loop_state_context": True},
            {"feature_mode": "summary_compact_luma16_pair_time"},
        )
    except ValueError as exc:
        assert "prior_action" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("closed_loop_state_context should reject non-prior-action feature modes")
