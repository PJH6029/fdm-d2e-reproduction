from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.key_event_taxonomy import build_key_event_taxonomy, classify_key_token, same_bin_extra_key_tokens
from fdm_d2e.io_utils import write_jsonl


def test_key_event_taxonomy_classifies_visible_and_same_bin_keys(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    rows = [
        {
            "sequence_id": "rec#0",
            "recording_id": "rec",
            "eval_split_tags": ["temporal"],
            "prior_action_tokens": ["NOOP"],
            "previous_event_tokens": ["MOUSE_DX_P1", "MOUSE_DY_Z0"],
            "ground_truth_tokens": ["KEY_PRESS_65", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
        },
        {
            "sequence_id": "rec#1",
            "recording_id": "rec",
            "eval_split_tags": ["temporal"],
            "prior_action_tokens": ["KEY_DOWN_65"],
            "previous_event_tokens": ["KEY_PRESS_65"],
            "ground_truth_tokens": ["KEY_PRESS_65", "KEY_RELEASE_65"],
        },
        {
            "sequence_id": "rec#2",
            "recording_id": "rec",
            "eval_split_tags": ["temporal"],
            "prior_action_tokens": ["NOOP"],
            "previous_event_tokens": ["KEY_PRESS_65", "KEY_RELEASE_65"],
            "ground_truth_tokens": ["NOOP"],
        },
    ]
    write_jsonl(target, rows)

    assert classify_key_token(rows[0], rows[1], "KEY_PRESS_65") == "visible_new_press"
    assert classify_key_token(rows[1], rows[2], "KEY_PRESS_65") == "press_then_release_while_held"
    assert classify_key_token(rows[1], rows[2], "KEY_RELEASE_65") == "visible_release"
    assert same_bin_extra_key_tokens(rows[1], rows[2]) == ["KEY_PRESS_65"]

    payload = build_key_event_taxonomy(target_paths=[target], max_rows=3)

    assert payload["rows"] == 3
    assert payload["total_key_tokens"] == 3
    assert payload["category_counts"]["visible_new_press"] == 1
    assert payload["category_counts"]["visible_release"] == 1
    assert payload["category_counts"]["press_then_release_while_held"] == 1
    metrics = payload["policy_metrics"]["state_delta_plus_same_bin_key_oracle_plus_previous_motion"]["all"]["paper_compatible"]
    assert metrics["keyboard"]["key_accuracy"] == 1.0


def test_key_event_taxonomy_identifies_same_bin_tap_from_no_state_change(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    rows = [
        {
            "sequence_id": "rec#0",
            "recording_id": "rec",
            "eval_split_tags": ["heldout_recording"],
            "prior_action_tokens": ["NOOP"],
            "previous_event_tokens": ["NOOP"],
            "ground_truth_tokens": ["KEY_PRESS_32", "KEY_RELEASE_32"],
        },
        {
            "sequence_id": "rec#1",
            "recording_id": "rec",
            "eval_split_tags": ["heldout_recording"],
            "prior_action_tokens": ["NOOP"],
            "previous_event_tokens": ["KEY_PRESS_32", "KEY_RELEASE_32"],
            "ground_truth_tokens": ["NOOP"],
        },
    ]
    write_jsonl(target, rows)

    payload = build_key_event_taxonomy(target_paths=[target])

    assert payload["category_counts"]["same_bin_tap_press"] == 1
    assert payload["category_counts"]["same_bin_tap_release"] == 1
    assert payload["hidden_or_repeat_key_tokens"] == 2
    assert payload["split_category_counts"]["heldout_recording"]["same_bin_tap_press"] == 1
