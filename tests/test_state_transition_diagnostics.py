from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.state_transition_diagnostics import (
    build_causal_keyboard_repeat_policy_matrix,
    build_key_repeat_prior_metrics,
    build_state_delta_oracle_metrics,
    merge_motion_and_categorical,
    state_delta_tokens,
)
from fdm_d2e.io_utils import write_jsonl


def test_state_delta_tokens_from_next_prior_state() -> None:
    row = {"sequence_id": "rec#0", "prior_action_tokens": ["KEY_DOWN_87"]}
    nxt = {"sequence_id": "rec#1", "prior_action_tokens": ["KEY_DOWN_65", "MOUSE_LEFT_DOWN"]}

    assert state_delta_tokens(row, nxt) == ["KEY_PRESS_65", "KEY_RELEASE_87", "MOUSE_LEFT_DOWN"]


def test_merge_preserves_duplicate_motion_but_dedupes_categories() -> None:
    out = merge_motion_and_categorical(
        ["MOUSE_DX_P1", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
        ["KEY_PRESS_65", "KEY_PRESS_65", "MOUSE_LEFT_DOWN"],
    )

    assert out == ["MOUSE_DX_P1", "MOUSE_DX_P1", "MOUSE_DY_Z0", "KEY_PRESS_65", "MOUSE_LEFT_DOWN"]


def test_state_delta_oracle_and_repeat_prior_metrics(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    # Train prior says held KEY_65 at hold=5/since=2 usually repeats.
    write_jsonl(
        train,
        [
            {
                "sequence_id": f"train#{idx}",
                "prior_action_tokens": ["KEY_DOWN_65"],
                "prior_key_hold_bins": {"65": 5},
                "prior_since_key_transition_bins": 2,
                "ground_truth_tokens": ["KEY_PRESS_65"],
            }
            for idx in range(3)
        ],
    )
    write_jsonl(
        target,
        [
            {
                "sequence_id": "rec#0",
                "prior_action_tokens": ["KEY_DOWN_65"],
                "prior_key_hold_bins": {"65": 5},
                "prior_since_key_transition_bins": 2,
                "previous_event_tokens": ["MOUSE_DX_P1", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
                "ground_truth_tokens": [
                    "KEY_PRESS_65",
                    "MOUSE_LEFT_DOWN",
                    "MOUSE_DX_P1",
                    "MOUSE_DX_P1",
                    "MOUSE_DY_Z0",
                ],
            },
            {
                "sequence_id": "rec#1",
                "prior_action_tokens": ["KEY_DOWN_65", "MOUSE_LEFT_DOWN"],
                "prior_key_hold_bins": {},
                "prior_since_key_transition_bins": 0,
                "previous_event_tokens": ["MOUSE_DX_Z0", "MOUSE_DY_Z0"],
                "ground_truth_tokens": ["MOUSE_DX_Z0", "MOUSE_DY_Z0"],
            },
        ],
    )

    oracle = build_state_delta_oracle_metrics(target_paths=[target], max_rows=2)
    repeat = build_key_repeat_prior_metrics(train_paths=[train], target_paths=[target], max_train_rows=3, max_target_rows=2)
    causal = build_causal_keyboard_repeat_policy_matrix(
        train_paths=[train],
        target_paths=[target],
        max_train_rows=3,
        max_target_rows=2,
        thresholds=[0.5],
    )

    assert oracle["policies"]["next_state_delta_plus_prev_motion"]["all"]["paper_compatible"]["mouse_button"]["button_accuracy"] == 1.0
    assert repeat["rows"] == 2
    assert repeat["policies"]["global_hold_since_th0.1"]["all"]["paper_compatible"]["keyboard"]["key_accuracy"] == 1.0
    assert repeat["policies"]["code_hold_mod_th0.1"]["all"]["paper_compatible"]["keyboard"]["key_accuracy"] == 1.0
    assert "code_hold_mod" in repeat["context_count"]
    assert causal["rows"] == 2
    assert causal["policies"]["global_hold_since_pressrelease_th0.5"]["all"]["paper_compatible"]["keyboard"]["key_accuracy"] == 1.0
