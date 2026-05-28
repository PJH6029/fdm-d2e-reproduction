from __future__ import annotations

import json
from pathlib import Path

from fdm_d2e.eval.key_repeat_clock_diagnostic import build_key_repeat_clock_diagnostic


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_repeat_clock_learns_per_key_repeat_phase(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    train_rows = []
    # Held W repeats every other bin when press age is odd/even under the learned context.
    for i in range(8):
        tokens = ["KEY_PRESS_87"] if i in {0, 2, 4, 6} else []
        train_rows.append(
            {
                "sequence_id": f"rec#%06d" % i,
                "recording_id": "Game/rec",
                "game": "Game",
                "prior_key_hold_bins": {"87": i + 1},
                "prior_since_key_transition_bins": i,
                "ground_truth_tokens": tokens,
                "eval_split_tags": ["temporal"],
            }
        )
    target_rows = []
    base_rows = []
    for i in range(4):
        target_rows.append(
            {
                "sequence_id": f"rec#%06d" % i,
                "recording_id": "Game/rec",
                "game": "Game",
                "prior_key_hold_bins": {"87": i + 1},
                "prior_since_key_transition_bins": i,
                "ground_truth_tokens": ["KEY_PRESS_87"] if i in {0, 2} else [],
                "eval_split_tags": ["temporal"],
            }
        )
        base_rows.append({"sequence_id": f"rec#%06d" % i, "predicted_tokens": []})
    _write(train, train_rows)
    _write(target, target_rows)
    _write(base, base_rows)
    payload = build_key_repeat_clock_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=None,
        max_target_rows=None,
        thresholds=[0.5],
        min_supports=[1],
        clock_modes=["teacher_forced"],
    )
    best = payload["ranked_policies"][0]
    assert payload["alignment"]["sequence_id_mismatches"] == 0
    assert best["keyboard_accuracy"] == 1.0


def test_predicted_clock_is_reported_separately(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    row = {
        "sequence_id": "rec#000000",
        "recording_id": "Game/rec",
        "game": "Game",
        "prior_key_hold_bins": {"65": 1},
        "prior_since_key_transition_bins": 1,
        "ground_truth_tokens": ["KEY_PRESS_65"],
        "eval_split_tags": ["temporal"],
    }
    _write(train, [row])
    _write(target, [row])
    _write(base, [{"sequence_id": "rec#000000", "predicted_tokens": []}])
    payload = build_key_repeat_clock_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        thresholds=[0.1],
        min_supports=[1],
        clock_modes=["predicted", "teacher_forced"],
    )
    assert payload["clock_modes"] == ["predicted", "teacher_forced"]
    assert any(row["policy"].startswith("clock_") for row in payload["ranked_policies"])
