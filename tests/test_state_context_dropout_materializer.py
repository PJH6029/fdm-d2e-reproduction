from __future__ import annotations

import json
from pathlib import Path

from scripts.materialize_state_context_dropout_train import materialize_state_context_dropout_train


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_state_context_dropout_masks_prior_fields_deterministically(tmp_path: Path) -> None:
    source = tmp_path / "shard_00" / "train_core.jsonl"
    rows = [
        {
            "sequence_id": "rec#0",
            "prior_action_tokens": ["KEY_DOWN_87", "MOUSE_LEFT_DOWN"],
            "prior_key_hold_bins": {"87": 3},
            "prior_button_hold_bins": {"LEFT": 2},
            "prior_since_key_transition_bins": 1,
            "prior_since_button_transition_bins": 1,
            "previous_event_tokens": ["KEY_PRESS_87"],
        },
        {
            "sequence_id": "rec#1",
            "prior_action_tokens": ["KEY_DOWN_65"],
            "prior_key_hold_bins": {"65": 4},
            "previous_event_tokens": ["NOOP"],
        },
    ]
    _write_jsonl(source, rows)

    summary = materialize_state_context_dropout_train(
        input_paths=[str(tmp_path / "shard_*" / "train_core.jsonl")],
        output_root=tmp_path / "out",
        dropout_rate=1.0,
        seed=123,
        summary_out=tmp_path / "summary.json",
        progress_out=tmp_path / "progress.json",
    )

    output = tmp_path / "out" / "shard_00" / "train_core.jsonl"
    out_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["rows"] == 2
    assert summary["dropped_rows"] == 2
    assert out_rows[0]["prior_action_tokens"] == ["NOOP"]
    assert out_rows[0]["prior_key_hold_bins"] == {}
    assert out_rows[0]["prior_button_hold_bins"] == {}
    assert out_rows[0]["prior_since_key_transition_bins"] is None
    assert out_rows[0]["previous_event_tokens"] == ["NOOP"]
    assert out_rows[0]["state_context_dropout"]["applied"] is True
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "pass"


def test_state_context_dropout_can_preserve_rows(tmp_path: Path) -> None:
    source = tmp_path / "train_core.jsonl"
    _write_jsonl(source, [{"sequence_id": "rec#0", "prior_action_tokens": ["KEY_DOWN_87"]}])

    materialize_state_context_dropout_train(
        input_paths=[str(source)],
        output_root=tmp_path / "out",
        dropout_rate=0.0,
        seed=123,
        summary_out=tmp_path / "summary.json",
    )

    row = json.loads((tmp_path / "out" / "shard_00" / "train_core.jsonl").read_text())
    assert row["prior_action_tokens"] == ["KEY_DOWN_87"]
    assert row["state_context_dropout"]["applied"] is False
