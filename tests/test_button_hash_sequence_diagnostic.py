from __future__ import annotations

from pathlib import Path

from fdm_d2e.eval.button_hash_sequence_diagnostic import build_button_hash_sequence_diagnostic


def write_jsonl(path: Path, rows: list[dict]) -> None:
    import json

    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_button_hash_sequence_diagnostic_learns_down_token(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    row = {
        "sequence_id": "train#000000",
        "recording_id": "game/train",
        "prior_button_hold_bins": {},
        "prior_since_button_transition_bins": 5,
        "previous_event_tokens": ["MOUSE_DX_Z0"],
        "prior_action_tokens": ["NOOP"],
        "ground_truth_tokens": ["MOUSE_LEFT_DOWN"],
    }
    write_jsonl(train, [row])
    write_jsonl(target, [{**row, "sequence_id": "target#000000", "recording_id": "game/target"}])
    write_jsonl(base, [{"sequence_id": "target#000000", "predicted_tokens": ["NOOP"]}])

    payload = build_button_hash_sequence_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        dim=2048,
        learning_rate=0.2,
        down_thresholds=[0.5],
        up_thresholds=[0.9],
    )

    assert payload["alignment"]["sequence_id_mismatches"] == 0
    assert payload["ranked_policies"][0]["mouse_button_accuracy"] > 0.0
    assert payload["ranked_policies"][-1]["policy"] == "base_all"


def test_button_hash_sequence_diagnostic_reports_alignment(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    row = {
        "sequence_id": "train#000000",
        "recording_id": "game/train",
        "prior_button_hold_bins": {"LEFT": 1},
        "prior_since_button_transition_bins": 0,
        "previous_event_tokens": ["MOUSE_LEFT_DOWN"],
        "prior_action_tokens": ["MOUSE_LEFT_DOWN"],
        "ground_truth_tokens": ["MOUSE_LEFT_UP"],
    }
    write_jsonl(train, [row])
    write_jsonl(target, [{**row, "sequence_id": "target#000000", "recording_id": "game/target"}])
    write_jsonl(base, [{"sequence_id": "wrong#000000", "predicted_tokens": []}])

    payload = build_button_hash_sequence_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        dim=2048,
        learning_rate=0.2,
        down_thresholds=[0.9],
        up_thresholds=[0.1],
    )

    assert payload["alignment"]["sequence_id_mismatches"] == 1
    assert payload["alignment"]["examples"]
