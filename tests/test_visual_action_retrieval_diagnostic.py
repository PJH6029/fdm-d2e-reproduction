from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.visual_action_retrieval_diagnostic import build_visual_action_retrieval_diagnostic
from fdm_d2e.io_utils import write_jsonl


def _row(seq: str, tokens: list[str]) -> dict:
    return {
        "sequence_id": seq,
        "recording_id": "d2e_480p:Game/rec",
        "game": "Game",
        "frame": {"features": [0.1, 0.2, 0.3, 0.4, 0.5], "grid8": [0.25] * 64},
        "next_frame_features": [0.2, 0.3, 0.4, 0.5, 0.6],
        "next_frame_grid8": [0.3] * 64,
        "frame_delta_features": [0.1] * 5,
        "prior_key_hold_bins": {"65": 4},
        "prior_since_key_transition_bins": 2,
        "ground_truth_tokens": tokens,
    }


def test_visual_action_retrieval_replaces_from_matching_visual_state(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    write_jsonl(train, [_row("train#000000", ["KEY_PRESS_65", "MOUSE_DX_P1"])] * 3)
    write_jsonl(target, [_row("target#000000", ["KEY_PRESS_65", "MOUSE_DX_P1"])])
    write_jsonl(base, [{"sequence_id": "target#000000", "predicted_tokens": []}])

    payload = build_visual_action_retrieval_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=3,
        max_target_rows=1,
        context_names=["state_visual_features"],
        thresholds=[0.2],
        min_supports=[1],
    )

    assert payload["alignment"]["sequence_id_mismatches"] == 0
    assert payload["ranked_policies"][0]["keyboard_accuracy"] == 1.0
    assert payload["context_count"]["state_visual_features"] == 1


def test_visual_action_retrieval_reports_alignment_mismatch(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    write_jsonl(train, [_row("train#000000", [])])
    write_jsonl(target, [_row("target#000000", [])])
    write_jsonl(base, [{"sequence_id": "other#000000", "predicted_tokens": []}])

    payload = build_visual_action_retrieval_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        context_names=["state_visual_features"],
        thresholds=[0.2],
        min_supports=[1],
    )

    assert payload["alignment"]["sequence_id_mismatches"] == 1
