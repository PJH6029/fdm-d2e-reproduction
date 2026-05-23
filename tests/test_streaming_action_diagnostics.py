from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.streaming_action_diagnostics import build_streaming_action_diagnostics, write_streaming_action_diagnostics
from fdm_d2e.io_utils import write_jsonl


def test_streaming_action_diagnostics_groups_by_game_and_split(tmp_path):
    preds = tmp_path / "preds.jsonl"
    targets = tmp_path / "targets.jsonl"
    write_jsonl(
        preds,
        [
            {"sequence_id": "a#0", "game": "GameA", "predicted_tokens": ["MOUSE_LEFT_DOWN", "MOUSE_DX_P1", "MOUSE_DY_Z0"]},
            {"sequence_id": "b#0", "game": "GameB", "predicted_tokens": ["KEY_PRESS_87", "MOUSE_DX_N1", "MOUSE_DY_Z0"]},
        ],
    )
    write_jsonl(
        targets,
        [
            {"sequence_id": "a#0", "game": "GameA", "eval_split_tags": ["temporal"], "ground_truth_tokens": ["MOUSE_DX_P1", "MOUSE_DY_Z0"]},
            {"sequence_id": "b#0", "game": "GameB", "eval_split_tags": ["heldout_game"], "ground_truth_tokens": ["KEY_PRESS_87", "MOUSE_DX_N1", "MOUSE_DY_Z0"]},
        ],
    )
    payload = build_streaming_action_diagnostics(prediction_paths=[preds], target_paths=[targets])
    assert payload["status"] == "pass"
    assert payload["groups"]["all"]["rows"] == 2
    assert payload["groups"]["game:GameA"]["mouse_button"]["no_button_false_positive_rate"] == 1.0
    assert payload["groups"]["eval_split:heldout_game"]["keyboard"]["accuracy"] == 1.0


def test_streaming_action_diagnostics_reports_alignment_warnings(tmp_path):
    preds = tmp_path / "preds.jsonl"
    targets = tmp_path / "targets.jsonl"
    write_jsonl(preds, [{"sequence_id": "a#0", "predicted_tokens": []}])
    write_jsonl(targets, [{"sequence_id": "b#0", "ground_truth_tokens": []}])
    payload = build_streaming_action_diagnostics(prediction_paths=[preds], target_paths=[targets])
    assert payload["status"] == "pass"
    assert payload["alignment"]["sequence_id_mismatches"] == 1
    assert payload["findings"][0]["code"] == "sequence_id_mismatches_detected"


def test_streaming_action_diagnostics_fails_on_missing_inputs(tmp_path):
    payload = build_streaming_action_diagnostics(prediction_paths=[tmp_path / "missing_pred"], target_paths=[tmp_path / "missing_target"])
    assert payload["status"] == "fail"
    assert {item["code"] for item in payload["findings"]} == {"missing_prediction_paths", "missing_target_paths"}


def test_streaming_action_diagnostics_writes_output(tmp_path):
    preds = tmp_path / "preds.jsonl"
    targets = tmp_path / "targets.jsonl"
    out = tmp_path / "out.json"
    write_jsonl(preds, [{"sequence_id": "a#0", "predicted_tokens": ["NOOP"]}])
    write_jsonl(targets, [{"sequence_id": "a#0", "ground_truth_tokens": ["NOOP"]}])
    payload = write_streaming_action_diagnostics(prediction_paths=[preds], target_paths=[targets], output_path=out)
    assert payload["schema"] == "g002_streaming_action_diagnostics.v1"
    assert out.is_file()
