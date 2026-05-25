from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.idm_alignment_shifts import build_idm_alignment_shift_diagnostics
from fdm_d2e.io_utils import write_jsonl


def test_alignment_shift_detects_model_target_offset(tmp_path: Path) -> None:
    preds = tmp_path / "preds.jsonl"
    targets = tmp_path / "targets.jsonl"
    write_jsonl(
        preds,
        [
            {"sequence_id": "rec#0", "predicted_tokens": ["KEY_B"]},
            {"sequence_id": "rec#1", "predicted_tokens": ["KEY_C"]},
            {"sequence_id": "rec#2", "predicted_tokens": ["KEY_C"]},
        ],
    )
    write_jsonl(
        targets,
        [
            {"sequence_id": "rec#0", "recording_id": "rec", "eval_split_tags": ["temporal"], "ground_truth_tokens": ["KEY_A"]},
            {"sequence_id": "rec#1", "recording_id": "rec", "eval_split_tags": ["temporal"], "ground_truth_tokens": ["KEY_B"]},
            {"sequence_id": "rec#2", "recording_id": "rec", "eval_split_tags": ["temporal"], "ground_truth_tokens": ["KEY_C"]},
        ],
    )
    payload = build_idm_alignment_shift_diagnostics(
        prediction_paths=[preds],
        target_paths=[targets],
        shifts=[-1, 0, 1],
        split_tags=["temporal"],
    )
    shifted = payload["diagnostics"]["model_vs_shifted_target"]["1"]["all"]["paper_compatible"]["keyboard"]["key_accuracy"]
    unshifted = payload["diagnostics"]["model_vs_shifted_target"]["0"]["all"]["paper_compatible"]["keyboard"]["key_accuracy"]
    autocorr = payload["diagnostics"]["target_autocorr"]["0"]["all"]["paper_compatible"]["keyboard"]["key_accuracy"]
    assert payload["status"] == "pass"
    assert shifted == 1.0
    assert shifted > unshifted
    assert autocorr == 1.0
    assert payload["diagnostics"]["pair_counts"]["1"] == 2


def test_alignment_shift_reports_sequence_mismatch(tmp_path: Path) -> None:
    preds = tmp_path / "preds.jsonl"
    targets = tmp_path / "targets.jsonl"
    write_jsonl(preds, [{"sequence_id": "pred#0", "predicted_tokens": ["NOOP"]}])
    write_jsonl(targets, [{"sequence_id": "target#0", "recording_id": "target", "ground_truth_tokens": ["NOOP"]}])

    payload = build_idm_alignment_shift_diagnostics(prediction_paths=[preds], target_paths=[targets], shifts=[0])

    assert payload["status"] == "fail"
    assert payload["findings"][0]["code"] == "sequence_id_mismatches_detected"
