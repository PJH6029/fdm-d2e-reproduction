from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.idm_alignment_shifts import build_idm_alignment_shift_diagnostics
from fdm_d2e.io_utils import write_jsonl
from fdm_d2e.reporting.g005_nep_offset_diagnostics import build_g005_nep_offset_summary


def test_g005_nep_summary_extracts_expected_shift_and_targets(tmp_path: Path) -> None:
    target = tmp_path / "targets.jsonl"
    write_jsonl(
        target,
        [
            {"sequence_id": "rec#0", "recording_id": "rec", "eval_split_tags": ["temporal"], "ground_truth_tokens": ["KEY_A", "MOUSE_LEFT_DOWN"]},
            {"sequence_id": "rec#1", "recording_id": "rec", "eval_split_tags": ["temporal"], "ground_truth_tokens": ["KEY_B"]},
            {"sequence_id": "rec#2", "recording_id": "rec", "eval_split_tags": ["temporal"], "ground_truth_tokens": ["KEY_A", "MOUSE_LEFT_DOWN"]},
        ],
    )
    diagnostics = build_idm_alignment_shift_diagnostics(target_paths=[target], shifts=[0, 1, 2], split_tags=["temporal"])
    contract = {
        "target_sequence": {
            "phase_1": {
                "primary_targets": {
                    "keyboard_accuracy": 0.9,
                    "mouse_button_accuracy": 0.9,
                }
            }
        }
    }

    summary = build_g005_nep_offset_summary(
        diagnostics_payload=diagnostics,
        contract_payload=contract,
        expected_nep_shift=2,
        source_label="unit",
    )

    assert summary["status"] == "pass"
    assert summary["expected_shift"]["shift_ms"] == 100
    assert summary["expected_shift"]["pair_count"] == 1
    assert summary["expected_shift"]["metrics"]["keyboard_accuracy"] == 1.0
    assert summary["expected_shift"]["metrics"]["mouse_button_accuracy"] == 1.0
    assert summary["expected_shift"]["paper_target_passes"] is True
    assert summary["nonzero_shifts_meeting_all_paper_targets"][0]["shift"] == 2
    assert "not a trained model" in summary["claim_boundary"]


def test_g005_nep_summary_reports_expected_shift_blockers(tmp_path: Path) -> None:
    target = tmp_path / "targets.jsonl"
    write_jsonl(
        target,
        [
            {"sequence_id": "rec#0", "recording_id": "rec", "ground_truth_tokens": ["KEY_A"]},
            {"sequence_id": "rec#1", "recording_id": "rec", "ground_truth_tokens": ["KEY_B"]},
            {"sequence_id": "rec#2", "recording_id": "rec", "ground_truth_tokens": ["KEY_C"]},
        ],
    )
    diagnostics = build_idm_alignment_shift_diagnostics(target_paths=[target], shifts=[0, 2])
    contract = {"target_sequence": {"phase_1": {"primary_targets": {"keyboard_accuracy": 0.9}}}}

    summary = build_g005_nep_offset_summary(
        diagnostics_payload=diagnostics,
        contract_payload=contract,
        expected_nep_shift=2,
    )

    assert summary["expected_shift"]["paper_target_passes"] is False
    codes = {finding["code"] for finding in summary["findings"]}
    assert "expected_nep_shift_target_autocorr_below_paper_targets" in codes
    assert summary["all_shift_metrics"][0]["shift"] == 0
    assert summary["all_shift_metrics"][0]["paper_target_passes"] is True
