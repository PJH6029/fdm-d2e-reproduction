from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.key_phase_repeat_diagnostic import build_key_phase_repeat_diagnostic, sequence_bin_index
from fdm_d2e.io_utils import write_jsonl


def test_sequence_bin_index_prefers_sequence_suffix() -> None:
    assert sequence_bin_index({"sequence_id": "rec#000123", "timestamp_ns": 0}) == 123
    assert sequence_bin_index({"sequence_id": "rec", "timestamp_ns": 150_000_000}) == 3


def test_phase_repeat_diagnostic_can_improve_base_key_accuracy(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    write_jsonl(
        train,
        [
            {
                "sequence_id": f"train#{idx:06d}",
                "prior_key_hold_bins": {"65": 10 + idx},
                "ground_truth_tokens": ["KEY_PRESS_65"] if idx % 2 == 0 else [],
            }
            for idx in range(20)
        ],
    )
    write_jsonl(
        target,
        [
            {"sequence_id": "target#000000", "prior_key_hold_bins": {"65": 30}, "ground_truth_tokens": ["KEY_PRESS_65"]},
            {"sequence_id": "target#000001", "prior_key_hold_bins": {"65": 31}, "ground_truth_tokens": []},
        ],
    )
    write_jsonl(base, [{"sequence_id": "target#000000", "predicted_tokens": []}, {"sequence_id": "target#000001", "predicted_tokens": []}])

    payload = build_key_phase_repeat_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=20,
        max_target_rows=2,
        periods=[2],
        thresholds=[0.5],
        min_support=1,
    )

    assert payload["alignment"]["sequence_id_mismatches"] == 0
    assert payload["base"]["keyboard_accuracy"] == 0.0
    assert payload["best_policy"]["keyboard_accuracy"] == 1.0


def test_phase_repeat_diagnostic_reports_alignment_mismatch(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    write_jsonl(train, [{"sequence_id": "train#0", "ground_truth_tokens": []}])
    write_jsonl(target, [{"sequence_id": "target#0", "ground_truth_tokens": []}])
    write_jsonl(base, [{"sequence_id": "other#0", "predicted_tokens": []}])

    payload = build_key_phase_repeat_diagnostic(
        train_paths=[train],
        target_paths=[target],
        base_prediction_paths=[base],
        max_train_rows=1,
        max_target_rows=1,
        periods=[2],
        thresholds=[0.5],
    )

    assert payload["alignment"]["sequence_id_mismatches"] == 1
