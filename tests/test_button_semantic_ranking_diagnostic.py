from __future__ import annotations

import json
from pathlib import Path

from fdm_d2e.eval.button_semantic_ranking_diagnostic import build_button_semantic_ranking_diagnostic


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_button_semantic_diagnostic_reports_mapping_and_offsets(tmp_path: Path) -> None:
    preds = tmp_path / "preds.jsonl"
    targets = tmp_path / "targets.jsonl"
    _write_jsonl(
        preds,
        [
            {"sequence_id": "r#000", "predicted_tokens": ["MOUSE_LEFT_DOWN"]},
            {"sequence_id": "r#001", "predicted_tokens": ["NOOP"]},
            {"sequence_id": "r#002", "predicted_tokens": ["MOUSE_LEFT_DOWN"]},
        ],
    )
    _write_jsonl(
        targets,
        [
            {"sequence_id": "r#000", "ground_truth_tokens": ["MOUSE_RIGHT_DOWN"]},
            {"sequence_id": "r#001", "ground_truth_tokens": ["MOUSE_LEFT_DOWN"]},
            {"sequence_id": "r#002", "ground_truth_tokens": []},
        ],
    )
    payload = build_button_semantic_ranking_diagnostic(
        prediction_paths=[preds],
        target_paths=[targets],
        offsets=[-1, 0, 1],
    )
    assert payload["status"] == "pass"
    assert payload["base"]["predicted_examples"] == 2
    assert payload["base"]["exact_true_positive_examples"] == 0
    assert payload["mapping_diagnostic"]["greedy_mapping"]["MOUSE_LEFT_DOWN"] == "MOUSE_RIGHT_DOWN"
    assert payload["mapping_diagnostic"]["greedy_metrics"]["exact_true_positive_examples"] == 1
    assert payload["best_offset_by_exact"]["offset"] == -1
    assert payload["best_offset_by_exact"]["exact_true_positive_examples"] == 1


def test_button_semantic_diagnostic_reports_alignment_examples(tmp_path: Path) -> None:
    preds = tmp_path / "preds.jsonl"
    targets = tmp_path / "targets.jsonl"
    _write_jsonl(preds, [{"sequence_id": "p#000", "predicted_tokens": ["MOUSE_LEFT_UP"]}])
    _write_jsonl(targets, [{"sequence_id": "t#000", "ground_truth_tokens": ["MOUSE_LEFT_UP"]}])
    payload = build_button_semantic_ranking_diagnostic(prediction_paths=[preds], target_paths=[targets])
    assert payload["alignment"]["sequence_id_mismatches"] == 1
    assert payload["alignment"]["examples"][0]["prediction_sequence_id"] == "p#000"
