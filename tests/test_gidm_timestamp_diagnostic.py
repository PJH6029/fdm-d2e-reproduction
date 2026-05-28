from __future__ import annotations

import json
from pathlib import Path

from fdm_d2e.eval.gidm_timestamp_diagnostic import build_gidm_base_offset_shift_diagnostic


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_base_offset_diagnostic_evaluates_manifest_base_shift(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "universe_row_id": "d2e_480p:Game/rec",
                        "timestamp_min_ns": 250_000_000,
                        "bin_index_min": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    targets = tmp_path / "targets.jsonl"
    _write_jsonl(
        predictions,
        [
            {"sequence_id": "rec#0", "predicted_tokens": []},
            {"sequence_id": "rec#1", "predicted_tokens": []},
            {"sequence_id": "rec#2", "predicted_tokens": []},
            {"sequence_id": "rec#3", "predicted_tokens": []},
            {"sequence_id": "rec#4", "predicted_tokens": ["KEY_PRESS_87"]},
        ],
    )
    _write_jsonl(
        targets,
        [
            {
                "sequence_id": "rec#0",
                "ground_truth_tokens": ["KEY_PRESS_87"],
                "eval_split_tags": ["temporal"],
            },
            {"sequence_id": "rec#1", "ground_truth_tokens": []},
            {"sequence_id": "rec#2", "ground_truth_tokens": []},
        ],
    )
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "paper_reported_targets": {
                    "aggregate": {
                        "keyboard_accuracy": 1.0,
                        "mouse_button_accuracy": 0.0,
                        "pearson_x": 0.0,
                        "pearson_y": 0.0,
                        "scale_ratio_x": 99.0,
                        "scale_ratio_y": 99.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    payload = build_gidm_base_offset_shift_diagnostic(
        manifest_path=manifest,
        prediction_path=predictions,
        target_path=targets,
        baseline_contract_path=contract,
        bin_ms=50,
        extra_shifts=[],
    )

    assert payload["status"] == "pass"
    assert payload["base_offsets"][0]["base_timestamp_ns"] == 200_000_000
    assert payload["base_offsets"][0]["base_shift_rows_round"] == 4
    assert 4 in payload["candidate_shifts"]
    best = payload["best_by_keyboard"][0]
    assert best["row_shift"] == 4
    assert best["keyboard_accuracy"] == 1.0
    assert best["overlap_rows"] == 1
