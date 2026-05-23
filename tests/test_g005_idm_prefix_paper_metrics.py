from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_g005_idm_prefix_paper_metrics import build_prefix_metrics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_build_g005_prefix_metrics_aligns_prediction_parts_to_target_chunks(tmp_path: Path):
    target_a = tmp_path / "shard_0" / "target_all_eval.jsonl"
    target_b = tmp_path / "shard_1" / "target_all_eval.jsonl"
    part_a = tmp_path / "parts" / "part_000" / "predictions.jsonl"
    part_b = tmp_path / "parts" / "part_001" / "predictions.jsonl"
    rows_a = [
        {
            "sequence_id": "a1",
            "eval_split_tags": ["temporal"],
            "ground_truth_tokens": ["KEY_PRESS_87", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
        },
        {
            "sequence_id": "a2",
            "eval_split_tags": ["temporal"],
            "ground_truth_tokens": ["MOUSE_LEFT_DOWN", "MOUSE_DX_Z0", "MOUSE_DY_Z0"],
        },
    ]
    rows_b = [
        {
            "sequence_id": "b1",
            "eval_split_tags": ["heldout_game"],
            "ground_truth_tokens": ["KEY_RELEASE_87", "MOUSE_DX_N1", "MOUSE_DY_Z0"],
        }
    ]
    _write_jsonl(target_a, rows_a)
    _write_jsonl(target_b, rows_b)
    _write_jsonl(part_a, [{"sequence_id": row["sequence_id"], "predicted_tokens": row["ground_truth_tokens"]} for row in rows_a])
    _write_jsonl(part_b, [{"sequence_id": row["sequence_id"], "predicted_tokens": row["ground_truth_tokens"]} for row in rows_b])

    payload = build_prefix_metrics(
        config={
            "model_name": "candidate",
            "target_records": str(target_a),
            "target_record_paths": [str(target_a), str(target_b)],
        },
        prediction_part_paths=[part_a, part_b],
        rows_per_part=10,
        split_tags=["temporal", "heldout_game"],
        empty_bins_as_correct=False,
    )

    assert payload["status"] == "pass"
    assert payload["alignment"]["rows_seen"] == 3
    assert payload["groups"]["all"]["paper_compatible"]["keyboard"]["key_accuracy"] == 1.0
    assert payload["groups"]["all"]["paper_compatible"]["mouse_button"]["button_accuracy"] == 1.0
    assert payload["groups"]["eval_split:temporal"]["rows"] == 2
    assert payload["groups"]["eval_split:heldout_game"]["rows"] == 1
