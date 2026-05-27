from __future__ import annotations

import json
from pathlib import Path

from scripts.materialize_chronological_prefix import main as materialize_main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_chronological_prefix_sorts_selected_prefix_rows(tmp_path: Path, monkeypatch) -> None:
    shard0 = tmp_path / "shard_0/target_all_eval.jsonl"
    shard1 = tmp_path / "shard_1/target_all_eval.jsonl"
    _write_jsonl(
        shard0,
        [
            {"sequence_id": "b#2", "recording_id": "b", "timestamp_ns": 2, "eval_split_tags": ["temporal"]},
            {"sequence_id": "a#2", "recording_id": "a", "timestamp_ns": 2, "eval_split_tags": ["heldout_game"]},
        ],
    )
    _write_jsonl(
        shard1,
        [
            {"sequence_id": "a#1", "recording_id": "a", "timestamp_ns": 1, "eval_split_tags": ["heldout_game"]},
            {"sequence_id": "b#1", "recording_id": "b", "timestamp_ns": 1, "eval_split_tags": ["temporal"]},
        ],
    )
    output = tmp_path / "chrono/target_all_eval.jsonl"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "materialize_chronological_prefix.py",
            "--input",
            str(tmp_path / "shard_*/target_all_eval.jsonl"),
            "--output",
            str(output),
            "--summary-out",
            str(summary),
            "--max-rows",
            "3",
        ],
    )

    assert materialize_main() == 0

    rows = _read_jsonl(output)
    assert [row["sequence_id"] for row in rows] == ["a#1", "a#2", "b#2"]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["rows"] == 3
    assert payload["input_order"]["per_recording_timestamp_violations"] == 1
    assert payload["output_order"]["per_recording_timestamp_violations"] == 0
    assert payload["output_sha256"]
