from __future__ import annotations

import json
from pathlib import Path

from scripts.materialize_balanced_prefix import materialize_balanced_prefix


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_materialize_balanced_prefix_caps_rows_per_recording(tmp_path: Path) -> None:
    rows = []
    for rec in ("rec-a", "rec-b", "rec-c"):
        for idx in range(3):
            rows.append({"sequence_id": f"{rec}-{idx}", "recording_id": rec, "timestamp_ns": idx})
    src = tmp_path / "train.jsonl"
    out = tmp_path / "out.jsonl"
    summary = tmp_path / "summary.json"
    _write_jsonl(src, rows)

    payload = materialize_balanced_prefix(
        input_patterns=[str(src)],
        output=out,
        summary_out=summary,
        balance_key="recording_id",
        max_rows=6,
        group_values=None,
        per_group_rows=None,
        max_per_group=2,
        source_label="unit",
    )

    selected = _read_jsonl(out)
    assert payload["status"] == "pass"
    assert len(selected) == 6
    assert payload["group_counts"] == {"rec-a": 2, "rec-b": 2, "rec-c": 2}


def test_materialize_balanced_prefix_balances_eval_split_tags(tmp_path: Path) -> None:
    rows = [
        {"sequence_id": "t0", "eval_split_tags": ["temporal"]},
        {"sequence_id": "t1", "eval_split_tags": ["temporal"]},
        {"sequence_id": "r0", "eval_split_tags": ["heldout_recording"]},
        {"sequence_id": "g0", "eval_split_tags": ["heldout_game"]},
        {"sequence_id": "rg", "eval_split_tags": ["heldout_recording", "heldout_game"]},
        {"sequence_id": "none", "eval_split_tags": []},
    ]
    src = tmp_path / "target.jsonl"
    out = tmp_path / "target_balanced.jsonl"
    summary = tmp_path / "target_summary.json"
    _write_jsonl(src, rows)

    payload = materialize_balanced_prefix(
        input_patterns=[str(src)],
        output=out,
        summary_out=summary,
        balance_key="eval_split_tags",
        max_rows=6,
        group_values=["temporal", "heldout_recording", "heldout_game"],
        per_group_rows=2,
        max_per_group=None,
        source_label="unit-target",
    )

    selected_ids = [row["sequence_id"] for row in _read_jsonl(out)]
    assert payload["status"] == "pass"
    assert selected_ids == ["t0", "t1", "r0", "g0", "rg"]
    assert payload["group_counts"] == {"heldout_game": 2, "heldout_recording": 2, "temporal": 2}
