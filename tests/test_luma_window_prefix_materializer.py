from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _row(recording_id: str, value: float, idx: int) -> dict:
    return {
        "recording_id": recording_id,
        "sequence_id": idx,
        "timestamp_ns": idx,
        "frame": {"luma16": [value for _ in range(16 * 16)]},
        "action": {"keyboard": [], "mouse_buttons": [], "mouse_dx": 0, "mouse_dy": 0},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_luma_window_prefix_materializer_keeps_train_target_windows_independent(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    train_out = tmp_path / "out" / "train.jsonl"
    target_out = tmp_path / "out" / "target.jsonl"
    summary = tmp_path / "summary.json"

    _write_jsonl(train, [_row("same-recording", value, idx) for idx, value in enumerate([10.0, 20.0, 30.0])])
    _write_jsonl(target, [_row("same-recording", value, idx) for idx, value in enumerate([100.0, 200.0])])

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/materialize_luma_window_prefix.py"),
            "--train-input",
            str(train),
            "--target-input",
            str(target),
            "--train-output",
            str(train_out),
            "--target-output",
            str(target_out),
            "--summary",
            str(summary),
            "--offsets",
            "0,1",
            "--luma-size",
            "16",
        ],
        cwd=ROOT,
        check=True,
    )

    train_rows = _read_jsonl(train_out)
    target_rows = _read_jsonl(target_out)
    assert [row["compact_luma_window_offsets"] for row in train_rows + target_rows] == [[0, 1]] * 5
    assert [plane[0] for plane in train_rows[-1]["compact_luma_window"]] == [30.0, 0.0]
    assert train_rows[-1]["compact_luma_window_mask"] == [1.0, 0.0]
    assert [plane[0] for plane in target_rows[0]["compact_luma_window"]] == [100.0, 200.0]
    assert target_rows[0]["compact_luma_window_mask"] == [1.0, 1.0]

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["train_output"]["rows"] == 3
    assert payload["target_output"]["rows"] == 2
    assert payload["offset_present_counts"] == {"0": 5, "1": 3}
    assert "not G005 completion evidence" in payload["claim_boundary"]
