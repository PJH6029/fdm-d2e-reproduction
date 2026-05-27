from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ensemble_idm_predictions.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_endpoint_ensemble_gates_button_source_by_context_detector(tmp_path: Path) -> None:
    event_rows = [
        {"sequence_id": "s1", "timestamp_ns": 1, "predicted_tokens": ["MOUSE_DX_P1", "KEY_PRESS_65", "MOUSE_LEFT_DOWN"]},
        {"sequence_id": "s2", "timestamp_ns": 2, "predicted_tokens": ["MOUSE_DX_P2", "KEY_PRESS_66"]},
    ]
    state_rows = [
        {"sequence_id": "s1", "timestamp_ns": 1, "predicted_tokens": ["MOUSE_DX_N1", "MOUSE_RIGHT_DOWN"]},
        {"sequence_id": "s2", "timestamp_ns": 2, "predicted_tokens": ["MOUSE_LEFT_DOWN"]},
    ]
    _write_jsonl(tmp_path / "event.jsonl", event_rows)
    _write_jsonl(tmp_path / "state.jsonl", state_rows)
    config = {
        "prediction_paths": {"event": "event.jsonl", "state": "state.jsonl"},
        "metadata_source": "event",
        "policies": {
            "mouse": {"mode": "source", "source": "event"},
            "keyboard": {"mode": "source", "source": "event"},
            "button": {"mode": "source_with_endpoint_gate", "source": "state", "gate_sources": ["event"]},
        },
        "output_path": "ensemble/predictions.jsonl",
        "summary_path": "ensemble/summary.json",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(tmp_path / "config.json"), "--root", str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert '"status": "pass"' in result.stdout
    rows = _read_jsonl(tmp_path / "ensemble/predictions.jsonl")
    assert rows[0]["predicted_tokens"] == ["MOUSE_DX_P1", "KEY_PRESS_65", "MOUSE_RIGHT_DOWN"]
    assert rows[1]["predicted_tokens"] == ["MOUSE_DX_P2", "KEY_PRESS_66"]
    summary = json.loads((tmp_path / "ensemble/summary.json").read_text())
    assert summary["endpoint_positive_rows"]["button"] == 1


def test_endpoint_ensemble_fails_on_sequence_mismatch(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "a.jsonl", [{"sequence_id": "s1", "predicted_tokens": ["NOOP"]}])
    _write_jsonl(tmp_path / "b.jsonl", [{"sequence_id": "s2", "predicted_tokens": ["NOOP"]}])
    config = {
        "prediction_paths": {"a": "a.jsonl", "b": "b.jsonl"},
        "metadata_source": "a",
        "policies": {
            "mouse": {"mode": "source", "source": "a"},
            "keyboard": {"mode": "source", "source": "a"},
            "button": {"mode": "intersection", "sources": ["a", "b"]},
        },
        "output_path": "out.jsonl",
        "summary_path": "summary.json",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(tmp_path / "config.json"), "--root", str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "fail"
    assert summary["mismatches"][0]["code"] == "sequence_id_mismatch"
