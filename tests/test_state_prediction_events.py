from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.state_prediction_events import convert_state_prediction_file
from fdm_d2e.io_utils import read_jsonl, write_jsonl


def test_state_predictions_convert_held_keys_to_press_release_events(tmp_path: Path) -> None:
    source = tmp_path / "state_preds.jsonl"
    out = tmp_path / "events.jsonl"
    write_jsonl(
        source,
        [
            {"sequence_id": "rec#0", "predicted_tokens": ["MOUSE_DX_P1", "MOUSE_DY_Z0", "KEY_DOWN_87"]},
            {"sequence_id": "rec#1", "predicted_tokens": ["MOUSE_DX_Z0", "MOUSE_DY_Z0", "KEY_DOWN_87"]},
            {"sequence_id": "rec#2", "predicted_tokens": ["MOUSE_DX_N1", "MOUSE_DY_Z0"]},
        ],
    )

    summary = convert_state_prediction_file(prediction_paths=[source], output_path=out)
    rows = read_jsonl(out)

    assert summary["status"] == "pass"
    assert rows[0]["predicted_tokens"] == ["MOUSE_DX_P1", "MOUSE_DY_Z0", "KEY_PRESS_87"]
    assert rows[1]["predicted_tokens"] == ["MOUSE_DX_Z0", "MOUSE_DY_Z0"]
    assert rows[2]["predicted_tokens"] == ["MOUSE_DX_N1", "MOUSE_DY_Z0", "KEY_RELEASE_87"]


def test_state_prediction_conversion_debounces_transitions(tmp_path: Path) -> None:
    source = tmp_path / "state_preds.jsonl"
    out = tmp_path / "events.jsonl"
    write_jsonl(
        source,
        [
            {"sequence_id": "rec#0", "predicted_tokens": ["KEY_DOWN_65"]},
            {"sequence_id": "rec#1", "predicted_tokens": []},
            {"sequence_id": "rec#2", "predicted_tokens": ["KEY_DOWN_65"]},
            {"sequence_id": "rec#3", "predicted_tokens": ["KEY_DOWN_65"]},
        ],
    )

    convert_state_prediction_file(prediction_paths=[source], output_path=out, key_press_rows=2)
    rows = read_jsonl(out)

    assert rows[0]["predicted_tokens"] == ["NOOP"]
    assert rows[1]["predicted_tokens"] == ["NOOP"]
    assert rows[2]["predicted_tokens"] == ["NOOP"]
    assert rows[3]["predicted_tokens"] == ["KEY_PRESS_65"]
