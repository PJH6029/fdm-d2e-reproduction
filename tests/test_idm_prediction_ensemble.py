from __future__ import annotations

import json
from pathlib import Path

from fdm_d2e.eval.idm_prediction_ensemble import ensemble_idm_predictions


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_ensemble_idm_predictions_combines_token_groups_without_targets(tmp_path: Path) -> None:
    context = tmp_path / "context.jsonl"
    buttons = tmp_path / "buttons.jsonl"
    _write(
        context,
        [
            {"sequence_id": "a", "model": "context", "predicted_tokens": ["KEY_PRESS_87", "MOUSE_DX_P2", "MOUSE_DY_N1", "MOUSE_LEFT_DOWN"]},
            {"sequence_id": "b", "model": "context", "predicted_tokens": ["MOUSE_DX_Z0", "MOUSE_DY_P1"]},
        ],
    )
    _write(
        buttons,
        [
            {"sequence_id": "a", "model": "buttons", "predicted_tokens": ["KEY_PRESS_65", "MOUSE_RIGHT_DOWN", "MOUSE_DX_N5"]},
            {"sequence_id": "b", "model": "buttons", "predicted_tokens": ["MOUSE_LEFT_UP"]},
        ],
    )
    output = tmp_path / "ensemble.jsonl"
    summary = tmp_path / "summary.json"

    payload = ensemble_idm_predictions(
        sources={"context": context, "buttons": buttons},
        group_sources={"keyboard": "context", "mouse_button": "buttons", "mouse_move": "context"},
        output_path=output,
        summary_out=summary,
        model_name="hybrid",
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert payload["rows"] == 2
    assert rows[0]["model"] == "hybrid"
    assert rows[0]["predicted_tokens"] == ["KEY_PRESS_87", "MOUSE_RIGHT_DOWN", "MOUSE_DX_P2", "MOUSE_DY_N1"]
    assert rows[1]["predicted_tokens"] == ["MOUSE_LEFT_UP", "MOUSE_DX_Z0", "MOUSE_DY_P1"]
    assert json.loads(summary.read_text())["token_counts"]["mouse_button"] == 2


def test_ensemble_idm_predictions_rejects_sequence_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write(a, [{"sequence_id": "a", "predicted_tokens": []}])
    _write(b, [{"sequence_id": "b", "predicted_tokens": []}])

    try:
        ensemble_idm_predictions(
            sources={"a": a, "b": b},
            group_sources={"keyboard": "a", "mouse_button": "b", "mouse_move": "a"},
            output_path=tmp_path / "out.jsonl",
            summary_out=tmp_path / "summary.json",
        )
    except ValueError as exc:
        assert "sequence_id mismatch" in str(exc)
    else:
        raise AssertionError("expected sequence mismatch")
