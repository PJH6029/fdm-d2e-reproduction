from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.plan_mlxp_reservation_payload import build_payload, find_free_window


def _board() -> dict:
    rows = []
    for hour in range(4):
        rows.append(
            {
                "slot_start": f"2026-06-01T0{hour}:00:00+09:00",
                "slot_end": f"2026-06-01T0{hour + 1}:00:00+09:00",
                "cells": [None] * 16,
            }
        )
    rows[0]["cells"][0] = {"reservation_id": "busy"}
    rows[1]["cells"][0] = {"reservation_id": "busy"}
    return {
        "now_iso": "2026-06-01T00:10:00+09:00",
        "slot_minutes": 60,
        "default_image_key": "base",
        "nodes": [{"node_id": "1", "gpu_count": 8}, {"node_id": "2", "gpu_count": 8}],
        "rows": rows,
    }


def test_find_free_window_prefers_requested_node_and_gpu_start():
    window = find_free_window(_board(), gpu_count=2, duration_hours=2, preferred_node_id="1", preferred_gpu_start=2)
    assert window["node_id"] == "1"
    assert window["gpu_start"] == 2
    assert window["start_at"] == "2026-06-01T00:00:00+09:00"
    assert window["end_at"] == "2026-06-01T02:00:00+09:00"


def test_find_free_window_skips_busy_cells():
    window = find_free_window(_board(), gpu_count=1, duration_hours=2, preferred_node_id="1", preferred_gpu_start=0)
    assert window["gpu_start"] != 0


def test_build_payload_uses_board_default_image_and_valid_shape():
    payload = build_payload(_board(), gpu_count=1, duration_hours=1, purpose="G003", preferred_node_id="2", preferred_gpu_start=0, actor_name="jeonghunpark", managed_image_key=None)
    assert payload["node_id"] == "2"
    assert payload["managed_image_key"] == "base"
    assert payload["purpose"] == "G003"


def test_plan_mlxp_reservation_payload_cli_writes_payload_and_validation(tmp_path: Path):
    board_path = tmp_path / "board.json"
    payload_path = tmp_path / "payload.json"
    validation_path = tmp_path / "validation.json"
    board_path.write_text(json.dumps(_board()))
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/plan_mlxp_reservation_payload.py",
            "--board-json",
            str(board_path),
            "--output",
            str(payload_path),
            "--validation-output",
            str(validation_path),
            "--gpu-count",
            "1",
            "--duration-hours",
            "2",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "payload" in completed.stdout
    payload = json.loads(payload_path.read_text())
    validation = json.loads(validation_path.read_text())
    assert payload["gpu_count"] == 1
    assert validation["status"] == "pass"
