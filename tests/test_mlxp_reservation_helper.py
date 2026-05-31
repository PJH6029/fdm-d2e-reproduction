from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import mlxp_reservation_helper as helper


def _payload() -> dict:
    return {
        "node_id": "1",
        "gpu_start": 2,
        "gpu_count": 1,
        "gpu_indices": [],
        "start_at": "2026-06-01T01:00:00+09:00",
        "end_at": "2026-06-01T13:00:00+09:00",
        "purpose": "Continuous GUI - FDM reproduction: G003",
        "managed_image_key": "base",
        "registry_profile_key": "",
        "image_path": "",
        "command": [],
        "args": [],
        "actor_name": "jeonghunpark",
    }


def test_validate_payload_rejects_conflicting_image_fields():
    payload = _payload()
    payload["image_path"] = "docker.io/pjh6029/custom:latest"
    errors = helper.validate_payload(payload)
    assert "managed_image_key and image_path must not both be set" in errors


def test_validate_payload_passes_g003_draft():
    assert helper.validate_payload(_payload()) == []


def test_create_requires_explicit_live_confirmation(tmp_path: Path, monkeypatch):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_payload()))
    monkeypatch.setenv("RESERVATION_API_TOKEN", "token")
    called = False

    def fake_request(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(helper, "request_json", fake_request)
    rc = helper.main(["create", "--payload", str(payload_path)])
    assert rc == 2
    assert called is False


def test_create_posts_when_confirmed(tmp_path: Path, monkeypatch):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_payload()))
    monkeypatch.setenv("RESERVATION_API_TOKEN", "token")
    captured = {}

    def fake_request(base_url, path, *, token, method="GET", payload=None, timeout=60):
        captured.update({"base_url": base_url, "path": path, "token": token, "method": method, "payload": payload})
        return {"reservation_id": "rsv-test"}

    monkeypatch.setattr(helper, "request_json", fake_request)
    rc = helper.main(["create", "--payload", str(payload_path), "--i-confirm-live-production-reservation"])
    assert rc == 0
    assert captured["path"] == "/api/projects/production/reservations"
    assert captured["method"] == "POST"
    assert captured["payload"]["gpu_count"] == 1


def test_validate_payload_cli_outputs_pass(tmp_path: Path):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_payload()))
    completed = subprocess.run(
        [sys.executable, "scripts/mlxp_reservation_helper.py", "validate-payload", "--payload", str(payload_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    data = json.loads(completed.stdout)
    assert data["status"] == "pass"
    assert data["payload_summary"]["gpu_count"] == 1


def test_summarize_board_counts_free_cells_by_node():
    board = {
        "project_id": "production",
        "now_iso": "2026-06-01T01:00:00+09:00",
        "slot_minutes": 60,
        "default_image_key": "base",
        "managed_images": {"base": {}},
        "registry_profiles": {},
        "nodes": [{"node_id": "1"}, {"node_id": "2"}],
        "rows": [
            {"cells": [None, {"reservation_id": "busy"}] + [None] * 14},
            {"cells": [{"reservation_id": "busy"}] * 8 + [None] * 8},
        ],
    }
    summary = helper.summarize_board(board)
    assert summary["free_cells_by_node"] == {"1": 7, "2": 16}
    assert summary["managed_image_keys"] == ["base"]


def test_board_command_writes_full_board_and_summary(tmp_path: Path, monkeypatch, capsys):
    board_payload = {
        "now_iso": "2026-06-01T01:00:00+09:00",
        "slot_minutes": 60,
        "nodes": [{"node_id": "1"}],
        "rows": [{"cells": [None] * 8}],
    }
    monkeypatch.setenv("RESERVATION_API_TOKEN", "token")

    def fake_request(base_url, path, *, token, method="GET", payload=None, timeout=60):
        assert path == "/api/projects/production/board?days=1"
        assert token == "token"
        return board_payload

    monkeypatch.setattr(helper, "request_json", fake_request)
    board_out = tmp_path / "board.json"
    summary_out = tmp_path / "summary.json"
    rc = helper.main(["board", "--output", str(board_out), "--summary-output", str(summary_out)])
    assert rc == 0
    assert json.loads(board_out.read_text())["rows"]
    assert json.loads(summary_out.read_text())["free_cells_by_node"] == {"1": 8}
    assert "free_cells_by_node" in capsys.readouterr().out
