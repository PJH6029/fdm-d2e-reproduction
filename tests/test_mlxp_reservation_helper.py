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
