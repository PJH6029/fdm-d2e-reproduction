from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import monitor_g003_fdm1_action_dataset_pod as monitor


def _config(root: Path) -> Path:
    path = root / "config.json"
    path.write_text(
        json.dumps(
            {
                "output_path": "artifacts/sources/audit.json",
                "omit_sha256_artifact_keys": ["action_slots"],
                "paths": {
                    "dataset_summary": "outputs/data/summary.json",
                    "action_slots": "outputs/data/action_slots.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_collect_status_reports_running_pid(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    pid = tmp_path / "outputs/cluster/run.pid"
    pid.parent.mkdir(parents=True)
    pid.write_text("123\n", encoding="utf-8")
    log = tmp_path / "artifacts/logs/run.log"
    log.parent.mkdir(parents=True)
    log.write_text("started\n", encoding="utf-8")
    monkeypatch.setattr(monitor, "pid_running", lambda value: value == 123)

    status = monitor.collect_status(root=tmp_path, pid_file="outputs/cluster/run.pid", log_path="artifacts/logs/run.log", completion_config_path=config)

    assert status["status"] == "running"
    assert status["pid"] == 123
    assert status["pid_running"] is True
    assert status["log_tail"] == ["started"]


def test_collect_status_requires_bundle_for_full_pass(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    audit = tmp_path / "artifacts/sources/audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({"status": "pass", "findings": []}), encoding="utf-8")
    monkeypatch.setattr(monitor, "pid_running", lambda value: False)

    status = monitor.collect_status(root=tmp_path, completion_config_path=config)
    assert status["status"] == "audit_pass_bundle_missing"

    bundle = tmp_path / "artifacts/sources/fdm1_g003_evidence_bundle_manifest.json"
    bundle.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    status = monitor.collect_status(root=tmp_path, completion_config_path=config)
    assert status["status"] == "pass"


def test_collect_status_flags_fatal_log_when_not_running(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    log = tmp_path / "artifacts/logs/run.log"
    log.parent.mkdir(parents=True)
    log.write_text("ok\nTraceback (most recent call last):\n", encoding="utf-8")
    monkeypatch.setattr(monitor, "pid_running", lambda value: False)

    status = monitor.collect_status(root=tmp_path, log_path="artifacts/logs/run.log", completion_config_path=config)
    assert status["status"] == "failed_or_interrupted"
    assert status["fatal_log_matches"]


def test_cli_writes_monitor_output(tmp_path: Path):
    config = _config(tmp_path)
    output = tmp_path / "monitor.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/monitor_g003_fdm1_action_dataset_pod.py",
            "--root",
            str(tmp_path),
            "--completion-config",
            str(config),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "G003 pod monitor" in completed.stdout
    data = json.loads(output.read_text())
    assert data["schema"] == "fdm1_g003_pod_monitor.v1"
    assert data["status"] == "incomplete"
