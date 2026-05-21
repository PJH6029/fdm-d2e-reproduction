from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from watch_g005_then_finalize import watch


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "output": "artifacts/aux/watcher.json",
        "allow_fail": False,
        "once": True,
        "poll_seconds": 0.01,
        "max_wait_seconds": -1.0,
        "pid_file": "outputs/cluster/g005_parent.pid",
        "watcher_pid_file": "outputs/cluster/g005_watcher.pid",
        "replace_existing_watcher": False,
        "g005_finalization_summary": "artifacts/aux/finalize.json",
        "g005_completion_config": "configs/eval/g005_completion.json",
        "g005_audit_output": "artifacts/aux/g005_audit.json",
        "run_summary": "artifacts/aux/run.json",
        "log_path": "artifacts/aux/run.log",
        "namespace_manifest_output": "artifacts/aux/namespace.json",
        "aux_candidates": "artifacts/sources/aux.json",
        "source_evidence": [],
        "eval_manifest_hashes": None,
        "completion_ready": False,
        "allow_template_namespace": False,
        "skip_namespace_build": False,
        "force_namespace": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")


def test_g005_watcher_waits_without_finalizing_while_parent_is_running(tmp_path: Path):
    _write_pid(tmp_path / "outputs/cluster/g005_parent.pid", os.getpid())
    write_json(tmp_path / "artifacts/aux/run.json", {"exit_code": None})
    calls: list[Namespace] = []
    payload = watch(_args(tmp_path), finalize_func=lambda ns: calls.append(ns) or {"status": "pass"})
    assert payload["status"] == "waiting_active_parent"
    assert payload["run"]["pid_running"] is True
    assert payload["run"]["run_summary_exists"] is True
    assert calls == []
    assert (tmp_path / "artifacts/aux/watcher.json").exists()
    assert not (tmp_path / "outputs/cluster/g005_watcher.pid").exists()


def test_g005_watcher_runs_finalizer_once_parent_is_inactive(tmp_path: Path):
    calls: list[Namespace] = []

    def fake_finalize(ns: Namespace) -> dict:
        calls.append(ns)
        return {"status": "pass", "g005_audit_status": "pass", "g005_audit_error_count": 0}

    payload = watch(_args(tmp_path, source_evidence=["artifacts/aux/source.json"], eval_manifest_hashes="artifacts/aux/hashes.json", completion_ready=True), finalize_func=fake_finalize)
    assert payload["status"] == "finalized_pass"
    assert payload["g005_audit_status"] == "pass"
    assert len(calls) == 1
    assert calls[0].summary_out == "artifacts/aux/finalize.json"
    assert calls[0].run_summary == "artifacts/aux/run.json"
    assert calls[0].source_evidence == ["artifacts/aux/source.json"]
    assert calls[0].eval_manifest_hashes == "artifacts/aux/hashes.json"
    assert calls[0].completion_ready is True


def test_g005_watcher_reports_failed_finalization(tmp_path: Path):
    payload = watch(_args(tmp_path), finalize_func=lambda ns: {"status": "fail", "g005_audit_status": "fail", "g005_audit_error_count": 2})
    assert payload["status"] == "finalized_fail"
    assert payload["findings"][0]["code"] == "g005_finalization_not_pass"


def test_g005_watcher_refuses_duplicate_running_watcher(tmp_path: Path):
    watcher = tmp_path / "outputs/cluster/g005_watcher.pid"
    watcher.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["sleep", "30"])
    try:
        watcher.write_text(f"{proc.pid}\n", encoding="utf-8")
        payload = watch(_args(tmp_path), finalize_func=lambda ns: {"status": "pass"})
        assert payload["status"] == "duplicate_watcher_running"
        assert payload["existing_pid"] == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=5)
