from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from watch_g004_then_plan_g005 import watch_and_plan


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "output": "artifacts/aux/chain.json",
        "allow_fail": False,
        "once": True,
        "poll_seconds": 0.01,
        "max_wait_seconds": -1.0,
        "watcher_pid_file": "outputs/cluster/chain.pid",
        "replace_existing_watcher": False,
        "g004_pid_file": "outputs/cluster/g004.pid",
        "g004_postrun_summary": "artifacts/fdm/g004_postrun_watcher_summary.json",
        "g004_finalization_summary": "artifacts/fdm/g004_finalize.json",
        "skip_split_stats": False,
        "force_split_stats": False,
        "g004_split_stats_config": "configs/eval/g004_split_stats.json",
        "g004_split_stats_summary": "artifacts/eval/g004_split.json",
        "g004_completion_config": "configs/eval/g004_completion.json",
        "g004_audit_output": "artifacts/fdm/g004_audit.json",
        "g004_run_summary": "artifacts/fdm/g004_run.json",
        "g004_log_path": "artifacts/fdm/g004.log",
        "g004_gpu_monitor": "artifacts/fdm/g004_gpu.csv",
        "g005_completion_config": "configs/eval/g005_completion.json",
        "g005_launch_readiness": "artifacts/aux/g005_launch.json",
        "g003_audit": "artifacts/idm/g003_audit.json",
        "g005_pid_file": "outputs/cluster/g005.pid",
        "source_evidence": [],
        "eval_manifest_hashes": None,
        "require_eval_manifest_hashes": False,
        "require_namespace_ready": False,
        "allow_overwrite_g005_run_summary": False,
    }
    data.update(overrides)
    return Namespace(**data)


def test_chain_waits_while_g004_parent_runs(tmp_path: Path):
    pid_path = tmp_path / "outputs/cluster/g004.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n")
    calls = {"finalize": 0, "plan": 0}
    payload = watch_and_plan(
        _args(tmp_path),
        finalize_func=lambda ns: calls.__setitem__("finalize", calls["finalize"] + 1) or {"status": "pass"},
        plan_func=lambda ns: calls.__setitem__("plan", calls["plan"] + 1) or {"status": "ready"},
    )
    assert payload["status"] == "waiting_g004_parent"
    assert payload["g004_run"]["pid_running"] is True
    assert calls == {"finalize": 0, "plan": 0}
    assert not (tmp_path / "outputs/cluster/chain.pid").exists()


def test_chain_uses_existing_g004_finalized_pass_and_reports_g005_ready(tmp_path: Path):
    write_json(
        tmp_path / "artifacts/fdm/g004_postrun_watcher_summary.json",
        {"status": "finalized_pass", "g004_audit_status": "pass", "g004_audit_error_count": 0},
    )
    finalized_calls: list[Namespace] = []
    planned_calls: list[Namespace] = []

    def fake_plan(ns: Namespace) -> dict:
        planned_calls.append(ns)
        return {"status": "ready", "findings": []}

    payload = watch_and_plan(
        _args(tmp_path, source_evidence=["artifacts/aux/source.json"], eval_manifest_hashes="artifacts/aux/hashes.json", require_eval_manifest_hashes=True),
        finalize_func=lambda ns: finalized_calls.append(ns) or {"status": "fail"},
        plan_func=fake_plan,
    )
    assert payload["status"] == "g005_launch_ready"
    assert payload["g004_finalization_source"] == "existing_g004_postrun_watcher"
    assert finalized_calls == []
    assert len(planned_calls) == 1
    assert planned_calls[0].source_evidence == ["artifacts/aux/source.json"]
    assert planned_calls[0].eval_manifest_hashes == "artifacts/aux/hashes.json"
    assert planned_calls[0].require_eval_manifest_hashes is True
    assert planned_calls[0].allow_precheckpoint is False


def test_chain_blocks_when_g004_finalization_fails(tmp_path: Path):
    plan_calls: list[Namespace] = []
    payload = watch_and_plan(
        _args(tmp_path),
        finalize_func=lambda ns: {"status": "fail", "g004_audit_status": "fail", "g004_audit_error_count": 2},
        plan_func=lambda ns: plan_calls.append(ns) or {"status": "ready"},
    )
    assert payload["status"] == "g004_finalization_not_pass"
    assert payload["findings"][0]["code"] == "g004_finalization_not_pass"
    assert plan_calls == []


def test_chain_reports_g005_not_ready_without_launching(tmp_path: Path):
    payload = watch_and_plan(
        _args(tmp_path),
        finalize_func=lambda ns: {"status": "pass", "g004_audit_status": "pass", "g004_audit_error_count": 0},
        plan_func=lambda ns: {"status": "blocked", "findings": [{"code": "missing_source"}]},
    )
    assert payload["status"] == "g005_launch_not_ready"
    assert payload["findings"][0]["code"] == "g005_launch_not_ready"


def test_chain_refuses_duplicate_running_chain_watcher(tmp_path: Path):
    watcher = tmp_path / "outputs/cluster/chain.pid"
    watcher.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["sleep", "30"])
    try:
        watcher.write_text(f"{proc.pid}\n")
        payload = watch_and_plan(_args(tmp_path), finalize_func=lambda ns: {"status": "pass"}, plan_func=lambda ns: {"status": "ready"})
        assert payload["status"] == "duplicate_chain_watcher_running"
        assert payload["existing_pid"] == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=5)
