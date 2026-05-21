from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from watch_g003_then_launch_g004 import watch_and_maybe_launch


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "output": "artifacts/fdm/chain.json",
        "allow_fail": False,
        "once": True,
        "launch": False,
        "start_g004_watcher": False,
        "poll_seconds": 0.01,
        "max_wait_seconds": -1.0,
        "watcher_pid_file": "outputs/cluster/chain.pid",
        "replace_existing_watcher": False,
        "goals_path": ".omx/ultragoal/goals.json",
        "g003_goal_id": "G003-d2e-only-idm",
        "require_g003_goal_checkpoint": False,
        "g003_postrun_summary": "artifacts/idm/g003_postrun_watcher_summary.json",
        "g003_finalization_summary": "artifacts/idm/g003_finalize.json",
        "skip_split_stats": False,
        "force_split_stats": False,
        "g003_split_stats_config": "configs/eval/g003_split_stats.json",
        "g003_split_stats_summary": "artifacts/eval/g003_split.json",
        "g003_completion_config": "configs/eval/g003_completion.json",
        "g003_audit_output": "artifacts/idm/g003_audit.json",
        "integrated_run_evidence": "artifacts/idm/integrated.json",
        "idm_summary": "artifacts/idm/idm_summary.json",
        "checkpoint_metadata": "outputs/idm/checkpoint_metadata.json",
        "metrics": "outputs/idm/metrics.json",
        "gpu_monitor": "artifacts/idm/gpu.csv",
        "attached_monitor_metadata": "artifacts/idm/gpu_meta.json",
        "train_run_summary": "artifacts/idm/train_run.json",
        "g003_nproc_per_node": 4,
        "expected_gpus": 4,
        "shard_root": "outputs/shards",
        "log_dir": "artifacts/sources",
        "data_universe": "artifacts/sources/universe.json",
        "data_output_dir": "outputs/data",
        "idm_output_dir": "outputs/idm",
        "g003_pid_file": "outputs/cluster/g003.pid",
        "num_shards": 1,
        "stale_seconds": 3600.0,
        "g004_launch_readiness": "artifacts/fdm/g004_launch.json",
        "fdm_config": "configs/model/fdm.json",
        "idm_predict_config": "configs/model/idm_predict.json",
        "fdm_labels": "outputs/idm/fdm_labels.jsonl",
        "g004_run_script": "scripts/run_g004.sh",
        "g004_run_summary": "artifacts/fdm/g004_run.json",
        "g004_pid_file": "outputs/cluster/g004.pid",
        "g004_log_path": "artifacts/fdm/g004.log",
        "g004_gpu_monitor": "artifacts/fdm/g004_gpu.csv",
        "g004_nproc_per_node": 4,
        "check_gpus": False,
        "g004_watcher_pid_file": "outputs/cluster/g004_watcher.pid",
        "g004_watcher_poll_seconds": 0.01,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_universe(root: Path) -> None:
    write_json(
        root / "artifacts/sources/universe.json",
        {
            "recordings": [
                {
                    "status": "included",
                    "source_id": "d2e_480p",
                    "game": "Game",
                    "recording_id": "rec",
                    "cross_resolution_key": "Game/rec",
                }
            ]
        },
    )


def test_chain_waits_while_g003_parent_runs(tmp_path: Path):
    _write_universe(tmp_path)
    pid_path = tmp_path / "outputs/cluster/g003.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n")
    calls = {"finalize": 0, "plan": 0, "launch": 0}

    payload = watch_and_maybe_launch(
        _args(tmp_path),
        finalize_func=lambda ns: calls.__setitem__("finalize", calls["finalize"] + 1) or {"status": "pass"},
        plan_func=lambda ns: calls.__setitem__("plan", calls["plan"] + 1) or {"status": "ready"},
        launch_func=lambda args, root: calls.__setitem__("launch", calls["launch"] + 1) or {"status": "launched"},
    )
    assert payload["status"] == "waiting_g003_parent"
    assert payload["g003_progress"]["pid_running"] is True
    assert calls == {"finalize": 0, "plan": 0, "launch": 0}
    assert not (tmp_path / "outputs/cluster/chain.pid").exists()


def test_chain_uses_existing_g003_finalized_pass_and_reports_launch_ready(tmp_path: Path):
    _write_universe(tmp_path)
    write_json(
        tmp_path / "artifacts/idm/g003_postrun_watcher_summary.json",
        {"status": "finalized_pass", "g003_audit_status": "pass", "g003_audit_error_count": 0},
    )
    finalized_calls: list[Namespace] = []
    planned_calls: list[Namespace] = []

    def fake_plan(ns: Namespace) -> dict:
        planned_calls.append(ns)
        return {"status": "ready", "error_count": 0, "recommended_command": "bash scripts/run_g004.sh"}

    payload = watch_and_maybe_launch(
        _args(tmp_path),
        finalize_func=lambda ns: finalized_calls.append(ns) or {"status": "fail"},
        plan_func=fake_plan,
    )
    assert payload["status"] == "g004_launch_ready"
    assert payload["g003_finalization_source"] == "existing_g003_postrun_watcher"
    assert finalized_calls == []
    assert len(planned_calls) == 1
    assert planned_calls[0].allow_precheckpoint is True


def test_chain_blocks_when_g003_finalization_fails(tmp_path: Path):
    _write_universe(tmp_path)
    plan_calls: list[Namespace] = []
    payload = watch_and_maybe_launch(
        _args(tmp_path),
        finalize_func=lambda ns: {"status": "fail", "g003_audit_status": "fail", "g003_audit_error_count": 2},
        plan_func=lambda ns: plan_calls.append(ns) or {"status": "ready"},
    )
    assert payload["status"] == "g003_finalization_not_pass"
    assert payload["findings"][0]["code"] == "g003_finalization_not_pass"
    assert plan_calls == []


def test_chain_retries_local_finalization_after_stale_existing_g003_finalized_fail(tmp_path: Path):
    _write_universe(tmp_path)
    write_json(
        tmp_path / "artifacts/idm/g003_postrun_watcher_summary.json",
        {"status": "finalized_fail", "g003_audit_status": "fail", "g003_audit_error_count": 4},
    )
    finalized_calls: list[Namespace] = []
    planned_calls: list[Namespace] = []

    payload = watch_and_maybe_launch(
        _args(tmp_path),
        finalize_func=lambda ns: finalized_calls.append(ns) or {"status": "pass", "g003_audit_status": "pass", "g003_audit_error_count": 0},
        plan_func=lambda ns: planned_calls.append(ns) or {"status": "ready", "error_count": 0},
    )

    assert payload["status"] == "g004_launch_ready"
    assert payload["g003_finalization_source"] == "local_finalize_g003"
    assert len(finalized_calls) == 1
    assert len(planned_calls) == 1


def test_chain_launches_g004_and_watcher_when_enabled(tmp_path: Path):
    _write_universe(tmp_path)
    launches: list[str] = []

    def fake_launch(args: Namespace, root: Path) -> dict:
        launches.append("g004")
        return {"status": "launched", "pid": 1234}

    def fake_watcher(args: Namespace, root: Path) -> dict:
        launches.append("watcher")
        return {"status": "launched", "pid": 5678}

    payload = watch_and_maybe_launch(
        _args(tmp_path, launch=True, start_g004_watcher=True),
        finalize_func=lambda ns: {"status": "pass", "g003_audit_status": "pass", "g003_audit_error_count": 0},
        plan_func=lambda ns: {"status": "ready", "error_count": 0},
        launch_func=fake_launch,
        launch_watcher_func=fake_watcher,
    )
    assert payload["status"] == "g004_launched"
    assert launches == ["g004", "watcher"]


def test_chain_refuses_duplicate_running_chain_watcher(tmp_path: Path):
    _write_universe(tmp_path)
    watcher = tmp_path / "outputs/cluster/chain.pid"
    watcher.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["sleep", "30"])
    try:
        watcher.write_text(f"{proc.pid}\n")
        payload = watch_and_maybe_launch(_args(tmp_path), finalize_func=lambda ns: {"status": "pass"}, plan_func=lambda ns: {"status": "ready"})
        assert payload["status"] == "duplicate_chain_watcher_running"
        assert payload["existing_pid"] == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=5)
