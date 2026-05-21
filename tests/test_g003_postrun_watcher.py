from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from watch_g003_then_finalize import watch


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "output": "artifacts/idm/watcher.json",
        "allow_fail": False,
        "once": True,
        "poll_seconds": 0.01,
        "max_wait_seconds": -1.0,
        "watcher_pid_file": "outputs/cluster/watcher.pid",
        "replace_existing_watcher": False,
        "g003_finalization_summary": "artifacts/idm/finalize.json",
        "skip_split_stats": False,
        "force_split_stats": False,
        "split_stats_config": "configs/eval/split_stats.json",
        "split_stats_summary": "artifacts/eval/split_summary.json",
        "g003_completion_config": "configs/eval/g003_completion.json",
        "g003_audit_output": "artifacts/idm/g003_audit.json",
        "integrated_run_evidence": "artifacts/idm/integrated.json",
        "idm_summary": "artifacts/idm/idm_summary.json",
        "checkpoint_metadata": "outputs/idm/checkpoint_metadata.json",
        "metrics": "outputs/idm/metrics.json",
        "gpu_monitor": "artifacts/idm/gpu.csv",
        "attached_monitor_metadata": "artifacts/idm/monitor_meta.json",
        "train_run_summary": "artifacts/idm/train_run.json",
        "nproc_per_node": 4,
        "expected_gpus": 4,
        "shard_root": "outputs/shards",
        "log_dir": "artifacts/sources",
        "data_universe": "artifacts/sources/universe.json",
        "data_output_dir": "outputs/data",
        "idm_output_dir": "outputs/idm",
        "pid_file": "outputs/cluster/parent.pid",
        "repair_pid_glob": None,
        "num_shards": 1,
        "stale_seconds": 3600.0,
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


def test_watcher_waits_without_finalizing_while_parent_is_running(tmp_path: Path):
    _write_universe(tmp_path)
    parent = tmp_path / "outputs/cluster/parent.pid"
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent.write_text(f"{os.getpid()}\n")
    calls: list[Namespace] = []
    payload = watch(_args(tmp_path), finalize_func=lambda ns: calls.append(ns) or {"status": "pass"})
    assert payload["status"] == "waiting_active_parent"
    assert payload["progress"]["pid_running"] is True
    assert calls == []
    assert (tmp_path / "artifacts/idm/watcher.json").exists()
    assert not (tmp_path / "outputs/cluster/watcher.pid").exists()


def test_watcher_runs_finalizer_once_parent_is_inactive(tmp_path: Path):
    _write_universe(tmp_path)
    calls: list[Namespace] = []

    def fake_finalize(ns: Namespace) -> dict:
        calls.append(ns)
        return {"status": "pass", "g003_audit_status": "pass", "g003_audit_error_count": 0}

    payload = watch(_args(tmp_path), finalize_func=fake_finalize)
    assert payload["status"] == "finalized_pass"
    assert payload["g003_audit_status"] == "pass"
    assert len(calls) == 1
    assert calls[0].allow_active_parent is False
    assert calls[0].summary_out == "artifacts/idm/finalize.json"
    assert calls[0].repair_pid_glob is None


def test_watcher_treats_running_repair_pid_as_active_before_finalizing(tmp_path: Path):
    _write_universe(tmp_path)
    repair_pid_file = tmp_path / "outputs/cluster/g003_accel64_shard_0_repair.pid"
    repair_pid_file.parent.mkdir(parents=True, exist_ok=True)
    repair_pid_file.write_text(f"{os.getpid()}\n")
    parent_pid_file = tmp_path / "outputs/cluster/g003_full_compact_accel64.pid"
    parent_pid_file.write_text("999999999\n")
    calls: list[Namespace] = []

    payload = watch(
        _args(
            tmp_path,
            pid_file="outputs/cluster/g003_full_compact_accel64.pid",
            repair_pid_glob="outputs/cluster/g003_accel64_shard_*_repair.pid",
        ),
        finalize_func=lambda ns: calls.append(ns) or {"status": "pass"},
    )

    assert payload["status"] == "waiting_active_parent"
    assert payload["progress"]["pid_running"] is True
    assert payload["progress"]["decoded_recording_variants"] == 0
    assert calls == []


def test_watcher_refuses_duplicate_running_watcher(tmp_path: Path):
    _write_universe(tmp_path)
    watcher = tmp_path / "outputs/cluster/watcher.pid"
    watcher.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["sleep", "30"])
    try:
        watcher.write_text(f"{proc.pid}\n")
        payload = watch(_args(tmp_path), finalize_func=lambda ns: {"status": "pass"})
        assert payload["status"] == "duplicate_watcher_running"
        assert payload["existing_pid"] == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=5)
