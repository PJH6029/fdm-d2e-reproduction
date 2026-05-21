from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.cluster.g003_monitor import (
    build_g003_live_health_report,
    build_g003_progress_report,
    write_g003_live_health_report,
    write_g003_progress_report,
)
from fdm_d2e.io_utils import write_json


def _universe(root: Path) -> Path:
    rows = []
    for idx in range(4):
        rows.append(
            {
                "status": "included",
                "source_id": "d2e_480p",
                "game": "Game",
                "recording_id": f"rec_{idx}",
                "cross_resolution_key": f"Game/rec_{idx}",
                "repo_id": "repo",
                "resolution_tier": "480p",
            }
        )
    path = root / "artifacts/sources/d2e_full_data_universe_manifest.json"
    write_json(path, {"schema": "data_universe_manifest.v1", "sources": [], "recordings": rows})
    return path


def test_g003_progress_counts_log_and_recording_summaries(tmp_path):
    universe = _universe(tmp_path)
    log_dir = tmp_path / "artifacts/sources"
    shard_root = tmp_path / "outputs/data/d2e_full_corpus_shards"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "d2e_full_corpus_shard_0.log").write_text('{"decoded": 1, "total_selected": 2, "universe_row_id": "d2e_480p:Game/rec_0"}\n')
    rec_summary = shard_root / "shard_1/by_recording/d2e_480p/Game/rec_1/decode_summary.json"
    write_json(rec_summary, {"schema": "d2e_full_recording_decode_summary.v1"})
    report = build_g003_progress_report(
        shard_root=shard_root,
        log_dir=log_dir,
        data_universe=universe,
        pid_file=tmp_path / "missing.pid",
        num_shards=2,
        stale_seconds=999999,
        now=1000.0,
    )
    assert report["decoded_recording_variants"] == 2
    assert report["expected_recording_variants"] == 4
    assert report["no_progress_shards"] == []
    assert report["status"] == "not_running_partial"


def test_g003_progress_detects_stale_no_progress_shard(tmp_path):
    universe = _universe(tmp_path)
    log_dir = tmp_path / "artifacts/sources"
    log_dir.mkdir(parents=True, exist_ok=True)
    stale = log_dir / "d2e_full_corpus_shard_0.log"
    stale.write_text("")
    # Keep default mtime near current, then use a far-future `now` to mark it stale.
    report = build_g003_progress_report(
        shard_root=tmp_path / "outputs/data/d2e_full_corpus_shards",
        log_dir=log_dir,
        data_universe=universe,
        pid_file=tmp_path / "missing.pid",
        num_shards=2,
        stale_seconds=10,
        now=stale.stat().st_mtime + 100,
    )
    assert report["status"] == "review_stale_shards"
    assert 0 in report["stale_shards"]
    assert 0 in report["no_progress_shards"]


def test_g003_progress_treats_stale_active_process_as_long_recording(tmp_path):
    universe = _universe(tmp_path)
    log_dir = tmp_path / "artifacts/sources"
    log_dir.mkdir(parents=True, exist_ok=True)
    stale = log_dir / "d2e_full_corpus_shard_0.log"
    stale.write_text('{"decoded": 1, "total_selected": 2, "universe_row_id": "d2e_480p:Game/rec_0"}\n')
    report = build_g003_progress_report(
        shard_root=tmp_path / "outputs/data/d2e_full_corpus_shards",
        log_dir=log_dir,
        data_universe=universe,
        pid_file=tmp_path / "missing.pid",
        num_shards=2,
        stale_seconds=10,
        now=stale.stat().st_mtime + 100,
        active_shard_processes={0},
    )
    assert report["status"] == "running"
    assert report["stale_shards"] == []
    assert report["long_running_shards"] == [0]
    assert report["active_shard_processes"] == [0]
    assert report["shards"][0]["status"] == "running_long_recording"
    assert report["quiet_active_shards"][0]["shard_index"] == 0
    assert report["recommendation"]["code"] == "continue_monitor_long_recordings"


def test_g003_progress_reports_rate_and_eta_from_log_mtime(tmp_path):
    universe = _universe(tmp_path)
    log_dir = tmp_path / "artifacts/sources"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "d2e_full_corpus_shard_0.log"
    log_path.write_text('{"decoded": 1, "total_selected": 2, "universe_row_id": "d2e_480p:Game/rec_0"}\n')
    os.utime(log_path, (100.0, 100.0))
    report = build_g003_progress_report(
        shard_root=tmp_path / "outputs/data/d2e_full_corpus_shards",
        log_dir=log_dir,
        data_universe=universe,
        pid_file=tmp_path / "missing.pid",
        num_shards=2,
        stale_seconds=999999,
        now=200.0,
    )
    assert report["elapsed_seconds_since_first_log"] == 100.0
    assert report["decoded_recording_variants_per_hour"] == 36.0
    assert report["eta_seconds_at_current_rate"] == 300.0
    assert report["max_seconds_since_log_update"] == 100.0
    assert report["recommendation"]["code"] == "plan_resume_if_parent_exited"


def test_g003_progress_complete_requires_shards_merge_and_metrics(tmp_path):
    universe = _universe(tmp_path)
    shard_root = tmp_path / "outputs/data/d2e_full_corpus_shards"
    for shard in range(2):
        write_json(shard_root / f"shard_{shard}/decode_summary.json", {"selected_recording_variants": 2})
    (tmp_path / "outputs/data/d2e_full_corpus").mkdir(parents=True)
    (tmp_path / "outputs/data/d2e_full_corpus/train_core.jsonl").write_text("{}\n")
    (tmp_path / "outputs/data/d2e_full_corpus/target_all_eval.jsonl").write_text("{}\n")
    write_json(tmp_path / "outputs/idm_streaming_d2e_full_compact/metrics.json", {"status": "pass"})
    report = build_g003_progress_report(
        shard_root=shard_root,
        log_dir=tmp_path / "artifacts/sources",
        data_universe=universe,
        output_dir=tmp_path / "outputs/data/d2e_full_corpus",
        idm_output_dir=tmp_path / "outputs/idm_streaming_d2e_full_compact",
        pid_file=tmp_path / "missing.pid",
        num_shards=2,
        now=1000.0,
    )
    assert report["status"] == "complete"
    assert report["complete_shards"] == 2


def test_g003_progress_write_report(tmp_path):
    universe = _universe(tmp_path)
    output = tmp_path / "progress.json"
    payload = write_g003_progress_report(output, data_universe=universe, shard_root=tmp_path / "shards", log_dir=tmp_path / "logs", pid_file=tmp_path / "missing.pid", num_shards=2)
    loaded = json.loads(output.read_text())
    assert loaded["schema"] == "g003_progress_report.v1"
    assert payload["expected_recording_variants"] == 4


def _write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")


def _g003_process_snapshot(*, extractors: list[int] | None = None, include_train: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"pid": 100, "ppid": 1, "cmdline": ["bash", "scripts/run_g003_d2e_full_idm_parallel.sh"]},
        {"pid": 200, "ppid": 1, "cmdline": ["python", "scripts/watch_g003_then_finalize.py"]},
        {"pid": 300, "ppid": 1, "cmdline": ["python", "scripts/attach_g003_gpu_monitor.py"]},
    ]
    for offset, shard_index in enumerate(extractors or []):
        rows.append(
            {
                "pid": 1000 + offset,
                "ppid": 100,
                "cmdline": [
                    "python",
                    "scripts/extract_d2e_full_corpus.py",
                    "--shard-index",
                    str(shard_index),
                    "--num-shards",
                    "2",
                ],
            }
        )
    if include_train:
        rows.append({"pid": 400, "ppid": 100, "cmdline": ["torchrun", "scripts/train_idm_streaming.py"]})
    return rows


def test_g003_live_health_reports_healthy_full_extractor_topology(tmp_path):
    universe = _universe(tmp_path)
    _write_pid(tmp_path / "outputs/cluster/g003_full_compact_parallel.pid", 100)
    _write_pid(tmp_path / "outputs/cluster/g003_postrun_watcher.pid", 200)
    _write_pid(tmp_path / "outputs/cluster/g003_attached_gpu_monitor.pid", 300)
    log_dir = tmp_path / "artifacts/sources"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "d2e_full_corpus_shard_0.log").write_text('{"decoded": 1, "universe_row_id": "d2e_480p:Game/rec_0"}\n')
    (log_dir / "d2e_full_corpus_shard_1.log").write_text('{"decoded": 1, "universe_row_id": "d2e_480p:Game/rec_1"}\n')
    report = build_g003_live_health_report(
        shard_root=tmp_path / "outputs/data/d2e_full_corpus_shards",
        log_dir=log_dir,
        data_universe=universe,
        pid_file=tmp_path / "outputs/cluster/g003_full_compact_parallel.pid",
        watcher_pid_file=tmp_path / "outputs/cluster/g003_postrun_watcher.pid",
        gpu_monitor_pid_file=tmp_path / "outputs/cluster/g003_attached_gpu_monitor.pid",
        num_shards=2,
        process_snapshot=_g003_process_snapshot(extractors=[0, 1]),
        now=1000.0,
    )
    assert report["status"] == "healthy_running"
    assert report["stage"] == "extracting"
    assert report["parent"]["running"] is True
    assert report["postrun_watcher"]["running"] is True
    assert report["gpu_monitor"]["running"] is True
    assert report["active_extractor_shards"] == [0, 1]
    assert report["warnings"] == []
    assert report["progress"]["recommendation"]["code"] == "continue_waiting"


def test_g003_live_health_warns_when_incomplete_shard_has_no_extractor(tmp_path):
    universe = _universe(tmp_path)
    _write_pid(tmp_path / "outputs/cluster/g003_full_compact_parallel.pid", 100)
    _write_pid(tmp_path / "outputs/cluster/g003_postrun_watcher.pid", 200)
    _write_pid(tmp_path / "outputs/cluster/g003_attached_gpu_monitor.pid", 300)
    report = build_g003_live_health_report(
        shard_root=tmp_path / "outputs/data/d2e_full_corpus_shards",
        log_dir=tmp_path / "artifacts/sources",
        data_universe=universe,
        pid_file=tmp_path / "outputs/cluster/g003_full_compact_parallel.pid",
        watcher_pid_file=tmp_path / "outputs/cluster/g003_postrun_watcher.pid",
        gpu_monitor_pid_file=tmp_path / "outputs/cluster/g003_attached_gpu_monitor.pid",
        num_shards=2,
        process_snapshot=_g003_process_snapshot(extractors=[0]),
        now=1000.0,
    )
    assert report["status"] == "warn_live_health"
    assert report["inactive_incomplete_shards"] == [1]
    assert report["warnings"][0]["code"] == "low_active_extractor_count"


def test_g003_live_health_treats_duplicate_wrapper_processes_as_observation(tmp_path):
    universe = _universe(tmp_path)
    _write_pid(tmp_path / "outputs/cluster/g003_full_compact_parallel.pid", 100)
    _write_pid(tmp_path / "outputs/cluster/g003_postrun_watcher.pid", 200)
    _write_pid(tmp_path / "outputs/cluster/g003_attached_gpu_monitor.pid", 300)
    report = build_g003_live_health_report(
        shard_root=tmp_path / "outputs/data/d2e_full_corpus_shards",
        log_dir=tmp_path / "artifacts/sources",
        data_universe=universe,
        pid_file=tmp_path / "outputs/cluster/g003_full_compact_parallel.pid",
        watcher_pid_file=tmp_path / "outputs/cluster/g003_postrun_watcher.pid",
        gpu_monitor_pid_file=tmp_path / "outputs/cluster/g003_attached_gpu_monitor.pid",
        num_shards=2,
        process_snapshot=_g003_process_snapshot(extractors=[0, 0, 1, 1]),
        now=1000.0,
    )
    assert report["status"] == "healthy_running"
    assert report["duplicate_active_shards"] == [0, 1]
    assert report["warnings"] == []
    assert report["observations"][0]["code"] == "duplicate_extractor_processes"


def test_g003_live_health_does_not_require_extractors_during_idm_training(tmp_path):
    universe = _universe(tmp_path)
    _write_pid(tmp_path / "outputs/cluster/g003_full_compact_parallel.pid", 100)
    _write_pid(tmp_path / "outputs/cluster/g003_postrun_watcher.pid", 200)
    _write_pid(tmp_path / "outputs/cluster/g003_attached_gpu_monitor.pid", 300)
    shard_root = tmp_path / "outputs/data/d2e_full_corpus_shards"
    for shard in range(2):
        write_json(shard_root / f"shard_{shard}/decode_summary.json", {"selected_recording_variants": 2})
    report = build_g003_live_health_report(
        shard_root=shard_root,
        log_dir=tmp_path / "artifacts/sources",
        data_universe=universe,
        pid_file=tmp_path / "outputs/cluster/g003_full_compact_parallel.pid",
        watcher_pid_file=tmp_path / "outputs/cluster/g003_postrun_watcher.pid",
        gpu_monitor_pid_file=tmp_path / "outputs/cluster/g003_attached_gpu_monitor.pid",
        num_shards=2,
        process_snapshot=_g003_process_snapshot(extractors=[], include_train=True),
        now=1000.0,
    )
    assert report["status"] == "healthy_running"
    assert report["stage"] == "idm_training"
    assert report["expected_active_extractors"] == 0
    assert not any(warning["code"] == "low_active_extractor_count" for warning in report["warnings"])


def test_g003_live_health_write_report(tmp_path):
    universe = _universe(tmp_path)
    output = tmp_path / "health.json"
    payload = write_g003_live_health_report(
        output,
        data_universe=universe,
        shard_root=tmp_path / "shards",
        log_dir=tmp_path / "logs",
        pid_file=tmp_path / "missing.pid",
        watcher_pid_file=tmp_path / "missing-watcher.pid",
        gpu_monitor_pid_file=tmp_path / "missing-monitor.pid",
        num_shards=2,
        process_snapshot=[],
    )
    loaded = json.loads(output.read_text())
    assert loaded["schema"] == "g003_live_health_report.v1"
    assert payload["claim_boundary"].startswith("Live health report only")

from fdm_d2e.cluster.g003_monitor import build_g003_resume_plan


def test_g003_resume_plan_defers_when_parent_is_active():
    progress = {
        "status": "running",
        "pid_running": True,
        "decoded_recording_variants": 5,
        "expected_recording_variants": 10,
        "complete_shards": 0,
        "num_shards": 2,
        "stale_shards": [],
        "no_progress_shards": [1],
        "shards": [
            {"shard_index": 0, "status": "running_or_pending"},
            {"shard_index": 1, "status": "running_or_pending"},
        ],
    }
    plan = build_g003_resume_plan(progress_report=progress, num_shards=2)
    assert plan["status"] == "defer_active_parent"
    assert plan["runnable"] is False
    assert plan["incomplete_shards"] == [0, 1]
    assert "extract_d2e_full_corpus.py" in plan["shard_commands"][0]["shell"]


def test_g003_resume_plan_ready_when_parent_is_not_running():
    progress = {
        "status": "not_running_partial",
        "pid_running": False,
        "decoded_recording_variants": 1,
        "expected_recording_variants": 4,
        "complete_shards": 1,
        "num_shards": 2,
        "stale_shards": [],
        "no_progress_shards": [],
        "shards": [
            {"shard_index": 0, "status": "complete"},
            {"shard_index": 1, "status": "running_or_pending"},
        ],
    }
    plan = build_g003_resume_plan(progress_report=progress, num_shards=2, uv_bin="/root/.local/bin/uv")
    assert plan["status"] == "ready_to_resume"
    assert plan["runnable"] is True
    assert plan["incomplete_shards"] == [1]
    assert plan["shard_commands"][0]["argv"][0] == "/root/.local/bin/uv"
    assert "merge_d2e_full_corpus_shards.py" in plan["followup_commands_after_all_shards_complete"]["merge"]["shell"]


def test_g003_resume_plan_noop_when_all_shards_complete():
    progress = {
        "status": "complete",
        "pid_running": False,
        "decoded_recording_variants": 4,
        "expected_recording_variants": 4,
        "complete_shards": 2,
        "num_shards": 2,
        "stale_shards": [],
        "no_progress_shards": [],
        "shards": [
            {"shard_index": 0, "status": "complete"},
            {"shard_index": 1, "status": "complete"},
        ],
    }
    plan = build_g003_resume_plan(progress_report=progress, num_shards=2)
    assert plan["status"] == "no_resume_needed"
    assert plan["shard_commands"] == []
