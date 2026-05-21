from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.cluster.g003_monitor import build_g003_progress_report, write_g003_progress_report
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
