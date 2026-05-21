from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from monitor_g005_aux_materialization import _pid_running, build_progress


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "aux_candidates": "artifacts/sources/aux.json",
        "namespace_root": "outputs/aux",
        "source_id": None,
        "pid_file": "outputs/cluster/materialize.pid",
        "materialization_summary": "artifacts/aux/materialize_summary.json",
        "watcher_summary": "artifacts/aux/watcher.json",
        "splits": ["train", "val", "test"],
        "max_files": 10,
        "output": "artifacts/aux/progress.json",
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_candidates(root: Path) -> None:
    write_json(
        root / "artifacts/sources/aux.json",
        {
            "candidates": [
                {"id": "partial_aux", "selection_status": "selected_candidate", "source_url": "https://example.test/a", "license_id": "cc0", "size_bytes": 100},
                {"id": "complete_aux", "selection_status": "selected_candidate", "source_url": "https://example.test/b", "license_id": "mit", "size_bytes": 20},
                {"id": "missing_aux", "selection_status": "selected_candidate", "source_url": "https://example.test/c", "license_id": "mit", "size_bytes": 20},
            ]
        },
    )


def test_monitor_reports_running_partial_and_complete_sources(tmp_path: Path):
    _write_candidates(tmp_path)
    pid = tmp_path / "outputs/cluster/materialize.pid"
    pid.parent.mkdir(parents=True, exist_ok=True)
    pid.write_text(f"{os.getpid()}\n", encoding="utf-8")
    partial_raw = tmp_path / "outputs/aux/partial_aux/raw"
    partial_raw.mkdir(parents=True)
    (partial_raw / "part.zip").write_bytes(b"x" * 7)
    complete_ns = tmp_path / "outputs/aux/complete_aux"
    (complete_ns / "raw").mkdir(parents=True)
    (complete_ns / "raw/data.zip").write_bytes(b"y" * 11)
    for split in ["train", "val", "test"]:
        split_dir = complete_ns / split
        split_dir.mkdir(parents=True)
        (split_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    payload = build_progress(_args(tmp_path))
    assert payload["status"] == "running"
    assert payload["pid_running"] is True
    assert payload["raw_total_bytes"] == 18
    assert payload["expected_raw_total_bytes"] == 140
    assert payload["raw_completion_ratio"] == 18 / 140
    assert payload["raw_remaining_expected_bytes"] == 122
    assert payload["recommendation"]["code"] == "continue_materialization"
    assert payload["partial_source_ids"] == ["partial_aux"]
    assert payload["completed_source_ids"] == ["complete_aux"]
    assert payload["missing_source_ids"] == ["missing_aux"]
    partial = next(row for row in payload["aux_sources"] if row["id"] == "partial_aux")
    assert partial["raw_completion_ratio"] == 0.07
    assert partial["raw_remaining_expected_bytes"] == 93


def test_monitor_passes_after_materialization_summary_pass(tmp_path: Path):
    _write_candidates(tmp_path)
    write_json(tmp_path / "artifacts/aux/materialize_summary.json", {"status": "pass"})
    payload = build_progress(_args(tmp_path, source_id=["missing_aux"]))
    assert payload["status"] == "pass"
    assert payload["materialization_summary_status"] == "pass"
    assert payload["recommendation"]["code"] == "run_integrity_and_namespace_gates"


def test_monitor_blocks_on_error_summary(tmp_path: Path):
    _write_candidates(tmp_path)
    write_json(tmp_path / "artifacts/aux/materialize_summary.json", {"status": "blocked"})
    payload = build_progress(_args(tmp_path, source_id=["missing_aux"]))
    assert payload["status"] == "blocked"
    assert payload["error_count"] == 1
    assert payload["findings"][0]["code"] == "materialization_summary_not_pass"
    assert payload["recommendation"]["code"] == "inspect_materialization_errors"


def test_monitor_converts_estimated_size_gib_to_bytes(tmp_path: Path):
    write_json(
        tmp_path / "artifacts/sources/aux.json",
        {
            "candidates": [
                {
                    "id": "gib_aux",
                    "selection_status": "selected_candidate",
                    "source_url": "https://example.test/gib",
                    "license_id": "cc0",
                    "estimated_size_gib": 0.5,
                }
            ]
        },
    )
    payload = build_progress(_args(tmp_path, source_id=["gib_aux"]))
    assert payload["expected_raw_total_bytes"] == 536870912
    assert payload["aux_sources"][0]["expected_size_bytes"] == 536870912


def test_monitor_tolerates_files_removed_during_download_scan(tmp_path: Path, monkeypatch):
    _write_candidates(tmp_path)
    raw_dir = tmp_path / "outputs/aux/partial_aux/raw"
    raw_dir.mkdir(parents=True)
    stable = raw_dir / "stable.bin"
    vanished = raw_dir / ".cache/huggingface/download/tmp.incomplete"
    stable.write_bytes(b"x" * 7)
    vanished.parent.mkdir(parents=True)

    def fake_iter_files(path: Path) -> list[Path]:
        if path == raw_dir:
            return [stable, vanished]
        return []

    monkeypatch.setattr("monitor_g005_aux_materialization._iter_files", fake_iter_files)

    payload = build_progress(_args(tmp_path, source_id=["partial_aux"]))
    source = payload["aux_sources"][0]
    assert payload["status"] == "blocked"
    assert source["raw_total_bytes"] == 7
    assert source["raw_file_count"] == 1
    assert source["raw_transient_missing_file_count"] == 1


def test_monitor_pid_running_treats_zombie_as_exited(monkeypatch):
    monkeypatch.setattr("monitor_g005_aux_materialization.os.kill", lambda pid, sig: None)
    monkeypatch.setattr("monitor_g005_aux_materialization._pid_is_zombie", lambda pid: True)
    assert _pid_running(12345) is False
