from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from promote_g003_accel64_to_canonical import build_promotion_mappings, promote


def _write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    _write(path, json.dumps(payload, sort_keys=True) + "\n")


def _configs() -> tuple[dict, dict]:
    source = {
        "paths": {
            "data_universe": "artifacts/sources/d2e_full_data_universe_manifest.json",
            "decode_summary": "artifacts/sources/d2e_full_corpus_decode_summary_accel64.json",
            "train_records": "outputs/data/d2e_full_corpus_accel64/train_core.jsonl",
            "target_records": "outputs/data/d2e_full_corpus_accel64/target_all_eval.jsonl",
            "checkpoint": "outputs/idm_streaming_d2e_full_compact_accel64/checkpoint.pt",
            "metrics": "outputs/idm_streaming_d2e_full_compact_accel64/metrics.json",
            "run_summary": "artifacts/idm/g003_d2e_full_idm_4xh200_train_run_accel64.json",
            "gpu_monitor": "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64.csv",
            "split_stats_summary": "artifacts/eval/g003_split_statistical_comparisons_accel64_summary.json",
            "summary": "artifacts/idm/idm_streaming_d2e_full_compact_accel64_summary.json",
        }
    }
    canonical = {
        "paths": {
            "data_universe": "artifacts/sources/d2e_full_data_universe_manifest.json",
            "decode_summary": "artifacts/sources/d2e_full_corpus_decode_summary.json",
            "train_records": "outputs/data/d2e_full_corpus/train_core.jsonl",
            "target_records": "outputs/data/d2e_full_corpus/target_all_eval.jsonl",
            "checkpoint": "outputs/idm_streaming_d2e_full_compact/checkpoint.pt",
            "metrics": "outputs/idm_streaming_d2e_full_compact/metrics.json",
            "run_summary": "artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json",
            "gpu_monitor": "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv",
            "split_stats_summary": "artifacts/eval/g003_split_statistical_comparisons_summary.json",
            "summary": "artifacts/idm/idm_streaming_d2e_full_compact_summary.json",
        }
    }
    return source, canonical


def _seed_source_files(root: Path) -> None:
    for rel in [
        "outputs/data/d2e_full_corpus_accel64/train_core.jsonl",
        "outputs/data/d2e_full_corpus_accel64/target_all_eval.jsonl",
        "outputs/data/d2e_full_corpus_accel64/target_temporal.jsonl",
        "outputs/idm_streaming_d2e_full_compact_accel64/checkpoint.pt",
        "outputs/idm_streaming_d2e_full_compact_accel64/metrics.json",
        "outputs/idm_streaming_d2e_full_compact_accel64/split_temporal_statistical_comparison.json",
        "artifacts/sources/d2e_full_corpus_decode_summary_accel64.json",
        "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64.csv",
        "artifacts/eval/g003_split_statistical_comparisons_accel64_summary.json",
        "artifacts/idm/idm_streaming_d2e_full_compact_accel64_summary.json",
        "artifacts/idm/g003_d2e_full_idm_run_accel64.json",
        "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64_attached.json",
        "artifacts/sources/g003_accel64/d2e_full_corpus_shard_0.log",
    ]:
        _write(root / rel)
    (root / "outputs/data/d2e_full_corpus_shards_accel64").mkdir(parents=True)


def _args(root: Path, **overrides) -> Namespace:
    source_config, canonical_config = _configs()
    _write_json(root / "source_config.json", source_config)
    _write_json(root / "canonical_config.json", canonical_config)
    _write_json(root / "artifacts/idm/g003_full_idm_completion_accel64_audit.json", {"status": "pass"})
    values = {
        "root": str(root),
        "source_config": "source_config.json",
        "canonical_config": "canonical_config.json",
        "source_audit": "artifacts/idm/g003_full_idm_completion_accel64_audit.json",
        "canonical_audit": "artifacts/idm/g003_full_idm_completion_audit.json",
        "output": "artifacts/idm/g003_accel64_promotion_manifest.json",
        "dry_run": False,
        "skip_source_audit_check": False,
        "allow_active_source": False,
        "allow_active_primary": False,
        "skip_finalize": True,
        "allow_fail": False,
        "source_pid_file": "outputs/cluster/g003_full_compact_accel64.pid",
        "primary_pid_file": "outputs/cluster/g003_full_compact_parallel.pid",
        "source_shard_root": "outputs/data/d2e_full_corpus_shards_accel64",
        "canonical_shard_root": "outputs/data/d2e_full_corpus_shards",
        "source_log_dir": "artifacts/sources/g003_accel64",
        "canonical_log_dir": "artifacts/sources",
        "backup_root": "artifacts/idm/g003_accel64_promotion_backups",
        "source_integrated_run_evidence": "artifacts/idm/g003_d2e_full_idm_run_accel64.json",
        "canonical_integrated_run_evidence": "artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json",
        "source_attached_monitor_metadata": "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64_attached.json",
        "canonical_attached_monitor_metadata": "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json",
        "canonical_idm_summary": "artifacts/idm/idm_streaming_d2e_full_compact_summary.json",
        "canonical_checkpoint_metadata": "outputs/idm_streaming_d2e_full_compact/checkpoint_metadata.json",
        "canonical_metrics": "outputs/idm_streaming_d2e_full_compact/metrics.json",
        "canonical_gpu_monitor": "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv",
        "canonical_train_run_summary": "artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json",
        "nproc_per_node": 4,
        "expected_gpus": 4,
    }
    values.update(overrides)
    return Namespace(**values)


def test_g003_accel64_promotion_maps_extra_canonical_artifacts(tmp_path):
    _seed_source_files(tmp_path)
    source_config, canonical_config = _configs()
    mappings = build_promotion_mappings(root=tmp_path, source_config=source_config, canonical_config=canonical_config)
    by_dest = {row["dest"]: row for row in mappings}

    assert "outputs/data/d2e_full_corpus/target_temporal.jsonl" in by_dest
    assert "outputs/idm_streaming_d2e_full_compact/split_temporal_statistical_comparison.json" in by_dest
    assert "artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json" in by_dest
    assert "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json" in by_dest
    assert "artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json" not in by_dest


def test_g003_accel64_promotion_refuses_active_primary_parent(tmp_path):
    _seed_source_files(tmp_path)
    _write(tmp_path / "outputs/cluster/g003_full_compact_parallel.pid", f"{os.getpid()}\n")
    payload = promote(_args(tmp_path))

    assert payload["status"] == "fail"
    assert any(finding["code"] == "primary_parent_still_running" for finding in payload["findings"])
    assert payload["actions"] == []


def test_g003_accel64_promotion_symlinks_sources_without_deleting_existing(tmp_path):
    _seed_source_files(tmp_path)
    _write(tmp_path / "outputs/data/d2e_full_corpus/train_core.jsonl", "old\n")
    payload = promote(_args(tmp_path))

    assert payload["status"] == "pass"
    train_link = tmp_path / "outputs/data/d2e_full_corpus/train_core.jsonl"
    assert train_link.is_symlink()
    assert train_link.resolve() == (tmp_path / "outputs/data/d2e_full_corpus_accel64/train_core.jsonl").resolve()
    backups = list((tmp_path / "artifacts/idm/g003_accel64_promotion_backups").glob("*/outputs/data/d2e_full_corpus/train_core.jsonl"))
    assert backups and backups[0].read_text() == "old\n"
    manifest = json.loads((tmp_path / "artifacts/idm/g003_accel64_promotion_manifest.json").read_text())
    assert manifest["claim_boundary"].startswith("Promotes an already-passing isolated accel64")
