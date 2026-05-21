from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json, write_jsonl
from finalize_g003_integrated_run import finalize


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "summary_out": "artifacts/idm/finalize.json",
        "allow_fail": False,
        "allow_active_parent": False,
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
        "num_shards": 1,
        "stale_seconds": 3600.0,
    }
    data.update(overrides)
    return Namespace(**data)


def _record(idx: int) -> dict:
    return {
        "sequence_id": f"seq_{idx}",
        "recording_id": "rec",
        "game": "Game",
        "eval_split_tags": ["temporal"],
        "ground_truth_tokens": ["KEY_PRESS_87", "MOUSE_LEFT_UP"],
    }


def _write_complete_fixture(root: Path) -> None:
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003-d2e-only-idm", "status": "in_progress"}]})
    write_json(
        root / "artifacts/sources/universe.json",
        {
            "schema": "data_universe_manifest.v1",
            "recordings": [
                {
                    "status": "included",
                    "source_id": "d2e_480p",
                    "game": "Game",
                    "recording_id": "rec",
                    "cross_resolution_key": "Game/rec",
                    "repo_id": "repo",
                    "resolution_tier": "480p",
                }
            ],
        },
    )
    write_json(root / "outputs/shards/shard_0/decode_summary.json", {"selected_recording_variants": 1})
    write_json(root / "artifacts/sources/decode_summary.json", {"selected_recording_variants": 1, "num_shards": 1, "failures": [], "counts": {"train_core": 2, "target_all_eval": 2}})
    rows = [_record(0), _record(1)]
    preds = [{"sequence_id": row["sequence_id"], "predicted_tokens": row["ground_truth_tokens"]} for row in rows]
    write_jsonl(root / "outputs/data/train.jsonl", rows)
    write_jsonl(root / "outputs/data/target.jsonl", rows)
    write_jsonl(root / "outputs/idm/pseudolabels.jsonl", preds)
    write_jsonl(root / "outputs/idm/predictions.jsonl", preds)
    for rel in ["outputs/idm/checkpoint.pt", "outputs/idm/streaming_stats.json", "outputs/idm/train_history.json", "outputs/idm/convergence_report.json", "outputs/idm/label_quality_report.json", "outputs/idm/statistical_comparison.json", "outputs/idm/resolved_config.json"]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".pt"):
            path.write_bytes(b"pt")
        else:
            write_json(path, {"status": "ok"})
    write_json(root / "outputs/idm/metrics.json", {"status": "ok"})
    write_json(
        root / "outputs/idm/checkpoint_metadata.json",
        {
            "source_namespace": "d2e_full_corpus",
            "data_universe": {"exists": True},
            "split_contract": {"exists": True},
            "distributed": {"enabled": True, "world_size": 4},
            "train_records": 2,
            "target_records": 2,
            "target_eval_split_tags": ["temporal"],
            "checkpoint_path": "outputs/idm/checkpoint.pt",
            "metrics_path": "outputs/idm/metrics.json",
            "label_quality_report_path": "outputs/idm/label_quality_report.json",
            "statistical_comparison_path": "outputs/idm/statistical_comparison.json",
            "convergence_report_path": "outputs/idm/convergence_report.json",
        },
    )
    write_json(root / "artifacts/idm/integrated.json", {"idm_nproc_per_node": 4})
    write_json(root / "artifacts/idm/idm_summary.json", {"schema": "streaming_idm_train_summary.v1"})
    write_json(root / "artifacts/idm/monitor_meta.json", {"samples": 1})
    gpu = root / "artifacts/idm/gpu.csv"
    gpu.parent.mkdir(parents=True, exist_ok=True)
    gpu.write_text(
        "sample_unix,parent_pid,timestamp,index\n"
        "1,2,now,0\n"
        "1,2,now,1\n"
        "1,2,now,2\n"
        "1,2,now,3\n"
    )
    endpoints = {
        "schema": "primary_endpoints.v1",
        "cluster_key": "recording_id",
        "bootstrap": {"n_resamples": 5, "seed": 1},
        "reference_baseline": "noop",
        "endpoints": [{"name": "keyboard_accuracy", "metric_path": ["keyboard", "accuracy"], "direction": "higher"}],
    }
    write_json(root / "configs/eval/endpoints.json", endpoints)
    write_json(
        root / "configs/eval/split_stats.json",
        {
            "model_name": "tiny",
            "predictions_path": "outputs/idm/predictions.jsonl",
            "ground_truth_path": "outputs/data/target.jsonl",
            "train_records_path": "outputs/data/train.jsonl",
            "output_dir": "outputs/idm",
            "summary_out": "artifacts/eval/split_summary.json",
            "endpoints": "configs/eval/endpoints.json",
            "baseline_names": ["noop"],
            "split_tags": ["temporal"],
        },
    )
    write_json(
        root / "configs/eval/g003_completion.json",
        {
            "goals_path": ".omx/ultragoal/goals.json",
            "goal_id": "G003-d2e-only-idm",
            "require_goal_checkpoint_complete": False,
            "expected_recording_variants": 1,
            "expected_shards": 1,
            "expected_nproc_per_node": 4,
            "required_target_eval_split_tags": ["temporal"],
            "paths": {
                "decode_summary": "artifacts/sources/decode_summary.json",
                "train_records": "outputs/data/train.jsonl",
                "target_records": "outputs/data/target.jsonl",
                "checkpoint": "outputs/idm/checkpoint.pt",
                "checkpoint_metadata": "outputs/idm/checkpoint_metadata.json",
                "resolved_config": "outputs/idm/resolved_config.json",
                "streaming_stats": "outputs/idm/streaming_stats.json",
                "train_history": "outputs/idm/train_history.json",
                "convergence_report": "outputs/idm/convergence_report.json",
                "pseudolabels": "outputs/idm/pseudolabels.jsonl",
                "predictions": "outputs/idm/predictions.jsonl",
                "metrics": "outputs/idm/metrics.json",
                "label_quality_report": "outputs/idm/label_quality_report.json",
                "statistical_comparison": "outputs/idm/statistical_comparison.json",
                "summary": "artifacts/idm/idm_summary.json",
                "run_summary": "artifacts/idm/train_run.json",
                "gpu_monitor": "artifacts/idm/gpu.csv",
                "split_stats_summary": "artifacts/eval/split_summary.json",
            },
            "metadata_expectations": {
                "source_namespace": "d2e_full_corpus",
                "data_universe.exists": True,
                "split_contract.exists": True,
                "distributed.enabled": True,
                "distributed.world_size": 4,
            },
        },
    )


def test_finalize_blocks_by_default_when_parent_is_running(tmp_path: Path):
    write_json(
        tmp_path / "artifacts/sources/universe.json",
        {"recordings": [{"status": "included", "source_id": "d2e_480p", "game": "Game", "recording_id": "rec", "cross_resolution_key": "Game/rec"}]},
    )
    pid_file = tmp_path / "outputs/cluster/parent.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()) + "\n")
    payload = finalize(_args(tmp_path))
    assert payload["status"] == "blocked_active_parent"
    assert payload["progress"]["log_dir"] == str(tmp_path / "artifacts/sources")
    assert payload["findings"][0]["code"] == "parent_still_running"
    assert payload["g003_audit_status"] is None


def test_finalize_builds_split_stats_train_summary_and_audit(tmp_path: Path):
    _write_complete_fixture(tmp_path)
    payload = finalize(_args(tmp_path))
    assert payload["status"] == "pass"
    assert payload["progress"]["log_dir"] == str(tmp_path / "artifacts/sources")
    assert payload["split_stats"]["status"] == "pass"
    assert payload["attached_train_summary_exit_code"] == 0
    assert payload["g003_audit_status"] == "pass"
    assert json.loads((tmp_path / "artifacts/idm/train_run.json").read_text())["exit_code"] == 0
    assert json.loads((tmp_path / "artifacts/idm/g003_audit.json").read_text())["status"] == "pass"
