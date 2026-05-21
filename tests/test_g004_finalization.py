from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json, write_jsonl
from finalize_g004_d2e_full_fdm import finalize


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "summary_out": "artifacts/fdm/finalize.json",
        "allow_fail": False,
        "skip_split_stats": False,
        "force_split_stats": False,
        "split_stats_config": "configs/eval/split_stats.json",
        "split_stats_summary": "artifacts/eval/split_summary.json",
        "g004_completion_config": "configs/eval/g004_completion.json",
        "g004_audit_output": "artifacts/fdm/g004_audit.json",
        "run_summary": "artifacts/fdm/run.json",
    }
    data.update(overrides)
    return Namespace(**data)


def _record(idx: int) -> dict:
    button = "MOUSE_LEFT_DOWN" if idx % 2 else "MOUSE_LEFT_UP"
    return {
        "sequence_id": f"seq_{idx}",
        "recording_id": "rec",
        "game": "Game",
        "eval_split_tags": ["temporal"],
        "ground_truth_tokens": ["KEY_PRESS_87", button],
    }


def _write_complete_fixture(root: Path) -> None:
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "complete"}, {"id": "G004", "status": "pending"}]})
    train = [_record(0), _record(1), _record(2)]
    target = [_record(3), _record(4)]
    preds = [{"sequence_id": row["sequence_id"], "predicted_tokens": row["ground_truth_tokens"]} for row in target]
    write_jsonl(root / "outputs/fdm/train.jsonl", train)
    write_jsonl(root / "outputs/fdm/target.jsonl", target)
    write_jsonl(root / "outputs/fdm/torch_model/predictions.jsonl", preds)
    write_json(
        root / "outputs/fdm/split.json",
        {
            "records_path": "outputs/fdm/train.jsonl",
            "target_records_source_path": "outputs/fdm/target.jsonl",
            "counts": {"pairs": 3, "train": 3, "target": 2, "mode": "explicit_target"},
            "prior_action_context": {
                "train_source": "idm_pseudolabel_previous_teacher_forced",
                "target_source": "d2e_ground_truth_previous_teacher_forced",
            },
        },
    )
    checkpoint = root / "outputs/fdm/torch_model/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"pt")
    for rel in ["outputs/fdm/resolved_config.json", "outputs/fdm/summary.json", "outputs/fdm/torch_train_summary.json", "outputs/fdm/torch_model/metrics.json", "outputs/fdm/torch_model/statistical_comparison.json", "artifacts/fdm/summary.json"]:
        write_json(root / rel, {"status": "ok"})
    write_json(root / "outputs/fdm/torch_model/convergence_report.json", {"num_validation_checkpoints": 1, "plateau_met": False})
    write_json(
        root / "outputs/fdm/checkpoint_metadata.json",
        {
            "label_source": "idm_pseudolabel",
            "oracle_ground_truth_control": False,
            "source_namespace": "d2e_full_corpus",
            "source_idm_metadata": {"exists": True},
            "data_universe": {"exists": True},
            "split_contract": {"exists": True},
            "torch_checkpoint_metadata": {
                "distributed": {"enabled": True, "world_size": 4},
                "feature_mode": "summary_causal_compact_grid8_time_prior_action",
            },
            "num_training_examples": 3,
            "target_examples": 2,
            "target_eval_split_tags": ["temporal"],
            "train_records_path": "outputs/fdm/train.jsonl",
            "target_records_path": "outputs/fdm/target.jsonl",
        },
    )
    write_json(
        root / "artifacts/fdm/run.json",
        {
            "exit_code": 0,
            "nproc_per_node": 4,
            "expected_gpus": 4,
            "gpu_monitor_status": {"rows": 4, "unique_gpu_indices": ["0", "1", "2", "3"], "expected_gpus": 4, "covers_expected_gpus": True},
        },
    )
    gpu = root / "artifacts/fdm/gpu.csv"
    gpu.parent.mkdir(parents=True, exist_ok=True)
    gpu.write_text(
        "timestamp,index,name\n"
        "now,0,H200\n"
        "now,1,H200\n"
        "now,2,H200\n"
        "now,3,H200\n"
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
            "model_name": "tiny_fdm",
            "predictions_path": "outputs/fdm/torch_model/predictions.jsonl",
            "ground_truth_path": "outputs/fdm/target.jsonl",
            "train_records_path": "outputs/fdm/train.jsonl",
            "output_dir": "outputs/fdm",
            "summary_out": "artifacts/eval/split_summary.json",
            "endpoints": "configs/eval/endpoints.json",
            "baseline_names": ["noop"],
            "split_tags": ["temporal"],
        },
    )
    write_json(
        root / "configs/eval/g004_completion.json",
        {
            "goals_path": ".omx/ultragoal/goals.json",
            "goal_id": "G004",
            "prerequisite_goals": ["G003"],
            "require_goal_checkpoint_complete": False,
            "expected_nproc_per_node": 4,
            "expected_gpus": 4,
            "min_gpu_monitor_rows": 4,
            "required_target_eval_split_tags": ["temporal"],
            "paths": {
                "fdm_train_records": "outputs/fdm/train.jsonl",
                "fdm_target_records": "outputs/fdm/target.jsonl",
                "split_summary": "outputs/fdm/split.json",
                "checkpoint_metadata": "outputs/fdm/checkpoint_metadata.json",
                "resolved_config": "outputs/fdm/resolved_config.json",
                "summary": "outputs/fdm/summary.json",
                "torch_train_summary": "outputs/fdm/torch_train_summary.json",
                "checkpoint": "outputs/fdm/torch_model/checkpoint.pt",
                "predictions": "outputs/fdm/torch_model/predictions.jsonl",
                "metrics": "outputs/fdm/torch_model/metrics.json",
                "statistical_comparison": "outputs/fdm/torch_model/statistical_comparison.json",
                "convergence_report": "outputs/fdm/torch_model/convergence_report.json",
                "artifact_summary": "artifacts/fdm/summary.json",
                "run_summary": "artifacts/fdm/run.json",
                "gpu_monitor": "artifacts/fdm/gpu.csv",
                "split_stats_summary": "artifacts/eval/split_summary.json",
            },
            "metadata_expectations": {
                "label_source": "idm_pseudolabel",
                "oracle_ground_truth_control": False,
                "source_namespace": "d2e_full_corpus",
                "source_idm_metadata.exists": True,
                "data_universe.exists": True,
                "split_contract.exists": True,
                "torch_checkpoint_metadata.distributed.enabled": True,
                "torch_checkpoint_metadata.distributed.world_size": 4,
                "torch_checkpoint_metadata.feature_mode": "summary_causal_compact_grid8_time_prior_action",
            },
            "split_summary_expectations": {
                "counts.mode": "explicit_target",
                "records_path": "outputs/fdm/train.jsonl",
                "target_records_source_path": "outputs/fdm/target.jsonl",
                "prior_action_context.train_source": "idm_pseudolabel_previous_teacher_forced",
                "prior_action_context.target_source": "d2e_ground_truth_previous_teacher_forced",
            },
        },
    )


def test_finalize_blocks_when_run_summary_missing(tmp_path: Path):
    payload = finalize(_args(tmp_path))
    assert payload["status"] == "blocked_missing_run_summary"
    assert payload["findings"][0]["code"] == "missing_run_summary"
    assert payload["g004_audit_status"] is None


def test_finalize_builds_split_stats_and_g004_audit(tmp_path: Path):
    _write_complete_fixture(tmp_path)
    payload = finalize(_args(tmp_path))
    assert payload["status"] == "pass"
    assert payload["split_stats"]["status"] == "pass"
    assert payload["g004_audit_status"] == "pass"
    assert json.loads((tmp_path / "artifacts/fdm/g004_audit.json").read_text())["status"] == "pass"
    assert json.loads((tmp_path / "artifacts/fdm/finalize.json").read_text())["status"] == "pass"
