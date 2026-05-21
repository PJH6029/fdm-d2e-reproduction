from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.g004_completion import validate_g004_full_fdm_completion


def _config() -> dict:
    paths = {
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
    }
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": "G004",
        "prerequisite_goals": ["G003"],
        "expected_nproc_per_node": 4,
        "expected_gpus": 4,
        "min_validation_checkpoints": 1,
        "required_target_eval_split_tags": ["temporal", "heldout_recording", "heldout_game"],
        "paths": paths,
        "metadata_expectations": {
            "label_source": "idm_pseudolabel",
            "oracle_ground_truth_control": False,
            "source_namespace": "d2e_full_corpus",
            "source_idm_metadata.exists": True,
            "data_universe.exists": True,
            "split_contract.exists": True,
            "torch_checkpoint_metadata.distributed.enabled": True,
            "torch_checkpoint_metadata.distributed.world_size": 4,
        },
    }


def _write_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join('{"x": %d}\n' % i for i in range(n)))


def _complete_fixture(root: Path) -> None:
    cfg = _config()
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "complete"}, {"id": "G004", "status": "complete"}]})
    _write_jsonl(root / cfg["paths"]["fdm_train_records"], 3)
    _write_jsonl(root / cfg["paths"]["fdm_target_records"], 2)
    _write_jsonl(root / cfg["paths"]["predictions"], 2)
    write_json(root / cfg["paths"]["split_summary"], {"counts": {"pairs": 3, "train": 3, "target": 2}})
    checkpoint = root / cfg["paths"]["checkpoint"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"pt")
    for key in ["resolved_config", "summary", "torch_train_summary", "metrics", "statistical_comparison", "artifact_summary"]:
        write_json(root / cfg["paths"][key], {"status": "ok"})
    write_json(root / cfg["paths"]["convergence_report"], {"num_validation_checkpoints": 1, "plateau_met": False})
    write_json(
        root / cfg["paths"]["checkpoint_metadata"],
        {
            "label_source": "idm_pseudolabel",
            "oracle_ground_truth_control": False,
            "source_namespace": "d2e_full_corpus",
            "source_idm_metadata": {"exists": True},
            "data_universe": {"exists": True},
            "split_contract": {"exists": True},
            "torch_checkpoint_metadata": {"distributed": {"enabled": True, "world_size": 4}},
            "num_training_examples": 3,
            "target_examples": 2,
            "target_eval_split_tags": ["temporal", "heldout_recording", "heldout_game"],
            "train_records_path": cfg["paths"]["fdm_train_records"],
            "target_records_path": cfg["paths"]["fdm_target_records"],
        },
    )
    write_json(root / cfg["paths"]["run_summary"], {"exit_code": 0, "nproc_per_node": 4, "expected_gpus": 4})
    write_json(root / cfg["paths"]["split_stats_summary"], {"status": "pass"})
    gpu = root / cfg["paths"]["gpu_monitor"]
    gpu.parent.mkdir(parents=True, exist_ok=True)
    gpu.write_text("timestamp,index\n")


def test_g004_completion_audit_passes_on_full_fixture(tmp_path: Path):
    _complete_fixture(tmp_path)
    payload = validate_g004_full_fdm_completion(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert payload["counts"]["fdm_train_records"] == 3


def test_g004_completion_audit_fails_on_prereq_and_prediction_count(tmp_path: Path):
    _complete_fixture(tmp_path)
    goals_path = tmp_path / ".omx/ultragoal/goals.json"
    write_json(goals_path, {"goals": [{"id": "G003", "status": "in_progress"}, {"id": "G004", "status": "pending"}]})
    _write_jsonl(tmp_path / _config()["paths"]["predictions"], 1)
    payload = validate_g004_full_fdm_completion(_config(), root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "goal_not_checkpointed_complete" in codes
    assert "prerequisite_goal_not_complete" in codes
    assert "predictions_count_mismatch" in codes
