from __future__ import annotations

import json
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
        "source_idm_metadata": "outputs/idm/checkpoint_metadata.json",
        "data_universe": "artifacts/sources/universe.json",
        "split_contract": "artifacts/sources/split_contract.json",
        "g003_completion_audit": "artifacts/idm/g003_audit.json",
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
        "expected_recording_variants": 3,
        "min_gpu_monitor_rows": 4,
        "require_gpu_monitor_covers_expected_gpus": True,
        "require_g003_completion_audit_pass": True,
        "min_validation_checkpoints": 1,
        "required_source_ids": ["d2e_480p", "d2e_original"],
        "required_resolution_tiers": ["480p", "original_fhd_qhd"],
        "expected_variants_by_source": {"d2e_480p": 2, "d2e_original": 1},
        "expected_variants_by_resolution_tier": {"480p": 2, "original_fhd_qhd": 1},
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
            "torch_checkpoint_metadata.feature_mode": "summary_causal_compact_grid8_time_prior_action",
        },
        "source_idm_metadata_expectations": {
            "source_namespace": "d2e_full_corpus",
            "data_universe.exists": True,
            "split_contract.exists": True,
            "distributed.enabled": True,
            "distributed.world_size": 4,
        },
        "split_summary_expectations": {
            "counts.mode": "explicit_target",
            "records_path": paths["fdm_train_records"],
            "target_records_source_path": paths["fdm_target_records"],
            "prior_action_context.train_source": "idm_pseudolabel_previous_teacher_forced",
            "prior_action_context.target_source": "d2e_ground_truth_previous_teacher_forced",
        },
        "resolved_config_expectations": {
            "config.records_path": "outputs/data/d2e_full_corpus/train_core.jsonl",
            "config.target_records_path": "outputs/data/d2e_full_corpus/target_all_eval.jsonl",
            "config.data_universe": paths["data_universe"],
            "config.split_contract": paths["split_contract"],
            "config.source_idm_metadata": paths["source_idm_metadata"],
            "config.source_namespace": "d2e_full_corpus",
        },
    }


def _write_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join('{"x": %d}\n' % i for i in range(n)))


def _complete_fixture(root: Path) -> None:
    cfg = _config()
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "complete"}, {"id": "G004", "status": "complete"}]})
    recordings = [
        {
            "status": "included",
            "source_id": "d2e_480p",
            "resolution_tier": "480p",
            "cross_resolution_key": "Game/rec_0",
        },
        {
            "status": "included",
            "source_id": "d2e_480p",
            "resolution_tier": "480p",
            "cross_resolution_key": "Game/rec_1",
        },
        {
            "status": "included",
            "source_id": "d2e_original",
            "resolution_tier": "original_fhd_qhd",
            "cross_resolution_key": "Game/rec_0",
        },
    ]
    write_json(
        root / cfg["paths"]["data_universe"],
        {
            "schema": "data_universe_manifest.v1",
            "decision_gates": {"full_success_requires_sources": ["d2e_480p", "d2e_original"]},
            "recordings": recordings,
        },
    )
    write_json(root / cfg["paths"]["split_contract"], {"schema": "split_contract.v1"})
    write_json(
        root / cfg["paths"]["source_idm_metadata"],
        {
            "schema": "idm_checkpoint_metadata.v1",
            "source_namespace": "d2e_full_corpus",
            "data_universe": {"exists": True, "path": cfg["paths"]["data_universe"]},
            "split_contract": {"exists": True, "path": cfg["paths"]["split_contract"]},
            "distributed": {"enabled": True, "world_size": 4},
            "source_ids": ["d2e_480p", "d2e_original"],
            "target_source_ids": ["d2e_480p", "d2e_original"],
            "resolution_tiers": ["480p", "original_fhd_qhd"],
            "target_resolution_tiers": ["480p", "original_fhd_qhd"],
        },
    )
    write_json(
        root / cfg["paths"]["g003_completion_audit"],
        {
            "schema": "g003_full_idm_completion_audit.v1",
            "status": "pass",
            "expected_recording_variants": 3,
            "data_universe_counts": {
                "included_recording_variants": 3,
                "source_ids": {"d2e_480p": 2, "d2e_original": 1},
                "resolution_tiers": {"480p": 2, "original_fhd_qhd": 1},
            },
            "error_count": 0,
        },
    )
    _write_jsonl(root / cfg["paths"]["fdm_train_records"], 3)
    _write_jsonl(root / cfg["paths"]["fdm_target_records"], 2)
    _write_jsonl(root / cfg["paths"]["predictions"], 2)
    write_json(
        root / cfg["paths"]["split_summary"],
        {
            "records_path": cfg["paths"]["fdm_train_records"],
            "target_records_source_path": cfg["paths"]["fdm_target_records"],
            "counts": {
                "pairs": 3,
                "train": 3,
                "target": 2,
                "mode": "explicit_target",
                "source_ids": {"d2e_480p": 2, "d2e_original": 1},
                "target_source_ids": {"d2e_480p": 1, "d2e_original": 1},
                "resolution_tiers": {"480p": 2, "original_fhd_qhd": 1},
                "target_resolution_tiers": {"480p": 1, "original_fhd_qhd": 1},
            },
            "prior_action_context": {
                "train_source": "idm_pseudolabel_previous_teacher_forced",
                "target_source": "d2e_ground_truth_previous_teacher_forced",
            },
        },
    )
    checkpoint = root / cfg["paths"]["checkpoint"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"pt")
    for key in ["summary", "torch_train_summary", "metrics", "statistical_comparison", "artifact_summary"]:
        write_json(root / cfg["paths"][key], {"status": "ok"})
    write_json(
        root / cfg["paths"]["resolved_config"],
        {
            "schema": "streaming_fdm_resolved_config.v1",
            "config": {
                "records_path": "outputs/data/d2e_full_corpus/train_core.jsonl",
                "target_records_path": "outputs/data/d2e_full_corpus/target_all_eval.jsonl",
                "data_universe": cfg["paths"]["data_universe"],
                "split_contract": cfg["paths"]["split_contract"],
                "source_idm_metadata": cfg["paths"]["source_idm_metadata"],
                "source_namespace": "d2e_full_corpus",
            },
        },
    )
    write_json(root / cfg["paths"]["convergence_report"], {"num_validation_checkpoints": 1, "plateau_met": False})
    write_json(
        root / cfg["paths"]["checkpoint_metadata"],
        {
            "label_source": "idm_pseudolabel",
            "oracle_ground_truth_control": False,
            "source_namespace": "d2e_full_corpus",
            "source_idm_metadata": {"exists": True, "path": cfg["paths"]["source_idm_metadata"]},
            "data_universe": {"exists": True, "path": cfg["paths"]["data_universe"]},
            "split_contract": {"exists": True, "path": cfg["paths"]["split_contract"]},
            "source_ids": ["d2e_480p", "d2e_original"],
            "target_source_ids": ["d2e_480p", "d2e_original"],
            "resolution_tiers": ["480p", "original_fhd_qhd"],
            "target_resolution_tiers": ["480p", "original_fhd_qhd"],
            "torch_checkpoint_metadata": {
                "distributed": {"enabled": True, "world_size": 4},
                "feature_mode": "summary_causal_compact_grid8_time_prior_action",
            },
            "num_training_examples": 3,
            "target_examples": 2,
            "target_eval_split_tags": ["temporal", "heldout_recording", "heldout_game"],
            "train_records_path": cfg["paths"]["fdm_train_records"],
            "target_records_path": cfg["paths"]["fdm_target_records"],
        },
    )
    write_json(
        root / cfg["paths"]["run_summary"],
        {
            "exit_code": 0,
            "nproc_per_node": 4,
            "expected_gpus": 4,
            "gpu_monitor_status": {
                "rows": 4,
                "unique_gpu_indices": ["0", "1", "2", "3"],
                "expected_gpus": 4,
                "covers_expected_gpus": True,
            },
        },
    )
    write_json(root / cfg["paths"]["split_stats_summary"], {"status": "pass"})
    gpu = root / cfg["paths"]["gpu_monitor"]
    gpu.parent.mkdir(parents=True, exist_ok=True)
    gpu.write_text(
        "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw\n"
        "now,0,H200,90,10,1,80,200\n"
        "now,1,H200,91,11,1,80,201\n"
        "now,2,H200,92,12,1,80,202\n"
        "now,3,H200,93,13,1,80,203\n"
    )


def test_g004_completion_audit_passes_on_full_fixture(tmp_path: Path):
    _complete_fixture(tmp_path)
    payload = validate_g004_full_fdm_completion(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert payload["counts"]["fdm_train_records"] == 3
    assert payload["data_universe_counts"]["source_ids"] == {"d2e_480p": 2, "d2e_original": 1}
    assert payload["gpu_monitor_status"]["covers_expected_gpus"] is True


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


def test_g004_completion_audit_rejects_recording_tail_split_mode(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    write_json(
        tmp_path / cfg["paths"]["split_summary"],
        {
            "records_path": cfg["paths"]["fdm_target_records"],
            "target_records_source_path": cfg["paths"]["fdm_target_records"],
            "counts": {"pairs": 3, "train": 3, "target": 2, "mode": "recording_tail"},
        },
    )
    payload = validate_g004_full_fdm_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "split_summary_expectation_mismatch" in codes


def test_g004_completion_audit_requires_both_d2e_source_tiers(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    metadata_path = tmp_path / cfg["paths"]["checkpoint_metadata"]
    metadata = json.loads(metadata_path.read_text())
    metadata["source_ids"] = ["d2e_480p"]
    metadata["target_source_ids"] = ["d2e_480p"]
    metadata["resolution_tiers"] = ["480p"]
    metadata["target_resolution_tiers"] = ["480p"]
    write_json(metadata_path, metadata)
    split_path = tmp_path / cfg["paths"]["split_summary"]
    split_summary = json.loads(split_path.read_text())
    split_summary["counts"]["source_ids"] = {"d2e_480p": 3}
    split_summary["counts"]["target_source_ids"] = {"d2e_480p": 2}
    split_summary["counts"]["resolution_tiers"] = {"480p": 3}
    split_summary["counts"]["target_resolution_tiers"] = {"480p": 2}
    write_json(split_path, split_summary)
    payload = validate_g004_full_fdm_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "metadata_missing_required_source_ids" in codes
    assert "metadata_missing_required_resolution_tiers" in codes
    assert "split_summary_missing_required_source_ids" in codes
    assert "split_summary_missing_required_resolution_tiers" in codes


def test_g004_completion_audit_rejects_nonpassing_g003_audit(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    write_json(
        tmp_path / cfg["paths"]["g003_completion_audit"],
        {
            "schema": "g003_full_idm_completion_audit.v1",
            "status": "fail",
            "expected_recording_variants": 3,
            "data_universe_counts": {
                "included_recording_variants": 2,
                "source_ids": {"d2e_480p": 2},
                "resolution_tiers": {"480p": 2},
            },
            "error_count": 3,
        },
    )
    payload = validate_g004_full_fdm_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "g003_completion_audit_not_pass" in codes
    assert "g003_data_universe_source_count_mismatch" in codes
    assert "g003_data_universe_resolution_tier_count_mismatch" in codes


def test_g004_completion_audit_rejects_partial_gpu_monitor(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    (tmp_path / cfg["paths"]["gpu_monitor"]).write_text(
        "timestamp,index,name\n"
        "now,0,H200\n"
        "now,1,H200\n"
    )
    write_json(
        tmp_path / cfg["paths"]["run_summary"],
        {
            "exit_code": 0,
            "nproc_per_node": 4,
            "expected_gpus": 4,
            "gpu_monitor_status": {
                "rows": 2,
                "unique_gpu_indices": ["0", "1"],
                "expected_gpus": 4,
                "covers_expected_gpus": False,
            },
        },
    )
    payload = validate_g004_full_fdm_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "gpu_monitor_too_few_rows" in codes
    assert "gpu_monitor_does_not_cover_expected_gpus" in codes
    assert "run_summary_gpu_monitor_missing_expected_gpus" in codes
