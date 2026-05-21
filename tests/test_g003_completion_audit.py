from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.g003_completion import validate_g003_full_idm_completion


def _config() -> dict:
    paths = {
        "decode_summary": "artifacts/sources/decode.json",
        "train_records": "outputs/train.jsonl",
        "target_records": "outputs/target.jsonl",
        "checkpoint": "outputs/checkpoint.pt",
        "checkpoint_metadata": "outputs/checkpoint_metadata.json",
        "resolved_config": "outputs/resolved_config.json",
        "streaming_stats": "outputs/streaming_stats.json",
        "train_history": "outputs/train_history.json",
        "convergence_report": "outputs/convergence_report.json",
        "pseudolabels": "outputs/pseudolabels.jsonl",
        "predictions": "outputs/predictions.jsonl",
        "metrics": "outputs/metrics.json",
        "label_quality_report": "outputs/label_quality_report.json",
        "statistical_comparison": "outputs/statistical_comparison.json",
        "summary": "artifacts/idm/summary.json",
        "run_summary": "artifacts/idm/run.json",
        "gpu_monitor": "artifacts/idm/gpu.csv",
        "split_stats_summary": "artifacts/eval/split_summary.json",
    }
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": "G003",
        "expected_recording_variants": 3,
        "expected_shards": 2,
        "expected_nproc_per_node": 4,
        "required_target_eval_split_tags": ["temporal", "heldout_recording", "heldout_game"],
        "paths": paths,
        "metadata_expectations": {
            "source_namespace": "d2e_full_corpus",
            "data_universe.exists": True,
            "split_contract.exists": True,
            "distributed.enabled": True,
            "distributed.world_size": 4,
        },
    }


def _write_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join('{"x": %d}\n' % i for i in range(n)))


def _complete_fixture(root: Path) -> None:
    cfg = _config()
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "complete"}]})
    write_json(root / cfg["paths"]["decode_summary"], {"selected_recording_variants": 3, "num_shards": 2, "failures": [], "counts": {"train_core": 2, "target_all_eval": 2}})
    _write_jsonl(root / cfg["paths"]["train_records"], 2)
    _write_jsonl(root / cfg["paths"]["target_records"], 2)
    _write_jsonl(root / cfg["paths"]["pseudolabels"], 2)
    _write_jsonl(root / cfg["paths"]["predictions"], 2)
    for key in ["checkpoint"]:
        path = root / cfg["paths"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pt")
    for key in ["resolved_config", "streaming_stats", "train_history", "convergence_report", "metrics", "label_quality_report", "statistical_comparison", "summary"]:
        write_json(root / cfg["paths"][key], {"status": "ok"})
    write_json(
        root / cfg["paths"]["checkpoint_metadata"],
        {
            "source_namespace": "d2e_full_corpus",
            "data_universe": {"exists": True},
            "split_contract": {"exists": True},
            "distributed": {"enabled": True, "world_size": 4},
            "train_records": 2,
            "target_records": 2,
            "target_eval_split_tags": ["temporal", "heldout_recording", "heldout_game"],
        },
    )
    write_json(root / cfg["paths"]["run_summary"], {"exit_code": 0, "nproc_per_node": 4})
    write_json(root / cfg["paths"]["split_stats_summary"], {"status": "pass"})
    gpu = root / cfg["paths"]["gpu_monitor"]
    gpu.parent.mkdir(parents=True, exist_ok=True)
    gpu.write_text("timestamp,index\n")


def test_g003_completion_audit_passes_on_full_fixture(tmp_path: Path):
    _complete_fixture(tmp_path)
    payload = validate_g003_full_idm_completion(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert payload["counts"]["train_records"] == 2


def test_g003_completion_audit_fails_on_partial_counts_and_goal(tmp_path: Path):
    _complete_fixture(tmp_path)
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "in_progress"}]})
    _write_jsonl(tmp_path / _config()["paths"]["predictions"], 1)
    payload = validate_g003_full_idm_completion(_config(), root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "goal_not_checkpointed_complete" in codes
    assert "predictions_count_mismatch" in codes


def test_g003_completion_audit_can_run_as_pre_checkpoint_evidence_gate(tmp_path: Path):
    _complete_fixture(tmp_path)
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "in_progress"}]})
    cfg = _config()
    cfg["require_goal_checkpoint_complete"] = False
    payload = validate_g003_full_idm_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "pass"
    assert payload["goal_status"] == "in_progress"
    assert payload["require_goal_checkpoint_complete"] is False
    assert "goal_not_checkpointed_complete" not in codes
