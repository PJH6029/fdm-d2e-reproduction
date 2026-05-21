from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.g003_completion import validate_g003_full_idm_completion


def _config() -> dict:
    paths = {
        "data_universe": "artifacts/sources/universe.json",
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
        "expected_gpus": 4,
        "min_gpu_monitor_rows": 4,
        "require_gpu_monitor_covers_expected_gpus": True,
        "required_target_eval_split_tags": ["temporal", "heldout_recording", "heldout_game"],
        "required_source_ids": ["d2e_480p", "d2e_original"],
        "required_resolution_tiers": ["480p", "original_fhd_qhd"],
        "expected_variants_by_source": {"d2e_480p": 2, "d2e_original": 1},
        "expected_variants_by_resolution_tier": {"480p": 2, "original_fhd_qhd": 1},
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
    recordings = [
        {
            "status": "included",
            "source_id": "d2e_480p",
            "resolution_tier": "480p",
            "game": "Game",
            "recording_id": "rec_0",
            "cross_resolution_key": "Game/rec_0",
        },
        {
            "status": "included",
            "source_id": "d2e_480p",
            "resolution_tier": "480p",
            "game": "Game",
            "recording_id": "rec_1",
            "cross_resolution_key": "Game/rec_1",
        },
        {
            "status": "included",
            "source_id": "d2e_original",
            "resolution_tier": "original_fhd_qhd",
            "game": "Game",
            "recording_id": "rec_0",
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
    write_json(
        root / cfg["paths"]["decode_summary"],
        {
            "selected_recording_variants": 3,
            "num_shards": 2,
            "failures": [],
            "counts": {"train_core": 2, "target_all_eval": 2},
            "recordings": [
                {"universe_row_id": f"{row['source_id']}:{row['cross_resolution_key']}", **row}
                for row in recordings
            ],
        },
    )
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
        "timestamp,index,name\n"
        "now,0,H200\n"
        "now,1,H200\n"
        "now,2,H200\n"
        "now,3,H200\n"
    )


def test_g003_completion_audit_passes_on_full_fixture(tmp_path: Path):
    _complete_fixture(tmp_path)
    payload = validate_g003_full_idm_completion(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert payload["counts"]["train_records"] == 2
    assert payload["data_universe_counts"]["source_ids"] == {"d2e_480p": 2, "d2e_original": 1}
    assert payload["decode_counts_by_resolution_tier"] == {"480p": 2, "original_fhd_qhd": 1}
    assert payload["gpu_monitor_status"]["covers_expected_gpus"] is True


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


def test_g003_completion_audit_requires_both_d2e_resolution_tiers(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    decode_path = tmp_path / cfg["paths"]["decode_summary"]
    decode = json.loads(decode_path.read_text())
    decode["recordings"] = [row for row in decode["recordings"] if row["source_id"] == "d2e_480p"]
    decode["selected_recording_variants"] = 2
    write_json(decode_path, decode)
    payload = validate_g003_full_idm_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "decode_missing_required_sources" in codes
    assert "decode_missing_required_resolution_tiers" in codes
    assert "decode_universe_row_id_mismatch" in codes


def test_g003_completion_audit_rejects_partial_gpu_monitor(tmp_path: Path):
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
    payload = validate_g003_full_idm_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "gpu_monitor_too_few_rows" in codes
    assert "gpu_monitor_does_not_cover_expected_gpus" in codes
    assert "run_summary_gpu_monitor_missing_expected_gpus" in codes


def test_g003_completion_audit_allows_configured_accelerated_shard_count(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    cfg["allowed_shard_counts"] = [2, 4]
    decode_path = tmp_path / cfg["paths"]["decode_summary"]
    decode = json.loads(decode_path.read_text())
    decode["num_shards"] = 4
    write_json(decode_path, decode)

    payload = validate_g003_full_idm_completion(cfg, root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["allowed_shard_counts"] == [2, 4]
