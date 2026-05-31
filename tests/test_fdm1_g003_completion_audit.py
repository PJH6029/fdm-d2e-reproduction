from __future__ import annotations

from pathlib import Path

from fdm_d2e.data.fdm1_alignment_report import build_alignment_report_from_jsonl
from fdm_d2e.data.fdm1_mouse_bins import build_fitted_mouse_bins
from fdm_d2e.data.fdm1_action_dataset import write_action_slot_dataset_streaming_from_jsonl
from fdm_d2e.io_utils import read_json, write_json, write_jsonl
from fdm_d2e.reporting.fdm1_g003_completion import validate_fdm1_g003_action_dataset_completion
from fdm_d2e.tokenization.fdm1_actions import ActionSlotTokenizer


def _config(root: Path) -> dict:
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": "G003-50ms-action-token-dataset-pipeline",
        "require_goal_checkpoint_complete": False,
        "expected_recording_variants": 2,
        "expected_split_mode": "fdm1-g002",
        "required_source_ids": ["d2e_480p"],
        "required_resolution_tiers": ["480p"],
        "expected_tokenization_config": "artifacts/sources/fitted_config.json",
        "required_nonzero_splits": ["train_core", "target_all_eval"],
        "min_unique_tokens": 2,
        "min_visual_rows": 2,
        "omit_sha256_artifact_keys": ["action_slots", "train_core_slots", "target_all_eval_slots"],
        "required_output_hash_roles": ["all", "train_core", "target_all_eval"],
        "paths": {
            "decode_summary": "artifacts/sources/decode.json",
            "fitted_mouse_bins": "artifacts/sources/bins.json",
            "fitted_tokenization_config": "artifacts/sources/fitted_config.json",
            "action_slots": "outputs/slots/action_slots.jsonl",
            "train_core_slots": "outputs/slots/splits/train_core.jsonl",
            "target_all_eval_slots": "outputs/slots/splits/target_all_eval.jsonl",
            "dataset_summary": "outputs/slots/dataset_summary.json",
            "overflow_summary": "outputs/slots/overflow_summary.json",
            "alignment_summary": "outputs/slots/alignment_summary.json",
            "sequence_pack": "outputs/slots/sequence_pack.json",
            "visual_alignment_audit": "artifacts/sources/visual.json",
            "visual_alignment_report": "artifacts/reports/visual.md",
        },
        "required_artifacts": [
            "decode_summary",
            "fitted_mouse_bins",
            "fitted_tokenization_config",
            "action_slots",
            "train_core_slots",
            "target_all_eval_slots",
            "dataset_summary",
            "overflow_summary",
            "alignment_summary",
            "sequence_pack",
            "visual_alignment_audit",
            "visual_alignment_report",
        ],
    }


def _records() -> list[dict]:
    return [
        {
            "schema": "d2e_window_record.v1",
            "sequence_id": "d2e_480p:Toy/rec0#000000",
            "recording_id": "d2e_480p:Toy/rec0",
            "game": "Toy",
            "split": "train_core",
            "eval_split_tags": [],
            "timestamp_ns": 0,
            "bin_index": 0,
            "frame": {"path": "toy.mkv#frame=0", "index": 0, "features": [0.1]},
            "events": [{"type": "mouse_move", "dx": 3, "dy": -2, "timestamp_ns": 1_000_000}],
        },
        {
            "schema": "d2e_window_record.v1",
            "sequence_id": "d2e_480p:Toy/rec0#000001",
            "recording_id": "d2e_480p:Toy/rec0",
            "game": "Toy",
            "split": "eval",
            "eval_split_tags": ["recording_test"],
            "timestamp_ns": 50_000_000,
            "bin_index": 1,
            "frame": {"path": "toy.mkv#frame=1", "index": 1, "features": [0.2]},
            "events": [{"type": "keyboard", "event_type": "press", "vk": 87, "timestamp_ns": 51_000_000}],
        },
    ]


def _complete_fixture(root: Path) -> dict:
    cfg = _config(root)
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003-50ms-action-token-dataset-pipeline", "status": "in_progress"}]})
    input_records = root / "outputs/windows/all_records.jsonl"
    write_jsonl(input_records, _records())
    build_fitted_mouse_bins(
        [input_records],
        bins_output_path=root / cfg["paths"]["fitted_mouse_bins"],
        fitted_config_path=root / cfg["paths"]["fitted_tokenization_config"],
    )
    write_action_slot_dataset_streaming_from_jsonl(
        [input_records],
        output_dir=root / "outputs/slots",
        tokenization_config_path=cfg["paths"]["fitted_tokenization_config"],
        tokenizer=ActionSlotTokenizer(k_event_slots=4),
    )
    build_alignment_report_from_jsonl(
        root / cfg["paths"]["action_slots"],
        markdown_path=root / cfg["paths"]["visual_alignment_report"],
        audit_path=root / cfg["paths"]["visual_alignment_audit"],
        max_rows=2,
    )
    write_json(
        root / cfg["paths"]["decode_summary"],
        {
            "schema": "d2e_full_corpus_decode_summary.v1",
            "split_mode": "fdm1-g002",
            "selected_recording_variants": 2,
            "source_ids": ["d2e_480p"],
            "resolution_tiers": ["480p"],
            "failures": [],
            "counts": {"all": 2, "train_core": 1, "target_all_eval": 1},
        },
    )
    return cfg


def test_fdm1_g003_completion_audit_passes_pre_checkpoint_fixture(tmp_path: Path):
    cfg = _complete_fixture(tmp_path)
    payload = validate_fdm1_g003_action_dataset_completion(cfg, root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert payload["goal_status"] == "in_progress"
    assert payload["require_goal_checkpoint_complete"] is False


def test_fdm1_g003_completion_audit_fails_on_mismatched_dataset_count(tmp_path: Path):
    cfg = _complete_fixture(tmp_path)
    decode = read_json(tmp_path / cfg["paths"]["decode_summary"])
    decode["counts"]["all"] = 3
    write_json(tmp_path / cfg["paths"]["decode_summary"], decode)
    payload = validate_fdm1_g003_action_dataset_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "dataset_record_count_mismatch" in codes
    assert "overflow_bin_count_mismatch" in codes


def test_fdm1_g003_completion_audit_requires_fitted_tokenization_config(tmp_path: Path):
    cfg = _complete_fixture(tmp_path)
    summary = read_json(tmp_path / cfg["paths"]["dataset_summary"])
    summary["tokenization_config"] = "configs/tokenization/fdm1_action_slots.json"
    write_json(tmp_path / cfg["paths"]["dataset_summary"], summary)
    payload = validate_fdm1_g003_action_dataset_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "dataset_tokenization_config_mismatch" in codes


def test_fdm1_g003_completion_audit_requires_dataset_output_hashes(tmp_path: Path):
    cfg = _complete_fixture(tmp_path)
    summary = read_json(tmp_path / cfg["paths"]["dataset_summary"])
    summary["output_hashes"].pop("target_all_eval", None)
    write_json(tmp_path / cfg["paths"]["dataset_summary"], summary)
    payload = validate_fdm1_g003_action_dataset_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "dataset_missing_output_hash" in codes
