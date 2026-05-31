from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fdm_d2e.data.fdm1_g003_finalization import finalize_fdm1_g003_action_dataset
from fdm_d2e.io_utils import read_json, write_json, write_jsonl


def _records() -> list[dict]:
    rows = []
    roles = [
        ("train_core", [], "D_IDM_LABELED_A"),
        ("train_core", [], "D_PSEUDO_B"),
        ("eval", ["recording_val"], "D_FDM_GT_EVAL"),
        ("eval", ["recording_test"], "not_in_pseudo_pool"),
        ("eval", ["heldout_game"], "not_in_pseudo_pool"),
        ("train_core", [], "D_IDM_LABELED_A"),
        ("train_core", [], "D_PSEUDO_B"),
        ("eval", ["recording_test", "heldout_game"], "D_FDM_GT_EVAL"),
    ]
    for idx, (split, tags, pseudo) in enumerate(roles):
        rows.append(
            {
                "schema": "d2e_window_record.v1",
                "sequence_id": f"d2e_480p:Toy/rec0#{idx:06d}",
                "recording_id": "d2e_480p:Toy/rec0",
                "game": "Toy",
                "split": split,
                "eval_split_tags": tags,
                "fdm1_pseudo_label_split": pseudo,
                "timestamp_ns": idx * 50_000_000,
                "bin_index": idx,
                "frame": {"path": f"toy.mkv#frame={idx}", "index": idx, "features": [float(idx)]},
                "events": [{"type": "mouse_move", "dx": idx + 1, "dy": -(idx + 1), "timestamp_ns": idx * 50_000_000 + 1_000_000}],
            }
        )
    return rows


def _write_fixture(root: Path) -> dict:
    records_path = root / "outputs/windows/all_records.jsonl"
    write_jsonl(records_path, _records())
    write_json(
        root / "artifacts/decode.json",
        {
            "schema": "d2e_full_corpus_decode_summary.v1",
            "split_mode": "fdm1-g002",
            "selected_recording_variants": 1,
            "source_ids": ["d2e_480p"],
            "resolution_tiers": ["480p"],
            "failures": [],
            "counts": {"all": len(_records()), "train_core": 4, "target_all_eval": 4},
        },
    )
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003-50ms-action-token-dataset-pipeline", "status": "in_progress"}]})
    completion_cfg = {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": "G003-50ms-action-token-dataset-pipeline",
        "require_goal_checkpoint_complete": False,
        "expected_recording_variants": 1,
        "expected_split_mode": "fdm1-g002",
        "required_source_ids": ["d2e_480p"],
        "required_resolution_tiers": ["480p"],
        "expected_tokenization_config": "artifacts/fitted_config.json",
        "required_nonzero_splits": ["train_core", "target_all_eval", "recording_val", "recording_test", "heldout_game", "pseudo_idm_labeled_a", "pseudo_pseudo_b", "pseudo_fdm_gt_eval"],
        "min_unique_tokens": 2,
        "min_visual_rows": 2,
        "paths": {
            "decode_summary": "artifacts/decode.json",
            "fitted_mouse_bins": "artifacts/bins.json",
            "fitted_tokenization_config": "artifacts/fitted_config.json",
            "action_slots": "outputs/slots/action_slots.jsonl",
            "train_core_slots": "outputs/slots/splits/train_core.jsonl",
            "target_all_eval_slots": "outputs/slots/splits/target_all_eval.jsonl",
            "recording_val_slots": "outputs/slots/splits/recording_val.jsonl",
            "recording_test_slots": "outputs/slots/splits/recording_test.jsonl",
            "heldout_game_slots": "outputs/slots/splits/heldout_game.jsonl",
            "pseudo_idm_labeled_a_slots": "outputs/slots/splits/pseudo_idm_labeled_a.jsonl",
            "pseudo_pseudo_b_slots": "outputs/slots/splits/pseudo_pseudo_b.jsonl",
            "pseudo_fdm_gt_eval_slots": "outputs/slots/splits/pseudo_fdm_gt_eval.jsonl",
            "dataset_summary": "outputs/slots/dataset_summary.json",
            "overflow_summary": "outputs/slots/overflow_summary.json",
            "alignment_summary": "outputs/slots/alignment_summary.json",
            "sequence_pack": "outputs/slots/sequence_pack.json",
            "visual_alignment_audit": "artifacts/visual.json",
            "visual_alignment_report": "artifacts/visual.md",
        },
        "required_artifacts": [
            "decode_summary", "fitted_mouse_bins", "fitted_tokenization_config", "action_slots", "train_core_slots", "target_all_eval_slots", "recording_val_slots", "recording_test_slots", "heldout_game_slots", "pseudo_idm_labeled_a_slots", "pseudo_pseudo_b_slots", "pseudo_fdm_gt_eval_slots", "dataset_summary", "overflow_summary", "alignment_summary", "sequence_pack", "visual_alignment_audit", "visual_alignment_report"
        ],
        "output_path": "artifacts/audit.json",
    }
    final_cfg = {
        "decoded_records": "outputs/windows/all_records.jsonl",
        "base_tokenization_config": str(Path.cwd() / "configs/tokenization/fdm1_action_slots.json"),
        "completion_config": "configs/final_completion.json",
        "action_output_dir": "outputs/slots",
        "k_event_slots": 4,
        "alignment_max_rows": 4,
        "paths": {
            "fitted_mouse_bins": "artifacts/bins.json",
            "fitted_tokenization_config": "artifacts/fitted_config.json",
            "action_slots": "outputs/slots/action_slots.jsonl",
            "dataset_summary": "outputs/slots/dataset_summary.json",
            "visual_alignment_audit": "artifacts/visual.json",
            "visual_alignment_report": "artifacts/visual.md",
        },
        "output_path": "artifacts/finalization.json",
    }
    (root / "configs").mkdir(exist_ok=True)
    write_json(root / "configs/final_completion.json", completion_cfg)
    write_json(root / "configs/finalization.json", final_cfg)
    return final_cfg


def test_finalize_fdm1_g003_action_dataset_runs_missing_steps(tmp_path: Path):
    cfg = _write_fixture(tmp_path)
    summary = finalize_fdm1_g003_action_dataset(cfg, root=tmp_path)
    assert summary["status"] == "pass"
    assert [step["status"] for step in summary["steps"][:3]] == ["ran", "ran", "ran"]
    assert read_json(tmp_path / "artifacts/audit.json")["status"] == "pass"
    assert (tmp_path / "outputs/slots/splits/pseudo_pseudo_b.jsonl").exists()


def test_finalize_fdm1_g003_action_dataset_skips_existing_outputs(tmp_path: Path):
    cfg = _write_fixture(tmp_path)
    first = finalize_fdm1_g003_action_dataset(cfg, root=tmp_path)
    second = finalize_fdm1_g003_action_dataset(cfg, root=tmp_path)
    assert first["status"] == "pass"
    assert second["status"] == "pass"
    assert second["steps"][0]["status"] == "skipped_existing"
    assert second["steps"][1]["status"] == "skipped_existing"


def test_finalize_g003_fdm1_action_dataset_cli(tmp_path: Path):
    _write_fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/finalize_g003_fdm1_action_dataset.py",
            "--config",
            str(tmp_path / "configs/finalization.json"),
            "--root",
            str(tmp_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "finalized FDM-1 G003 action dataset" in completed.stdout
    assert read_json(tmp_path / "artifacts/finalization.json")["status"] == "pass"
