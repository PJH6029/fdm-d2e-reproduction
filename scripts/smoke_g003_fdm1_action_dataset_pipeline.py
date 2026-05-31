#!/usr/bin/env python3
"""Synthetic end-to-end smoke for the G003 action-slot finalization path.

This does not use D2E data and must never be used as completion evidence for
G003.  It verifies that the post-extraction pipeline stages can compose:
synthetic decoded windows -> fitted mouse bins -> action-slot packs -> visual
alignment -> completion audit -> evidence bundle -> monitor -> checkpoint handoff.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fdm_d2e.data.fdm1_g003_finalization import finalize_fdm1_g003_action_dataset
from fdm_d2e.io_utils import read_json, write_json, write_jsonl

from scripts.build_fdm1_g003_checkpoint_handoff import build_handoff
from scripts.build_fdm1_g003_evidence_bundle import build_bundle
from scripts.monitor_g003_fdm1_action_dataset_pod import collect_status

GOAL_ID = "G003-50ms-action-token-dataset-pipeline"
DEFAULT_ROOT = ".omx/tmp/fdm1_g003_pipeline_smoke"
DEFAULT_SUMMARY_OUT = "artifacts/sources/fdm1_g003_pipeline_smoke_summary.json"


def _records() -> list[dict[str, Any]]:
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
    rows: list[dict[str, Any]] = []
    for idx, (split, tags, pseudo) in enumerate(roles):
        rows.append(
            {
                "schema": "d2e_window_record.v1",
                "sequence_id": f"d2e_480p:SmokeGame/rec0#{idx:06d}",
                "recording_id": "d2e_480p:SmokeGame/rec0",
                "game": "SmokeGame",
                "split": split,
                "eval_split_tags": tags,
                "fdm1_pseudo_label_split": pseudo,
                "timestamp_ns": idx * 50_000_000,
                "bin_index": idx,
                "frame": {"path": f"smoke.mkv#frame={idx}", "index": idx, "features": [float(idx)], "grid8": [0.0] * 192},
                "events": [
                    {"type": "mouse_move", "dx": idx + 1, "dy": -(idx + 1), "timestamp_ns": idx * 50_000_000 + 1_000_000},
                    {"type": "keyboard", "event_type": "press" if idx % 2 == 0 else "release", "vk": 87, "timestamp_ns": idx * 50_000_000 + 2_000_000},
                ],
            }
        )
    return rows


def write_fixture(root: Path) -> dict[str, Any]:
    records = _records()
    write_jsonl(root / "outputs/windows/all_records.jsonl", records)
    write_json(
        root / "artifacts/sources/fdm1_d2e_480p_window_records_decode_summary.json",
        {
            "schema": "d2e_full_corpus_decode_summary.v1",
            "split_mode": "fdm1-g002",
            "selected_recording_variants": 1,
            "source_ids": ["d2e_480p"],
            "resolution_tiers": ["480p"],
            "failures": [],
            "counts": {"all": len(records), "train_core": 4, "target_all_eval": 4},
        },
    )
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": GOAL_ID, "status": "in_progress"}]})
    completion_cfg = {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": GOAL_ID,
        "require_goal_checkpoint_complete": False,
        "expected_recording_variants": 1,
        "expected_split_mode": "fdm1-g002",
        "required_source_ids": ["d2e_480p"],
        "required_resolution_tiers": ["480p"],
        "expected_tokenization_config": "artifacts/sources/fdm1_action_slots_fitted_config.json",
        "required_nonzero_splits": ["train_core", "target_all_eval", "recording_val", "recording_test", "heldout_game", "pseudo_idm_labeled_a", "pseudo_pseudo_b", "pseudo_fdm_gt_eval"],
        "min_unique_tokens": 2,
        "min_visual_rows": 2,
        "omit_sha256_artifact_keys": ["action_slots", "train_core_slots", "target_all_eval_slots", "recording_val_slots", "recording_test_slots", "heldout_game_slots", "pseudo_idm_labeled_a_slots", "pseudo_pseudo_b_slots", "pseudo_fdm_gt_eval_slots"],
        "required_output_hash_roles": ["all", "train_core", "target_all_eval", "recording_val", "recording_test", "heldout_game", "pseudo_idm_labeled_a", "pseudo_pseudo_b", "pseudo_fdm_gt_eval"],
        "paths": {
            "decode_summary": "artifacts/sources/fdm1_d2e_480p_window_records_decode_summary.json",
            "fitted_mouse_bins": "artifacts/sources/fdm1_g003_fitted_mouse_bins.json",
            "fitted_tokenization_config": "artifacts/sources/fdm1_action_slots_fitted_config.json",
            "action_slots": "outputs/data/fdm1_action_slots/action_slots.jsonl",
            "train_core_slots": "outputs/data/fdm1_action_slots/splits/train_core.jsonl",
            "target_all_eval_slots": "outputs/data/fdm1_action_slots/splits/target_all_eval.jsonl",
            "recording_val_slots": "outputs/data/fdm1_action_slots/splits/recording_val.jsonl",
            "recording_test_slots": "outputs/data/fdm1_action_slots/splits/recording_test.jsonl",
            "heldout_game_slots": "outputs/data/fdm1_action_slots/splits/heldout_game.jsonl",
            "pseudo_idm_labeled_a_slots": "outputs/data/fdm1_action_slots/splits/pseudo_idm_labeled_a.jsonl",
            "pseudo_pseudo_b_slots": "outputs/data/fdm1_action_slots/splits/pseudo_pseudo_b.jsonl",
            "pseudo_fdm_gt_eval_slots": "outputs/data/fdm1_action_slots/splits/pseudo_fdm_gt_eval.jsonl",
            "dataset_summary": "outputs/data/fdm1_action_slots/dataset_summary.json",
            "overflow_summary": "outputs/data/fdm1_action_slots/overflow_summary.json",
            "alignment_summary": "outputs/data/fdm1_action_slots/alignment_summary.json",
            "sequence_pack": "outputs/data/fdm1_action_slots/sequence_pack.json",
            "visual_alignment_audit": "artifacts/sources/fdm1_g003_action_alignment_visual_check.json",
            "visual_alignment_report": "artifacts/reports/fdm1_g003_action_alignment_visual_check.md",
        },
        "required_artifacts": [
            "decode_summary", "fitted_mouse_bins", "fitted_tokenization_config", "action_slots", "train_core_slots", "target_all_eval_slots", "recording_val_slots", "recording_test_slots", "heldout_game_slots", "pseudo_idm_labeled_a_slots", "pseudo_pseudo_b_slots", "pseudo_fdm_gt_eval_slots", "dataset_summary", "overflow_summary", "alignment_summary", "sequence_pack", "visual_alignment_audit", "visual_alignment_report"
        ],
        "output_path": "artifacts/sources/fdm1_g003_action_dataset_completion_audit.json",
    }
    final_cfg = {
        "decoded_records": "outputs/windows/all_records.jsonl",
        "base_tokenization_config": str((Path.cwd() / "configs/tokenization/fdm1_action_slots.json").resolve()),
        "completion_config": "configs/eval/fdm1_g003_action_dataset_completion.yaml",
        "action_output_dir": "outputs/data/fdm1_action_slots",
        "k_event_slots": 4,
        "alignment_max_rows": 4,
        "paths": {
            "fitted_mouse_bins": "artifacts/sources/fdm1_g003_fitted_mouse_bins.json",
            "fitted_tokenization_config": "artifacts/sources/fdm1_action_slots_fitted_config.json",
            "action_slots": "outputs/data/fdm1_action_slots/action_slots.jsonl",
            "dataset_summary": "outputs/data/fdm1_action_slots/dataset_summary.json",
            "visual_alignment_audit": "artifacts/sources/fdm1_g003_action_alignment_visual_check.json",
            "visual_alignment_report": "artifacts/reports/fdm1_g003_action_alignment_visual_check.md",
        },
        "output_path": "artifacts/sources/fdm1_g003_action_dataset_finalization_summary.json",
    }
    write_json(root / "configs/eval/fdm1_g003_action_dataset_completion.yaml", completion_cfg)
    write_json(root / "configs/data/fdm1_g003_action_dataset_finalization.yaml", final_cfg)
    return final_cfg


def run_smoke(root: Path, *, force: bool = False) -> dict[str, Any]:
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    final_cfg = write_fixture(root)
    finalization = finalize_fdm1_g003_action_dataset(final_cfg, root=root, force=True)
    completion_cfg = read_json(root / "configs/eval/fdm1_g003_action_dataset_completion.yaml")
    bundle = build_bundle(completion_cfg, root=root)
    monitor = collect_status(root=root, completion_config_path="configs/eval/fdm1_g003_action_dataset_completion.yaml")
    write_json(root / "artifacts/cluster/fdm1_g003_action_dataset_pod_monitor.json", monitor)
    handoff = build_handoff(root=root)
    summary = {
        "schema": "fdm1_g003_pipeline_smoke_summary.v1",
        "status": "pass" if finalization.get("status") == "pass" and bundle.get("status") == "pass" and monitor.get("status") == "pass" and handoff.get("status") == "ready_to_checkpoint" else "fail",
        "root": str(root),
        "finalization_status": finalization.get("status"),
        "completion_audit_status": finalization.get("completion_audit_status"),
        "evidence_bundle_status": bundle.get("status"),
        "monitor_status": monitor.get("status"),
        "handoff_status": handoff.get("status"),
        "records": len(_records()),
        "claim_boundary": "Synthetic smoke only. This is not D2E full-corpus G003 completion evidence and must not be used for OMX checkpointing.",
    }
    write_json(root / "artifacts/sources/fdm1_g003_pipeline_smoke_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a synthetic G003 post-extraction action-slot pipeline smoke.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_smoke(Path(args.root), force=args.force)
    write_json(args.summary_out, summary)
    print(f"G003 synthetic pipeline smoke: status={summary['status']} root={summary['root']} summary={args.summary_out}")
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
