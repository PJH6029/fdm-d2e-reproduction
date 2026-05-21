#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

DEFAULT_PATTERNS = [
    "README.md",
    "docs/d2e_full_data_universe.md",
    "docs/d2e_full_split_contract.md",
    "docs/d2e_full_idm_pipeline.md",
    "docs/d2e_full_fdm_pipeline.md",
    "docs/auxiliary_data_plan.md",
    "docs/source_validation.md",
    "docs/mlxp_resource_plan.md",
    "docs/d2e_real_ingestion.md",
    "docs/baselines_statistics.md",
    "docs/idm_research_track.md",
    "docs/fdm_research_track.md",
    "docs/ablation_scaling.md",
    "docs/harness_selection_and_execution.md",
    "docs/runtime_sdk_adapter.md",
    "docs/live_open_game_suite.md",
    "docs/final_quality_gates.md",
    "docs/g006_evaluation_readiness.md",
    "docs/g003_progress_monitoring.md",
    "docs/final_research_report.md",
    "docs/evidence_index.md",
    "docs/failure_analysis.md",
    "docs/reproducibility_runbook.md",
    "docs/cluster_runbook.md",
    "configs/data/d2e_real_multi_shooter64.yaml",
    "configs/data/d2e_full_corpus.yaml",
    "configs/eval/primary_endpoints.yaml",
    "configs/eval/final_quality_gates.yaml",
    "configs/eval/g006_evaluation_readiness.yaml",
    "configs/eval/g006_completion.yaml",
    "configs/eval/g007_completion.yaml",
    "configs/eval/g006_final_artifacts.yaml",
    "configs/eval/g003_split_statistics.yaml",
    "configs/eval/g004_split_statistics.yaml",
    "configs/eval/g003_full_idm_completion.yaml",
    "configs/eval/g004_full_fdm_completion.yaml",
    "configs/eval/g005_aux_completion.yaml",
    "configs/eval/g008_live_suite_completion.yaml",
    "configs/eval/g009_completion.yaml",
    "configs/harness/g008_game_harness.yaml",
    "configs/harness/g008_live_open_game_suite.yaml",
    "configs/runtime/game_adapter_contract_fixture.yaml",
    "configs/runtime/game_adapter_demo.yaml",
    "configs/model/idm_torch_shooter64_surface_motion.yaml",
    "configs/model/idm_streaming_d2e_full_compact.yaml",
    "configs/model/idm_streaming_d2e_full_compact_predict_fdm_train.yaml",
    "configs/model/idm_predict_shooter64_train_button_recall.yaml",
    "configs/model/fdm_streaming_d2e_full_compact.yaml",
    "configs/model/fdm_shooter64_surface_motion_fulltrain_bth05.yaml",
    "configs/model/fdm_bth05_d2e_train_scale_calibrated.yaml",
    "configs/model/fdm_bth05_d2e_train_prediction_scale_calibrated.yaml",
    "configs/model/fdm_bth05_predict_train_for_scale.yaml",
    "scripts/prepare_d2e_real.py",
    "scripts/build_data_universe_manifest.py",
    "scripts/build_split_contract.py",
    "scripts/build_aux_dataset_plan.py",
    "scripts/build_g005_aux_namespace_manifest.py",
    "scripts/plan_g005_launch.py",
    "scripts/finalize_g005_aux_best_model.py",
    "scripts/watch_g005_then_finalize.py",
    "scripts/extract_d2e_real_multi.py",
    "scripts/extract_d2e_full_corpus.py",
    "scripts/merge_d2e_full_corpus_shards.py",
    "scripts/train_idm_torch.py",
    "scripts/train_idm_streaming.py",
    "scripts/predict_idm_streaming.py",
    "scripts/predict_idm_torch.py",
    "scripts/train_fdm_real.py",
    "scripts/train_fdm_streaming.py",
    "scripts/calibrate_fdm_predictions.py",
    "scripts/run_g003_d2e_full_idm_parallel.sh",
    "scripts/run_g003_idm_training_4xh200.sh",
    "scripts/attach_g003_gpu_monitor.py",
    "scripts/build_g003_attached_train_run_summary.py",
    "scripts/finalize_g003_integrated_run.py",
    "scripts/run_g004_d2e_full_fdm_4xh200.sh",
    "scripts/plan_g004_launch.py",
    "scripts/finalize_g004_d2e_full_fdm.py",
    "scripts/watch_g004_then_finalize.py",
    "scripts/summarize_ablation_scaling.py",
    "scripts/run_game_harness_eval.py",
    "scripts/validate_live_game_suite.py",
    "scripts/plan_g008_readiness.py",
    "scripts/finalize_g008_live_suite.py",
    "scripts/validate_final_quality_gates.py",
    "scripts/validate_g006_evaluation_readiness.py",
    "scripts/validate_g006_completion.py",
    "scripts/plan_g006_readiness.py",
    "scripts/watch_g006_then_finalize.py",
    "scripts/finalize_g006_evaluation.py",
    "scripts/validate_g007_completion.py",
    "scripts/build_g006_final_eval_artifacts.py",
    "scripts/build_split_statistical_comparisons.py",
    "scripts/validate_g003_full_idm_completion.py",
    "scripts/validate_g004_full_fdm_completion.py",
    "scripts/validate_g005_aux_completion.py",
    "scripts/validate_g008_live_suite_completion.py",
    "scripts/validate_g009_completion.py",
    "scripts/plan_g009_readiness.py",
    "scripts/finalize_g009_report_package.py",
    "scripts/monitor_g003_progress.py",
    "scripts/audit_g003_live_health.py",
    "scripts/watch_g003_then_finalize.py",
    "scripts/plan_g003_resume.py",
    "scripts/run_runtime_replay_adapter.py",
    "scripts/audit_claim_boundaries.py",
    "scripts/build_repro_package_manifest.py",
    "src/fdm_d2e/training/calibrated_fdm.py",
    "src/fdm_d2e/training/streaming_fdm.py",
    "src/fdm_d2e/training/streaming_idm.py",
    "src/fdm_d2e/training/train_fdm.py",
    "src/fdm_d2e/training/torch_idm.py",
    "src/fdm_d2e/data/full_corpus.py",
    "src/fdm_d2e/rollout/game_harness.py",
    "src/fdm_d2e/rollout/live_suite.py",
    "src/fdm_d2e/runtime/sdk.py",
    "src/fdm_d2e/reporting/claim_audit.py",
    "src/fdm_d2e/reporting/quality_gates.py",
    "src/fdm_d2e/reporting/evaluation_readiness.py",
    "src/fdm_d2e/reporting/g006_completion.py",
    "src/fdm_d2e/reporting/g007_completion.py",
    "src/fdm_d2e/reporting/final_eval.py",
    "src/fdm_d2e/reporting/g003_completion.py",
    "src/fdm_d2e/reporting/g004_completion.py",
    "src/fdm_d2e/reporting/g005_completion.py",
    "src/fdm_d2e/reporting/g008_completion.py",
    "src/fdm_d2e/reporting/g009_completion.py",
    "src/fdm_d2e/cluster/g003_monitor.py",
    "src/fdm_d2e/eval/statistics.py",
    "src/fdm_d2e/eval/split_statistics.py",
    "src/fdm_d2e/eval/action_metrics.py",
    "tests/test_calibrated_fdm_contract.py",
    "tests/test_full_corpus_extraction_contract.py",
    "tests/test_streaming_fdm_contract.py",
    "tests/test_streaming_idm_contract.py",
    "tests/test_game_harness_contract.py",
    "tests/test_live_game_suite_contract.py",
    "tests/test_runtime_sdk_adapter.py",
    "tests/test_claim_boundary_audit.py",
    "tests/test_final_quality_gates.py",
    "tests/test_g006_evaluation_readiness.py",
    "tests/test_g006_completion_audit.py",
    "tests/test_g006_finalization.py",
    "tests/test_g007_completion_audit.py",
    "tests/test_g006_final_artifacts.py",
    "tests/test_split_statistics.py",
    "tests/test_g003_completion_audit.py",
    "tests/test_g004_completion_audit.py",
    "tests/test_g004_launch_planner.py",
    "tests/test_g004_finalization.py",
    "tests/test_g005_aux_completion_audit.py",
    "tests/test_g005_namespace_manifest_builder.py",
    "tests/test_g005_finalization.py",
    "tests/test_g008_live_suite_completion_audit.py",
    "tests/test_g008_live_suite_finalization.py",
    "tests/test_g008_readiness_planner.py",
    "tests/test_g009_completion_audit.py",
    "tests/test_g009_finalization.py",
    "tests/test_g009_readiness_planner.py",
    "tests/test_g003_monitor.py",
    "tests/test_g003_postrun_watcher.py",
    "tests/test_g003_attached_run_evidence.py",
    "tests/test_g003_integrated_finalization.py",
    "tests/test_training_run_scripts.py",
    "tests/test_aux_dataset_plan.py",
    "tests/test_eval_statistics.py",
    "artifacts/sources/d2e_multi_decode_shooter64_summary.json",
    "artifacts/sources/d2e_full_data_universe_manifest.json",
    "artifacts/sources/d2e_full_split_contract.json",
    "artifacts/sources/d2e_full_split_leakage_report.json",
    "artifacts/sources/d2e_full_temporal_split_manifest.json",
    "artifacts/sources/d2e_full_heldout_recording_split_manifest.json",
    "artifacts/sources/d2e_full_heldout_game_split_manifest.json",
    "artifacts/sources/aux_game_action_dataset_candidates.json",
    "outputs/data/d2e_full_corpus/*.jsonl",
    "outputs/idm_streaming_d2e_full_compact/*",
    "outputs/fdm_streaming_d2e_full_compact/*",
    "artifacts/idm/idm_streaming_d2e_full_compact*.json",
    "outputs/idm_streaming_d2e_full_compact/fdm_train_core_pseudolabels/*",
    "artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel*.json",
    "artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json",
    "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor*",
    "artifacts/idm/g003_full_compact_parallel_progress*.json",
    "artifacts/idm/g003_live_health_report*.json",
    "artifacts/idm/g003_resume_plan*.json",
    "artifacts/idm/g003_postrun_watcher_summary*.json",
    "artifacts/idm/g003_integrated_finalization_summary*.json",
    "artifacts/idm/g003_full_idm_completion_audit.json",
    "artifacts/fdm/g004_d2e_full_fdm_4xh200_run*.json",
    "artifacts/fdm/g004_launch_readiness*.json",
    "artifacts/fdm/g004_postrun_watcher_summary*.json",
    "artifacts/fdm/g004_d2e_full_fdm_finalization_summary*.json",
    "artifacts/fdm/g004_full_fdm_completion_audit.json",
    "artifacts/aux/*.json",
    "artifacts/aux/g005_launch_readiness*.json",
    "artifacts/aux/g005_postrun_watcher_summary*.json",
    "artifacts/aux/g005_aux_finalization_summary*.json",
    "artifacts/aux/g005_aux_completion_audit.json",
    "artifacts/harness/g008_live_suite_completion_audit.json",
    "artifacts/eval/final_*.json",
    "artifacts/eval/g006_final_artifact_build_summary.json",
    "artifacts/eval/g006_readiness_plan*.json",
    "artifacts/eval/g006_postrun_watcher_summary*.json",
    "artifacts/eval/g006_finalization_summary*.json",
    "artifacts/eval/g00[34]_split_statistical_comparisons_summary.json",
    "artifacts/eval/g006_evaluation_readiness_audit.json",
    "artifacts/eval/g006_completion_audit.json",
    "artifacts/idm/shooter64_surface_motion_selected/summary.json",
    "artifacts/idm/shooter64_surface_motion_selected/checkpoint_metadata.json",
    "artifacts/idm/shooter64_surface_motion_selected/metrics.json",
    "artifacts/idm/shooter64_surface_motion_selected/statistical_comparison.json",
    "artifacts/idm/idm_torch_shooter64_surface_motion_sweep_h200.json",
    "artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/summary.json",
    "artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/checkpoint_metadata.json",
    "artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/metrics.json",
    "artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/statistical_comparison.json",
    "artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/predictions.jsonl",
    "artifacts/fdm/fdm_bth05_d2e_train_prediction_scale_calibrated_h200/summary.json",
    "artifacts/fdm/fdm_bth05_d2e_train_prediction_scale_calibrated_h200/checkpoint_metadata.json",
    "artifacts/fdm/fdm_bth05_d2e_train_prediction_scale_calibrated_h200/metrics.json",
    "artifacts/fdm/fdm_bth05_d2e_train_prediction_scale_calibrated_h200/statistical_comparison.json",
    "artifacts/fdm/fdm_shooter64_fulltrain_button_sweep_h200.json",
    "artifacts/fdm/fdm_shooter64_recall_beta_sweep_h200.json",
    "artifacts/fdm/fdm_knn_shooter64_surface_sweep_h200.json",
    "artifacts/ablation_scaling/g007_ablation_scaling_summary.json",
    "artifacts/harness/g008_game_harness_eval.json",
    "artifacts/harness/g008_readiness_plan*.json",
    "artifacts/harness/g008_live_open_game_suite_protocol.json",
    "artifacts/harness/g008_live_open_game_suite_finalization_summary*.json",
    "artifacts/harness/g008_live_open_game_suite_evidence_validation*.json",
    "artifacts/runtime/g007_completion_audit.json",
    "artifacts/runtime/g007_runtime_fixture_predictions.jsonl",
    "artifacts/runtime/g007_runtime_replay_adapter_contract.json",
    "artifacts/reproducibility/claim_boundary_audit.json",
    "artifacts/reproducibility/final_quality_gate_audit.json",
    "artifacts/reproducibility/g009_readiness_plan*.json",
    "artifacts/reproducibility/g009_completion_audit.json",
    "artifacts/reproducibility/g009_finalization_summary*.json",
    "artifacts/reproducibility/final_cleanup_review.md",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_paths(patterns: Iterable[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        if not matches and not any(ch in pattern for ch in "*?[]"):
            raise FileNotFoundError(f"required manifest path is missing: {pattern}")
        for path in matches:
            if path.is_file():
                paths[str(path)] = path
    return [paths[key] for key in sorted(paths)]


def classify(path: Path) -> str:
    text = str(path)
    if text.startswith("docs/") or text == "README.md":
        return "documentation"
    if text.startswith("configs/"):
        return "configuration"
    if text.startswith("scripts/"):
        return "reproduction_script"
    if text.startswith("artifacts/"):
        return "evidence_artifact"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a reproducibility package manifest with hashes for key evidence artifacts.")
    parser.add_argument("--output", default="artifacts/reproducibility/package_manifest.json")
    parser.add_argument("--patterns", nargs="*", default=DEFAULT_PATTERNS)
    args = parser.parse_args()
    entries = []
    for path in iter_paths(args.patterns):
        entries.append({"path": str(path), "kind": classify(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {
        "schema": "repro_package_manifest.v1",
        "generated_at_utc": "deterministic-manifest-no-wall-clock",
        "entry_count": len(entries),
        "entries": entries,
        "notes": [
            "D2E-derived artifacts are research/non-commercial and must follow upstream terms.",
            "This manifest supports a scaled reproduction report, not an FDM-1 parity claim.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out} entries={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
