from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from finalize_g006_evaluation import finalize


SPLITS = ["temporal", "heldout_recording", "heldout_game"]
ENDPOINTS = [
    "keyboard_accuracy",
    "mouse_button_accuracy",
    "mouse_button_precision",
    "mouse_button_f1",
    "no_button_false_positive_rate",
    "mouse_move_pearson",
    "mouse_move_scale_ratio_distance",
]
AXES = ["action", "game", "resolution", "source", "calibration"]
CLAIMS = ["d2e_only_idm", "d2e_only_fdm", "d2e_aux_comparison", "live_open_game_suite", "negative_results"]
FORBIDDEN = ["fdm1_parity", "commercial_game_control_without_live_open_suite", "robotics_transfer", "car_control_transfer"]


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "summary_out": "artifacts/eval/g006_finalize.json",
        "allow_fail": False,
        "skip_build": False,
        "build_config": "configs/eval/g006_build.json",
        "build_summary_out": "artifacts/eval/g006_build_summary.json",
        "readiness_config": "configs/eval/g006_readiness.json",
        "readiness_output": "artifacts/eval/g006_readiness_audit.json",
        "g006_completion_config": "configs/eval/g006_completion.json",
        "g006_audit_output": "artifacts/eval/g006_completion_audit.json",
    }
    data.update(overrides)
    return Namespace(**data)


def _build_config() -> dict:
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "endpoint_statistics_path": "artifacts/eval/final_endpoint_statistics.json",
        "failure_analysis_path": "artifacts/eval/final_failure_analysis.json",
        "claim_taxonomy_path": "artifacts/eval/final_claim_taxonomy.json",
        "prerequisite_goals": ["G003-d2e-only-idm", "G004-d2e-only-fdm-4xh200", "G005-aux-data-best-model"],
        "required_splits": SPLITS,
        "required_endpoints": ENDPOINTS,
        "required_comparison_fields": [
            "split",
            "endpoint",
            "model",
            "reference",
            "candidate_value",
            "baseline_value",
            "delta",
            "p_value",
            "p_adjusted_holm",
            "reject_holm_0_05",
            "artifact_path",
            "artifact_sha256",
        ],
        "comparison_sources": [
            {"id": f"fdm_{split}", "path": f"artifacts/eval/{split}_stats.json", "split": split, "model_namespace": "d2e_only_fdm"}
            for split in SPLITS
        ],
        "metadata_sources": ["artifacts/eval/idm_metadata.json", "artifacts/eval/fdm_metadata.json"],
        "required_failure_axes": AXES,
        "required_claim_taxonomy": CLAIMS,
        "required_claim_states": {
            "d2e_only_idm": "claimable",
            "d2e_only_fdm": "claimable",
            "d2e_aux_comparison": "claimable",
            "live_open_game_suite": "not_claimed_until_g008",
            "negative_results": "documented",
        },
        "claim_states_requiring_evidence": ["claimable", "documented"],
        "require_non_rejections": True,
        "require_examples": True,
        "claim_taxonomy": {
            "evidence_paths": [
                "artifacts/eval/idm_metadata.json",
                "artifacts/eval/fdm_metadata.json",
                "artifacts/aux/d2e_aux_ablation_summary.json",
            ]
        },
    }


def _readiness_config() -> dict:
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "endpoint_statistics_path": "artifacts/eval/final_endpoint_statistics.json",
        "failure_analysis_path": "artifacts/eval/final_failure_analysis.json",
        "claim_taxonomy_path": "artifacts/eval/final_claim_taxonomy.json",
        "prerequisite_goals": ["G003-d2e-only-idm", "G004-d2e-only-fdm-4xh200", "G005-aux-data-best-model"],
        "required_splits": SPLITS,
        "required_endpoints": ENDPOINTS,
        "required_comparison_fields": [
            "split",
            "endpoint",
            "model",
            "reference",
            "candidate_value",
            "baseline_value",
            "delta",
            "p_value",
            "p_adjusted_holm",
            "reject_holm_0_05",
            "artifact_path",
            "artifact_sha256",
        ],
        "required_failure_axes": AXES,
        "required_claim_taxonomy": CLAIMS,
        "require_non_rejections": True,
        "require_examples": True,
    }


def _completion_config() -> dict:
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": "G006-evaluation-failure-analysis",
        "prerequisite_goals": ["G003-d2e-only-idm", "G004-d2e-only-fdm-4xh200", "G005-aux-data-best-model"],
        "require_goal_checkpoint_complete": False,
        "expected_recording_variants": 3,
        "require_d2e_only_completion_audits_pass": True,
        "require_g005_completion_audit_pass": True,
        "expected_variants_by_source": {"d2e_480p": 2, "d2e_original": 1},
        "expected_variants_by_resolution_tier": {"480p": 2, "original_fhd_qhd": 1},
        "required_splits": SPLITS,
        "required_endpoints": ENDPOINTS,
        "required_failure_axes": AXES,
        "required_claim_taxonomy": CLAIMS,
        "required_claim_states": {
            "d2e_only_idm": "claimable",
            "d2e_only_fdm": "claimable",
            "d2e_aux_comparison": "claimable",
            "live_open_game_suite": "not_claimed_until_g008",
            "negative_results": "documented",
        },
        "claim_states_requiring_evidence": ["claimable", "documented"],
        "required_forbidden_claims": FORBIDDEN,
        "paths": {
            "endpoint_statistics": "artifacts/eval/final_endpoint_statistics.json",
            "failure_analysis": "artifacts/eval/final_failure_analysis.json",
            "claim_taxonomy": "artifacts/eval/final_claim_taxonomy.json",
            "readiness_audit": "artifacts/eval/g006_readiness_audit.json",
            "build_summary": "artifacts/eval/g006_build_summary.json",
            "g003_completion_audit": "artifacts/idm/g003_audit.json",
            "g004_completion_audit": "artifacts/fdm/g004_audit.json",
            "g005_completion_audit": "artifacts/aux/g005_audit.json",
            "failure_doc": "docs/failure.md",
        },
        "endpoint_expectations": {"status": "pass"},
        "failure_expectations": {"status": "pass"},
        "taxonomy_expectations": {"status": "pass"},
        "readiness_expectations": {"status": "pass"},
        "build_summary_expectations": {
            "status": "pass",
            "statuses.endpoint_statistics": "pass",
            "statuses.failure_analysis": "pass",
            "statuses.claim_taxonomy": "pass",
        },
    }


def _write_configs(root: Path) -> None:
    write_json(root / "configs/eval/g006_build.json", _build_config())
    write_json(root / "configs/eval/g006_readiness.json", _readiness_config())
    write_json(root / "configs/eval/g006_completion.json", _completion_config())
    doc = root / "docs/failure.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("failure analysis fixture")


def _write_complete_fixture(root: Path) -> None:
    _write_configs(root)
    write_json(
        root / ".omx/ultragoal/goals.json",
        {
            "goals": [
                {"id": "G003-d2e-only-idm", "status": "complete"},
                {"id": "G004-d2e-only-fdm-4xh200", "status": "complete"},
                {"id": "G005-aux-data-best-model", "status": "complete"},
                {"id": "G006-evaluation-failure-analysis", "status": "pending"},
            ]
        },
    )
    for split in SPLITS:
        comparisons = []
        for idx, endpoint in enumerate(ENDPOINTS):
            comparisons.append(
                {
                    "endpoint": endpoint,
                    "model": "d2e_only_fdm",
                    "reference": "noop" if idx < 5 else "last_seen_train",
                    "candidate_value": 0.7,
                    "baseline_value": 0.5,
                    "delta": 0.2,
                    "p_value": 0.01 if idx else 0.2,
                    "p_adjusted_holm": 0.02 if idx else 0.4,
                    "reject_holm_0_05": bool(idx),
                    "status": "computed",
                    "num_clusters": 5,
                }
            )
        write_json(root / f"artifacts/eval/{split}_stats.json", {"schema": "stat_comparison.v1", "comparisons": comparisons})
    write_json(
        root / "artifacts/eval/idm_metadata.json",
        {
            "schema": "idm_checkpoint_metadata.v1",
            "source_ids": ["d2e_480p"],
            "target_source_ids": ["d2e_480p"],
            "target_resolution_tiers": ["480p", "original"],
            "target_games": ["Apex"],
            "calibration": {"mode": "global_threshold_streaming"},
        },
    )
    write_json(
        root / "artifacts/eval/fdm_metadata.json",
        {
            "schema": "fdm_checkpoint_metadata.v1",
            "label_source": "idm_pseudolabel",
            "source_ids": ["d2e_480p"],
            "target_source_ids": ["d2e_480p"],
            "target_resolution_tiers": ["480p"],
            "target_games": ["Apex"],
        },
    )
    write_json(root / "artifacts/aux/d2e_aux_ablation_summary.json", {"status": "pass"})
    d2e_audit_counts = {
        "included_recording_variants": 3,
        "source_ids": {"d2e_480p": 2, "d2e_original": 1},
        "resolution_tiers": {"480p": 2, "original_fhd_qhd": 1},
    }
    write_json(
        root / "artifacts/idm/g003_audit.json",
        {
            "schema": "g003_full_idm_completion_audit.v1",
            "status": "pass",
            "error_count": 0,
            "data_universe_counts": d2e_audit_counts,
            "decode_counts_by_source": {"d2e_480p": 2, "d2e_original": 1},
            "decode_counts_by_resolution_tier": {"480p": 2, "original_fhd_qhd": 1},
        },
    )
    write_json(
        root / "artifacts/fdm/g004_audit.json",
        {
            "schema": "g004_full_fdm_completion_audit.v1",
            "status": "pass",
            "error_count": 0,
            "data_universe_counts": d2e_audit_counts,
        },
    )
    write_json(root / "artifacts/aux/g005_audit.json", {"schema": "g005_aux_completion_audit.v1", "status": "pass", "error_count": 0})


def test_finalize_g006_builds_readiness_and_completion_audits(tmp_path: Path):
    _write_complete_fixture(tmp_path)
    payload = finalize(_args(tmp_path))
    assert payload["status"] == "pass"
    assert payload["build_summary_status"] == "pass"
    assert payload["readiness_status"] == "pass"
    assert payload["g006_audit_status"] == "pass"
    assert json.loads((tmp_path / "artifacts/eval/g006_build_summary.json").read_text())["status"] == "pass"
    assert json.loads((tmp_path / "artifacts/eval/g006_readiness_audit.json").read_text())["status"] == "pass"
    assert json.loads((tmp_path / "artifacts/eval/g006_completion_audit.json").read_text())["status"] == "pass"


def test_finalize_g006_records_non_terminal_missing_inputs(tmp_path: Path):
    _write_configs(tmp_path)
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003-d2e-only-idm", "status": "in_progress"}]})
    payload = finalize(_args(tmp_path))
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "g006_final_artifact_build_not_pass" in codes
    assert "g006_evaluation_readiness_not_pass" in codes
    assert "g006_completion_audit_not_pass" in codes
