from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.final_eval import build_g006_final_artifacts


ENDPOINTS = [
    "keyboard_accuracy",
    "mouse_button_accuracy",
    "mouse_button_precision",
    "mouse_button_f1",
    "no_button_false_positive_rate",
    "mouse_move_pearson",
    "mouse_move_scale_ratio_distance",
]
SPLITS = ["temporal", "heldout_recording", "heldout_game"]


def _config(root: Path) -> dict:
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
        "required_failure_axes": ["action", "game", "resolution", "source", "calibration"],
        "required_claim_taxonomy": ["d2e_only_idm", "d2e_only_fdm", "d2e_aux_comparison", "live_open_game_suite", "negative_results"],
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


def _write_complete_fixture(root: Path) -> None:
    write_json(
        root / ".omx/ultragoal/goals.json",
        {
            "goals": [
                {"id": "G003-d2e-only-idm", "status": "complete"},
                {"id": "G004-d2e-only-fdm-4xh200", "status": "complete"},
                {"id": "G005-aux-data-best-model", "status": "complete"},
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


def test_build_g006_final_artifacts_passes_with_split_coverage_and_non_rejection(tmp_path: Path):
    _write_complete_fixture(tmp_path)
    summary = build_g006_final_artifacts(_config(tmp_path), root=tmp_path)
    assert summary["status"] == "pass"
    endpoint = json.loads((tmp_path / "artifacts/eval/final_endpoint_statistics.json").read_text())
    failure = json.loads((tmp_path / "artifacts/eval/final_failure_analysis.json").read_text())
    taxonomy = json.loads((tmp_path / "artifacts/eval/final_claim_taxonomy.json").read_text())
    assert endpoint["status"] == "pass"
    assert len(endpoint["comparisons"]) == len(SPLITS) * len(ENDPOINTS)
    assert failure["status"] == "pass"
    assert failure["non_rejections"]
    assert taxonomy["status"] == "pass"
    assert {claim["id"] for claim in taxonomy["claims"]} >= {"d2e_only_idm", "d2e_only_fdm", "negative_results"}
    aux_claim = next(claim for claim in taxonomy["claims"] if claim["id"] == "d2e_aux_comparison")
    assert aux_claim["state"] == "claimable"
    assert aux_claim["evidence_paths"]


def test_build_g006_final_artifacts_fails_without_prereqs_and_split_sources(tmp_path: Path):
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003-d2e-only-idm", "status": "in_progress"}]})
    summary = build_g006_final_artifacts(_config(tmp_path), root=tmp_path)
    endpoint = json.loads((tmp_path / "artifacts/eval/final_endpoint_statistics.json").read_text())
    codes = {item["code"] for item in endpoint["findings"]}
    assert summary["status"] == "fail"
    assert "prerequisite_goal_not_complete" in codes
    assert "missing_comparison_source" in codes
    assert "missing_required_splits" in codes
