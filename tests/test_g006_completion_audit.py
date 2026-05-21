from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.g006_completion import validate_g006_completion


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


def _config() -> dict:
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": "G006",
        "prerequisite_goals": ["G003", "G004", "G005"],
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
            "endpoint_statistics": "artifacts/eval/endpoint.json",
            "failure_analysis": "artifacts/eval/failure.json",
            "claim_taxonomy": "artifacts/eval/taxonomy.json",
            "readiness_audit": "artifacts/eval/readiness.json",
            "build_summary": "artifacts/eval/build.json",
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


def _complete_fixture(root: Path) -> None:
    cfg = _config()
    write_json(
        root / cfg["goals_path"],
        {
            "goals": [
                {"id": "G003", "status": "complete"},
                {"id": "G004", "status": "complete"},
                {"id": "G005", "status": "complete"},
                {"id": "G006", "status": "complete"},
            ]
        },
    )
    comparisons = []
    for split in SPLITS:
        for idx, endpoint in enumerate(ENDPOINTS):
            comparisons.append({"split": split, "endpoint": endpoint, "reject_holm_0_05": bool(idx), "p_adjusted_holm": 0.02 if idx else 0.4})
    write_json(root / cfg["paths"]["endpoint_statistics"], {"status": "pass", "required_splits": SPLITS, "required_endpoints": ENDPOINTS, "comparisons": comparisons})
    write_json(
        root / cfg["paths"]["failure_analysis"],
        {
            "status": "pass",
            "axes": {axis: ["example"] for axis in AXES},
            "non_rejections": [{"endpoint": "keyboard_accuracy"}],
            "examples": [{"endpoint": "keyboard_accuracy", "reason": "documented non-rejection"}],
        },
    )
    write_json(
        root / cfg["paths"]["claim_taxonomy"],
        {
            "status": "pass",
            "claims": [
                {"id": "d2e_only_idm", "state": "claimable", "evidence_paths": ["outputs/idm/metadata.json"]},
                {"id": "d2e_only_fdm", "state": "claimable", "evidence_paths": ["outputs/fdm/metadata.json"]},
                {"id": "d2e_aux_comparison", "state": "claimable", "evidence_paths": ["artifacts/aux/ablation.json"]},
                {"id": "live_open_game_suite", "state": "not_claimed_until_g008"},
                {"id": "negative_results", "state": "documented", "evidence_paths": ["artifacts/eval/failure.json"]},
            ],
            "forbidden_claims": FORBIDDEN,
        },
    )
    write_json(root / cfg["paths"]["readiness_audit"], {"status": "pass"})
    write_json(root / cfg["paths"]["build_summary"], {"status": "pass", "statuses": {"endpoint_statistics": "pass", "failure_analysis": "pass", "claim_taxonomy": "pass"}})
    doc = root / cfg["paths"]["failure_doc"]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("failure analysis")


def test_g006_completion_audit_passes_on_full_fixture(tmp_path: Path):
    _complete_fixture(tmp_path)
    payload = validate_g006_completion(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert payload["required_claim_states"]["d2e_aux_comparison"] == "claimable"


def test_g006_completion_audit_fails_on_missing_goal_claims_and_failures(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    write_json(tmp_path / cfg["goals_path"], {"goals": [{"id": "G003", "status": "in_progress"}, {"id": "G006", "status": "pending"}]})
    failure = {"status": "pass", "axes": {"action": ["mouse"]}, "non_rejections": [], "examples": []}
    write_json(tmp_path / cfg["paths"]["failure_analysis"], failure)
    write_json(
        tmp_path / cfg["paths"]["claim_taxonomy"],
        {
            "status": "pass",
            "claims": [{"id": "d2e_only_idm", "state": "claimable", "evidence_paths": []}],
            "forbidden_claims": ["fdm1_parity"],
        },
    )
    payload = validate_g006_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "goal_not_checkpointed_complete" in codes
    assert "prerequisite_goal_not_complete" in codes
    assert "failure_analysis_missing_axes" in codes
    assert "failure_analysis_missing_non_rejections" in codes
    assert "failure_analysis_missing_examples" in codes
    assert "claim_taxonomy_missing_claims" in codes
    assert "claim_taxonomy_state_mismatch" in codes
    assert "claim_taxonomy_missing_evidence_for_state" in codes
    assert "claim_taxonomy_missing_forbidden_claims" in codes
