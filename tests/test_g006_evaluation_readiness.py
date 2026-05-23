from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.evaluation_readiness import validate_g006_evaluation_readiness, write_g006_evaluation_readiness


def _config() -> dict:
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "endpoint_statistics_path": "artifacts/eval/final_endpoint_statistics.json",
        "failure_analysis_path": "artifacts/eval/final_failure_analysis.json",
        "claim_taxonomy_path": "artifacts/eval/final_claim_taxonomy.json",
        "output_path": "artifacts/eval/g006_evaluation_readiness_audit.json",
        "prerequisite_goals": ["G003", "G004"],
        "required_splits": ["temporal", "heldout_recording", "heldout_game"],
        "required_endpoints": ["keyboard_accuracy", "mouse_button_f1"],
        "required_comparison_fields": ["split", "endpoint", "model", "reference", "candidate_value", "baseline_value", "delta", "p_value", "p_adjusted_holm", "reject_holm_0_05", "artifact_path", "artifact_sha256"],
        "required_failure_axes": ["action", "game", "resolution", "source", "calibration"],
        "required_claim_taxonomy": ["d2e_only_idm", "d2e_only_fdm", "negative_results"],
        "require_non_rejections": True,
        "require_examples": True,
    }


def _complete_fixture(root: Path) -> None:
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "complete"}, {"id": "G004", "status": "complete"}]})
    comparisons = []
    for split in ["temporal", "heldout_recording", "heldout_game"]:
        for endpoint in ["keyboard_accuracy", "mouse_button_f1"]:
            comparisons.append(
                {
                    "split": split,
                    "endpoint": endpoint,
                    "model": "fdm_d2e_only",
                    "reference": "noop",
                    "candidate_value": 0.7,
                    "baseline_value": 0.5,
                    "delta": 0.2,
                    "p_value": 0.01,
                    "p_adjusted_holm": 0.02,
                    "reject_holm_0_05": True,
                    "artifact_path": "artifacts/eval/source.json",
                    "artifact_sha256": "abc",
                }
            )
    write_json(root / "artifacts/eval/final_endpoint_statistics.json", {"schema": "final_endpoint_statistics.v1", "status": "pass", "comparisons": comparisons})
    write_json(
        root / "artifacts/eval/final_failure_analysis.json",
        {
            "schema": "final_failure_analysis.v1",
            "status": "pass",
            "axes": ["action", "game", "resolution", "source", "calibration"],
            "non_rejections": [{"endpoint": "scale", "reason": "documented"}],
            "examples": [{"id": "ex1"}],
        },
    )
    write_json(
        root / "artifacts/eval/final_claim_taxonomy.json",
        {"schema": "final_claim_taxonomy.v1", "status": "pass", "claims": [{"id": "d2e_only_idm"}, {"id": "d2e_only_fdm"}, {"id": "negative_results"}]},
    )


def test_g006_readiness_passes_with_split_endpoint_failure_and_claim_artifacts(tmp_path):
    _complete_fixture(tmp_path)
    payload = validate_g006_evaluation_readiness(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert payload["artifacts"]["endpoint_statistics"]["exists"] is True


def test_g006_readiness_allows_explicit_unavailable_comparison_rows(tmp_path):
    _complete_fixture(tmp_path)
    endpoint = json.loads((tmp_path / "artifacts/eval/final_endpoint_statistics.json").read_text())
    endpoint["comparisons"][0].update(
        {
            "status": "no_shared_clusters",
            "candidate_value": 0.12,
            "baseline_value": None,
            "delta": None,
            "p_value": None,
            "p_adjusted_holm": None,
            "reject_holm_0_05": False,
            "stat_test_available": False,
            "unavailable_reason": "statistical_test_unavailable:no_shared_clusters:candidate_clusters=4:reference_clusters=0",
        }
    )
    write_json(tmp_path / "artifacts/eval/final_endpoint_statistics.json", endpoint)

    payload = validate_g006_evaluation_readiness(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0


def test_g006_readiness_fails_when_prereqs_and_final_artifacts_are_missing(tmp_path):
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "in_progress"}]})
    payload = validate_g006_evaluation_readiness(_config(), root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "prerequisite_goal_not_complete" in codes
    assert "missing_endpoint_statistics" in codes
    assert "missing_failure_analysis" in codes
    assert "missing_claim_taxonomy" in codes


def test_g006_readiness_detects_missing_splits_endpoints_and_failure_axes(tmp_path):
    _complete_fixture(tmp_path)
    endpoint = json.loads((tmp_path / "artifacts/eval/final_endpoint_statistics.json").read_text())
    endpoint["comparisons"] = endpoint["comparisons"][:1]
    write_json(tmp_path / "artifacts/eval/final_endpoint_statistics.json", endpoint)
    failure = json.loads((tmp_path / "artifacts/eval/final_failure_analysis.json").read_text())
    failure["axes"] = ["action"]
    failure["non_rejections"] = []
    write_json(tmp_path / "artifacts/eval/final_failure_analysis.json", failure)
    payload = validate_g006_evaluation_readiness(_config(), root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert "endpoint_statistics_missing_splits" in codes
    assert "endpoint_statistics_missing_endpoints" in codes
    assert "failure_analysis_missing_axes" in codes
    assert "failure_analysis_missing_non_rejections" in codes


def test_g006_readiness_writes_current_audit(tmp_path):
    _complete_fixture(tmp_path)
    payload = write_g006_evaluation_readiness(_config(), root=tmp_path)
    written = json.loads((tmp_path / "artifacts/eval/g006_evaluation_readiness_audit.json").read_text())
    assert payload["schema"] == "g006_evaluation_readiness_audit.v1"
    assert written["status"] == "pass"
