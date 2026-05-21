from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from plan_g006_readiness import build_readiness_plan


SPLITS = ["temporal", "heldout_recording", "heldout_game"]
ENDPOINTS = ["keyboard_accuracy", "mouse_button_f1"]


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "build_config": "configs/eval/g006_build.json",
        "build_summary_out": "artifacts/eval/g006_build_summary.json",
        "readiness_config": "configs/eval/g006_readiness.json",
        "readiness_output": "artifacts/eval/g006_readiness.json",
        "g006_completion_config": "configs/eval/g006_completion.json",
        "g006_audit_output": "artifacts/eval/g006_completion.json",
        "require_existing_final_outputs": False,
        "allow_precheckpoint": False,
        "output": "artifacts/eval/g006_plan.json",
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_configs(root: Path) -> None:
    comparison_sources = [
        {"id": f"idm_{split}", "path": f"artifacts/eval/{split}_idm.json", "split": split, "model_namespace": "d2e_only_idm"}
        for split in SPLITS
    ]
    write_json(
        root / "configs/eval/g006_build.json",
        {
            "goals_path": ".omx/ultragoal/goals.json",
            "endpoint_statistics_path": "artifacts/eval/final_endpoint_statistics.json",
            "failure_analysis_path": "artifacts/eval/final_failure_analysis.json",
            "claim_taxonomy_path": "artifacts/eval/final_claim_taxonomy.json",
            "prerequisite_goals": ["G003", "G004", "G005"],
            "required_splits": SPLITS,
            "required_endpoints": ENDPOINTS,
            "comparison_sources": comparison_sources,
            "metadata_sources": ["artifacts/eval/idm_metadata.json", "artifacts/eval/fdm_metadata.json"],
            "required_claim_states": {
                "d2e_only_idm": "claimable",
                "d2e_only_fdm": "claimable",
                "d2e_aux_comparison": "claimable",
                "live_open_game_suite": "not_claimed_until_g008",
                "negative_results": "documented",
            },
            "claim_states_requiring_evidence": ["claimable", "documented"],
            "claim_taxonomy": {
                "evidence_paths": [
                    "artifacts/eval/idm_metadata.json",
                    "artifacts/eval/fdm_metadata.json",
                    "artifacts/aux/d2e_aux_ablation_summary.json",
                    "artifacts/harness/g008_live_open_game_suite_evidence_validation.json",
                ]
            },
        },
    )
    write_json(
        root / "configs/eval/g006_readiness.json",
        {
            "goals_path": ".omx/ultragoal/goals.json",
            "prerequisite_goals": ["G003", "G004", "G005"],
            "required_splits": SPLITS,
            "required_endpoints": ENDPOINTS,
        },
    )
    write_json(
        root / "configs/eval/g006_completion.json",
        {
            "goals_path": ".omx/ultragoal/goals.json",
            "goal_id": "G006",
            "prerequisite_goals": ["G003", "G004", "G005"],
        },
    )


def _write_ready_fixture(root: Path) -> None:
    _write_configs(root)
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "complete"}, {"id": "G004", "status": "complete"}, {"id": "G005", "status": "complete"}]})
    for split in SPLITS:
        write_json(root / f"artifacts/eval/{split}_idm.json", {"comparisons": [{"endpoint": "keyboard_accuracy"}]})
    write_json(root / "artifacts/eval/idm_metadata.json", {"status": "ok"})
    write_json(root / "artifacts/eval/fdm_metadata.json", {"status": "ok"})
    write_json(root / "artifacts/aux/d2e_aux_ablation_summary.json", {"status": "pass"})


def test_g006_readiness_plan_ready_with_sources_and_prereqs(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    payload = build_readiness_plan(_args(tmp_path))
    assert payload["status"] == "ready"
    assert payload["findings"] == []
    assert payload["comparison_sources"][0]["exists"] is True
    assert any(item["code"] == "optional_live_suite_evidence_missing_until_g008" for item in payload["warnings"])
    assert payload["commands"]["finalize"] == ["uv", "run", "python", "scripts/finalize_g006_evaluation.py"]


def test_g006_readiness_plan_blocks_missing_sources_and_prereqs(tmp_path: Path):
    _write_configs(tmp_path)
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "in_progress"}]})
    payload = build_readiness_plan(_args(tmp_path))
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "blocked"
    assert "prerequisite_goal_not_complete" in codes
    assert "missing_comparison_source" in codes
    assert "missing_metadata_source" in codes
    assert "missing_claim_evidence_path" in codes


def test_g006_readiness_plan_allows_precheckpoint_diagnostics(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "in_progress"}]})
    payload = build_readiness_plan(_args(tmp_path, allow_precheckpoint=True))
    assert payload["status"] == "ready"
    assert any(item["code"] == "prerequisite_goal_not_complete" for item in payload["warnings"])


def test_g006_readiness_plan_can_require_existing_final_outputs(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    payload = build_readiness_plan(_args(tmp_path, require_existing_final_outputs=True))
    assert payload["status"] == "blocked"
    assert any(item["code"] == "existing_final_output_not_pass" for item in payload["findings"])
