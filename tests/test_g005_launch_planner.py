from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from plan_g005_launch import build_launch_readiness


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "g005_completion_config": "configs/eval/g005_completion.json",
        "g003_audit": "artifacts/idm/g003_audit.json",
        "g004_audit": "artifacts/fdm/g004_audit.json",
        "pid_file": "outputs/cluster/g005.pid",
        "source_evidence": [],
        "eval_manifest_hashes": None,
        "require_eval_manifest_hashes": False,
        "require_namespace_ready": False,
        "allow_precheckpoint": False,
        "allow_overwrite": False,
        "output": "artifacts/aux/g005_launch.json",
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_ready_fixture(root: Path) -> None:
    write_json(
        root / ".omx/ultragoal/goals.json",
        {
            "goals": [
                {"id": "G003-d2e-only-idm", "status": "complete"},
                {"id": "G004-d2e-only-fdm-4xh200", "status": "complete"},
                {"id": "G005-aux-data-best-model", "status": "pending"},
            ]
        },
    )
    write_json(root / "artifacts/idm/g003_audit.json", {"status": "pass", "error_count": 0})
    write_json(root / "artifacts/fdm/g004_audit.json", {"status": "pass", "error_count": 0})
    write_json(
        root / "artifacts/sources/aux.json",
        {
            "user_decision": {"d2e_aux_may_be_primary": True},
            "claim_boundary": {"no_d2e_aux_claim_before_d2e_only_gates": True},
            "storage_policy": {"fits_cap_with_selected_candidates": True},
            "candidates": [{"id": "aux_a", "selection_status": "selected_candidate"}],
        },
    )
    aux_doc = root / "docs/aux.md"
    aux_doc.parent.mkdir(parents=True, exist_ok=True)
    aux_doc.write_text("aux plan", encoding="utf-8")
    write_json(
        root / "configs/eval/g005_completion.json",
        {
            "goals_path": ".omx/ultragoal/goals.json",
            "goal_id": "G005-aux-data-best-model",
            "prerequisite_goals": ["G003-d2e-only-idm", "G004-d2e-only-fdm-4xh200"],
            "paths": {
                "aux_candidates": "artifacts/sources/aux.json",
                "aux_plan_doc": "docs/aux.md",
                "namespace_manifest": "artifacts/aux/namespace.json",
                "run_summary": "artifacts/aux/run.json",
            },
            "aux_candidate_expectations": {
                "user_decision.d2e_aux_may_be_primary": True,
                "claim_boundary.no_d2e_aux_claim_before_d2e_only_gates": True,
                "storage_policy.fits_cap_with_selected_candidates": True,
            },
        },
    )


def test_g005_launch_readiness_passes_after_d2e_only_gates(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    payload = build_launch_readiness(_args(tmp_path))
    assert payload["status"] == "ready"
    assert payload["selected_aux_candidate_ids"] == ["aux_a"]
    assert payload["findings"] == []
    assert any(item["code"] == "namespace_manifest_not_built_yet" for item in payload["warnings"])
    assert payload["commands"]["postrun_watcher"][0:3] == ["uv", "run", "python"]


def test_g005_launch_readiness_blocks_before_d2e_only_gates(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    write_json(root := tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003-d2e-only-idm", "status": "in_progress"}]})
    write_json(tmp_path / "artifacts/fdm/g004_audit.json", {"status": "fail", "error_count": 2})
    payload = build_launch_readiness(_args(tmp_path))
    codes = {item["code"] for item in payload["findings"]}
    assert root.exists()
    assert payload["status"] == "blocked"
    assert "prerequisite_goal_not_complete" in codes
    assert "d2e_only_audit_not_pass" in codes


def test_g005_launch_readiness_allows_precheckpoint_diagnostics(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003-d2e-only-idm", "status": "in_progress"}]})
    payload = build_launch_readiness(_args(tmp_path, allow_precheckpoint=True))
    assert payload["status"] == "ready"
    assert any(item["code"] == "prerequisite_goal_not_complete" for item in payload["warnings"])


def test_g005_launch_readiness_requires_namespace_when_requested(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    payload = build_launch_readiness(_args(tmp_path, require_namespace_ready=True))
    assert payload["status"] == "blocked"
    assert any(item["code"] == "namespace_manifest_not_completion_ready" for item in payload["findings"])


def test_g005_launch_readiness_blocks_existing_run_summary(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    write_json(tmp_path / "artifacts/aux/run.json", {"exit_code": 0})
    payload = build_launch_readiness(_args(tmp_path))
    assert payload["status"] == "blocked"
    assert any(item["code"] == "run_summary_already_exists" for item in payload["findings"])
