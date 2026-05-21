from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import sha256_file, write_json
from plan_g009_readiness import build_readiness_plan


PREREQS = [
    "G001-data-universe-audit",
    "G002-split-leakage-contract",
    "G003-d2e-only-idm",
    "G004-d2e-only-fdm-4xh200",
    "G005-aux-data-best-model",
    "G006-evaluation-failure-analysis",
    "G007-runtime-sdk-adapter",
    "G008-live-game-suite",
]
DOC_KEYS = ["final_report", "evidence_index", "reproducibility_runbook", "failure_analysis_doc", "final_quality_doc"]


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "g009_completion_config": "configs/eval/g009.json",
        "final_quality_config": "configs/eval/final_quality.json",
        "summary_out": "artifacts/repro/g009_summary.json",
        "g009_audit_output": "artifacts/repro/g009_audit.json",
        "require_existing_final_outputs": False,
        "allow_precheckpoint": False,
        "output": "artifacts/repro/g009_plan.json",
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_text(root: Path, rel_path: str, text: str = "substantial document text") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_configs(root: Path) -> dict[str, str]:
    paths = {
        "final_report": "docs/final.md",
        "evidence_index": "docs/evidence.md",
        "reproducibility_runbook": "docs/runbook.md",
        "failure_analysis_doc": "docs/failure.md",
        "final_quality_doc": "docs/quality.md",
        "package_manifest": "artifacts/repro/manifest.json",
        "claim_boundary_audit": "artifacts/repro/claim.json",
        "final_quality_audit": "artifacts/repro/final_quality.json",
    }
    write_json(
        root / "configs/eval/g009.json",
        {
            "schema": "g009_completion_config.v1",
            "goals_path": ".omx/ultragoal/goals.json",
            "goal_id": "G009",
            "prerequisite_goals": PREREQS,
            "output_path": "artifacts/repro/g009_audit.json",
            "require_goal_checkpoint_complete": False,
            "min_doc_bytes": 8,
            "paths": paths,
            "required_docs": DOC_KEYS,
            "manifest_expectations": {"schema": "repro_package_manifest.v1"},
            "claim_boundary_expectations": {"status": "pass"},
            "final_quality_expectations": {"schema": "final_quality_gate_audit.v1", "status": "pass"},
            "required_manifest_paths": [paths[key] for key in DOC_KEYS] + [paths["claim_boundary_audit"], paths["final_quality_audit"]],
        },
    )
    write_json(
        root / "configs/eval/final_quality.json",
        {
            "schema": "final_quality_gates_config.v1",
            "goals_path": ".omx/ultragoal/goals.json",
            "package_manifest_path": paths["package_manifest"],
            "claim_boundary_audit_path": paths["claim_boundary_audit"],
            "output_path": paths["final_quality_audit"],
        },
    )
    return paths


def _write_ready_fixture(root: Path, *, write_outputs: bool = False) -> dict[str, str]:
    paths = _write_configs(root)
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": goal, "status": "complete"} for goal in [*PREREQS, "G009"]]})
    for key in DOC_KEYS:
        _write_text(root, paths[key])
    if write_outputs:
        write_json(root / paths["claim_boundary_audit"], {"schema": "claim_boundary_audit.v1", "status": "pass"})
        write_json(root / paths["final_quality_audit"], {"schema": "final_quality_gate_audit.v1", "status": "pass"})
        entries = []
        for rel in [paths[key] for key in DOC_KEYS] + [paths["claim_boundary_audit"], paths["final_quality_audit"]]:
            path = root / rel
            entries.append({"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size})
        write_json(root / paths["package_manifest"], {"schema": "repro_package_manifest.v1", "entries": entries})
        write_json(root / "artifacts/repro/g009_audit.json", {"schema": "g009_completion_audit.v1", "status": "pass", "error_count": 0})
    return paths


def test_g009_readiness_plan_ready_to_finalize_with_docs_and_prereqs(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    payload = build_readiness_plan(_args(tmp_path))
    assert payload["status"] == "ready"
    assert payload["findings"] == []
    assert any(item["code"] == "refreshable_output_missing" for item in payload["warnings"])
    assert payload["commands"]["finalize"] == ["uv", "run", "python", "scripts/finalize_g009_report_package.py"]
    assert "does not refresh audits" in payload["claim_boundary"]


def test_g009_readiness_plan_blocks_missing_prereqs_and_docs(tmp_path: Path):
    paths = _write_ready_fixture(tmp_path)
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G001-data-universe-audit", "status": "complete"}, {"id": "G003-d2e-only-idm", "status": "in_progress"}]})
    (tmp_path / paths["final_report"]).unlink()
    payload = build_readiness_plan(_args(tmp_path))
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "blocked"
    assert "prerequisite_goal_not_complete" in codes
    assert "missing_required_doc" in codes


def test_g009_readiness_plan_allows_precheckpoint_diagnostics(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003-d2e-only-idm", "status": "in_progress"}]})
    payload = build_readiness_plan(_args(tmp_path, allow_precheckpoint=True))
    assert payload["status"] == "ready"
    assert any(item["code"] == "prerequisite_goal_not_complete" for item in payload["warnings"])


def test_g009_readiness_plan_can_require_existing_final_outputs(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    payload = build_readiness_plan(_args(tmp_path, require_existing_final_outputs=True))
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "blocked"
    assert "refreshable_output_missing" in codes
    assert "g009_completion_audit_not_pass" in codes
    assert "completion_projection_not_pass" in codes


def test_g009_readiness_plan_passes_when_existing_outputs_are_current(tmp_path: Path):
    _write_ready_fixture(tmp_path, write_outputs=True)
    payload = build_readiness_plan(_args(tmp_path, require_existing_final_outputs=True))
    assert payload["status"] == "ready"
    assert payload["findings"] == []
    assert payload["refreshable_outputs"]["final_quality_audit"]["status"] == "pass"
