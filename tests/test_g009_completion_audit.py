from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import sha256_file, write_json
from fdm_d2e.reporting.g009_completion import validate_g009_completion


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


def _config() -> dict:
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
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": "G009",
        "prerequisite_goals": PREREQS,
        "min_doc_bytes": 8,
        "paths": paths,
        "required_docs": ["final_report", "evidence_index", "reproducibility_runbook", "failure_analysis_doc", "final_quality_doc"],
        "manifest_expectations": {"schema": "repro_package_manifest.v1"},
        "claim_boundary_expectations": {"status": "pass"},
        "final_quality_expectations": {"schema": "final_quality_gate_audit.v1"},
        "required_manifest_paths": [rel for key, rel in paths.items() if key not in {"package_manifest", "final_quality_audit"}],
    }


def _write_text(root: Path, rel: str, text: str = "substantial document text") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_complete_fixture(root: Path) -> None:
    cfg = _config()
    write_json(root / cfg["goals_path"], {"goals": [{"id": goal, "status": "complete"} for goal in [*PREREQS, "G009"]]})
    for key in cfg["required_docs"]:
        _write_text(root, cfg["paths"][key])
    write_json(root / cfg["paths"]["claim_boundary_audit"], {"schema": "claim_boundary_audit.v1", "status": "pass"})
    write_json(root / cfg["paths"]["final_quality_audit"], {"schema": "final_quality_gate_audit.v1", "status": "pass"})
    entries = []
    for rel in cfg["required_manifest_paths"]:
        path = root / rel
        entries.append({"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json(root / cfg["paths"]["package_manifest"], {"schema": "repro_package_manifest.v1", "entries": entries})


def test_g009_completion_audit_passes_on_full_fixture(tmp_path: Path):
    _write_complete_fixture(tmp_path)
    payload = validate_g009_completion(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0


def test_g009_completion_audit_fails_on_prereq_claim_and_manifest_hash(tmp_path: Path):
    _write_complete_fixture(tmp_path)
    cfg = _config()
    write_json(tmp_path / cfg["goals_path"], {"goals": [{"id": "G001-data-universe-audit", "status": "complete"}, {"id": "G009", "status": "pending"}]})
    write_json(tmp_path / cfg["paths"]["claim_boundary_audit"], {"schema": "claim_boundary_audit.v1", "status": "fail"})
    _write_text(tmp_path, cfg["paths"]["final_report"], "changed after manifest")
    payload = validate_g009_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "goal_not_checkpointed_complete" in codes
    assert "prerequisite_goal_not_complete" in codes
    assert "claim_boundary_expectation_mismatch" in codes
    assert "package_manifest_hash_mismatch" in codes
