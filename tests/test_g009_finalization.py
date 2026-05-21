from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from finalize_g009_report_package import finalize


GOALS = [
    "G001-data-universe-audit",
    "G002-split-leakage-contract",
    "G003-d2e-only-idm",
    "G004-d2e-only-fdm-4xh200",
    "G005-aux-data-best-model",
    "G006-evaluation-failure-analysis",
    "G007-runtime-sdk-adapter",
    "G008-live-game-suite",
    "G009-report-repro-package",
]
REPORT_DOCS = [
    "README.md",
    "docs/final_research_report.md",
    "docs/evidence_index.md",
    "docs/reproducibility_runbook.md",
    "docs/fdm_research_track.md",
    "docs/harness_selection_and_execution.md",
    "docs/runtime_sdk_adapter.md",
]


def _args(root: Path, **overrides) -> Namespace:
    patterns = [
        *REPORT_DOCS,
        "docs/failure_analysis.md",
        "docs/final_quality_gates.md",
        "artifacts/g001.json",
        "artifacts/harness/g008_live_open_game_suite_evidence_validation.json",
        "artifacts/reproducibility/claim_boundary_audit.json",
        "artifacts/reproducibility/final_quality_gate_audit.json",
        "artifacts/reproducibility/g009_completion_audit.json",
    ]
    data = {
        "root": str(root),
        "summary_out": "artifacts/reproducibility/g009_finalization_summary.json",
        "allow_fail": False,
        "goals_path": ".omx/ultragoal/goals.json",
        "claim_audit_output": "artifacts/reproducibility/claim_boundary_audit.json",
        "final_quality_config": "configs/eval/final_quality.json",
        "final_quality_output": "artifacts/reproducibility/final_quality_gate_audit.json",
        "package_manifest_output": "artifacts/reproducibility/package_manifest.json",
        "package_patterns": patterns,
        "g009_completion_config": "configs/eval/g009.json",
        "g009_audit_output": "artifacts/reproducibility/g009_completion_audit.json",
    }
    data.update(overrides)
    return Namespace(**data)


def _write_text(root: Path, rel_path: str, text: str = "substantial document text") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_common_docs(root: Path) -> None:
    for rel in REPORT_DOCS:
        _write_text(root, rel, "current full-corpus ultragoal is not complete\nsubstantial document text")
    _write_text(root, "docs/failure_analysis.md", "substantial failure analysis document")
    _write_text(root, "docs/final_quality_gates.md", "substantial quality gate document")
    # G007 complete while G008 incomplete cases require this explicit notice.
    _write_text(root, "docs/runtime_sdk_adapter.md", "no g008 live-suite claim\nsubstantial runtime document")


def _write_configs(root: Path) -> None:
    write_json(
        root / "configs/eval/final_quality.json",
        {
            "goals_path": ".omx/ultragoal/goals.json",
            "package_manifest_path": "artifacts/reproducibility/package_manifest.json",
            "claim_boundary_audit_path": "artifacts/reproducibility/claim_boundary_audit.json",
            "live_suite_evidence_validation_path": "artifacts/harness/g008_live_open_game_suite_evidence_validation.json",
            "require_all_goals_complete": True,
            "require_live_suite_pass": True,
            "goal_gates": [
                {"id": "G001-data-universe-audit", "requires_status": "complete", "required_paths": ["artifacts/g001.json"]},
                {
                    "id": "G008-live-game-suite",
                    "requires_status": "complete",
                    "required_paths": ["artifacts/harness/g008_live_open_game_suite_evidence_validation.json"],
                    "json_assertions": [
                        {
                            "path": "artifacts/harness/g008_live_open_game_suite_evidence_validation.json",
                            "json_path": "quality_gate.status",
                            "equals": "pass",
                        }
                    ],
                },
            ],
        },
    )
    write_json(
        root / "configs/eval/g009.json",
        {
            "goals_path": ".omx/ultragoal/goals.json",
            "goal_id": "G009-report-repro-package",
            "prerequisite_goals": GOALS[:-1],
            "require_goal_checkpoint_complete": False,
            "min_doc_bytes": 8,
            "paths": {
                "final_report": "docs/final_research_report.md",
                "evidence_index": "docs/evidence_index.md",
                "reproducibility_runbook": "docs/reproducibility_runbook.md",
                "failure_analysis_doc": "docs/failure_analysis.md",
                "final_quality_doc": "docs/final_quality_gates.md",
                "package_manifest": "artifacts/reproducibility/package_manifest.json",
                "claim_boundary_audit": "artifacts/reproducibility/claim_boundary_audit.json",
                "final_quality_audit": "artifacts/reproducibility/final_quality_gate_audit.json",
            },
            "required_docs": ["final_report", "evidence_index", "reproducibility_runbook", "failure_analysis_doc", "final_quality_doc"],
            "manifest_expectations": {"schema": "repro_package_manifest.v1"},
            "claim_boundary_expectations": {"status": "pass"},
            "final_quality_expectations": {"schema": "final_quality_gate_audit.v1", "status": "pass"},
            "required_manifest_paths": [
                "docs/final_research_report.md",
                "docs/evidence_index.md",
                "docs/reproducibility_runbook.md",
                "docs/failure_analysis.md",
                "docs/final_quality_gates.md",
                "artifacts/reproducibility/claim_boundary_audit.json",
                "artifacts/reproducibility/final_quality_gate_audit.json",
            ],
        },
    )


def _write_fixture(root: Path, *, complete: bool = True) -> None:
    _write_common_docs(root)
    _write_configs(root)
    statuses = "complete" if complete else "pending"
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": goal, "status": statuses} for goal in GOALS]})
    _write_text(root, "artifacts/g001.json", "{}")
    write_json(root / "artifacts/harness/g008_live_open_game_suite_evidence_validation.json", {"quality_gate": {"status": "pass"}})


def test_g009_finalizer_refreshes_claim_quality_manifest_and_completion(tmp_path: Path):
    _write_fixture(tmp_path)
    payload = finalize(_args(tmp_path))
    assert payload["status"] == "pass"
    assert payload["claim_audit_status"] == "pass"
    assert payload["final_quality_status"] == "pass"
    assert payload["g009_audit_status"] == "pass"
    assert json.loads((tmp_path / "artifacts/reproducibility/package_manifest.json").read_text())["entry_count"] >= 7
    assert json.loads((tmp_path / "artifacts/reproducibility/g009_completion_audit.json").read_text())["status"] == "pass"


def test_g009_finalizer_records_non_terminal_incomplete_goal_state(tmp_path: Path):
    _write_fixture(tmp_path, complete=False)
    payload = finalize(_args(tmp_path))
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "final_quality_gate_audit_not_pass" in codes
    assert "g009_completion_audit_not_pass" in codes
    audit = json.loads((tmp_path / "artifacts/reproducibility/g009_completion_audit.json").read_text())
    assert audit["status"] == "fail"
