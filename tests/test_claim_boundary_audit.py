from __future__ import annotations

import json
from pathlib import Path

from fdm_d2e.reporting.claim_audit import audit_claim_boundaries


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _goals(path: Path, *, g008: str = "pending") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "activeGoalId": "G003-d2e-only-idm",
                "goals": [
                    {"id": "G001-data-universe-audit", "status": "complete"},
                    {"id": "G002-split-leakage-contract", "status": "complete"},
                    {"id": "G003-d2e-only-idm", "status": "in_progress"},
                    {"id": "G007-runtime-sdk-adapter", "status": "complete"},
                    {"id": "G008-live-game-suite", "status": g008},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_claim_audit_requires_historical_notice_when_full_corpus_incomplete(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _goals(tmp_path / ".omx/ultragoal/goals.json")
    for path in [
        "README.md",
        "docs/fdm_research_track.md",
        "docs/harness_selection_and_execution.md",
        "docs/runtime_sdk_adapter.md",
    ]:
        _write(tmp_path / path, "not an fdm-1 parity claim; does not by itself prove live game control\n")
    _write(tmp_path / "docs/final_research_report.md", "final report without current notice\n")
    _write(tmp_path / "docs/evidence_index.md", "evidence index without current notice\n")
    _write(tmp_path / "docs/reproducibility_runbook.md", "runbook without current notice\n")

    payload = audit_claim_boundaries()

    assert payload["status"] == "fail"
    assert {item["code"] for item in payload["findings"]} >= {"missing_incomplete_full_corpus_notice"}


def test_claim_audit_passes_with_incomplete_and_no_live_claim_notices(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _goals(tmp_path / ".omx/ultragoal/goals.json")
    notice = "Current full-corpus ultragoal is not complete; historical bounded evidence only.\n"
    for path in [
        "README.md",
        "docs/final_research_report.md",
        "docs/evidence_index.md",
        "docs/reproducibility_runbook.md",
        "docs/fdm_research_track.md",
        "docs/harness_selection_and_execution.md",
    ]:
        _write(tmp_path / path, notice + "not an fdm-1 parity claim\n")
    _write(tmp_path / "docs/runtime_sdk_adapter.md", notice + "No G008 live-suite claim; does not by itself prove live game control.\n")

    payload = audit_claim_boundaries()

    assert payload["status"] == "pass"
    assert payload["findings"] == []


def test_claim_audit_rejects_positive_fdm1_parity_claim(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _goals(tmp_path / ".omx/ultragoal/goals.json", g008="complete")
    for path in [
        "README.md",
        "docs/final_research_report.md",
        "docs/evidence_index.md",
        "docs/reproducibility_runbook.md",
        "docs/fdm_research_track.md",
        "docs/harness_selection_and_execution.md",
        "docs/runtime_sdk_adapter.md",
    ]:
        _write(tmp_path / path, "safe report\n")
    _write(tmp_path / "docs/final_research_report.md", "This matches FDM-1.\n")

    payload = audit_claim_boundaries()

    assert payload["status"] == "fail"
    assert any(item["code"] == "fdm1_parity_positive" for item in payload["findings"])
