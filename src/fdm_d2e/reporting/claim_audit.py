from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_REPORT_PATHS = [
    "README.md",
    "docs/final_research_report.md",
    "docs/evidence_index.md",
    "docs/reproducibility_runbook.md",
    "docs/fdm_research_track.md",
    "docs/harness_selection_and_execution.md",
    "docs/runtime_sdk_adapter.md",
]

FORBIDDEN_PATTERNS = {
    "fdm1_parity_positive": re.compile(r"\b(matches|matched|equivalent to|same as)\s+fdm-?1\b", re.IGNORECASE),
    "commercial_game_success_positive": re.compile(
        r"\b(proves?|demonstrates?|achieves?|supports?)\s+(live\s+)?commercial[- ]game\s+(control|play)",
        re.IGNORECASE,
    ),
    "robotics_transfer_positive": re.compile(r"\b(proves?|demonstrates?|achieves?|supports?)\s+robotics\s+transfer\b", re.IGNORECASE),
}

INCOMPLETE_FULL_CORPUS_NOTICE = "current full-corpus ultragoal is not complete"
HISTORICAL_NOTICE = "historical bounded"
NO_LIVE_CLAIM_NOTICE = "no g008 live-suite claim"


def _load_goals(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _goal_statuses(goals: dict[str, Any]) -> dict[str, str]:
    return {str(goal["id"]): str(goal["status"]) for goal in goals.get("goals", [])}


def _read_lower(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").lower()


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 32) : start].lower()
    return any(marker in prefix for marker in ["not ", "no ", "does not ", "do not ", "must not "])


def audit_claim_boundaries(
    *,
    goals_path: str | Path = ".omx/ultragoal/goals.json",
    report_paths: list[str] | None = None,
) -> dict[str, Any]:
    goals = _load_goals(goals_path)
    statuses = _goal_statuses(goals)
    report_paths = report_paths or DEFAULT_REPORT_PATHS
    findings: list[dict[str, Any]] = []
    checked_paths: list[str] = []

    for path_text in report_paths:
        path = Path(path_text)
        if not path.exists():
            findings.append({"severity": "error", "path": path_text, "code": "missing_report_path"})
            continue
        checked_paths.append(path_text)
        text = path.read_text(encoding="utf-8")
        for code, pattern in FORBIDDEN_PATTERNS.items():
            if any(not _is_negated(text, match.start()) for match in pattern.finditer(text)):
                findings.append({"severity": "error", "path": path_text, "code": code})

    full_corpus_done = all(status == "complete" for status in statuses.values())
    if not full_corpus_done:
        for path_text in ["docs/final_research_report.md", "docs/evidence_index.md", "docs/reproducibility_runbook.md"]:
            path = Path(path_text)
            if not path.exists():
                continue
            text = _read_lower(path)
            if INCOMPLETE_FULL_CORPUS_NOTICE not in text and HISTORICAL_NOTICE not in text:
                findings.append(
                    {
                        "severity": "error",
                        "path": path_text,
                        "code": "missing_incomplete_full_corpus_notice",
                        "detail": f"add '{INCOMPLETE_FULL_CORPUS_NOTICE}' or '{HISTORICAL_NOTICE}'",
                    }
                )

    if statuses.get("G007-runtime-sdk-adapter") == "complete" and statuses.get("G008-live-game-suite") != "complete":
        runtime_text = _read_lower("docs/runtime_sdk_adapter.md") if Path("docs/runtime_sdk_adapter.md").exists() else ""
        if NO_LIVE_CLAIM_NOTICE not in runtime_text and "does not by itself prove live game control" not in runtime_text:
            findings.append(
                {
                    "severity": "error",
                    "path": "docs/runtime_sdk_adapter.md",
                    "code": "missing_no_live_suite_claim_notice",
                }
            )

    payload = {
        "schema": "claim_boundary_audit.v1",
        "goals_path": str(goals_path),
        "active_goal_id": goals.get("activeGoalId"),
        "goal_statuses": statuses,
        "checked_paths": checked_paths,
        "findings": findings,
        "status": "pass" if not any(item["severity"] == "error" for item in findings) else "fail",
    }
    return payload
