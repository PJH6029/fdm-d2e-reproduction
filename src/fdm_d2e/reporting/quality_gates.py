from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import sha256_file, write_json


DEFAULT_COMPLETE_STATUSES = {"complete"}


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _status_counts(goals: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for goal in goals:
        status = str(goal.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _file_evidence(path: Path, rel_path: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    return {"path": rel_path, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _manifest_paths(package_manifest: dict[str, Any] | None) -> set[str]:
    if not package_manifest:
        return set()
    return {str(entry.get("path")) for entry in package_manifest.get("entries", [])}


def _get_nested(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _check_json_assertions(root: Path, assertions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for assertion in assertions:
        path_text = str(assertion["path"])
        json_path = str(assertion["json_path"])
        path = root / path_text
        payload = _load_json_if_exists(path)
        if payload is None:
            findings.append({"severity": "error", "code": "json_assertion_file_missing", "path": path_text, "json_path": json_path})
            continue
        actual = _get_nested(payload, json_path)
        expected = assertion.get("equals")
        if "equals" in assertion and actual != expected:
            findings.append(
                {
                    "severity": "error",
                    "code": "json_assertion_mismatch",
                    "path": path_text,
                    "json_path": json_path,
                    "expected": expected,
                    "actual": actual,
                }
            )
        if assertion.get("truthy") and not actual:
            findings.append({"severity": "error", "code": "json_assertion_not_truthy", "path": path_text, "json_path": json_path, "actual": actual})
    return findings


def validate_final_quality_gates(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    """Validate final ultragoal evidence without mutating goal state.

    This is a completion-audit helper: it records exactly which goal/artifact
    requirements are proven by current files and which remain missing. A pass is
    intentionally strict and requires all configured goal statuses, artifacts,
    package-manifest coverage, and claim/live-suite audit checks to pass.
    """

    root_path = Path(root)
    goals_path = root_path / str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goals_payload = _load_json_if_exists(goals_path)
    goals = list((goals_payload or {}).get("goals", []))
    goals_by_id = {str(goal.get("id")): goal for goal in goals}
    package_manifest_path = root_path / str(config.get("package_manifest_path", "artifacts/reproducibility/package_manifest.json"))
    package_manifest_rel_path = str(config.get("package_manifest_path", "artifacts/reproducibility/package_manifest.json"))
    package_manifest = _load_json_if_exists(package_manifest_path)
    manifest_entries = _manifest_paths(package_manifest)

    findings: list[dict[str, Any]] = []
    goal_reports: list[dict[str, Any]] = []
    required_paths_seen: list[str] = []

    if goals_payload is None:
        findings.append({"severity": "error", "code": "missing_goals_file", "path": str(config.get("goals_path"))})
    if package_manifest is None:
        findings.append({"severity": "error", "code": "missing_package_manifest", "path": str(config.get("package_manifest_path"))})

    complete_statuses = set(config.get("complete_statuses", list(DEFAULT_COMPLETE_STATUSES)))
    for gate in config.get("goal_gates", []):
        goal_id = str(gate["id"])
        goal = goals_by_id.get(goal_id)
        status = str(goal.get("status", "missing")) if goal else "missing"
        goal_findings: list[dict[str, Any]] = []
        requires_status = str(gate.get("requires_status", "complete"))
        if status != requires_status:
            goal_findings.append(
                {
                    "severity": "error",
                    "code": "goal_status_not_complete",
                    "goal_id": goal_id,
                    "expected_status": requires_status,
                    "actual_status": status,
                }
            )
        path_reports = []
        for rel_path in gate.get("required_paths", []):
            rel_path = str(rel_path)
            required_paths_seen.append(rel_path)
            evidence = _file_evidence(root_path / rel_path, rel_path)
            path_reports.append(evidence)
            if not evidence["exists"]:
                goal_findings.append({"severity": "error", "code": "missing_required_artifact", "goal_id": goal_id, "path": rel_path})
            elif (
                rel_path != package_manifest_rel_path
                and gate.get("require_package_manifest_coverage", True)
                and package_manifest is not None
                and rel_path not in manifest_entries
            ):
                goal_findings.append({"severity": "error", "code": "artifact_not_in_package_manifest", "goal_id": goal_id, "path": rel_path})
        goal_findings.extend(_check_json_assertions(root_path, list(gate.get("json_assertions", []))))
        goal_status = "pass" if not any(item["severity"] == "error" for item in goal_findings) else "fail"
        goal_reports.append(
            {
                "goal_id": goal_id,
                "title": gate.get("title"),
                "actual_status": status,
                "required_status": requires_status,
                "status": goal_status,
                "artifacts": path_reports,
                "findings": goal_findings,
            }
        )
        findings.extend(goal_findings)

    if config.get("require_all_goals_complete", True) and goals:
        not_complete = [str(goal.get("id")) for goal in goals if str(goal.get("status")) not in complete_statuses]
        if not_complete:
            findings.append({"severity": "error", "code": "not_all_ultragoal_stories_complete", "goal_ids": not_complete})

    claim_path_text = str(config.get("claim_boundary_audit_path", "artifacts/reproducibility/claim_boundary_audit.json"))
    claim_payload = _load_json_if_exists(root_path / claim_path_text)
    if claim_payload is None:
        findings.append({"severity": "error", "code": "missing_claim_boundary_audit", "path": claim_path_text})
    elif claim_payload.get("status") != "pass":
        findings.append({"severity": "error", "code": "claim_boundary_audit_not_pass", "path": claim_path_text, "status": claim_payload.get("status")})

    live_validation_path = config.get("live_suite_evidence_validation_path")
    live_payload = _load_json_if_exists(root_path / str(live_validation_path)) if live_validation_path else None
    if config.get("require_live_suite_pass", False):
        if live_payload is None:
            findings.append({"severity": "error", "code": "missing_live_suite_evidence_validation", "path": str(live_validation_path)})
        elif _get_nested(live_payload, "quality_gate.status") != "pass":
            findings.append(
                {
                    "severity": "error",
                    "code": "live_suite_evidence_not_pass",
                    "path": str(live_validation_path),
                    "status": _get_nested(live_payload, "quality_gate.status"),
                }
            )

    if package_manifest is not None and config.get("require_package_manifest_for_configured_paths", True):
        missing_from_manifest = sorted({path for path in required_paths_seen if path != package_manifest_rel_path and path not in manifest_entries and (root_path / path).exists()})
        if missing_from_manifest:
            findings.append({"severity": "error", "code": "configured_artifacts_missing_from_manifest", "paths": missing_from_manifest})

    error_count = sum(1 for item in findings if item.get("severity") == "error")
    payload = {
        "schema": "final_quality_gate_audit.v1",
        "status": "pass" if error_count == 0 else "fail",
        "goals_path": str(config.get("goals_path", ".omx/ultragoal/goals.json")),
        "goal_status_counts": _status_counts(goals),
        "goal_reports": goal_reports,
        "package_manifest": _file_evidence(package_manifest_path, str(config.get("package_manifest_path", "artifacts/reproducibility/package_manifest.json"))),
        "claim_boundary_audit_status": claim_payload.get("status") if claim_payload else None,
        "live_suite_validation_status": _get_nested(live_payload, "quality_gate.status") if live_payload else None,
        "findings": findings,
        "error_count": error_count,
        "claim_boundary": "A pass is required before completing the aggregate ultragoal; failing/pending results are expected while G003-G009 remain incomplete.",
    }
    return payload


def write_final_quality_gate_audit(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_final_quality_gates(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload
