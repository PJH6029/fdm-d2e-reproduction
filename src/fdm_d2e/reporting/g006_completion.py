from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import sha256_file, write_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _file_status(path: Path, rel_path: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    return {"path": rel_path, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _get(data: dict[str, Any] | None, dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _goal_statuses(root: Path, goals_path: str) -> dict[str, str]:
    payload = _load_json(root / goals_path) or {}
    return {str(goal.get("id")): str(goal.get("status")) for goal in payload.get("goals", [])}


def _assert_expectations(payload: dict[str, Any] | None, expectations: dict[str, Any], *, source: str, findings: list[dict[str, Any]]) -> None:
    for dotted, expected in expectations.items():
        actual = _get(payload, dotted)
        if actual != expected:
            findings.append(
                {
                    "severity": "error",
                    "code": "json_expectation_mismatch",
                    "source": source,
                    "json_path": dotted,
                    "expected": expected,
                    "actual": actual,
                }
            )


def validate_g006_completion(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goal_id = str(config.get("goal_id", "G006-evaluation-failure-analysis"))
    statuses = _goal_statuses(root_path, goals_path)
    goal_status = statuses.get(goal_id, "missing")
    require_goal_checkpoint = bool(config.get("require_goal_checkpoint_complete", True))
    if require_goal_checkpoint and goal_status != "complete":
        findings.append({"severity": "error", "code": "goal_not_checkpointed_complete", "goal_id": goal_id, "actual": goal_status})
    prereq_report = {}
    for prereq in config.get("prerequisite_goals", []):
        actual = statuses.get(str(prereq), "missing")
        prereq_report[str(prereq)] = actual
        if actual != "complete":
            findings.append({"severity": "error", "code": "prerequisite_goal_not_complete", "goal_id": str(prereq), "actual": actual})

    paths = {key: str(value) for key, value in dict(config.get("paths", {})).items()}
    artifacts = {key: _file_status(root_path / rel_path, rel_path) for key, rel_path in paths.items()}
    for key, evidence in artifacts.items():
        if not evidence["exists"]:
            findings.append({"severity": "error", "code": "missing_required_artifact", "artifact_key": key, "path": evidence["path"]})

    endpoint = _load_json(root_path / paths.get("endpoint_statistics", "")) if paths.get("endpoint_statistics") else None
    failure = _load_json(root_path / paths.get("failure_analysis", "")) if paths.get("failure_analysis") else None
    taxonomy = _load_json(root_path / paths.get("claim_taxonomy", "")) if paths.get("claim_taxonomy") else None
    readiness = _load_json(root_path / paths.get("readiness_audit", "")) if paths.get("readiness_audit") else None
    build_summary = _load_json(root_path / paths.get("build_summary", "")) if paths.get("build_summary") else None

    _assert_expectations(endpoint, dict(config.get("endpoint_expectations", {})), source="endpoint_statistics", findings=findings)
    _assert_expectations(failure, dict(config.get("failure_expectations", {})), source="failure_analysis", findings=findings)
    _assert_expectations(taxonomy, dict(config.get("taxonomy_expectations", {})), source="claim_taxonomy", findings=findings)
    _assert_expectations(readiness, dict(config.get("readiness_expectations", {})), source="readiness_audit", findings=findings)
    _assert_expectations(build_summary, dict(config.get("build_summary_expectations", {})), source="build_summary", findings=findings)

    required_splits = set(str(item) for item in config.get("required_splits", []))
    required_endpoints = set(str(item) for item in config.get("required_endpoints", []))
    endpoint_splits = set(str(item) for item in (endpoint or {}).get("required_splits", []) or [])
    endpoint_endpoints = set(str(item) for item in (endpoint or {}).get("required_endpoints", []) or [])
    if endpoint is not None:
        rows = [row for row in endpoint.get("comparisons", []) or [] if isinstance(row, dict)]
        endpoint_splits.update(str(row.get("split")) for row in rows if row.get("split") is not None)
        endpoint_endpoints.update(str(row.get("endpoint")) for row in rows if row.get("endpoint") is not None)
    missing_splits = sorted(required_splits - endpoint_splits)
    missing_endpoints = sorted(required_endpoints - endpoint_endpoints)
    if missing_splits:
        findings.append({"severity": "error", "code": "endpoint_statistics_missing_required_splits", "missing": missing_splits})
    if missing_endpoints:
        findings.append({"severity": "error", "code": "endpoint_statistics_missing_required_endpoints", "missing": missing_endpoints})

    required_failure_axes = set(str(item) for item in config.get("required_failure_axes", []))
    axes = (failure or {}).get("axes", {}) if isinstance((failure or {}).get("axes", {}), dict) else {}
    missing_axes = sorted(axis for axis in required_failure_axes if not axes.get(axis))
    if missing_axes:
        findings.append({"severity": "error", "code": "failure_analysis_missing_axes", "missing": missing_axes})
    if bool(config.get("require_non_rejections", True)) and not ((failure or {}).get("non_rejections") or []):
        findings.append({"severity": "error", "code": "failure_analysis_missing_non_rejections"})
    if bool(config.get("require_failure_examples", True)) and not ((failure or {}).get("examples") or []):
        findings.append({"severity": "error", "code": "failure_analysis_missing_examples"})

    required_claims = set(str(item) for item in config.get("required_claim_taxonomy", []))
    claim_rows = [item for item in (taxonomy or {}).get("claims", []) or [] if isinstance(item, dict)]
    actual_claims = {str(item.get("id")) for item in claim_rows}
    missing_claims = sorted(required_claims - actual_claims)
    if missing_claims:
        findings.append({"severity": "error", "code": "claim_taxonomy_missing_claims", "missing": missing_claims})
    claim_by_id = {str(item.get("id")): item for item in claim_rows}
    for claim_id, expected_state in dict(config.get("required_claim_states", {})).items():
        claim_row = claim_by_id.get(str(claim_id), {})
        actual_state = claim_row.get("state")
        if actual_state != expected_state:
            findings.append(
                {
                    "severity": "error",
                    "code": "claim_taxonomy_state_mismatch",
                    "claim_id": str(claim_id),
                    "expected": expected_state,
                    "actual": actual_state,
                }
            )
    states_requiring_evidence = {str(item) for item in config.get("claim_states_requiring_evidence", [])}
    for claim_id, claim_row in sorted(claim_by_id.items()):
        state = str(claim_row.get("state"))
        if state in states_requiring_evidence and not claim_row.get("evidence_paths"):
            findings.append(
                {
                    "severity": "error",
                    "code": "claim_taxonomy_missing_evidence_for_state",
                    "claim_id": claim_id,
                    "state": state,
                }
            )
    forbidden = set(str(item) for item in (taxonomy or {}).get("forbidden_claims", []) or [])
    required_forbidden = set(str(item) for item in config.get("required_forbidden_claims", []))
    missing_forbidden = sorted(required_forbidden - forbidden)
    if missing_forbidden:
        findings.append({"severity": "error", "code": "claim_taxonomy_missing_forbidden_claims", "missing": missing_forbidden})

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g006_completion_audit.v1",
        "status": "pass" if not errors else "fail",
        "goal_id": goal_id,
        "goal_status": goal_status,
        "require_goal_checkpoint_complete": require_goal_checkpoint,
        "prerequisite_goal_statuses": prereq_report,
        "required_splits": sorted(required_splits),
        "required_endpoints": sorted(required_endpoints),
        "required_failure_axes": sorted(required_failure_axes),
        "required_claim_taxonomy": sorted(required_claims),
        "required_claim_states": dict(config.get("required_claim_states", {})),
        "claim_states_requiring_evidence": sorted(str(item) for item in config.get("claim_states_requiring_evidence", [])),
        "artifacts": artifacts,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "This audit is required before checkpointing G006 complete; endpoint statistics, failure analysis, claim taxonomy, readiness, and final artifact build summary must all pass with G003/G004/G005 prerequisites complete.",
    }


def write_g006_completion_audit(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_g006_completion(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload
