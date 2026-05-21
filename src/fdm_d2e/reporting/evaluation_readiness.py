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


def _goal_statuses(root: Path, goals_path: str) -> dict[str, str]:
    payload = _load_json(root / goals_path) or {}
    return {str(goal.get("id")): str(goal.get("status")) for goal in payload.get("goals", [])}


def _comparison_key(row: dict[str, Any], field_names: list[str]) -> str | None:
    for name in field_names:
        value = row.get(name)
        if value is not None:
            return str(value)
    return None


def _validate_endpoint_statistics(payload: dict[str, Any] | None, config: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if payload is None:
        return [{"severity": "error", "code": "missing_endpoint_statistics"}]
    if payload.get("status") != "pass":
        findings.append({"severity": "error", "code": "endpoint_statistics_not_pass", "actual": payload.get("status")})
    comparisons = payload.get("comparisons") or payload.get("endpoint_tables") or []
    if not isinstance(comparisons, list) or not comparisons:
        findings.append({"severity": "error", "code": "endpoint_statistics_no_comparisons"})
        comparisons = []
    required_splits = set(config.get("required_splits", []))
    required_endpoints = set(config.get("required_endpoints", []))
    seen_splits = {_comparison_key(row, ["split", "eval_split", "split_name"]) for row in comparisons if isinstance(row, dict)}
    seen_endpoints = {_comparison_key(row, ["endpoint", "metric", "metric_name"]) for row in comparisons if isinstance(row, dict)}
    missing_splits = sorted(required_splits - {item for item in seen_splits if item})
    missing_endpoints = sorted(required_endpoints - {item for item in seen_endpoints if item})
    if missing_splits:
        findings.append({"severity": "error", "code": "endpoint_statistics_missing_splits", "missing": missing_splits})
    if missing_endpoints:
        findings.append({"severity": "error", "code": "endpoint_statistics_missing_endpoints", "missing": missing_endpoints})
    required_fields = list(config.get("required_comparison_fields", []))
    for idx, row in enumerate(comparisons):
        if not isinstance(row, dict):
            findings.append({"severity": "error", "code": "endpoint_comparison_not_object", "index": idx})
            continue
        missing = [field for field in required_fields if row.get(field) is None]
        if missing:
            findings.append({"severity": "error", "code": "endpoint_comparison_missing_fields", "index": idx, "missing": missing})
    return findings


def _validate_failure_analysis(payload: dict[str, Any] | None, config: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if payload is None:
        return [{"severity": "error", "code": "missing_failure_analysis"}]
    if payload.get("status") != "pass":
        findings.append({"severity": "error", "code": "failure_analysis_not_pass", "actual": payload.get("status")})
    required_axes = set(config.get("required_failure_axes", []))
    axes = payload.get("axes") or payload.get("failure_axes") or []
    if isinstance(axes, dict):
        seen_axes = set(axes.keys())
    else:
        seen_axes = {str(axis) for axis in axes}
    missing_axes = sorted(required_axes - seen_axes)
    if missing_axes:
        findings.append({"severity": "error", "code": "failure_analysis_missing_axes", "missing": missing_axes})
    if config.get("require_non_rejections", True) and not payload.get("non_rejections"):
        findings.append({"severity": "error", "code": "failure_analysis_missing_non_rejections"})
    if config.get("require_examples", True) and not payload.get("examples"):
        findings.append({"severity": "error", "code": "failure_analysis_missing_examples"})
    return findings


def _validate_claim_taxonomy(payload: dict[str, Any] | None, config: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if payload is None:
        return [{"severity": "error", "code": "missing_claim_taxonomy"}]
    if payload.get("status") != "pass":
        findings.append({"severity": "error", "code": "claim_taxonomy_not_pass", "actual": payload.get("status")})
    required_claims = set(config.get("required_claim_taxonomy", []))
    claims = payload.get("claims") or []
    if isinstance(claims, dict):
        seen = set(claims.keys())
    else:
        seen = {str(item.get("id") or item.get("claim") or item) for item in claims}
    missing = sorted(required_claims - seen)
    if missing:
        findings.append({"severity": "error", "code": "claim_taxonomy_missing_claims", "missing": missing})
    return findings


def validate_g006_evaluation_readiness(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    statuses = _goal_statuses(root_path, goals_path)
    findings: list[dict[str, Any]] = []

    for goal_id in config.get("prerequisite_goals", []):
        if statuses.get(goal_id) != "complete":
            findings.append({"severity": "error", "code": "prerequisite_goal_not_complete", "goal_id": goal_id, "actual": statuses.get(goal_id, "missing")})

    endpoint_path = str(config.get("endpoint_statistics_path", "artifacts/eval/final_endpoint_statistics.json"))
    failure_path = str(config.get("failure_analysis_path", "artifacts/eval/final_failure_analysis.json"))
    taxonomy_path = str(config.get("claim_taxonomy_path", "artifacts/eval/final_claim_taxonomy.json"))
    endpoint_payload = _load_json(root_path / endpoint_path)
    failure_payload = _load_json(root_path / failure_path)
    taxonomy_payload = _load_json(root_path / taxonomy_path)

    findings.extend(_validate_endpoint_statistics(endpoint_payload, config))
    findings.extend(_validate_failure_analysis(failure_payload, config))
    findings.extend(_validate_claim_taxonomy(taxonomy_payload, config))

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g006_evaluation_readiness_audit.v1",
        "status": "pass" if not errors else "fail",
        "goals_path": goals_path,
        "prerequisite_goal_statuses": {goal_id: statuses.get(goal_id, "missing") for goal_id in config.get("prerequisite_goals", [])},
        "artifacts": {
            "endpoint_statistics": _file_status(root_path / endpoint_path, endpoint_path),
            "failure_analysis": _file_status(root_path / failure_path, failure_path),
            "claim_taxonomy": _file_status(root_path / taxonomy_path, taxonomy_path),
        },
        "required_splits": list(config.get("required_splits", [])),
        "required_endpoints": list(config.get("required_endpoints", [])),
        "required_failure_axes": list(config.get("required_failure_axes", [])),
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "This readiness audit does not complete G006; G006 is complete only after final endpoint statistics, failure analysis, and claim taxonomy pass with G003/G004 prerequisites complete.",
    }


def write_g006_evaluation_readiness(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_g006_evaluation_readiness(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload
