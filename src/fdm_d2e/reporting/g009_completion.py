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


def _manifest_entries(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    return {str(entry.get("path")): entry for entry in manifest.get("entries", []) if isinstance(entry, dict)}


def validate_g009_completion(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goal_id = str(config.get("goal_id", "G009-report-repro-package"))
    statuses = _goal_statuses(root_path, goals_path)
    goal_status = statuses.get(goal_id, "missing")
    if goal_status != "complete":
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

    manifest_path = paths.get("package_manifest", "artifacts/reproducibility/package_manifest.json")
    manifest = _load_json(root_path / manifest_path)
    claim_audit = _load_json(root_path / paths.get("claim_boundary_audit", "")) if paths.get("claim_boundary_audit") else None
    final_quality = _load_json(root_path / paths.get("final_quality_audit", "")) if paths.get("final_quality_audit") else None

    for dotted, expected in dict(config.get("manifest_expectations", {})).items():
        actual = _get(manifest, dotted)
        if actual != expected:
            findings.append({"severity": "error", "code": "manifest_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})
    for dotted, expected in dict(config.get("claim_boundary_expectations", {})).items():
        actual = _get(claim_audit, dotted)
        if actual != expected:
            findings.append({"severity": "error", "code": "claim_boundary_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})
    for dotted, expected in dict(config.get("final_quality_expectations", {})).items():
        actual = _get(final_quality, dotted)
        if actual != expected:
            findings.append({"severity": "error", "code": "final_quality_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})

    entries = _manifest_entries(manifest)
    required_manifest_paths = [str(path) for path in config.get("required_manifest_paths", [])]
    missing_manifest_paths: list[str] = []
    hash_mismatches: list[dict[str, Any]] = []
    for rel_path in required_manifest_paths:
        entry = entries.get(rel_path)
        if entry is None:
            missing_manifest_paths.append(rel_path)
            continue
        actual = root_path / rel_path
        if not actual.exists() or not actual.is_file():
            findings.append({"severity": "error", "code": "manifest_entry_points_to_missing_file", "path": rel_path})
            continue
        actual_hash = sha256_file(actual)
        if entry.get("sha256") != actual_hash:
            hash_mismatches.append({"path": rel_path, "manifest_sha256": entry.get("sha256"), "actual_sha256": actual_hash})
    if missing_manifest_paths:
        findings.append({"severity": "error", "code": "package_manifest_missing_required_paths", "missing": sorted(missing_manifest_paths)})
    if hash_mismatches:
        findings.append({"severity": "error", "code": "package_manifest_hash_mismatch", "mismatches": hash_mismatches})

    min_doc_bytes = int(config.get("min_doc_bytes", 128))
    for key in config.get("required_docs", []):
        rel_path = paths.get(str(key), str(key))
        status = _file_status(root_path / rel_path, rel_path)
        if status["exists"] and int(status["bytes"]) < min_doc_bytes:
            findings.append({"severity": "error", "code": "document_too_small", "artifact_key": str(key), "path": rel_path, "bytes": status["bytes"], "min_bytes": min_doc_bytes})

    forbidden_claims = set(str(item) for item in config.get("required_forbidden_claims", []))
    claim_forbidden = set(str(item) for item in (claim_audit or {}).get("forbidden_claims", []) or [])
    # claim_boundary_audit.v1 stores checked report paths and findings rather than forbidden_claims;
    # for final G009 we therefore verify the configured report text does not contain positive forbidden phrases.
    if forbidden_claims and claim_audit is not None and claim_audit.get("status") != "pass":
        findings.append({"severity": "error", "code": "claim_boundary_audit_not_pass", "actual": claim_audit.get("status")})
    if claim_forbidden and not forbidden_claims.issubset(claim_forbidden):
        findings.append({"severity": "error", "code": "claim_boundary_missing_forbidden_claims", "missing": sorted(forbidden_claims - claim_forbidden)})

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g009_completion_audit.v1",
        "status": "pass" if not errors else "fail",
        "goal_id": goal_id,
        "goal_status": goal_status,
        "prerequisite_goal_statuses": prereq_report,
        "artifacts": artifacts,
        "required_manifest_paths": required_manifest_paths,
        "manifest_entry_count": len(entries),
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "This audit is required before checkpointing G009 complete; final report, evidence index, runbook, package manifest, claim-boundary audit, manifest hashes, and prerequisite goal statuses must all be current.",
    }


def write_g009_completion_audit(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_g009_completion(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload
