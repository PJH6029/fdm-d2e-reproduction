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


def _package_manifest_evidence(path: Path, rel_path: str, package_manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize the package manifest without hashing it.

    The package manifest includes the final-quality audit hash, so placing the
    package-manifest hash back into the final-quality audit creates an
    unsatisfiable hash cycle. The gate still loads and validates manifest
    coverage; this evidence row intentionally records only stable metadata.
    """

    if not path.exists() or not path.is_file():
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None, "entry_count": 0}
    return {
        "path": rel_path,
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": None,
        "schema": (package_manifest or {}).get("schema"),
        "entry_count": len((package_manifest or {}).get("entries", [])),
        "hash_omitted_reason": "avoid_final_quality_package_manifest_hash_cycle",
    }


def _external_manifest_entries(external_manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not external_manifest:
        return {}
    entries = external_manifest.get("entries", [])
    if isinstance(entries, dict):
        return {str(key): dict(value) for key, value in entries.items() if isinstance(value, dict)}
    return {str(entry.get("path")): dict(entry) for entry in entries if isinstance(entry, dict) and entry.get("path")}


def _external_evidence(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return normalized external evidence only when it is strong enough.

    Final packages intentionally do not git-track multi-GB/TB full-corpus JSONL
    artifacts. They still need positive proof that those artifacts exist at the
    MLXP/PVC location used for training/evaluation. Accept either a full sha256
    or a deterministic run-specific fingerprint, but require non-zero bytes and
    an explicit external storage URI.
    """

    if not entry:
        return None
    exists = bool(entry.get("exists", True))
    bytes_value = int(entry.get("bytes") or 0)
    sha = entry.get("sha256")
    fingerprint = entry.get("fingerprint")
    storage_uri = entry.get("storage_uri")
    if not exists or bytes_value <= 0 or not storage_uri or not (sha or fingerprint):
        return None
    return {
        "path": str(entry.get("path")),
        "exists": True,
        "bytes": bytes_value,
        "sha256": sha,
        "fingerprint": fingerprint,
        "fingerprint_type": entry.get("fingerprint_type"),
        "storage_uri": storage_uri,
        "source_artifact": entry.get("source_artifact"),
        "proof": entry.get("proof"),
    }


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
        if "in" in assertion:
            allowed = list(assertion.get("in") or [])
            if actual not in allowed:
                findings.append(
                    {
                        "severity": "error",
                        "code": "json_assertion_not_in_allowed_values",
                        "path": path_text,
                        "json_path": json_path,
                        "allowed": allowed,
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
    external_manifest_rel_path = config.get("external_artifact_manifest_path")
    external_manifest = _load_json_if_exists(root_path / str(external_manifest_rel_path)) if external_manifest_rel_path else None
    external_entries = _external_manifest_entries(external_manifest)

    findings: list[dict[str, Any]] = []
    goal_reports: list[dict[str, Any]] = []
    required_paths_seen: list[str] = []

    if goals_payload is None:
        findings.append({"severity": "error", "code": "missing_goals_file", "path": str(config.get("goals_path"))})
    if package_manifest is None:
        findings.append({"severity": "error", "code": "missing_package_manifest", "path": str(config.get("package_manifest_path"))})
    if external_manifest_rel_path and external_manifest is None:
        findings.append({"severity": "error", "code": "missing_external_artifact_manifest", "path": str(external_manifest_rel_path)})
    elif external_manifest_rel_path and external_manifest is not None:
        if external_manifest.get("schema") != "external_artifact_manifest.v1":
            findings.append(
                {
                    "severity": "error",
                    "code": "external_artifact_manifest_schema_mismatch",
                    "path": str(external_manifest_rel_path),
                    "actual": external_manifest.get("schema"),
                }
            )
        if external_manifest.get("status") != "pass":
            findings.append(
                {
                    "severity": "error",
                    "code": "external_artifact_manifest_not_pass",
                    "path": str(external_manifest_rel_path),
                    "status": external_manifest.get("status"),
                    "error_count": external_manifest.get("error_count"),
                }
            )

    complete_statuses = set(config.get("complete_statuses", list(DEFAULT_COMPLETE_STATUSES)))
    allow_in_progress_goal_ids = {str(item) for item in config.get("allow_in_progress_goal_ids", [])}
    for gate in config.get("goal_gates", []):
        goal_id = str(gate["id"])
        goal = goals_by_id.get(goal_id)
        status = str(goal.get("status", "missing")) if goal else "missing"
        goal_findings: list[dict[str, Any]] = []
        requires_status = str(gate.get("requires_status", "complete"))
        status_allowed_by_final_precheckpoint = (
            status == "in_progress" and requires_status == "complete" and goal_id in allow_in_progress_goal_ids
        )
        if status != requires_status and not status_allowed_by_final_precheckpoint:
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
        external_paths = {str(path) for path in gate.get("external_artifact_paths", [])}
        for rel_path in gate.get("required_paths", []):
            rel_path = str(rel_path)
            required_paths_seen.append(rel_path)
            if rel_path == package_manifest_rel_path:
                evidence = _package_manifest_evidence(root_path / rel_path, rel_path, package_manifest)
            else:
                evidence = _file_evidence(root_path / rel_path, rel_path)
            external = None
            if not evidence["exists"] and rel_path in external_paths:
                external = _external_evidence(external_entries.get(rel_path))
                if external:
                    evidence = {
                        **evidence,
                        "external_satisfied": True,
                        "external": external,
                    }
            path_reports.append(evidence)
            if not evidence["exists"] and not evidence.get("external_satisfied"):
                goal_findings.append({"severity": "error", "code": "missing_required_artifact", "goal_id": goal_id, "path": rel_path})
                if rel_path in external_paths:
                    goal_findings.append(
                        {
                            "severity": "error",
                            "code": "external_artifact_evidence_missing_or_weak",
                            "goal_id": goal_id,
                            "path": rel_path,
                            "external_manifest_path": str(external_manifest_rel_path),
                        }
                    )
            elif (
                rel_path != package_manifest_rel_path
                and not evidence.get("external_satisfied")
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
        not_complete = [
            str(goal.get("id"))
            for goal in goals
            if str(goal.get("status")) not in complete_statuses
            and not (str(goal.get("id")) in allow_in_progress_goal_ids and str(goal.get("status")) == "in_progress")
        ]
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
        "package_manifest": _package_manifest_evidence(
            package_manifest_path,
            str(config.get("package_manifest_path", "artifacts/reproducibility/package_manifest.json")),
            package_manifest,
        ),
        "external_artifact_manifest": (
            _file_evidence(root_path / str(external_manifest_rel_path), str(external_manifest_rel_path))
            if external_manifest_rel_path
            else None
        ),
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
