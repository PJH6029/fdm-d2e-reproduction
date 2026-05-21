#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import sha256_file, write_json
from fdm_d2e.reporting.g009_completion import validate_g009_completion


DEFAULT_OUTPUT = "artifacts/reproducibility/g009_readiness_plan.json"
REFRESHABLE_OUTPUT_KEYS = {"package_manifest", "claim_boundary_audit", "final_quality_audit"}


def _path(root: Path, value: str | Path | None) -> Path:
    p = Path(value or "")
    return p if p.is_absolute() else root / p


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"schema": "invalid_json", "error": str(exc)}


def _file_status(root: Path, rel_path: str | Path | None) -> dict[str, Any]:
    if not rel_path:
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    p = _path(root, rel_path)
    if not p.exists() or not p.is_file():
        return {"path": str(rel_path), "exists": False, "bytes": 0, "sha256": None}
    return {"path": str(rel_path), "exists": True, "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def _goal_statuses(root: Path, goals_path: str) -> dict[str, str]:
    payload = _load_json(_path(root, goals_path)) or {}
    return {str(goal.get("id")): str(goal.get("status")) for goal in payload.get("goals", [])}


def _json_path(data: dict[str, Any] | None, dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _audit_status(root: Path, rel_path: str | Path | None) -> dict[str, Any]:
    payload = _load_json(_path(root, rel_path))
    return {
        **_file_status(root, rel_path),
        "schema": payload.get("schema") if isinstance(payload, dict) else None,
        "status": payload.get("status") if isinstance(payload, dict) else None,
        "error_count": payload.get("error_count") if isinstance(payload, dict) else None,
        "entry_count": payload.get("entry_count") if isinstance(payload, dict) else None,
    }


def _manifest_entries(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, dict):
        return {}
    return {str(entry.get("path")): entry for entry in manifest.get("entries", []) if isinstance(entry, dict)}


def _manifest_required_path_statuses(root: Path, manifest_path: str | None, required_paths: list[str]) -> list[dict[str, Any]]:
    manifest = _load_json(_path(root, manifest_path)) if manifest_path else None
    entries = _manifest_entries(manifest)
    statuses = []
    for rel_path in required_paths:
        entry = entries.get(rel_path)
        file_status = _file_status(root, rel_path)
        status = {
            "path": rel_path,
            "in_manifest": entry is not None,
            "file_exists": file_status["exists"],
            "manifest_sha256": entry.get("sha256") if isinstance(entry, dict) else None,
            "actual_sha256": file_status.get("sha256"),
            "hash_matches": bool(entry) and entry.get("sha256") == file_status.get("sha256"),
        }
        statuses.append(status)
    return statuses


def build_readiness_plan(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    config = load_config(_path(root, args.g009_completion_config))
    final_quality_config = load_config(_path(root, args.final_quality_config))
    goals_path = str(config.get("goals_path", final_quality_config.get("goals_path", ".omx/ultragoal/goals.json")))
    statuses = _goal_statuses(root, goals_path)
    findings: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    prereq_statuses: dict[str, str] = {}
    for goal_id in config.get("prerequisite_goals", []):
        actual = statuses.get(str(goal_id), "missing")
        prereq_statuses[str(goal_id)] = actual
        if actual != "complete":
            item = {"severity": "error", "code": "prerequisite_goal_not_complete", "goal_id": str(goal_id), "actual": actual}
            if args.allow_precheckpoint:
                warnings.append({**item, "severity": "warning"})
            else:
                findings.append(item)

    paths = {str(key): str(value) for key, value in dict(config.get("paths", {})).items()}
    doc_statuses: dict[str, dict[str, Any]] = {}
    min_doc_bytes = int(config.get("min_doc_bytes", 128))
    for key in config.get("required_docs", []):
        rel_path = paths.get(str(key), str(key))
        status = _file_status(root, rel_path)
        doc_statuses[str(key)] = status
        if not status["exists"]:
            findings.append({"severity": "error", "code": "missing_required_doc", "artifact_key": str(key), "path": status["path"]})
        elif int(status["bytes"]) < min_doc_bytes:
            findings.append(
                {
                    "severity": "error",
                    "code": "document_too_small",
                    "artifact_key": str(key),
                    "path": status["path"],
                    "bytes": status["bytes"],
                    "min_bytes": min_doc_bytes,
                }
            )

    refreshable_outputs = {key: _audit_status(root, paths.get(key)) for key in sorted(REFRESHABLE_OUTPUT_KEYS)}
    for key, status in refreshable_outputs.items():
        if not status["exists"]:
            item = {"severity": "warning", "code": "refreshable_output_missing", "artifact_key": key, "path": status["path"]}
            if args.require_existing_final_outputs:
                findings.append({**item, "severity": "error"})
            else:
                warnings.append(item)

    current_g009_audit = _audit_status(root, config.get("output_path", args.g009_audit_output))
    current_finalization_summary = _audit_status(root, args.summary_out)
    if args.require_existing_final_outputs:
        required_statuses = {
            "claim_boundary_audit": "pass",
            "final_quality_audit": "pass",
        }
        for key, expected in required_statuses.items():
            actual = refreshable_outputs[key].get("status")
            if actual != expected:
                findings.append(
                    {
                        "severity": "error",
                        "code": "refreshable_output_status_not_pass",
                        "artifact_key": key,
                        "path": refreshable_outputs[key]["path"],
                        "expected": expected,
                        "actual": actual,
                    }
                )
        if current_g009_audit.get("status") != "pass":
            findings.append(
                {
                    "severity": "error",
                    "code": "g009_completion_audit_not_pass",
                    "path": current_g009_audit["path"],
                    "actual": current_g009_audit.get("status"),
                }
            )

    manifest_path = paths.get("package_manifest", str(final_quality_config.get("package_manifest_path", "artifacts/reproducibility/package_manifest.json")))
    manifest_required_paths = [str(path) for path in config.get("required_manifest_paths", [])]
    manifest_path_statuses = _manifest_required_path_statuses(root, manifest_path, manifest_required_paths)
    manifest_hash_mismatches = [item for item in manifest_path_statuses if item["in_manifest"] and item["file_exists"] and not item["hash_matches"]]
    manifest_missing_required = [item for item in manifest_path_statuses if not item["in_manifest"]]
    if manifest_hash_mismatches:
        warnings.append({"severity": "warning", "code": "package_manifest_hash_mismatch_until_refreshed", "count": len(manifest_hash_mismatches)})
    if manifest_missing_required and refreshable_outputs.get("package_manifest", {}).get("exists"):
        warnings.append({"severity": "warning", "code": "package_manifest_missing_required_paths_until_refreshed", "count": len(manifest_missing_required)})

    completion_projection = validate_g009_completion(config, root=root)
    if args.require_existing_final_outputs and completion_projection.get("status") != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "completion_projection_not_pass",
                "error_count": completion_projection.get("error_count"),
            }
        )

    commands = {
        "finalize": ["uv", "run", "python", "scripts/finalize_g009_report_package.py"],
        "finalize_allow_fail": ["uv", "run", "python", "scripts/finalize_g009_report_package.py", "--allow-fail"],
        "validate_completion": ["uv", "run", "python", "scripts/validate_g009_completion.py"],
        "build_package_manifest": ["uv", "run", "python", "scripts/build_repro_package_manifest.py", "--output", manifest_path],
        "validate_final_quality": ["uv", "run", "python", "scripts/validate_final_quality_gates.py"],
    }
    status = "ready" if not findings else "blocked"
    return {
        "schema": "g009_readiness_plan.v1",
        "status": status,
        "root": str(root),
        "allow_precheckpoint": bool(args.allow_precheckpoint),
        "require_existing_final_outputs": bool(args.require_existing_final_outputs),
        "goal_statuses": statuses,
        "prerequisite_goal_statuses": prereq_statuses,
        "required_docs": doc_statuses,
        "refreshable_outputs": refreshable_outputs,
        "current_g009_audit": current_g009_audit,
        "current_finalization_summary": current_finalization_summary,
        "manifest_required_paths": manifest_path_statuses,
        "completion_projection_status": completion_projection.get("status"),
        "completion_projection_error_count": completion_projection.get("error_count"),
        "commands": commands,
        "findings": findings,
        "warnings": warnings,
        "claim_boundary": "G009 readiness planning is read-only; it does not refresh audits, rebuild the package, checkpoint G009, or complete the aggregate goal.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan/check G009 final report and reproducibility package readiness without refreshing artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--g009-completion-config", default="configs/eval/g009_completion.yaml")
    parser.add_argument("--final-quality-config", default="configs/eval/final_quality_gates.yaml")
    parser.add_argument("--summary-out", default="artifacts/reproducibility/g009_finalization_summary.json")
    parser.add_argument("--g009-audit-output", default="artifacts/reproducibility/g009_completion_audit.json")
    parser.add_argument("--require-existing-final-outputs", action="store_true")
    parser.add_argument("--allow-precheckpoint", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_readiness_plan(args)
    write_json(_path(Path(args.root).resolve(), args.output), payload)
    print(f"g009 readiness plan: status={payload['status']} findings={len(payload['findings'])} warnings={len(payload['warnings'])} output={args.output}")
    return 0 if payload["status"] == "ready" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
