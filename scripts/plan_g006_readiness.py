#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json


DEFAULT_OUTPUT = "artifacts/eval/g006_readiness_plan.json"


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
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
        return {"path": rel_path, "exists": False, "bytes": 0}
    p = _path(root, rel_path)
    return {"path": str(rel_path), "exists": p.exists() and p.is_file(), "bytes": p.stat().st_size if p.exists() and p.is_file() else 0}


def _goal_statuses(root: Path, goals_path: str) -> dict[str, str]:
    payload = _load_json(_path(root, goals_path)) or {}
    return {str(goal.get("id")): str(goal.get("status")) for goal in payload.get("goals", [])}


def _status_payload(root: Path, rel_path: str | Path) -> dict[str, Any]:
    payload = _load_json(_path(root, rel_path))
    return {
        **_file_status(root, rel_path),
        "status": payload.get("status") if isinstance(payload, dict) else None,
        "error_count": payload.get("error_count") if isinstance(payload, dict) else None,
    }


def build_readiness_plan(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    build_config = load_config(_path(root, args.build_config))
    readiness_config = load_config(_path(root, args.readiness_config))
    completion_config = load_config(_path(root, args.g006_completion_config))
    goals_path = str(build_config.get("goals_path", readiness_config.get("goals_path", ".omx/ultragoal/goals.json")))
    statuses = _goal_statuses(root, goals_path)
    findings: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    prereq_ids = list(dict.fromkeys([*build_config.get("prerequisite_goals", []), *readiness_config.get("prerequisite_goals", []), *completion_config.get("prerequisite_goals", [])]))
    prereq_statuses: dict[str, str] = {}
    for goal_id in prereq_ids:
        actual = statuses.get(str(goal_id), "missing")
        prereq_statuses[str(goal_id)] = actual
        if actual != "complete":
            item = {"severity": "error", "code": "prerequisite_goal_not_complete", "goal_id": str(goal_id), "actual": actual}
            if args.allow_precheckpoint:
                warnings.append({**item, "severity": "warning"})
            else:
                findings.append(item)

    comparison_sources = []
    for source in build_config.get("comparison_sources", []):
        status = _file_status(root, source.get("path"))
        comparison_sources.append({"id": source.get("id"), "split": source.get("split"), "model_namespace": source.get("model_namespace"), **status})
        if not status["exists"]:
            findings.append({"severity": "error", "code": "missing_comparison_source", "source_id": source.get("id"), "path": status["path"]})

    metadata_sources = []
    for rel_path in build_config.get("metadata_sources", []):
        status = _file_status(root, rel_path)
        metadata_sources.append(status)
        if not status["exists"]:
            findings.append({"severity": "error", "code": "missing_metadata_source", "path": status["path"]})

    required_states = {str(item) for item in build_config.get("claim_states_requiring_evidence", [])}
    claim_states = {str(k): str(v) for k, v in dict(build_config.get("required_claim_states", {})).items()}
    live_requires_evidence = claim_states.get("live_open_game_suite") in required_states
    claim_evidence_paths = []
    for rel_path in dict(build_config.get("claim_taxonomy", {})).get("evidence_paths", []):
        status = _file_status(root, rel_path)
        claim_evidence_paths.append(status)
        if not status["exists"]:
            path_text = str(status["path"])
            if not live_requires_evidence and ("harness" in path_text or "live" in path_text):
                warnings.append({"severity": "warning", "code": "optional_live_suite_evidence_missing_until_g008", "path": status["path"]})
            else:
                findings.append({"severity": "error", "code": "missing_claim_evidence_path", "path": status["path"]})

    final_outputs = {
        "endpoint_statistics": _status_payload(root, build_config.get("endpoint_statistics_path", "artifacts/eval/final_endpoint_statistics.json")),
        "failure_analysis": _status_payload(root, build_config.get("failure_analysis_path", "artifacts/eval/final_failure_analysis.json")),
        "claim_taxonomy": _status_payload(root, build_config.get("claim_taxonomy_path", "artifacts/eval/final_claim_taxonomy.json")),
        "build_summary": _status_payload(root, args.build_summary_out),
        "readiness_audit": _status_payload(root, args.readiness_output),
        "completion_audit": _status_payload(root, args.g006_audit_output),
    }
    if args.require_existing_final_outputs:
        for key, status in final_outputs.items():
            if status["status"] != "pass":
                findings.append({"severity": "error", "code": "existing_final_output_not_pass", "artifact_key": key, "path": status["path"], "status": status["status"]})

    commands = {
        "build_final_artifacts": ["uv", "run", "python", "scripts/build_g006_final_eval_artifacts.py"],
        "finalize": ["uv", "run", "python", "scripts/finalize_g006_evaluation.py"],
        "watch_then_finalize": ["uv", "run", "python", "scripts/watch_g006_then_finalize.py", "--output", "artifacts/eval/g006_postrun_watcher_summary.json"],
        "validate_readiness": ["uv", "run", "python", "scripts/validate_g006_evaluation_readiness.py"],
        "validate_completion": ["uv", "run", "python", "scripts/validate_g006_completion.py"],
    }
    status = "ready" if not findings else "blocked"
    return {
        "schema": "g006_readiness_plan.v1",
        "status": status,
        "root": str(root),
        "allow_precheckpoint": bool(args.allow_precheckpoint),
        "goal_statuses": statuses,
        "prerequisite_goal_statuses": prereq_statuses,
        "required_splits": build_config.get("required_splits", []),
        "required_endpoints": build_config.get("required_endpoints", []),
        "comparison_sources": comparison_sources,
        "metadata_sources": metadata_sources,
        "claim_evidence_paths": claim_evidence_paths,
        "final_outputs": final_outputs,
        "commands": commands,
        "findings": findings,
        "warnings": warnings,
        "claim_boundary": "G006 readiness planning is read-only; it does not build final artifacts, checkpoint G006, or weaken G003/G004/G005 prerequisites.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan/check G006 evaluation/failure-analysis readiness without building artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--build-config", default="configs/eval/g006_final_artifacts.yaml")
    parser.add_argument("--build-summary-out", default="artifacts/eval/g006_final_artifact_build_summary.json")
    parser.add_argument("--readiness-config", default="configs/eval/g006_evaluation_readiness.yaml")
    parser.add_argument("--readiness-output", default="artifacts/eval/g006_evaluation_readiness_audit.json")
    parser.add_argument("--g006-completion-config", default="configs/eval/g006_completion.yaml")
    parser.add_argument("--g006-audit-output", default="artifacts/eval/g006_completion_audit.json")
    parser.add_argument("--require-existing-final-outputs", action="store_true")
    parser.add_argument("--allow-precheckpoint", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_readiness_plan(args)
    write_json(_path(Path(args.root).resolve(), args.output), payload)
    print(f"g006 readiness plan: status={payload['status']} findings={len(payload['findings'])} warnings={len(payload['warnings'])} output={args.output}")
    return 0 if payload["status"] == "ready" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
