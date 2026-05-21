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


def validate_g007_completion(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goal_id = str(config.get("goal_id", "G007-runtime-sdk-adapter"))
    statuses = _goal_statuses(root_path, goals_path)
    goal_status = statuses.get(goal_id, "missing")
    if goal_status != "complete":
        findings.append({"severity": "error", "code": "goal_not_checkpointed_complete", "goal_id": goal_id, "actual": goal_status})

    paths = {key: str(value) for key, value in dict(config.get("paths", {})).items()}
    artifacts = {key: _file_status(root_path / rel_path, rel_path) for key, rel_path in paths.items()}
    for key, evidence in artifacts.items():
        if not evidence["exists"]:
            findings.append({"severity": "error", "code": "missing_required_artifact", "artifact_key": key, "path": evidence["path"]})

    contract = _load_json(root_path / paths.get("contract_evidence", "")) if paths.get("contract_evidence") else None
    fixture_config = _load_json(root_path / paths.get("contract_config", "")) if paths.get("contract_config") else None
    demo_config = _load_json(root_path / paths.get("demo_config", "")) if paths.get("demo_config") else None

    _assert_expectations(contract, dict(config.get("contract_expectations", {})), source="contract_evidence", findings=findings)
    _assert_expectations(fixture_config, dict(config.get("fixture_config_expectations", {})), source="contract_config", findings=findings)
    _assert_expectations(demo_config, dict(config.get("demo_config_expectations", {})), source="demo_config", findings=findings)

    if contract is not None:
        if str(contract.get("mode")) not in set(config.get("allowed_contract_modes", ["deterministic_replay_dry_run"])):
            findings.append({"severity": "error", "code": "unexpected_runtime_contract_mode", "actual": contract.get("mode")})
        if int(contract.get("blocked_actions", 0)) != int(config.get("expected_blocked_actions", 0)):
            findings.append({"severity": "error", "code": "blocked_action_count_mismatch", "expected": int(config.get("expected_blocked_actions", 0)), "actual": contract.get("blocked_actions")})
        if int(contract.get("applied_actions", 0)) < int(config.get("min_applied_actions", 1)):
            findings.append({"severity": "error", "code": "too_few_applied_actions", "expected_min": int(config.get("min_applied_actions", 1)), "actual": contract.get("applied_actions")})
        if _get(contract, "safety.require_focus") is not True:
            findings.append({"severity": "error", "code": "runtime_focus_guard_not_enabled"})
        if _get(contract, "safety.kill_switch_path") in (None, ""):
            findings.append({"severity": "error", "code": "runtime_kill_switch_missing"})
        if _get(contract, "latency.schema") != "runtime_latency_summary.v1":
            findings.append({"severity": "error", "code": "runtime_latency_schema_missing", "actual": _get(contract, "latency.schema")})
        targets = contract.get("adapter_targets") or []
        boundaries = {str(target.get("claim_boundary")) for target in targets if isinstance(target, dict)}
        if "deterministic_sdk_contract_only" not in boundaries:
            findings.append({"severity": "error", "code": "runtime_contract_boundary_missing", "actual": sorted(boundaries)})
        notes = str(contract.get("notes", "")).lower()
        for phrase in config.get("required_contract_note_phrases", []):
            if str(phrase).lower() not in notes:
                findings.append({"severity": "error", "code": "runtime_contract_note_missing_phrase", "phrase": str(phrase)})

    if demo_config is not None:
        targets = demo_config.get("adapter_targets") or []
        if len(targets) < int(config.get("min_demo_targets", 3)):
            findings.append({"severity": "error", "code": "too_few_demo_targets", "expected_min": int(config.get("min_demo_targets", 3)), "actual": len(targets)})
        for idx, target in enumerate(targets):
            if target.get("claim_boundary") != "open_source_offline_target_candidate":
                findings.append({"severity": "error", "code": "demo_target_boundary_mismatch", "index": idx, "actual": target.get("claim_boundary")})
            if target.get("license_probe_required") is not True:
                findings.append({"severity": "error", "code": "demo_target_license_probe_not_required", "index": idx})

    doc_path = root_path / paths.get("runtime_doc", "") if paths.get("runtime_doc") else None
    if doc_path and doc_path.exists():
        text = doc_path.read_text(encoding="utf-8").lower()
        for phrase in config.get("required_doc_phrases", []):
            if str(phrase).lower() not in text:
                findings.append({"severity": "error", "code": "runtime_doc_missing_phrase", "phrase": str(phrase)})

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g007_completion_audit.v1",
        "status": "pass" if not errors else "fail",
        "goal_id": goal_id,
        "goal_status": goal_status,
        "artifacts": artifacts,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "This audit is required to preserve G007 as a reusable safe adapter contract only; it does not prove G008 live game control or any commercial-game claim.",
    }


def write_g007_completion_audit(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_g007_completion(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload
