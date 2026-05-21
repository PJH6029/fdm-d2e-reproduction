#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json
from fdm_d2e.rollout.live_suite import run_live_suite_validation


DEFAULT_OUTPUT = "artifacts/harness/g008_readiness_plan.json"


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
        return {"path": rel_path, "exists": False, "bytes": 0}
    p = _path(root, rel_path)
    return {"path": str(rel_path), "exists": p.exists() and p.is_file(), "bytes": p.stat().st_size if p.exists() and p.is_file() else 0}


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


def _executable_status(command_template: str | None) -> dict[str, Any]:
    if not command_template:
        return {"command": command_template, "executable": None, "found": False, "resolved": None}
    try:
        parts = shlex.split(command_template)
    except ValueError as exc:
        return {"command": command_template, "executable": None, "found": False, "resolved": None, "error": str(exc)}
    executable = parts[0] if parts else None
    if not executable:
        return {"command": command_template, "executable": None, "found": False, "resolved": None}
    resolved = shutil.which(executable)
    absolute_exists = Path(executable).exists() if "/" in executable else False
    return {"command": command_template, "executable": executable, "found": bool(resolved or absolute_exists), "resolved": resolved or (executable if absolute_exists else None)}


def _game_system_checks(config: dict[str, Any], *, skip_system_checks: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for game in config.get("games", []) or []:
        if not isinstance(game, dict) or not bool(game.get("enabled", True)):
            continue
        status = _executable_status(game.get("launch_command_template"))
        checks.append(
            {
                "game_id": game.get("id"),
                "name": game.get("name"),
                "skip_system_checks": bool(skip_system_checks),
                "launch": status,
                "window_title_pattern": game.get("window_title_pattern"),
                "tasks": [task.get("id") for task in game.get("tasks", []) if isinstance(task, dict)],
            }
        )
    return checks


def _control_backend_checks(config: dict[str, Any], *, skip_system_checks: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for backend in config.get("allowed_control_backends", []) or []:
        backend_text = str(backend)
        if backend_text in {"native_os_input", "os_input", "pyautogui_os_input"}:
            checks.append({"backend": backend_text, "skip_system_checks": bool(skip_system_checks), "requires_binary": False, "available": True})
            continue
        resolved = shutil.which(backend_text)
        checks.append({"backend": backend_text, "skip_system_checks": bool(skip_system_checks), "requires_binary": True, "available": bool(resolved), "resolved": resolved})
    return checks


def build_readiness_plan(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    suite_config = load_config(_path(root, args.suite_config))
    completion_config = load_config(_path(root, args.g008_completion_config))
    protocol = run_live_suite_validation(suite_config, None, root=root)
    findings: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if protocol.get("quality_gate", {}).get("status") != "protocol_ready":
        findings.append({"severity": "error", "code": "protocol_not_ready", "status": protocol.get("quality_gate", {}).get("status")})

    goals_path = str(completion_config.get("goals_path", ".omx/ultragoal/goals.json"))
    statuses = _goal_statuses(root, goals_path)
    prereq_statuses: dict[str, str] = {}
    for goal_id in completion_config.get("prerequisite_goals", []):
        actual = statuses.get(str(goal_id), "missing")
        prereq_statuses[str(goal_id)] = actual
        if actual != "complete":
            item = {"severity": "error", "code": "prerequisite_goal_not_complete", "goal_id": str(goal_id), "actual": actual}
            if args.allow_precheckpoint:
                warnings.append({**item, "severity": "warning"})
            else:
                findings.append(item)

    paths = {str(key): str(value) for key, value in dict(completion_config.get("paths", {})).items()}
    required_before_run = {
        "suite_config": paths.get("suite_config", args.suite_config),
        "trained_checkpoint_metadata": paths.get("trained_checkpoint_metadata"),
        "runtime_adapter_contract": paths.get("runtime_adapter_contract"),
        "live_suite_doc": paths.get("live_suite_doc"),
    }
    artifacts = {key: _file_status(root, value) for key, value in required_before_run.items()}
    for key, status in artifacts.items():
        if not status["exists"]:
            findings.append({"severity": "error", "code": "missing_required_pre_run_artifact", "artifact_key": key, "path": status["path"]})

    evidence_validation = _file_status(root, paths.get("evidence_validation"))
    if evidence_validation["exists"] and not args.allow_overwrite_evidence:
        findings.append({"severity": "error", "code": "evidence_validation_already_exists", "path": evidence_validation["path"]})
    elif not evidence_validation["exists"]:
        warnings.append({"severity": "warning", "code": "evidence_validation_not_collected_yet", "path": evidence_validation["path"]})

    checkpoint_metadata = _load_json(_path(root, paths.get("trained_checkpoint_metadata")))
    for dotted, expected in dict(completion_config.get("checkpoint_expectations", {})).items():
        actual = _json_path(checkpoint_metadata, dotted)
        if checkpoint_metadata is not None and actual != expected:
            findings.append({"severity": "error", "code": "checkpoint_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})

    game_checks = _game_system_checks(suite_config, skip_system_checks=args.skip_system_checks)
    backend_checks = _control_backend_checks(suite_config, skip_system_checks=args.skip_system_checks)
    if not args.skip_system_checks:
        for check in game_checks:
            if check["launch"]["found"] is not True:
                findings.append({"severity": "error", "code": "game_launch_executable_missing", "game_id": check["game_id"], "executable": check["launch"].get("executable")})
        if not any(check.get("available") for check in backend_checks):
            findings.append({"severity": "error", "code": "no_allowed_control_backend_available", "allowed_control_backends": [check.get("backend") for check in backend_checks]})
    else:
        warnings.append({"severity": "warning", "code": "system_checks_skipped"})

    commands = {
        "protocol": ["uv", "run", "python", "scripts/validate_live_game_suite.py", "--config", args.suite_config],
        "finalize_with_evidence": ["uv", "run", "python", "scripts/finalize_g008_live_suite.py", "--evidence", "artifacts/harness/<run>/live_suite_evidence.json"],
        "validate_completion": ["uv", "run", "python", "scripts/validate_g008_live_suite_completion.py"],
    }
    status = "ready" if not findings else "blocked"
    return {
        "schema": "g008_readiness_plan.v1",
        "status": status,
        "root": str(root),
        "allow_precheckpoint": bool(args.allow_precheckpoint),
        "skip_system_checks": bool(args.skip_system_checks),
        "goal_statuses": statuses,
        "prerequisite_goal_statuses": prereq_statuses,
        "protocol_status": protocol.get("quality_gate", {}).get("status"),
        "planned_games": protocol.get("quality_gate", {}).get("planned_games"),
        "planned_tasks": protocol.get("quality_gate", {}).get("planned_tasks"),
        "planned_seeded_episodes": protocol.get("quality_gate", {}).get("planned_seeded_episodes"),
        "required_artifacts_before_run": artifacts,
        "evidence_validation_artifact": evidence_validation,
        "game_system_checks": game_checks,
        "control_backend_checks": backend_checks,
        "commands": commands,
        "findings": findings,
        "warnings": warnings,
        "claim_boundary": "G008 readiness planning is read-only; it does not launch games, collect live evidence, checkpoint G008, or permit commercial-game/live-control claims.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan/check G008 live open-source graphical game-suite readiness without launching games.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--suite-config", default="configs/harness/g008_live_open_game_suite.yaml")
    parser.add_argument("--g008-completion-config", default="configs/eval/g008_live_suite_completion.yaml")
    parser.add_argument("--allow-precheckpoint", action="store_true")
    parser.add_argument("--skip-system-checks", action="store_true")
    parser.add_argument("--allow-overwrite-evidence", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_readiness_plan(args)
    write_json(_path(Path(args.root).resolve(), args.output), payload)
    print(f"g008 readiness plan: status={payload['status']} findings={len(payload['findings'])} warnings={len(payload['warnings'])} output={args.output}")
    return 0 if payload["status"] == "ready" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
