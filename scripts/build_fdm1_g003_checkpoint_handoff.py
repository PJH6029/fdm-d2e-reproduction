#!/usr/bin/env python3
"""Build a guarded OMX checkpoint handoff for G003 after the pod run passes.

This script never mutates `.omx/ultragoal/ledger.jsonl`. It prepares a durable
handoff JSON only when monitor, completion audit, evidence bundle, and ultragoal
state all prove that the active G003 story is ready to checkpoint. A fresh Codex
`get_goal` snapshot may be supplied to validate the aggregate goal is still
active before emitting the exact checkpoint command.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import sha256_file, write_json

GOAL_ID = "G003-50ms-action-token-dataset-pipeline"
DEFAULT_MONITOR = "artifacts/cluster/fdm1_g003_action_dataset_pod_monitor.json"
DEFAULT_AUDIT = "artifacts/sources/fdm1_g003_action_dataset_completion_audit.json"
DEFAULT_BUNDLE = "artifacts/sources/fdm1_g003_evidence_bundle_manifest.json"
DEFAULT_GOALS = ".omx/ultragoal/goals.json"
DEFAULT_OUTPUT = "artifacts/cluster/fdm1_g003_checkpoint_handoff.json"
DEFAULT_CODEX_GOAL_HINT = ".omx/tmp/fdm1_g003_get_goal_snapshot.json"


def _path(root: str | Path, rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid_json: {exc}"
    if not isinstance(data, dict):
        return None, "invalid_json: top-level is not object"
    return data, None


def _record(path: Path, *, root: Path) -> dict[str, Any]:
    try:
        display = str(path.relative_to(root))
    except Exception:
        display = str(path)
    entry: dict[str, Any] = {"path": display, "exists": path.exists(), "bytes": 0, "sha256": None}
    if path.exists() and path.is_file():
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    return entry


def _codex_goal_status(snapshot: dict[str, Any]) -> tuple[str | None, str | None]:
    if "goal" in snapshot and isinstance(snapshot["goal"], dict):
        goal = snapshot["goal"]
        return goal.get("status"), goal.get("objective")
    if "raw" in snapshot and isinstance(snapshot["raw"], dict) and isinstance(snapshot["raw"].get("goal"), dict):
        goal = snapshot["raw"]["goal"]
        return goal.get("status"), goal.get("objective")
    return snapshot.get("status"), snapshot.get("objective")


def build_handoff(
    *,
    root: str | Path = ".",
    monitor_path: str | Path = DEFAULT_MONITOR,
    audit_path: str | Path = DEFAULT_AUDIT,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    goals_path: str | Path = DEFAULT_GOALS,
    codex_goal_json: str | Path | None = None,
    codex_goal_hint: str | Path = DEFAULT_CODEX_GOAL_HINT,
) -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, Any] | None] = {}
    paths = {
        "monitor": _path(root_path, monitor_path),
        "completion_audit": _path(root_path, audit_path),
        "evidence_bundle": _path(root_path, bundle_path),
        "goals": _path(root_path, goals_path),
    }
    for key, path in paths.items():
        data, err = _load_json(path)
        loaded[key] = data
        if err:
            findings.append({"severity": "error", "code": f"{key}_{err}", "path": str(path)})

    monitor = loaded["monitor"] or {}
    audit = loaded["completion_audit"] or {}
    bundle = loaded["evidence_bundle"] or {}
    goals = loaded["goals"] or {}

    if monitor.get("status") != "pass":
        findings.append({"severity": "error", "code": "monitor_not_pass", "actual": monitor.get("status")})
    if audit.get("status") != "pass":
        findings.append({"severity": "error", "code": "completion_audit_not_pass", "actual": audit.get("status")})
    if bundle.get("status") != "pass":
        findings.append({"severity": "error", "code": "evidence_bundle_not_pass", "actual": bundle.get("status")})

    goal_obj = None
    for item in goals.get("goals", []) if isinstance(goals.get("goals"), list) else []:
        if isinstance(item, dict) and item.get("id") == GOAL_ID:
            goal_obj = item
            break
    if goal_obj is None:
        findings.append({"severity": "error", "code": "g003_goal_missing_from_goals_json"})
    elif goal_obj.get("status") != "in_progress":
        findings.append({"severity": "error", "code": "g003_goal_not_in_progress", "actual": goal_obj.get("status")})

    aggregate_objective = goals.get("codexObjective") if isinstance(goals, dict) else None
    codex_goal_path = _path(root_path, codex_goal_json) if codex_goal_json else None
    if codex_goal_path:
        codex, err = _load_json(codex_goal_path)
        if err:
            findings.append({"severity": "error", "code": f"codex_goal_{err}", "path": str(codex_goal_path)})
        else:
            codex_status, codex_objective = _codex_goal_status(codex or {})
            if codex_status != "active":
                findings.append({"severity": "error", "code": "codex_goal_not_active", "actual": codex_status})
            if aggregate_objective and codex_objective != aggregate_objective:
                findings.append({"severity": "error", "code": "codex_goal_objective_mismatch", "actual": codex_objective, "expected": aggregate_objective})

    evidence = (
        "G003 FDM-1 action-token dataset pipeline passed: "
        f"monitor={Path(monitor_path)}, completion_audit={Path(audit_path)} status=pass, "
        f"evidence_bundle={Path(bundle_path)} status=pass; "
        "ROADMAP.md remains canonical and large JSONL slot packs are represented by write-time output hashes."
    )
    codex_goal_arg = str(codex_goal_json or codex_goal_hint)
    checkpoint_command = [
        "omx",
        "ultragoal",
        "checkpoint",
        "--goal-id",
        GOAL_ID,
        "--status",
        "complete",
        "--evidence",
        evidence,
        "--codex-goal-json",
        codex_goal_arg,
    ]
    errors = [f for f in findings if f.get("severity") == "error"]
    return {
        "schema": "fdm1_g003_checkpoint_handoff.v1",
        "status": "ready_to_checkpoint" if not errors else "blocked",
        "goal_id": GOAL_ID,
        "canonical_roadmap": "ROADMAP.md",
        "artifacts": {key: _record(path, root=root_path) for key, path in paths.items()},
        "codex_goal_json": str(codex_goal_json) if codex_goal_json else None,
        "codex_goal_hint": str(codex_goal_hint),
        "evidence": evidence,
        "checkpoint_command": checkpoint_command,
        "checkpoint_command_shell": " ".join(shlex.quote(part) for part in checkpoint_command),
        "findings": findings,
        "claim_boundary": "Handoff only; it does not mutate OMX or Codex goal state. Run the command only after saving a fresh get_goal snapshot.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the guarded G003 OMX checkpoint handoff after a passing pod run.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--monitor", default=DEFAULT_MONITOR)
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--goals", default=DEFAULT_GOALS)
    parser.add_argument("--codex-goal-json")
    parser.add_argument("--codex-goal-hint", default=DEFAULT_CODEX_GOAL_HINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args(argv)
    payload = build_handoff(
        root=args.root,
        monitor_path=args.monitor,
        audit_path=args.audit,
        bundle_path=args.bundle,
        goals_path=args.goals,
        codex_goal_json=args.codex_goal_json,
        codex_goal_hint=args.codex_goal_hint,
    )
    write_json(_path(args.root, args.output), payload)
    print(f"G003 checkpoint handoff: status={payload['status']} errors={len([f for f in payload['findings'] if f.get('severity') == 'error'])} output={args.output}")
    if payload["status"] == "ready_to_checkpoint":
        print(payload["checkpoint_command_shell"])
    return 0 if payload["status"] == "ready_to_checkpoint" or args.allow_blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
