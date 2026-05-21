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


DEFAULT_OUTPUT = "artifacts/aux/g005_launch_readiness.json"


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


def _file_status(root: Path, rel_path: str | Path) -> dict[str, Any]:
    p = _path(root, rel_path)
    return {
        "path": str(rel_path),
        "exists": p.exists() and p.is_file(),
        "bytes": p.stat().st_size if p.exists() and p.is_file() else 0,
    }


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


def _selected_candidates(aux_candidates: dict[str, Any] | None) -> list[str]:
    if not aux_candidates:
        return []
    return [
        str(row["id"])
        for row in aux_candidates.get("candidates", []) or []
        if isinstance(row, dict) and row.get("id") and row.get("selection_status") == "selected_candidate"
    ]


def _audit_status(root: Path, rel_path: str | Path) -> dict[str, Any]:
    payload = _load_json(_path(root, rel_path))
    return {
        **_file_status(root, rel_path),
        "status": payload.get("status") if isinstance(payload, dict) else None,
        "error_count": payload.get("error_count") if isinstance(payload, dict) else None,
    }


def build_launch_readiness(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    config = load_config(_path(root, args.g005_completion_config))
    findings: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    statuses = _goal_statuses(root, goals_path)
    prereq_statuses: dict[str, str] = {}
    for prereq in config.get("prerequisite_goals", []):
        actual = statuses.get(str(prereq), "missing")
        prereq_statuses[str(prereq)] = actual
        if actual != "complete":
            item = {"severity": "error", "code": "prerequisite_goal_not_complete", "goal_id": str(prereq), "actual": actual}
            if args.allow_precheckpoint:
                warnings.append({**item, "severity": "warning"})
            else:
                findings.append(item)

    d2e_only_audits = {
        "g003": _audit_status(root, args.g003_audit),
        "g004": _audit_status(root, args.g004_audit),
    }
    for name, audit in d2e_only_audits.items():
        if audit["status"] != "pass":
            item = {"severity": "error", "code": "d2e_only_audit_not_pass", "audit": name, "path": audit["path"], "status": audit["status"], "error_count": audit["error_count"]}
            if args.allow_precheckpoint:
                warnings.append({**item, "severity": "warning"})
            else:
                findings.append(item)

    paths = {key: str(value) for key, value in dict(config.get("paths", {})).items()}
    required_inputs = {
        "aux_candidates": paths.get("aux_candidates", "artifacts/sources/aux_game_action_dataset_candidates.json"),
        "aux_plan_doc": paths.get("aux_plan_doc", "docs/auxiliary_data_plan.md"),
    }
    input_artifacts = {key: _file_status(root, value) for key, value in required_inputs.items()}
    for key, status in input_artifacts.items():
        if not status["exists"]:
            findings.append({"severity": "error", "code": "missing_required_input", "artifact_key": key, "path": status["path"]})

    aux_candidates = _load_json(_path(root, required_inputs["aux_candidates"]))
    selected = _selected_candidates(aux_candidates)
    if not selected:
        findings.append({"severity": "error", "code": "no_selected_aux_candidates", "path": required_inputs["aux_candidates"]})
    for dotted, expected in dict(config.get("aux_candidate_expectations", {})).items():
        actual = _json_path(aux_candidates, dotted)
        if actual != expected:
            findings.append({"severity": "error", "code": "aux_candidate_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})

    source_evidence = [_file_status(root, item) for item in args.source_evidence]
    for status in source_evidence:
        if not status["exists"]:
            findings.append({"severity": "error", "code": "missing_source_evidence", "path": status["path"]})
    eval_hashes = _file_status(root, args.eval_manifest_hashes) if args.eval_manifest_hashes else None
    if args.require_eval_manifest_hashes and (eval_hashes is None or not eval_hashes["exists"]):
        findings.append({"severity": "error", "code": "missing_eval_manifest_hashes", "path": args.eval_manifest_hashes})

    namespace_path = paths.get("namespace_manifest", "artifacts/aux/g005_aux_namespace_manifest.json")
    namespace = _load_json(_path(root, namespace_path))
    namespace_status = {
        **_file_status(root, namespace_path),
        "completion_ready": namespace.get("completion_ready") if isinstance(namespace, dict) else None,
    }
    if args.require_namespace_ready and namespace_status["completion_ready"] is not True:
        findings.append({"severity": "error", "code": "namespace_manifest_not_completion_ready", "path": namespace_path, "completion_ready": namespace_status["completion_ready"]})
    elif not namespace_status["exists"]:
        warnings.append({"severity": "warning", "code": "namespace_manifest_not_built_yet", "path": namespace_path})

    aux_examples_path = paths.get("aux_examples_summary")
    aux_examples_status = None
    if aux_examples_path:
        aux_examples = _load_json(_path(root, aux_examples_path))
        aux_examples_status = {
            **_file_status(root, aux_examples_path),
            "status": aux_examples.get("status") if isinstance(aux_examples, dict) else None,
            "error_count": aux_examples.get("error_count") if isinstance(aux_examples, dict) else None,
            "total_examples": aux_examples.get("total_examples") if isinstance(aux_examples, dict) else None,
        }
        if not aux_examples_status["exists"]:
            findings.append({"severity": "error", "code": "missing_aux_examples_summary", "path": aux_examples_path})
        elif aux_examples_status["status"] != "pass":
            findings.append(
                {
                    "severity": "error",
                    "code": "aux_examples_not_pass",
                    "path": aux_examples_path,
                    "status": aux_examples_status["status"],
                    "error_count": aux_examples_status["error_count"],
                }
            )

    run_summary_path = paths.get("run_summary", "artifacts/aux/g005_d2e_aux_train_run.json")
    run_summary_status = _file_status(root, run_summary_path)
    if run_summary_status["exists"] and not args.allow_overwrite:
        findings.append({"severity": "error", "code": "run_summary_already_exists", "path": run_summary_path})

    namespace_args: list[str] = []
    for source in args.source_evidence:
        namespace_args += ["--source-evidence", source]
    if args.eval_manifest_hashes:
        namespace_args += ["--eval-manifest-hashes", args.eval_manifest_hashes]
    namespace_command = ["uv", "run", "python", "scripts/build_g005_aux_namespace_manifest.py", *namespace_args, "--completion-ready"]
    finalizer_command = ["uv", "run", "python", "scripts/finalize_g005_aux_best_model.py", *namespace_args, "--completion-ready"]
    watcher_command = [
        "uv",
        "run",
        "python",
        "scripts/watch_g005_then_finalize.py",
        "--pid-file",
        args.pid_file,
        "--output",
        "artifacts/aux/g005_postrun_watcher_summary.json",
        *namespace_args,
        "--completion-ready",
    ]
    status = "ready" if not findings else "blocked"
    payload = {
        "schema": "g005_launch_readiness.v1",
        "status": status,
        "root": str(root),
        "allow_precheckpoint": bool(args.allow_precheckpoint),
        "goal_statuses": statuses,
        "prerequisite_goal_statuses": prereq_statuses,
        "d2e_only_audits": d2e_only_audits,
        "selected_aux_candidate_ids": selected,
        "input_artifacts": input_artifacts,
        "source_evidence": source_evidence,
        "eval_manifest_hashes": eval_hashes,
        "namespace_manifest": namespace_status,
        "aux_examples": aux_examples_status,
        "run_summary": run_summary_status,
        "commands": {
            "build_namespace_manifest": namespace_command,
            "postrun_watcher": watcher_command,
            "finalize_after_run": finalizer_command,
        },
        "findings": findings,
        "warnings": warnings,
        "claim_boundary": "G005 launch readiness is a preflight only; it does not train, checkpoint G005, or weaken G003/G004 D2E-only prerequisites.",
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan/check G005 D2E+aux best-model launch readiness without launching training.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--g005-completion-config", default="configs/eval/g005_aux_completion.yaml")
    parser.add_argument("--g003-audit", default="artifacts/idm/g003_full_idm_completion_audit.json")
    parser.add_argument("--g004-audit", default="artifacts/fdm/g004_full_fdm_completion_audit.json")
    parser.add_argument("--pid-file", default="outputs/cluster/g005_d2e_aux_best.pid")
    parser.add_argument("--source-evidence", action="append", default=[])
    parser.add_argument("--eval-manifest-hashes")
    parser.add_argument("--require-eval-manifest-hashes", action="store_true")
    parser.add_argument("--require-namespace-ready", action="store_true")
    parser.add_argument("--allow-precheckpoint", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_launch_readiness(args)
    write_json(_path(Path(args.root).resolve(), args.output), payload)
    print(f"g005 launch readiness: status={payload['status']} findings={len(payload['findings'])} warnings={len(payload['warnings'])} output={args.output}")
    return 0 if payload["status"] == "ready" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
