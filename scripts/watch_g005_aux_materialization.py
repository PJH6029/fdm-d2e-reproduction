#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdm_d2e.io_utils import write_json
from build_g005_aux_examples import build_examples as build_aux_examples
from build_g005_aux_namespace_manifest import build_manifest as build_namespace_manifest
from build_g005_aux_source_evidence import build_evidence as build_source_evidence
from plan_g005_launch import build_launch_readiness as plan_g005_launch
from validate_g005_aux_materialization_integrity import build_integrity as build_materialization_integrity
from validate_g005_aux_runtime_env import validate_runtime_env


DEFAULT_OUTPUT = "artifacts/aux/g005_aux_materialization_watcher_summary.json"


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _pid_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not _pid_is_zombie(pid)


def _pid_is_zombie(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="ignore")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return False
    try:
        tail = stat.rsplit(")", 1)[1].strip().split()
    except IndexError:
        return False
    return bool(tail and tail[0] == "Z")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"schema": "invalid_json", "error": str(exc)}
    return payload if isinstance(payload, dict) else {"schema": "unexpected_json", "payload_type": type(payload).__name__}


def _file_status(path: Path, rel_path: str | Path) -> dict[str, Any]:
    return {"path": str(rel_path), "exists": path.exists() and path.is_file(), "bytes": path.stat().st_size if path.exists() and path.is_file() else 0}


def _iter_tree_files(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    files: list[Path] = []
    for item in path.rglob("*"):
        try:
            if item.is_file():
                files.append(item)
        except OSError:
            continue
    return files


def _safe_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except (FileNotFoundError, OSError):
        return None


def _tree_status(root: Path, rel_path: str | Path) -> dict[str, Any]:
    path = _path(root, rel_path)
    files = _iter_tree_files(path)
    sizes = [_safe_file_size(item) for item in files]
    existing_sizes = [int(size) for size in sizes if size is not None]
    return {
        "path": str(rel_path),
        "exists": path.exists() and path.is_dir(),
        "file_count": len(existing_sizes),
        "bytes": sum(existing_sizes),
        "transient_missing_file_count": len(sizes) - len(existing_sizes),
    }


def _materialization_snapshot(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    pid = _read_pid(_path(root, args.pid_file))
    summary_path = _path(root, args.materialization_summary)
    log_path = _path(root, args.materialization_log)
    return {
        "pid": pid,
        "pid_running": _pid_running(pid),
        "materialization_summary": _load_json(summary_path),
        "artifacts": {
            "materialization_summary": _file_status(summary_path, args.materialization_summary),
            "materialization_log": _file_status(log_path, args.materialization_log),
            "namespace_root": _tree_status(root, args.namespace_root),
        },
    }


def _source_evidence_args(args: argparse.Namespace, root: Path) -> Namespace:
    return Namespace(
        root=str(root),
        aux_candidates=args.aux_candidates,
        namespace_root=args.namespace_root,
        source_id=None,
        required_splits=list(args.required_splits),
        max_files=args.max_files,
        output=args.source_evidence_output,
        allow_fail=True,
    )


def _integrity_args(args: argparse.Namespace, root: Path) -> Namespace:
    return Namespace(
        root=str(root),
        aux_candidates=args.aux_candidates,
        namespace_root=args.namespace_root,
        materialization_summary=args.materialization_summary,
        source_id=None,
        required_splits=list(args.required_splits),
        output=args.integrity_output,
        allow_fail=True,
    )


def _plan_args(args: argparse.Namespace, root: Path) -> Namespace:
    return Namespace(
        root=str(root),
        g005_completion_config=args.g005_completion_config,
        g003_audit=args.g003_audit,
        g004_audit=args.g004_audit,
        pid_file=args.g005_pid_file,
        source_evidence=[args.source_evidence_output],
        eval_manifest_hashes=args.eval_manifest_hashes,
        require_eval_manifest_hashes=True,
        require_namespace_ready=True,
        allow_precheckpoint=False,
        allow_overwrite=args.allow_overwrite_g005_run_summary,
        output=args.g005_launch_readiness_output,
        allow_fail=True,
    )


def _aux_examples_args(args: argparse.Namespace, root: Path) -> Namespace:
    return Namespace(
        root=str(root),
        action_registry=args.action_registry,
        namespace_root=args.namespace_root,
        examples_root=args.examples_root,
        source_id=None,
        required_splits=list(args.required_splits),
        max_examples_per_source=args.max_examples_per_source,
        allow_incomplete_raw=False,
        output=args.aux_examples_output,
        allow_fail=True,
    )


def _runtime_env_args(args: argparse.Namespace, root: Path) -> Namespace:
    return Namespace(
        root=str(root),
        action_registry=args.action_registry,
        output=args.runtime_env_output,
        allow_fail=True,
    )


def _base_payload(args: argparse.Namespace, root: Path, *, started_at: float) -> dict[str, Any]:
    return {
        "schema": "g005_aux_materialization_watcher.v1",
        "root": str(root),
        "started_at_unix": started_at,
        "output": args.output,
        "pid_file": args.pid_file,
        "watcher_pid_file": args.watcher_pid_file,
        "poll_seconds": float(args.poll_seconds),
        "max_wait_seconds": float(args.max_wait_seconds),
        "claim_boundary": "Watches selected auxiliary source materialization and builds provenance/readiness artifacts only; it never launches G005 training, checkpoints goals, or weakens G003/G004 D2E-only prerequisites.",
    }


def _write_summary(root: Path, output: str | Path, payload: dict[str, Any]) -> None:
    write_json(_path(root, output), payload)


def _call_namespace_builder(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    payload = build_namespace_manifest(
        aux_candidates_path=str(_path(root, args.aux_candidates)),
        source_evidence_paths=[str(_path(root, args.source_evidence_output))],
        eval_manifest_hashes_path=str(_path(root, args.eval_manifest_hashes)),
        completion_ready_requested=True,
        allow_template=False,
    )
    write_json(_path(root, args.namespace_manifest_output), payload)
    return payload


def watch(
    args: argparse.Namespace,
    *,
    integrity_func: Callable[[argparse.Namespace], dict[str, Any]] = build_materialization_integrity,
    source_evidence_func: Callable[[argparse.Namespace], dict[str, Any]] = build_source_evidence,
    aux_examples_func: Callable[[argparse.Namespace], dict[str, Any]] = build_aux_examples,
    runtime_env_func: Callable[[argparse.Namespace], dict[str, Any]] = validate_runtime_env,
    namespace_func: Callable[[argparse.Namespace, Path], dict[str, Any]] = _call_namespace_builder,
    plan_func: Callable[[argparse.Namespace], dict[str, Any]] = plan_g005_launch,
    sleep_func: Callable[[float], None] = time.sleep,
    time_func: Callable[[], float] = time.time,
) -> dict[str, Any]:
    root = Path(args.root).resolve()
    started = time_func()
    base = _base_payload(args, root, started_at=started)
    watcher_pid_path = _path(root, args.watcher_pid_file) if args.watcher_pid_file else None
    if watcher_pid_path is not None:
        existing_pid = _read_pid(watcher_pid_path)
        if existing_pid and existing_pid != os.getpid() and _pid_running(existing_pid) and not args.replace_existing_watcher:
            payload = {
                **base,
                "status": "duplicate_watcher_running",
                "existing_pid": existing_pid,
                "findings": [{"severity": "warning", "code": "duplicate_watcher_running", "pid": existing_pid}],
            }
            _write_summary(root, args.output, payload)
            return payload
        watcher_pid_path.parent.mkdir(parents=True, exist_ok=True)
        watcher_pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    try:
        while True:
            now = time_func()
            elapsed = max(0.0, now - started)
            snapshot = _materialization_snapshot(args, root)
            if snapshot["pid_running"]:
                payload = {
                    **base,
                    "status": "waiting_active_materialization",
                    "elapsed_seconds": elapsed,
                    "materialization": snapshot,
                    "findings": [],
                }
                _write_summary(root, args.output, payload)
                if args.once:
                    return payload
                if float(args.max_wait_seconds) >= 0 and elapsed >= float(args.max_wait_seconds):
                    payload["status"] = "timeout_waiting_active_materialization"
                    payload["findings"] = [{"severity": "error", "code": "timeout_waiting_active_materialization", "elapsed_seconds": elapsed}]
                    _write_summary(root, args.output, payload)
                    return payload
                sleep_func(float(args.poll_seconds))
                continue

            materialization_summary = snapshot.get("materialization_summary")
            if not isinstance(materialization_summary, dict) or materialization_summary.get("status") != "pass":
                payload = {
                    **base,
                    "status": "materialization_not_pass",
                    "elapsed_seconds": elapsed,
                    "materialization": snapshot,
                    "findings": [
                        {
                            "severity": "error",
                            "code": "materialization_not_pass",
                            "summary_status": materialization_summary.get("status") if isinstance(materialization_summary, dict) else None,
                        }
                    ],
                }
                _write_summary(root, args.output, payload)
                return payload

            integrity = integrity_func(_integrity_args(args, root))
            write_json(_path(root, args.integrity_output), integrity)
            if integrity.get("status") != "pass":
                payload = {
                    **base,
                    "status": "materialization_integrity_not_pass",
                    "elapsed_seconds": elapsed,
                    "materialization": snapshot,
                    "materialization_integrity_status": integrity.get("status"),
                    "materialization_integrity_error_count": integrity.get("error_count"),
                    "findings": [{"severity": "error", "code": "materialization_integrity_not_pass", "error_count": integrity.get("error_count")}],
                }
                _write_summary(root, args.output, payload)
                return payload

            source_evidence = source_evidence_func(_source_evidence_args(args, root))
            write_json(_path(root, args.source_evidence_output), source_evidence)
            if source_evidence.get("status") != "pass":
                payload = {
                    **base,
                    "status": "source_evidence_not_pass",
                    "elapsed_seconds": elapsed,
                    "materialization": snapshot,
                    "materialization_integrity_status": integrity.get("status"),
                    "source_evidence_status": source_evidence.get("status"),
                    "source_evidence_error_count": source_evidence.get("error_count"),
                    "findings": [{"severity": "error", "code": "source_evidence_not_pass", "error_count": source_evidence.get("error_count")}],
                }
                _write_summary(root, args.output, payload)
                return payload

            aux_examples = aux_examples_func(_aux_examples_args(args, root))
            write_json(_path(root, args.aux_examples_output), aux_examples)
            if aux_examples.get("status") != "pass":
                payload = {
                    **base,
                    "status": "aux_examples_not_pass",
                    "elapsed_seconds": elapsed,
                    "materialization": snapshot,
                    "materialization_integrity_status": integrity.get("status"),
                    "source_evidence_status": source_evidence.get("status"),
                    "aux_examples_status": aux_examples.get("status"),
                    "aux_examples_error_count": aux_examples.get("error_count"),
                    "findings": [{"severity": "error", "code": "aux_examples_not_pass", "error_count": aux_examples.get("error_count")}],
                }
                _write_summary(root, args.output, payload)
                return payload

            runtime_env = runtime_env_func(_runtime_env_args(args, root))
            write_json(_path(root, args.runtime_env_output), runtime_env)
            if runtime_env.get("status") != "pass":
                payload = {
                    **base,
                    "status": "runtime_env_not_pass",
                    "elapsed_seconds": elapsed,
                    "materialization": snapshot,
                    "materialization_integrity_status": integrity.get("status"),
                    "source_evidence_status": source_evidence.get("status"),
                    "aux_examples_status": aux_examples.get("status"),
                    "runtime_env_status": runtime_env.get("status"),
                    "runtime_env_error_count": runtime_env.get("error_count"),
                    "findings": [{"severity": "error", "code": "runtime_env_not_pass", "error_count": runtime_env.get("error_count")}],
                }
                _write_summary(root, args.output, payload)
                return payload

            namespace_manifest = namespace_func(args, root)
            if namespace_manifest.get("completion_ready") is not True:
                payload = {
                    **base,
                    "status": "namespace_not_ready",
                    "elapsed_seconds": elapsed,
                    "materialization": snapshot,
                    "materialization_integrity_status": integrity.get("status"),
                    "source_evidence_status": source_evidence.get("status"),
                    "aux_examples_status": aux_examples.get("status"),
                    "runtime_env_status": runtime_env.get("status"),
                    "namespace_completion_ready": namespace_manifest.get("completion_ready"),
                    "findings": [{"severity": "error", "code": "namespace_not_ready", "completion_ready": namespace_manifest.get("completion_ready")}],
                }
                _write_summary(root, args.output, payload)
                return payload

            plan = plan_func(_plan_args(args, root))
            write_json(_path(root, args.g005_launch_readiness_output), plan)
            if plan.get("status") != "ready":
                payload = {
                    **base,
                    "status": "g005_launch_not_ready",
                    "elapsed_seconds": elapsed,
                    "materialization": snapshot,
                    "materialization_integrity_status": integrity.get("status"),
                    "source_evidence_status": source_evidence.get("status"),
                    "aux_examples_status": aux_examples.get("status"),
                    "runtime_env_status": runtime_env.get("status"),
                    "namespace_completion_ready": namespace_manifest.get("completion_ready"),
                    "g005_launch_plan_status": plan.get("status"),
                    "g005_launch_plan_finding_count": len(plan.get("findings", [])),
                    "findings": [{"severity": "error", "code": "g005_launch_not_ready", "finding_count": len(plan.get("findings", []))}],
                }
                _write_summary(root, args.output, payload)
                return payload

            payload = {
                **base,
                "status": "g005_launch_ready",
                "elapsed_seconds": elapsed,
                "materialization": snapshot,
                "materialization_integrity_status": integrity.get("status"),
                "source_evidence_status": source_evidence.get("status"),
                "aux_examples_status": aux_examples.get("status"),
                "runtime_env_status": runtime_env.get("status"),
                "namespace_completion_ready": namespace_manifest.get("completion_ready"),
                "g005_launch_plan_status": plan.get("status"),
                "findings": [],
            }
            _write_summary(root, args.output, payload)
            return payload
    finally:
        if watcher_pid_path is not None and _read_pid(watcher_pid_path) == os.getpid():
            try:
                watcher_pid_path.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch selected G005 aux source materialization, then build fail-closed source/namespace/readiness evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--once", action="store_true", help="Write one waiting sample and exit if materialization is still active.")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-wait-seconds", type=float, default=-1.0, help="Negative means wait indefinitely.")
    parser.add_argument("--pid-file", default="outputs/cluster/g005_aux_materialization.pid")
    parser.add_argument("--watcher-pid-file", default="outputs/cluster/g005_aux_materialization_watcher.pid")
    parser.add_argument("--replace-existing-watcher", action="store_true")
    parser.add_argument("--materialization-summary", default="artifacts/aux/g005_aux_materialization_execute_summary.json")
    parser.add_argument("--materialization-log", default="artifacts/aux/g005_aux_materialization_execute.log")
    parser.add_argument("--namespace-root", default="outputs/aux")
    parser.add_argument("--examples-root", default="outputs/aux_examples")
    parser.add_argument("--required-splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-examples-per-source", type=int)
    parser.add_argument("--aux-candidates", default="artifacts/sources/aux_game_action_dataset_candidates.json")
    parser.add_argument("--action-registry", default="artifacts/aux/g005_aux_action_registry.json")
    parser.add_argument("--source-evidence-output", default="artifacts/aux/g005_aux_source_materialization_evidence.json")
    parser.add_argument("--integrity-output", default="artifacts/aux/g005_aux_materialization_integrity.json")
    parser.add_argument("--aux-examples-output", default="artifacts/aux/g005_aux_examples_summary.json")
    parser.add_argument("--runtime-env-output", default="artifacts/aux/g005_aux_runtime_env.json")
    parser.add_argument("--eval-manifest-hashes", default="artifacts/aux/d2e_eval_manifest_hashes.json")
    parser.add_argument("--namespace-manifest-output", default="artifacts/aux/g005_aux_namespace_manifest.json")
    parser.add_argument("--g005-launch-readiness-output", default="artifacts/aux/g005_launch_readiness.json")
    parser.add_argument("--g005-completion-config", default="configs/eval/g005_aux_completion.yaml")
    parser.add_argument("--g003-audit", default="artifacts/idm/g003_full_idm_completion_audit.json")
    parser.add_argument("--g004-audit", default="artifacts/fdm/g004_full_fdm_completion_audit.json")
    parser.add_argument("--g005-pid-file", default="outputs/cluster/g005_d2e_aux_best.pid")
    parser.add_argument("--allow-overwrite-g005-run-summary", action="store_true")
    args = parser.parse_args()
    payload = watch(args)
    print(f"g005 aux materialization watcher: status={payload['status']} output={args.output}")
    terminal_ok = payload["status"] in {"waiting_active_materialization", "duplicate_watcher_running", "g005_launch_ready"}
    return 0 if terminal_ok or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
