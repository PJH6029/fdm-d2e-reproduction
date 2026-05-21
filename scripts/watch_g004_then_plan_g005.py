#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdm_d2e.io_utils import write_json
from finalize_g004_d2e_full_fdm import finalize as finalize_g004
from plan_g005_launch import build_launch_readiness as plan_g005_launch


DEFAULT_OUTPUT = "artifacts/aux/g004_to_g005_readiness_chain_summary.json"


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
    return True


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - malformed handoff is surfaced
        return {"schema": "invalid_json", "error": str(exc)}


def _file_status(path: Path, rel_path: str) -> dict[str, Any]:
    return {"path": rel_path, "exists": path.exists() and path.is_file(), "bytes": path.stat().st_size if path.exists() and path.is_file() else 0}


def _run_snapshot(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    pid = _read_pid(_path(root, args.g004_pid_file))
    run_summary_path = _path(root, args.g004_run_summary)
    log_path = _path(root, args.g004_log_path)
    gpu_monitor_path = _path(root, args.g004_gpu_monitor)
    run_summary = _load_json(run_summary_path)
    return {
        "pid": pid,
        "pid_running": _pid_running(pid),
        "run_summary_exists": run_summary_path.exists() and run_summary_path.is_file(),
        "run_summary_status": run_summary.get("exit_code") if isinstance(run_summary, dict) else None,
        "run_summary": run_summary,
        "artifacts": {
            "run_summary": _file_status(run_summary_path, args.g004_run_summary),
            "log": _file_status(log_path, args.g004_log_path),
            "gpu_monitor": _file_status(gpu_monitor_path, args.g004_gpu_monitor),
        },
    }


def _g004_finalizer_args(args: argparse.Namespace, root: Path) -> Namespace:
    return Namespace(
        root=str(root),
        summary_out=args.g004_finalization_summary,
        allow_fail=True,
        skip_split_stats=args.skip_split_stats,
        force_split_stats=args.force_split_stats,
        split_stats_config=args.g004_split_stats_config,
        split_stats_summary=args.g004_split_stats_summary,
        g004_completion_config=args.g004_completion_config,
        g004_audit_output=args.g004_audit_output,
        run_summary=args.g004_run_summary,
    )


def _g005_plan_args(args: argparse.Namespace, root: Path) -> Namespace:
    return Namespace(
        root=str(root),
        g005_completion_config=args.g005_completion_config,
        g003_audit=args.g003_audit,
        g004_audit=args.g004_audit_output,
        pid_file=args.g005_pid_file,
        source_evidence=list(args.source_evidence),
        eval_manifest_hashes=args.eval_manifest_hashes,
        require_eval_manifest_hashes=args.require_eval_manifest_hashes,
        require_namespace_ready=args.require_namespace_ready,
        allow_precheckpoint=False,
        allow_overwrite=args.allow_overwrite_g005_run_summary,
        output=args.g005_launch_readiness,
        allow_fail=True,
    )


def _g004_finalization_from_existing_watcher(args: argparse.Namespace, root: Path) -> dict[str, Any] | None:
    summary = _load_json(_path(root, args.g004_postrun_summary))
    if not summary:
        return None
    status = summary.get("status")
    if status not in {"finalized_pass", "finalized_fail"}:
        return None
    return {
        "schema": "g004_finalization_reference.v1",
        "source": args.g004_postrun_summary,
        "status": "pass" if status == "finalized_pass" else "fail",
        "watcher_status": status,
        "g004_audit_status": summary.get("g004_audit_status"),
        "g004_audit_error_count": summary.get("g004_audit_error_count"),
        "payload": summary,
    }


def _g004_finalization_passed(payload: dict[str, Any]) -> bool:
    return payload.get("status") == "pass" and payload.get("g004_audit_status") == "pass"


def _base_payload(args: argparse.Namespace, root: Path, *, started_at: float) -> dict[str, Any]:
    return {
        "schema": "g004_to_g005_readiness_chain.v1",
        "root": str(root),
        "started_at_unix": started_at,
        "output": args.output,
        "watcher_pid_file": args.watcher_pid_file,
        "poll_seconds": float(args.poll_seconds),
        "max_wait_seconds": float(args.max_wait_seconds),
        "claim_boundary": "Fail-closed chain from G004 finalization to G005 readiness planning; it never launches G005, checkpoints goals, or weakens G003/G004 D2E-only prerequisites.",
    }


def watch_and_plan(
    args: argparse.Namespace,
    *,
    finalize_func: Callable[[argparse.Namespace], dict[str, Any]] = finalize_g004,
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
                "status": "duplicate_chain_watcher_running",
                "existing_pid": existing_pid,
                "findings": [{"severity": "warning", "code": "duplicate_chain_watcher_running", "pid": existing_pid}],
            }
            write_json(_path(root, args.output), payload)
            return payload
        watcher_pid_path.parent.mkdir(parents=True, exist_ok=True)
        watcher_pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    try:
        while True:
            now = time_func()
            elapsed = max(0.0, now - started)
            snapshot = _run_snapshot(args, root)
            if snapshot["pid_running"]:
                payload = {**base, "status": "waiting_g004_parent", "elapsed_seconds": elapsed, "g004_run": snapshot, "findings": []}
                write_json(_path(root, args.output), payload)
                if args.once:
                    return payload
                if float(args.max_wait_seconds) >= 0 and elapsed >= float(args.max_wait_seconds):
                    payload["status"] = "timeout_waiting_g004_parent"
                    payload["findings"] = [{"severity": "error", "code": "timeout_waiting_g004_parent", "elapsed_seconds": elapsed}]
                    write_json(_path(root, args.output), payload)
                    return payload
                sleep_func(float(args.poll_seconds))
                continue

            finalization = _g004_finalization_from_existing_watcher(args, root)
            source = "existing_g004_postrun_watcher"
            if finalization is None:
                source = "local_finalize_g004"
                finalization = finalize_func(_g004_finalizer_args(args, root))
            if not _g004_finalization_passed(finalization):
                payload = {
                    **base,
                    "status": "g004_finalization_not_pass",
                    "elapsed_seconds": elapsed,
                    "g004_run": snapshot,
                    "g004_finalization_source": source,
                    "g004_finalization": finalization,
                    "findings": [
                        {
                            "severity": "error",
                            "code": "g004_finalization_not_pass",
                            "source": source,
                            "status": finalization.get("status"),
                            "g004_audit_status": finalization.get("g004_audit_status"),
                        }
                    ],
                }
                write_json(_path(root, args.output), payload)
                return payload

            plan = plan_func(_g005_plan_args(args, root))
            if plan.get("status") != "ready":
                payload = {
                    **base,
                    "status": "g005_launch_not_ready",
                    "elapsed_seconds": elapsed,
                    "g004_run": snapshot,
                    "g004_finalization_source": source,
                    "g004_finalization_status": finalization.get("status"),
                    "g004_audit_status": finalization.get("g004_audit_status"),
                    "g005_launch_plan": plan,
                    "findings": [{"severity": "error", "code": "g005_launch_not_ready", "finding_count": len(plan.get("findings", []))}],
                }
                write_json(_path(root, args.output), payload)
                return payload

            payload = {
                **base,
                "status": "g005_launch_ready",
                "elapsed_seconds": elapsed,
                "g004_run": snapshot,
                "g004_finalization_source": source,
                "g004_finalization_status": finalization.get("status"),
                "g004_audit_status": finalization.get("g004_audit_status"),
                "g005_launch_plan": plan,
                "findings": [],
            }
            write_json(_path(root, args.output), payload)
            return payload
    finally:
        if watcher_pid_path is not None and _read_pid(watcher_pid_path) == os.getpid():
            try:
                watcher_pid_path.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch G004, then run fail-closed G005 launch readiness planning after G004 audit pass.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--once", action="store_true", help="Write one waiting sample and exit if G004 is still active.")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-wait-seconds", type=float, default=-1.0, help="Negative means wait indefinitely.")
    parser.add_argument("--watcher-pid-file", default="outputs/cluster/g004_to_g005_readiness_chain.pid")
    parser.add_argument("--replace-existing-watcher", action="store_true")
    parser.add_argument("--g004-pid-file", default="outputs/cluster/g004_d2e_full_fdm_4xh200.pid")
    parser.add_argument("--g004-postrun-summary", default="artifacts/fdm/g004_postrun_watcher_summary.json")
    parser.add_argument("--g004-finalization-summary", default="artifacts/fdm/g004_d2e_full_fdm_finalization_summary.json")
    parser.add_argument("--skip-split-stats", action="store_true")
    parser.add_argument("--force-split-stats", action="store_true")
    parser.add_argument("--g004-split-stats-config", default="configs/eval/g004_split_statistics.yaml")
    parser.add_argument("--g004-split-stats-summary", default="artifacts/eval/g004_split_statistical_comparisons_summary.json")
    parser.add_argument("--g004-completion-config", default="configs/eval/g004_full_fdm_completion.yaml")
    parser.add_argument("--g004-audit-output", default="artifacts/fdm/g004_full_fdm_completion_audit.json")
    parser.add_argument("--g004-run-summary", default="artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json")
    parser.add_argument("--g004-log-path", default="artifacts/fdm/g004_d2e_full_fdm_4xh200.log")
    parser.add_argument("--g004-gpu-monitor", default="artifacts/fdm/g004_d2e_full_fdm_4xh200_gpu_monitor.csv")
    parser.add_argument("--g005-completion-config", default="configs/eval/g005_aux_completion.yaml")
    parser.add_argument("--g005-launch-readiness", default="artifacts/aux/g005_launch_readiness.json")
    parser.add_argument("--g003-audit", default="artifacts/idm/g003_full_idm_completion_audit.json")
    parser.add_argument("--g005-pid-file", default="outputs/cluster/g005_d2e_aux_best.pid")
    parser.add_argument("--source-evidence", action="append", default=[])
    parser.add_argument("--eval-manifest-hashes")
    parser.add_argument("--require-eval-manifest-hashes", action="store_true")
    parser.add_argument("--require-namespace-ready", action="store_true")
    parser.add_argument("--allow-overwrite-g005-run-summary", action="store_true")
    args = parser.parse_args()
    payload = watch_and_plan(args)
    print(f"g004->g005 readiness chain: status={payload['status']} output={args.output}")
    terminal_ok = payload["status"] in {"waiting_g004_parent", "duplicate_chain_watcher_running", "g005_launch_ready"}
    return 0 if terminal_ok or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
