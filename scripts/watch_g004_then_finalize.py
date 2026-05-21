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
from finalize_g004_d2e_full_fdm import finalize as finalize_g004


DEFAULT_OUTPUT = "artifacts/fdm/g004_postrun_watcher_summary.json"


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
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"schema": "invalid_json", "error": str(exc)}


def _file_status(path: Path, rel_path: str) -> dict[str, Any]:
    return {"path": rel_path, "exists": path.exists() and path.is_file(), "bytes": path.stat().st_size if path.exists() and path.is_file() else 0}


def _run_snapshot(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    run_summary_path = _path(root, args.run_summary)
    log_path = _path(root, args.log_path)
    gpu_monitor_path = _path(root, args.gpu_monitor)
    run_summary = _load_json(run_summary_path)
    return {
        "pid": _read_pid(_path(root, args.pid_file)),
        "pid_running": _pid_running(_read_pid(_path(root, args.pid_file))),
        "run_summary_exists": run_summary_path.exists() and run_summary_path.is_file(),
        "run_summary_status": run_summary.get("exit_code") if isinstance(run_summary, dict) else None,
        "run_summary": run_summary,
        "artifacts": {
            "run_summary": _file_status(run_summary_path, args.run_summary),
            "log": _file_status(log_path, args.log_path),
            "gpu_monitor": _file_status(gpu_monitor_path, args.gpu_monitor),
        },
    }


def _finalizer_args(args: argparse.Namespace, root: Path) -> Namespace:
    return Namespace(
        root=str(root),
        summary_out=args.g004_finalization_summary,
        allow_fail=True,
        skip_split_stats=args.skip_split_stats,
        force_split_stats=args.force_split_stats,
        split_stats_config=args.split_stats_config,
        split_stats_summary=args.split_stats_summary,
        g004_completion_config=args.g004_completion_config,
        g004_audit_output=args.g004_audit_output,
        run_summary=args.run_summary,
    )


def _base_payload(args: argparse.Namespace, root: Path, *, started_at: float) -> dict[str, Any]:
    return {
        "schema": "g004_postrun_watcher.v1",
        "root": str(root),
        "started_at_unix": started_at,
        "output": args.output,
        "pid_file": args.pid_file,
        "watcher_pid_file": args.watcher_pid_file,
        "poll_seconds": float(args.poll_seconds),
        "max_wait_seconds": float(args.max_wait_seconds),
        "claim_boundary": "Watches for the G004 parent to exit and then runs local finalization only; it never checkpoints G004 or mutates OMX/Codex goal state.",
    }


def _write_summary(root: Path, output: str | Path, payload: dict[str, Any]) -> None:
    write_json(_path(root, output), payload)


def watch(
    args: argparse.Namespace,
    *,
    finalize_func: Callable[[argparse.Namespace], dict[str, Any]] = finalize_g004,
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
            snapshot = _run_snapshot(args, root)
            if snapshot["pid_running"]:
                payload = {
                    **base,
                    "status": "waiting_active_parent",
                    "elapsed_seconds": elapsed,
                    "run": snapshot,
                    "findings": [],
                }
                _write_summary(root, args.output, payload)
                if args.once:
                    return payload
                if float(args.max_wait_seconds) >= 0 and elapsed >= float(args.max_wait_seconds):
                    payload["status"] = "timeout_waiting_active_parent"
                    payload["findings"] = [{"severity": "error", "code": "timeout_waiting_active_parent", "elapsed_seconds": elapsed}]
                    _write_summary(root, args.output, payload)
                    return payload
                sleep_func(float(args.poll_seconds))
                continue

            finalization = finalize_func(_finalizer_args(args, root))
            status = "finalized_pass" if finalization.get("status") == "pass" else "finalized_fail"
            payload = {
                **base,
                "status": status,
                "elapsed_seconds": elapsed,
                "run": snapshot,
                "g004_finalization_summary": args.g004_finalization_summary,
                "g004_finalization_status": finalization.get("status"),
                "g004_audit_status": finalization.get("g004_audit_status"),
                "g004_audit_error_count": finalization.get("g004_audit_error_count"),
                "findings": [] if status == "finalized_pass" else [{"severity": "error", "code": "g004_finalization_not_pass", "status": finalization.get("status")}],
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
    parser = argparse.ArgumentParser(description="Watch G004 until the parent exits, then run the non-mutating finalizer.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--once", action="store_true", help="Write one status sample and exit if the G004 parent is still active.")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-wait-seconds", type=float, default=-1.0, help="Negative means wait indefinitely.")
    parser.add_argument("--pid-file", default="outputs/cluster/g004_d2e_full_fdm_4xh200.pid")
    parser.add_argument("--watcher-pid-file", default="outputs/cluster/g004_postrun_watcher.pid")
    parser.add_argument("--replace-existing-watcher", action="store_true")
    parser.add_argument("--g004-finalization-summary", default="artifacts/fdm/g004_d2e_full_fdm_finalization_summary.json")
    parser.add_argument("--skip-split-stats", action="store_true")
    parser.add_argument("--force-split-stats", action="store_true")
    parser.add_argument("--split-stats-config", default="configs/eval/g004_split_statistics.yaml")
    parser.add_argument("--split-stats-summary", default="artifacts/eval/g004_split_statistical_comparisons_summary.json")
    parser.add_argument("--g004-completion-config", default="configs/eval/g004_full_fdm_completion.yaml")
    parser.add_argument("--g004-audit-output", default="artifacts/fdm/g004_full_fdm_completion_audit.json")
    parser.add_argument("--run-summary", default="artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json")
    parser.add_argument("--log-path", default="artifacts/fdm/g004_d2e_full_fdm_4xh200.log")
    parser.add_argument("--gpu-monitor", default="artifacts/fdm/g004_d2e_full_fdm_4xh200_gpu_monitor.csv")
    args = parser.parse_args()
    payload = watch(args)
    print(f"g004 postrun watcher: status={payload['status']} output={args.output}")
    terminal_ok = payload["status"] in {"waiting_active_parent", "duplicate_watcher_running", "finalized_pass"}
    return 0 if terminal_ok or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
