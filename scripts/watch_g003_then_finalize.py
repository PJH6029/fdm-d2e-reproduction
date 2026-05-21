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

from fdm_d2e.cluster.g003_monitor import build_g003_progress_report
from fdm_d2e.io_utils import write_json
from finalize_g003_integrated_run import finalize as finalize_g003


DEFAULT_OUTPUT = "artifacts/idm/g003_postrun_watcher_summary.json"


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
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


def _write_summary(root: Path, output: str | Path, payload: dict[str, Any]) -> None:
    write_json(_path(root, output), payload)


def _progress(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    return build_g003_progress_report(
        shard_root=_path(root, args.shard_root),
        log_dir=_path(root, args.log_dir),
        data_universe=_path(root, args.data_universe),
        output_dir=_path(root, args.data_output_dir),
        idm_output_dir=_path(root, args.idm_output_dir),
        pid_file=_path(root, args.pid_file),
        num_shards=int(args.num_shards),
        stale_seconds=float(args.stale_seconds),
    )


def _finalizer_args(args: argparse.Namespace, root: Path) -> Namespace:
    return Namespace(
        root=str(root),
        summary_out=args.g003_finalization_summary,
        allow_fail=True,
        allow_active_parent=False,
        skip_split_stats=args.skip_split_stats,
        force_split_stats=args.force_split_stats,
        split_stats_config=args.split_stats_config,
        split_stats_summary=args.split_stats_summary,
        g003_completion_config=args.g003_completion_config,
        g003_audit_output=args.g003_audit_output,
        integrated_run_evidence=args.integrated_run_evidence,
        idm_summary=args.idm_summary,
        checkpoint_metadata=args.checkpoint_metadata,
        metrics=args.metrics,
        gpu_monitor=args.gpu_monitor,
        attached_monitor_metadata=args.attached_monitor_metadata,
        train_run_summary=args.train_run_summary,
        nproc_per_node=args.nproc_per_node,
        expected_gpus=args.expected_gpus,
        shard_root=args.shard_root,
        log_dir=args.log_dir,
        data_universe=args.data_universe,
        data_output_dir=args.data_output_dir,
        idm_output_dir=args.idm_output_dir,
        pid_file=args.pid_file,
        num_shards=args.num_shards,
        stale_seconds=args.stale_seconds,
    )


def _base_payload(args: argparse.Namespace, root: Path, *, started_at: float) -> dict[str, Any]:
    return {
        "schema": "g003_postrun_watcher.v1",
        "root": str(root),
        "started_at_unix": started_at,
        "output": args.output,
        "pid_file": args.pid_file,
        "watcher_pid_file": args.watcher_pid_file,
        "poll_seconds": float(args.poll_seconds),
        "max_wait_seconds": float(args.max_wait_seconds),
        "claim_boundary": "Watches for the G003 parent to exit and then runs local finalization only; it never checkpoints G003 or mutates OMX/Codex goal state.",
    }


def watch(
    args: argparse.Namespace,
    *,
    finalize_func: Callable[[argparse.Namespace], dict[str, Any]] = finalize_g003,
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
            progress = _progress(args, root)
            elapsed = max(0.0, now - started)
            if progress.get("pid_running"):
                payload = {
                    **base,
                    "status": "waiting_active_parent",
                    "elapsed_seconds": elapsed,
                    "progress": {
                        "status": progress.get("status"),
                        "pid": progress.get("pid"),
                        "pid_running": progress.get("pid_running"),
                        "decoded_recording_variants": progress.get("decoded_recording_variants"),
                        "expected_recording_variants": progress.get("expected_recording_variants"),
                        "complete_shards": progress.get("complete_shards"),
                        "num_shards": progress.get("num_shards"),
                        "stale_shards": progress.get("stale_shards"),
                        "no_progress_shards": progress.get("no_progress_shards"),
                    },
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
                "progress": {
                    "status": progress.get("status"),
                    "pid": progress.get("pid"),
                    "pid_running": progress.get("pid_running"),
                    "decoded_recording_variants": progress.get("decoded_recording_variants"),
                    "expected_recording_variants": progress.get("expected_recording_variants"),
                    "complete_shards": progress.get("complete_shards"),
                    "num_shards": progress.get("num_shards"),
                    "stale_shards": progress.get("stale_shards"),
                    "no_progress_shards": progress.get("no_progress_shards"),
                },
                "g003_finalization_summary": args.g003_finalization_summary,
                "g003_finalization_status": finalization.get("status"),
                "g003_audit_status": finalization.get("g003_audit_status"),
                "g003_audit_error_count": finalization.get("g003_audit_error_count"),
                "findings": [] if status == "finalized_pass" else [{"severity": "error", "code": "g003_finalization_not_pass", "status": finalization.get("status")}],
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
    parser = argparse.ArgumentParser(description="Watch G003 until the parent exits, then run the non-mutating integrated finalizer.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--once", action="store_true", help="Write one status sample and exit if the G003 parent is still active.")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-wait-seconds", type=float, default=-1.0, help="Negative means wait indefinitely.")
    parser.add_argument("--watcher-pid-file", default="outputs/cluster/g003_postrun_watcher.pid")
    parser.add_argument("--replace-existing-watcher", action="store_true")
    parser.add_argument("--g003-finalization-summary", default="artifacts/idm/g003_integrated_finalization_summary.json")
    parser.add_argument("--skip-split-stats", action="store_true")
    parser.add_argument("--force-split-stats", action="store_true")
    parser.add_argument("--split-stats-config", default="configs/eval/g003_split_statistics.yaml")
    parser.add_argument("--split-stats-summary", default="artifacts/eval/g003_split_statistical_comparisons_summary.json")
    parser.add_argument("--g003-completion-config", default="configs/eval/g003_full_idm_completion.yaml")
    parser.add_argument("--g003-audit-output", default="artifacts/idm/g003_full_idm_completion_audit.json")
    parser.add_argument("--integrated-run-evidence", default="artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json")
    parser.add_argument("--idm-summary", default="artifacts/idm/idm_streaming_d2e_full_compact_summary.json")
    parser.add_argument("--checkpoint-metadata", default="outputs/idm_streaming_d2e_full_compact/checkpoint_metadata.json")
    parser.add_argument("--metrics", default="outputs/idm_streaming_d2e_full_compact/metrics.json")
    parser.add_argument("--gpu-monitor", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv")
    parser.add_argument("--attached-monitor-metadata", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json")
    parser.add_argument("--train-run-summary", default="artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--shard-root", default="outputs/data/d2e_full_corpus_shards")
    parser.add_argument("--log-dir", default="artifacts/sources")
    parser.add_argument("--data-universe", default="artifacts/sources/d2e_full_data_universe_manifest.json")
    parser.add_argument("--data-output-dir", default="outputs/data/d2e_full_corpus")
    parser.add_argument("--idm-output-dir", default="outputs/idm_streaming_d2e_full_compact")
    parser.add_argument("--pid-file", default="outputs/cluster/g003_full_compact_parallel.pid")
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--stale-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    payload = watch(args)
    print(f"g003 postrun watcher: status={payload['status']} output={args.output}")
    terminal_ok = payload["status"] in {"waiting_active_parent", "duplicate_watcher_running", "finalized_pass"}
    return 0 if terminal_ok or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
