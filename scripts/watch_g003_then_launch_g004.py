#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdm_d2e.cluster.g003_monitor import build_g003_progress_report
from fdm_d2e.io_utils import write_json
from fdm_d2e.process_liveness import pid_running
from finalize_g003_integrated_run import finalize as finalize_g003
from plan_g004_launch import plan_launch as plan_g004_launch


DEFAULT_OUTPUT = "artifacts/fdm/g003_to_g004_chain_summary.json"


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
    return pid_running(pid)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - invalid JSON is reported in payload
        return {"schema": "invalid_json", "error": str(exc)}


def _write_summary(root: Path, output: str | Path, payload: dict[str, Any]) -> None:
    write_json(_path(root, output), payload)


def _progress(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    return build_g003_progress_report(
        shard_root=_path(root, args.shard_root),
        log_dir=_path(root, args.log_dir),
        data_universe=_path(root, args.data_universe),
        output_dir=_path(root, args.data_output_dir),
        idm_output_dir=_path(root, args.idm_output_dir),
        pid_file=_path(root, args.g003_pid_file),
        num_shards=int(args.num_shards),
        stale_seconds=float(args.stale_seconds),
    )


def _progress_summary(progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": progress.get("status"),
        "pid": progress.get("pid"),
        "pid_running": progress.get("pid_running"),
        "decoded_recording_variants": progress.get("decoded_recording_variants"),
        "expected_recording_variants": progress.get("expected_recording_variants"),
        "complete_shards": progress.get("complete_shards"),
        "num_shards": progress.get("num_shards"),
        "stale_shards": progress.get("stale_shards"),
        "no_progress_shards": progress.get("no_progress_shards"),
    }


def _g003_finalizer_args(args: argparse.Namespace, root: Path) -> Namespace:
    return Namespace(
        root=str(root),
        summary_out=args.g003_finalization_summary,
        allow_fail=True,
        allow_active_parent=False,
        skip_split_stats=args.skip_split_stats,
        force_split_stats=args.force_split_stats,
        split_stats_config=args.g003_split_stats_config,
        split_stats_summary=args.g003_split_stats_summary,
        g003_completion_config=args.g003_completion_config,
        g003_audit_output=args.g003_audit_output,
        integrated_run_evidence=args.integrated_run_evidence,
        idm_summary=args.idm_summary,
        checkpoint_metadata=args.checkpoint_metadata,
        metrics=args.metrics,
        gpu_monitor=args.gpu_monitor,
        attached_monitor_metadata=args.attached_monitor_metadata,
        train_run_summary=args.train_run_summary,
        nproc_per_node=args.g003_nproc_per_node,
        expected_gpus=args.expected_gpus,
        shard_root=args.shard_root,
        log_dir=args.log_dir,
        data_universe=args.data_universe,
        data_output_dir=args.data_output_dir,
        idm_output_dir=args.idm_output_dir,
        pid_file=args.g003_pid_file,
        num_shards=args.num_shards,
        stale_seconds=args.stale_seconds,
    )


def _g004_plan_args(args: argparse.Namespace, root: Path) -> Namespace:
    return Namespace(
        root=str(root),
        output=args.g004_launch_readiness,
        goals_path=args.goals_path,
        g003_goal_id=args.g003_goal_id,
        g003_completion_config=args.g003_completion_config,
        g003_audit=args.g003_audit_output,
        skip_refresh_g003_audit=False,
        allow_precheckpoint=not args.require_g003_goal_checkpoint,
        fdm_config=args.fdm_config,
        idm_predict_config=args.idm_predict_config,
        fdm_labels=args.fdm_labels,
        g004_run_script=args.g004_run_script,
        g004_run_summary=args.g004_run_summary,
        nproc_per_node=args.g004_nproc_per_node,
        expected_gpus=args.expected_gpus,
        check_gpus=args.check_gpus,
        allow_fail=True,
    )


def _g003_finalization_from_existing_watcher(args: argparse.Namespace, root: Path) -> dict[str, Any] | None:
    summary = _load_json(_path(root, args.g003_postrun_summary))
    if not summary:
        return None
    status = summary.get("status")
    if status != "finalized_pass":
        return None
    return {
        "schema": "g003_finalization_reference.v1",
        "source": args.g003_postrun_summary,
        "status": "pass",
        "watcher_status": status,
        "g003_audit_status": summary.get("g003_audit_status"),
        "g003_audit_error_count": summary.get("g003_audit_error_count"),
        "payload": summary,
    }


def _g003_finalization_passed(payload: dict[str, Any]) -> bool:
    return payload.get("status") == "pass" and payload.get("g003_audit_status") == "pass"


def _base_payload(args: argparse.Namespace, root: Path, *, started_at: float) -> dict[str, Any]:
    return {
        "schema": "g003_to_g004_chain.v1",
        "root": str(root),
        "started_at_unix": started_at,
        "output": args.output,
        "watcher_pid_file": args.watcher_pid_file,
        "poll_seconds": float(args.poll_seconds),
        "max_wait_seconds": float(args.max_wait_seconds),
        "launch_enabled": bool(args.launch),
        "start_g004_watcher": bool(args.start_g004_watcher),
        "require_g003_goal_checkpoint": bool(args.require_g003_goal_checkpoint),
        "claim_boundary": "Fail-closed chain from G003 finalization to G004 D2E-only FDM launch; it never checkpoints G003/G004 and never launches G004 unless G003 finalization plus audit pass and G004 launch readiness is ready.",
    }


def _launch_g004(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    pid_path = _path(root, args.g004_pid_file)
    existing_pid = _read_pid(pid_path)
    if _pid_running(existing_pid):
        return {"status": "already_running", "pid": existing_pid, "pid_file": args.g004_pid_file}
    env = os.environ.copy()
    env.update(
        {
            "CONFIG": args.fdm_config,
            "IDM_PREDICT_CONFIG": args.idm_predict_config,
            "NPROC_PER_NODE": str(args.g004_nproc_per_node),
            "EXPECTED_GPUS": str(args.expected_gpus),
            "LOG_PATH": args.g004_log_path,
            "RUN_SUMMARY": args.g004_run_summary,
            "GPU_MONITOR_LOG": args.g004_gpu_monitor,
            "PID_FILE": args.g004_pid_file,
        }
    )
    Path(root / args.g004_pid_file).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["bash", args.g004_run_script], cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
    return {
        "status": "launched",
        "pid": proc.pid,
        "pid_file": args.g004_pid_file,
        "command": ["bash", args.g004_run_script],
        "env": {key: env[key] for key in ["CONFIG", "IDM_PREDICT_CONFIG", "NPROC_PER_NODE", "EXPECTED_GPUS", "LOG_PATH", "RUN_SUMMARY", "GPU_MONITOR_LOG", "PID_FILE"]},
    }


def _launch_g004_watcher(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    pid_path = _path(root, args.g004_watcher_pid_file)
    existing_pid = _read_pid(pid_path)
    if _pid_running(existing_pid):
        return {"status": "already_running", "pid": existing_pid, "pid_file": args.g004_watcher_pid_file}
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/watch_g004_then_finalize.py",
        "--poll-seconds",
        str(args.g004_watcher_poll_seconds),
        "--pid-file",
        args.g004_pid_file,
        "--watcher-pid-file",
        args.g004_watcher_pid_file,
    ]
    proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
    return {"status": "launched", "pid": proc.pid, "pid_file": args.g004_watcher_pid_file, "command": cmd}


def watch_and_maybe_launch(
    args: argparse.Namespace,
    *,
    finalize_func: Callable[[argparse.Namespace], dict[str, Any]] = finalize_g003,
    plan_func: Callable[[argparse.Namespace], dict[str, Any]] = plan_g004_launch,
    launch_func: Callable[[argparse.Namespace, Path], dict[str, Any]] = _launch_g004,
    launch_watcher_func: Callable[[argparse.Namespace, Path], dict[str, Any]] = _launch_g004_watcher,
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
            _write_summary(root, args.output, payload)
            return payload
        watcher_pid_path.parent.mkdir(parents=True, exist_ok=True)
        watcher_pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    try:
        while True:
            now = time_func()
            elapsed = max(0.0, now - started)
            progress = _progress(args, root)
            if progress.get("pid_running"):
                payload = {
                    **base,
                    "status": "waiting_g003_parent",
                    "elapsed_seconds": elapsed,
                    "g003_progress": _progress_summary(progress),
                    "findings": [],
                }
                _write_summary(root, args.output, payload)
                if args.once:
                    return payload
                if float(args.max_wait_seconds) >= 0 and elapsed >= float(args.max_wait_seconds):
                    payload["status"] = "timeout_waiting_g003_parent"
                    payload["findings"] = [{"severity": "error", "code": "timeout_waiting_g003_parent", "elapsed_seconds": elapsed}]
                    _write_summary(root, args.output, payload)
                    return payload
                sleep_func(float(args.poll_seconds))
                continue

            finalization = _g003_finalization_from_existing_watcher(args, root)
            source = "existing_g003_postrun_watcher"
            if finalization is None:
                source = "local_finalize_g003"
                finalization = finalize_func(_g003_finalizer_args(args, root))
            if not _g003_finalization_passed(finalization):
                payload = {
                    **base,
                    "status": "g003_finalization_not_pass",
                    "elapsed_seconds": elapsed,
                    "g003_progress": _progress_summary(progress),
                    "g003_finalization_source": source,
                    "g003_finalization": finalization,
                    "findings": [{"severity": "error", "code": "g003_finalization_not_pass", "source": source, "status": finalization.get("status"), "g003_audit_status": finalization.get("g003_audit_status")}],
                }
                _write_summary(root, args.output, payload)
                return payload

            plan = plan_func(_g004_plan_args(args, root))
            if plan.get("status") != "ready":
                payload = {
                    **base,
                    "status": "g004_launch_not_ready",
                    "elapsed_seconds": elapsed,
                    "g003_progress": _progress_summary(progress),
                    "g003_finalization_source": source,
                    "g003_finalization_status": finalization.get("status"),
                    "g003_audit_status": finalization.get("g003_audit_status"),
                    "g004_launch_plan": plan,
                    "findings": [{"severity": "error", "code": "g004_launch_not_ready", "error_count": plan.get("error_count")}],
                }
                _write_summary(root, args.output, payload)
                return payload

            if not args.launch:
                payload = {
                    **base,
                    "status": "g004_launch_ready",
                    "elapsed_seconds": elapsed,
                    "g003_progress": _progress_summary(progress),
                    "g003_finalization_source": source,
                    "g003_finalization_status": finalization.get("status"),
                    "g003_audit_status": finalization.get("g003_audit_status"),
                    "g004_launch_plan": plan,
                    "findings": [],
                }
                _write_summary(root, args.output, payload)
                return payload

            launch = launch_func(args, root)
            watcher = launch_watcher_func(args, root) if args.start_g004_watcher else None
            launch_ok = launch.get("status") in {"launched", "already_running"}
            payload = {
                **base,
                "status": "g004_launched" if launch_ok else "g004_launch_failed",
                "elapsed_seconds": elapsed,
                "g003_progress": _progress_summary(progress),
                "g003_finalization_source": source,
                "g003_finalization_status": finalization.get("status"),
                "g003_audit_status": finalization.get("g003_audit_status"),
                "g004_launch_plan": plan,
                "g004_launch": launch,
                "g004_watcher": watcher,
                "findings": [] if launch_ok else [{"severity": "error", "code": "g004_launch_failed", "launch_status": launch.get("status")}],
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
    parser = argparse.ArgumentParser(description="Fail-closed watcher that launches G004 only after G003 finalization/audit pass.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--once", action="store_true", help="Write one waiting sample and exit if G003 is still active.")
    parser.add_argument("--launch", action="store_true", help="Actually launch G004 when all gates pass; otherwise stop at launch_ready.")
    parser.add_argument("--start-g004-watcher", action="store_true", help="Start watch_g004_then_finalize.py after launching G004.")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-wait-seconds", type=float, default=-1.0, help="Negative means wait indefinitely.")
    parser.add_argument("--watcher-pid-file", default="outputs/cluster/g003_to_g004_chain_watcher.pid")
    parser.add_argument("--replace-existing-watcher", action="store_true")
    parser.add_argument("--goals-path", default=".omx/ultragoal/goals.json")
    parser.add_argument("--g003-goal-id", default="G003-d2e-only-idm")
    parser.add_argument("--require-g003-goal-checkpoint", action="store_true", help="Also require G003 OMX checkpoint before planning G004; audit pass is always required.")
    parser.add_argument("--g003-postrun-summary", default="artifacts/idm/g003_postrun_watcher_summary.json")
    parser.add_argument("--g003-finalization-summary", default="artifacts/idm/g003_integrated_finalization_summary.json")
    parser.add_argument("--skip-split-stats", action="store_true")
    parser.add_argument("--force-split-stats", action="store_true")
    parser.add_argument("--g003-split-stats-config", default="configs/eval/g003_split_statistics.yaml")
    parser.add_argument("--g003-split-stats-summary", default="artifacts/eval/g003_split_statistical_comparisons_summary.json")
    parser.add_argument("--g003-completion-config", default="configs/eval/g003_full_idm_completion.yaml")
    parser.add_argument("--g003-audit-output", default="artifacts/idm/g003_full_idm_completion_audit.json")
    parser.add_argument("--integrated-run-evidence", default="artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json")
    parser.add_argument("--idm-summary", default="artifacts/idm/idm_streaming_d2e_full_compact_summary.json")
    parser.add_argument("--checkpoint-metadata", default="outputs/idm_streaming_d2e_full_compact/checkpoint_metadata.json")
    parser.add_argument("--metrics", default="outputs/idm_streaming_d2e_full_compact/metrics.json")
    parser.add_argument("--gpu-monitor", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv")
    parser.add_argument("--attached-monitor-metadata", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json")
    parser.add_argument("--train-run-summary", default="artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json")
    parser.add_argument("--g003-nproc-per-node", type=int, default=4)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--shard-root", default="outputs/data/d2e_full_corpus_shards")
    parser.add_argument("--log-dir", default="artifacts/sources")
    parser.add_argument("--data-universe", default="artifacts/sources/d2e_full_data_universe_manifest.json")
    parser.add_argument("--data-output-dir", default="outputs/data/d2e_full_corpus")
    parser.add_argument("--idm-output-dir", default="outputs/idm_streaming_d2e_full_compact")
    parser.add_argument("--g003-pid-file", default="outputs/cluster/g003_full_compact_parallel.pid")
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--stale-seconds", type=float, default=3600.0)
    parser.add_argument("--g004-launch-readiness", default="artifacts/fdm/g004_launch_readiness.json")
    parser.add_argument("--fdm-config", default="configs/model/fdm_streaming_d2e_full_compact.yaml")
    parser.add_argument("--idm-predict-config", default="configs/model/idm_streaming_d2e_full_compact_predict_fdm_train.yaml")
    parser.add_argument("--fdm-labels", default="outputs/idm_streaming_d2e_full_compact/fdm_train_core_pseudolabels/pseudolabels.jsonl")
    parser.add_argument("--g004-run-script", default="scripts/run_g004_d2e_full_fdm_4xh200.sh")
    parser.add_argument("--g004-run-summary", default="artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json")
    parser.add_argument("--g004-pid-file", default="outputs/cluster/g004_d2e_full_fdm_4xh200.pid")
    parser.add_argument("--g004-log-path", default="artifacts/fdm/g004_d2e_full_fdm_4xh200.log")
    parser.add_argument("--g004-gpu-monitor", default="artifacts/fdm/g004_d2e_full_fdm_4xh200_gpu_monitor.csv")
    parser.add_argument("--g004-nproc-per-node", type=int, default=4)
    parser.add_argument("--check-gpus", action="store_true")
    parser.add_argument("--g004-watcher-pid-file", default="outputs/cluster/g004_postrun_watcher.pid")
    parser.add_argument("--g004-watcher-poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    payload = watch_and_maybe_launch(args)
    print(f"g003->g004 chain: status={payload['status']} output={args.output}")
    terminal_ok = payload["status"] in {"waiting_g003_parent", "duplicate_chain_watcher_running", "g004_launch_ready", "g004_launched"}
    return 0 if terminal_ok or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
