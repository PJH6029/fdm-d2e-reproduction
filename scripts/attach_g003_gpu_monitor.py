#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.process_liveness import pid_running

QUERY_FIELDS = [
    "timestamp",
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
]


def _pid_running(pid: int) -> bool:
    return pid_running(pid)


def _read_pid(pid_file: Path) -> int | None:
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _query_gpu(nvidia_smi_bin: str) -> list[list[str]]:
    cmd = [
        nvidia_smi_bin,
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    rows: list[list[str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        rows.append([part.strip() for part in next(csv.reader([line]))])
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_monitor(args: argparse.Namespace) -> dict[str, Any]:
    pid_file = Path(args.pid_file)
    output = Path(args.output)
    metadata_out = Path(args.metadata_out)
    monitor_pid_file = Path(args.monitor_pid_file) if args.monitor_pid_file else None
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    if monitor_pid_file:
        monitor_pid_file.parent.mkdir(parents=True, exist_ok=True)
        existing_monitor_pid = _read_pid(monitor_pid_file)
        if existing_monitor_pid and existing_monitor_pid != os.getpid() and _pid_running(existing_monitor_pid) and not args.force:
            payload = {
                "schema": "g003_attached_gpu_monitor.v1",
                "pid_file": str(pid_file),
                "parent_pid": _read_pid(pid_file),
                "output": str(output),
                "started_at_unix": time.time(),
                "ended_at_unix": time.time(),
                "interval_seconds": float(args.interval_seconds),
                "samples": 0,
                "exit_reason": "existing_monitor_running",
                "monitor_pid": os.getpid(),
                "existing_monitor_pid": existing_monitor_pid,
                "errors": [],
                "claim_boundary": "Existing attached monitor detected; no second monitor was started and no G003 completion is implied.",
            }
            _write_json(metadata_out, payload)
            return payload
        monitor_pid_file.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    parent_pid = _read_pid(pid_file)
    started_at = time.time()
    samples = 0
    errors: list[str] = []
    exit_reason = "parent_pid_missing"
    write_header = not output.exists() or output.stat().st_size == 0 or args.truncate
    mode = "w" if args.truncate else "a"
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["sample_unix", "parent_pid", *QUERY_FIELDS])
            handle.flush()
        while True:
            parent_pid = _read_pid(pid_file)
            if parent_pid is None:
                exit_reason = "parent_pid_missing"
                break
            if not _pid_running(parent_pid):
                exit_reason = "parent_exited"
                break
            try:
                gpu_rows = _query_gpu(args.nvidia_smi_bin)
                now = time.time()
                for row in gpu_rows:
                    writer.writerow([f"{now:.6f}", parent_pid, *row])
                samples += 1
                handle.flush()
            except Exception as exc:  # pragma: no cover - exercised by pod/runtime failures.
                errors.append(repr(exc))
                if not args.keep_running_on_error:
                    exit_reason = "nvidia_smi_error"
                    break
            if args.max_samples is not None and samples >= int(args.max_samples):
                exit_reason = "max_samples"
                break
            time.sleep(float(args.interval_seconds))
    payload = {
        "schema": "g003_attached_gpu_monitor.v1",
        "pid_file": str(pid_file),
        "parent_pid": parent_pid,
        "output": str(output),
        "started_at_unix": started_at,
        "ended_at_unix": time.time(),
        "interval_seconds": float(args.interval_seconds),
        "samples": samples,
        "exit_reason": exit_reason,
        "monitor_pid": os.getpid(),
        "existing_monitor_pid": None,
        "errors": errors,
        "claim_boundary": "Attached monitor evidence only; G003 completion still requires full decode, training, metrics, split statistics, and completion audit pass.",
    }
    _write_json(metadata_out, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach a GPU monitor to the already-running G003 parent PID without restarting it.")
    parser.add_argument("--pid-file", default="outputs/cluster/g003_full_compact_parallel.pid")
    parser.add_argument("--output", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv")
    parser.add_argument("--metadata-out", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json")
    parser.add_argument("--monitor-pid-file", default="outputs/cluster/g003_attached_gpu_monitor.pid")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--nvidia-smi-bin", default="nvidia-smi")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--force", action="store_true", help="Start a new monitor even if monitor-pid-file points at a live process.")
    parser.add_argument("--keep-running-on-error", action="store_true")
    args = parser.parse_args()
    payload = run_monitor(args)
    print(f"g003 attached gpu monitor: exit_reason={payload['exit_reason']} samples={payload['samples']} output={payload['output']}")
    if payload["exit_reason"] == "existing_monitor_running":
        return 0
    return 0 if payload["samples"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
