#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _load_env_file(path: Path) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    loaded: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ.setdefault(key, value)
        loaded.append(key)
    return loaded


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _floatish(value: str) -> float | None:
    cleaned = value.strip().replace("%", "").replace("MiB", "").replace("W", "").strip()
    if not cleaned or cleaned in {"[Not Supported]", "N/A"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _latest_gpu_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for raw_row in csv.reader(handle):
            row = [cell.strip() for cell in raw_row]
            if row:
                rows.append(row)
    if not rows:
        return []
    data_rows = [row for row in rows if not any(cell.lower() == "timestamp" for cell in row)]
    if not data_rows:
        return []
    latest_reversed: list[list[str]] = []
    seen_indices: set[str] = set()
    for row in reversed(data_rows):
        if len(row) < 2:
            continue
        index = row[1]
        if index in seen_indices:
            break
        seen_indices.add(index)
        latest_reversed.append(row)
    latest = list(reversed(latest_reversed))
    parsed: list[dict[str, Any]] = []
    for row in latest:
        if len(row) < 8:
            continue
        parsed.append(
            {
                "timestamp": row[0],
                "index": row[1],
                "name": row[2],
                "utilization_gpu_pct": _floatish(row[3]),
                "utilization_memory_pct": _floatish(row[4]),
                "memory_used_mib": _floatish(row[5]),
                "memory_total_mib": _floatish(row[6]),
                "power_draw_w": _floatish(row[7]),
            }
        )
    return parsed


def _rank_progress(progress_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not progress_dir.exists() or not progress_dir.is_dir():
        return rows
    now = time.time()
    for path in sorted(progress_dir.glob("train_rank*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        row = dict(payload)
        row["path"] = str(path)
        updated = row.get("updated_at_epoch")
        try:
            row["heartbeat_age_sec"] = now - float(updated)
        except (TypeError, ValueError):
            row["heartbeat_age_sec"] = None
        rows.append(row)
    return rows


def _process_running(pattern: str) -> bool | None:
    if not pattern:
        return None
    try:
        result = subprocess.run(["pgrep", "-af", pattern], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return None
    return result.returncode == 0


def _history_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not payload:
        return []
    history = payload.get("history")
    return [row for row in history if isinstance(row, dict)] if isinstance(history, list) else []


def _wandb_config(args: argparse.Namespace, env_keys: list[str]) -> dict[str, Any]:
    return {
        "schema": "wandb_training_sidecar_config.v1",
        "train_history": str(args.train_history),
        "rank_progress_dir": str(args.rank_progress_dir),
        "gpu_monitor": str(args.gpu_monitor),
        "run_summary": str(args.run_summary),
        "checkpoint": str(args.checkpoint),
        "metadata": str(args.metadata),
        "poll_seconds": args.poll_seconds,
        "finish_on_run_summary": args.finish_on_run_summary,
        "env_keys_loaded": sorted(set(env_keys)),
        "claim_boundary": "Sidecar logs existing training artifacts and monitor files; it does not train, evaluate, or mutate checkpoints.",
    }


def _log_epoch_rows(run: Any, rows: list[dict[str, Any]], seen_epochs: set[int]) -> int:
    logged = 0
    for row in rows:
        epoch_raw = row.get("epoch")
        try:
            epoch = int(epoch_raw)
        except (TypeError, ValueError):
            continue
        if epoch in seen_epochs:
            continue
        metrics = {
            "train/epoch": epoch,
            "train/loss": row.get("loss"),
            "train/loss_sum": row.get("loss_sum"),
            "train/examples": row.get("examples"),
            "train/batches": row.get("batches"),
        }
        run.log(metrics, step=epoch)
        seen_epochs.add(epoch)
        logged += 1
    return logged


def _log_progress(run: Any, progress_rows: list[dict[str, Any]], gpu_rows: list[dict[str, Any]], *, loop_index: int) -> None:
    payload: dict[str, Any] = {"sidecar/loop": loop_index, "sidecar/time": time.time()}
    for row in progress_rows:
        rank = row.get("rank")
        if rank is None:
            continue
        prefix = f"rank/{rank}"
        payload[f"{prefix}/batches"] = row.get("batches")
        payload[f"{prefix}/examples"] = row.get("examples")
        payload[f"{prefix}/loss"] = row.get("loss")
        payload[f"{prefix}/heartbeat_age_sec"] = row.get("heartbeat_age_sec")
    for row in gpu_rows:
        index = row.get("index")
        if index is None:
            continue
        prefix = f"gpu/{index}"
        payload[f"{prefix}/utilization_gpu_pct"] = row.get("utilization_gpu_pct")
        payload[f"{prefix}/utilization_memory_pct"] = row.get("utilization_memory_pct")
        payload[f"{prefix}/memory_used_mib"] = row.get("memory_used_mib")
        payload[f"{prefix}/power_draw_w"] = row.get("power_draw_w")
    run.log(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Log D2E training artifacts to Weights & Biases without touching the trainer.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--train-history", required=True)
    parser.add_argument("--rank-progress-dir", required=True)
    parser.add_argument("--gpu-monitor", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid-file")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", default="g005-idm-paper-target")
    parser.add_argument("--job-type", default="train-sidecar")
    parser.add_argument("--tags", default="g005,idm,d2e,4xh200,sidecar")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--process-pattern", default="train_idm_video.py")
    parser.add_argument("--finish-on-run-summary", action="store_true")
    args = parser.parse_args()

    if args.pid_file:
        pid_path = Path(args.pid_file)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    env_keys = _load_env_file(Path(args.env_file))
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        raise SystemExit("WANDB_PROJECT is required in environment or .env")

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise SystemExit("wandb is not installed; run with `uv run --with wandb ...` or install the train extra") from exc

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    wandb_dir = Path(os.environ.setdefault("WANDB_DIR", "outputs/wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=args.run_name,
        group=args.group,
        job_type=args.job_type,
        tags=tags,
        config=_wandb_config(args, env_keys),
        dir=str(wandb_dir),
        resume=os.environ.get("WANDB_RESUME", "allow"),
    )

    start = time.time()
    seen_epochs: set[int] = set()
    loop_index = 0
    final_status = "running"
    try:
        while True:
            loop_index += 1
            history = _history_rows(Path(args.train_history))
            logged_epochs = _log_epoch_rows(run, history, seen_epochs)
            progress = _rank_progress(Path(args.rank_progress_dir))
            gpu_rows = _latest_gpu_rows(Path(args.gpu_monitor))
            _log_progress(run, progress, gpu_rows, loop_index=loop_index)
            process_running = _process_running(args.process_pattern)
            summary = {
                "schema": "wandb_training_sidecar_status.v1",
                "status": final_status,
                "wandb_run_id": getattr(run, "id", None),
                "wandb_run_url": getattr(run, "url", None),
                "project": project,
                "entity_configured": bool(os.environ.get("WANDB_ENTITY")),
                "run_name": args.run_name,
                "loop_index": loop_index,
                "logged_epochs_total": sorted(seen_epochs),
                "logged_epochs_this_loop": logged_epochs,
                "last_epoch": max(seen_epochs) if seen_epochs else None,
                "rank_progress": progress,
                "gpu_latest": gpu_rows,
                "process_running": process_running,
                "checkpoint_exists": Path(args.checkpoint).exists(),
                "metadata_exists": Path(args.metadata).exists(),
                "run_summary_exists": Path(args.run_summary).exists(),
                "updated_at_epoch": time.time(),
                "claim_boundary": "W&B sidecar status contains no secrets and is logging-only evidence.",
            }
            if args.finish_on_run_summary and summary["run_summary_exists"]:
                final_status = "complete"
                summary["status"] = final_status
                _write_json(Path(args.output), summary)
                break
            if args.max_seconds > 0 and time.time() - start >= args.max_seconds:
                final_status = "max_seconds_reached"
                summary["status"] = final_status
                _write_json(Path(args.output), summary)
                break
            _write_json(Path(args.output), summary)
            time.sleep(max(1.0, args.poll_seconds))
    finally:
        run.finish()
        if args.pid_file:
            pid_path = Path(args.pid_file)
            if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pid_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
