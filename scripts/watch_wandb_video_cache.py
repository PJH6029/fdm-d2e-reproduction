#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        if not key:
            continue
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))
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


def _cache_status(cache_dir: Path) -> dict[str, Any]:
    manifests = sorted(cache_dir.glob("*.manifest.json"))
    rows = 0
    bytes_total = 0
    chunks = 0
    readable_manifests = 0
    for manifest_path in manifests:
        payload = _read_json(manifest_path)
        if not payload:
            continue
        readable_manifests += 1
        rows += int(payload.get("rows", 0) or 0)
        bytes_total += int(payload.get("bytes", 0) or 0)
        manifest_chunks = payload.get("chunks", [])
        if isinstance(manifest_chunks, list):
            chunks += len(manifest_chunks)
    return {
        "manifest_count": len(manifests),
        "readable_manifest_count": readable_manifests,
        "rows": rows,
        "bytes": bytes_total,
        "chunks": chunks,
    }


def _process_running(pattern: str) -> bool | None:
    if not pattern:
        return None
    try:
        result = subprocess.run(["pgrep", "-af", pattern], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return None
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Log video-cache materialization progress to Weights & Biases.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid-file")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", default="g005-idm-paper-target")
    parser.add_argument("--job-type", default="video-cache")
    parser.add_argument("--tags", default="g005,idm,d2e,video-cache,sidecar")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--finish-rows", type=int, default=0)
    parser.add_argument("--finish-manifests", type=int, default=0)
    parser.add_argument("--process-pattern", default="precompute_video_idm_cache.py")
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
        raise SystemExit("wandb is not installed; run with `uv run --with wandb ...`") from exc

    wandb_dir = Path(os.environ.setdefault("WANDB_DIR", "outputs/wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=args.run_name,
        group=args.group,
        job_type=args.job_type,
        tags=[tag.strip() for tag in args.tags.split(",") if tag.strip()],
        dir=str(wandb_dir),
        resume=os.environ.get("WANDB_RESUME", "allow"),
        config={
            "schema": "wandb_video_cache_sidecar_config.v1",
            "cache_dir": args.cache_dir,
            "summary": args.summary,
            "run_summary": args.run_summary,
            "finish_rows": args.finish_rows,
            "finish_manifests": args.finish_manifests,
            "env_keys_loaded": sorted(set(env_keys)),
            "claim_boundary": "Sidecar logs cache manifest progress only; it does not write cache chunks.",
        },
    )
    loop_index = 0
    final_status = "running"
    try:
        while True:
            loop_index += 1
            cache = _cache_status(Path(args.cache_dir))
            process_running = _process_running(args.process_pattern)
            run_summary = _read_json(Path(args.run_summary))
            summary = _read_json(Path(args.summary))
            run.log(
                {
                    "cache/manifest_count": cache["manifest_count"],
                    "cache/readable_manifest_count": cache["readable_manifest_count"],
                    "cache/rows": cache["rows"],
                    "cache/bytes": cache["bytes"],
                    "cache/chunks": cache["chunks"],
                    "cache/process_running": process_running,
                    "cache/run_summary_exists": Path(args.run_summary).exists(),
                    "cache/summary_exists": Path(args.summary).exists(),
                    "sidecar/loop": loop_index,
                    "sidecar/time": time.time(),
                }
            )
            done_by_rows = bool(args.finish_rows and cache["rows"] >= args.finish_rows)
            done_by_manifests = bool(args.finish_manifests and cache["manifest_count"] >= args.finish_manifests)
            done_by_summary = bool(run_summary and run_summary.get("status") == "pass")
            if done_by_rows or done_by_manifests or done_by_summary:
                final_status = "complete"
            payload = {
                "schema": "wandb_video_cache_sidecar_status.v1",
                "status": final_status,
                "wandb_run_id": getattr(run, "id", None),
                "wandb_run_url": getattr(run, "url", None),
                "project": project,
                "entity_configured": bool(os.environ.get("WANDB_ENTITY")),
                "run_name": args.run_name,
                "loop_index": loop_index,
                "cache": cache,
                "process_running": process_running,
                "summary_status": summary.get("status") if summary else None,
                "run_summary_status": run_summary.get("status") if run_summary else None,
                "updated_at_epoch": time.time(),
                "claim_boundary": "W&B cache sidecar status contains no secrets and is logging-only evidence.",
            }
            _write_json(Path(args.output), payload)
            if final_status == "complete":
                break
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
