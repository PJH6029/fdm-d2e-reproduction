#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
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
    latest_ts = data_rows[-1][0]
    latest = [row for row in data_rows if row and row[0] == latest_ts]
    parsed: list[dict[str, Any]] = []
    for row in latest:
        if len(row) < 8:
            continue
        # Supported monitor shapes:
        # - epoch,timestamp,index,util_gpu,util_mem,mem_used,mem_total,power
        # - timestamp,index,name,util_gpu,util_mem,mem_used,mem_total,power
        has_epoch_prefix = "/" in row[1] or "-" in row[1]
        offset = 2 if has_epoch_prefix else 1
        parsed.append(
            {
                "timestamp": row[0],
                "index": row[offset],
                "utilization_gpu_pct": _floatish(row[offset + 1 if has_epoch_prefix else offset + 2]),
                "utilization_memory_pct": _floatish(row[offset + 2 if has_epoch_prefix else offset + 3]),
                "memory_used_mib": _floatish(row[offset + 3 if has_epoch_prefix else offset + 4]),
                "memory_total_mib": _floatish(row[offset + 4 if has_epoch_prefix else offset + 5]),
                "power_draw_w": _floatish(row[offset + 5 if has_epoch_prefix else offset + 6]),
            }
        )
    return parsed


def _mcap_status(predicted_dir: Path, planned_paths: list[Path] | None = None) -> dict[str, Any]:
    if planned_paths:
        final_paths = planned_paths
        temp_paths = []
        for path in planned_paths:
            temp_paths.extend(path.parent.glob(path.name + ".tmp.*"))
    else:
        final_paths = [Path(path) for path in glob.glob(str(predicted_dir / "**" / "*.mcap"), recursive=True)]
        temp_paths = [Path(path) for path in glob.glob(str(predicted_dir / "**" / "*.tmp.*"), recursive=True)]
    nonzero_final = [path for path in final_paths if path.exists() and path.stat().st_size > 0]
    zero_final = [path for path in final_paths if path.exists() and path.stat().st_size == 0]
    temp_bytes = sum(path.stat().st_size for path in temp_paths if path.exists())
    final_bytes = sum(path.stat().st_size for path in nonzero_final if path.exists())
    return {
        "predicted_dir": str(predicted_dir),
        "planned_path_scoped": bool(planned_paths),
        "final_mcap_count": len(nonzero_final),
        "zero_final_mcap_count": len(zero_final),
        "temp_output_count": len(temp_paths),
        "final_mcap_bytes": final_bytes,
        "temp_output_bytes": temp_bytes,
    }


def _planned_prediction_paths(manifest: dict[str, Any]) -> list[Path]:
    rows = manifest.get("recordings", [])
    if not isinstance(rows, list):
        return []
    paths: list[Path] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        chunk_paths = row.get("prediction_mcap_paths")
        if isinstance(chunk_paths, list) and chunk_paths:
            paths.extend(Path(str(path)) for path in chunk_paths)
        elif row.get("prediction_mcap_path"):
            paths.append(Path(str(row["prediction_mcap_path"])))
    return paths


def _planned_prediction_outputs(manifest: dict[str, Any]) -> dict[str, int | bool | None]:
    rows = manifest.get("recordings", [])
    if not isinstance(rows, list):
        return {"planned_recordings": None, "planned_outputs": None, "chunked": False}
    planned_outputs = 0
    chunked = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        paths = row.get("prediction_mcap_paths")
        if isinstance(paths, list) and paths:
            planned_outputs += len(paths)
            chunked = True
        elif row.get("prediction_mcap_path"):
            planned_outputs += 1
    return {"planned_recordings": len(rows), "planned_outputs": planned_outputs, "chunked": chunked}


def _process_running(pattern: str) -> bool | None:
    if not pattern:
        return None
    try:
        result = subprocess.run(["pgrep", "-af", pattern], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return None
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Log released G-IDM MCAP inference progress to Weights & Biases.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--predicted-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpu-monitor", required=True)
    parser.add_argument("--pipeline-summary", required=True)
    parser.add_argument("--inference-summary", required=True)
    parser.add_argument("--wrapper-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid-file")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", default="g006-gidm-exact-split")
    parser.add_argument("--job-type", default="gidm-inference-sidecar")
    parser.add_argument("--tags", default="g006,gidm,d2e,inference,sidecar")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--process-pattern", default="run_g006_gidm_exact_split_pipeline.py")
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

    manifest = _read_json(Path(args.manifest)) or {}
    planned = _planned_prediction_outputs(manifest)
    planned_paths = _planned_prediction_paths(manifest)
    planned_recordings = planned["planned_recordings"]
    planned_outputs = planned["planned_outputs"]
    manifest_chunked = planned["chunked"]
    target_rows = manifest.get("target_rows")
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
            "schema": "wandb_gidm_inference_sidecar_config.v1",
            "predicted_dir": args.predicted_dir,
            "manifest": args.manifest,
            "planned_recordings": planned_recordings,
            "planned_outputs": planned_outputs,
            "manifest_chunked": manifest_chunked,
            "target_rows": target_rows,
            "env_keys_loaded": sorted(set(env_keys)),
            "claim_boundary": "Sidecar logs released G-IDM inference progress only; it does not run inference or mutate outputs.",
        },
    )
    loop_index = 0
    final_status = "running"
    try:
        while True:
            loop_index += 1
            mcap = _mcap_status(Path(args.predicted_dir), planned_paths=planned_paths)
            gpu_rows = _latest_gpu_rows(Path(args.gpu_monitor))
            pipeline_summary = _read_json(Path(args.pipeline_summary))
            inference_summary = _read_json(Path(args.inference_summary))
            wrapper_summary = _read_json(Path(args.wrapper_summary))
            process_running = _process_running(args.process_pattern)
            progress = None
            if planned_outputs:
                progress = mcap["final_mcap_count"] / float(planned_outputs)
            metrics: dict[str, Any] = {
                "gidm/final_mcap_count": mcap["final_mcap_count"],
                "gidm/zero_final_mcap_count": mcap["zero_final_mcap_count"],
                "gidm/temp_output_count": mcap["temp_output_count"],
                "gidm/final_mcap_bytes": mcap["final_mcap_bytes"],
                "gidm/temp_output_bytes": mcap["temp_output_bytes"],
                "gidm/planned_recordings": planned_recordings,
                "gidm/planned_outputs": planned_outputs,
                "gidm/manifest_chunked": manifest_chunked,
                "gidm/progress_fraction": progress,
                "gidm/process_running": process_running,
                "sidecar/loop": loop_index,
                "sidecar/time": time.time(),
            }
            for row in gpu_rows:
                index = row.get("index")
                if index is None:
                    continue
                prefix = f"gpu/{index}"
                metrics[f"{prefix}/utilization_gpu_pct"] = row.get("utilization_gpu_pct")
                metrics[f"{prefix}/utilization_memory_pct"] = row.get("utilization_memory_pct")
                metrics[f"{prefix}/memory_used_mib"] = row.get("memory_used_mib")
                metrics[f"{prefix}/power_draw_w"] = row.get("power_draw_w")
            run.log(metrics)
            pipeline_status = pipeline_summary.get("status") if pipeline_summary else None
            wrapper_status = wrapper_summary.get("status") if wrapper_summary else None
            if pipeline_status in {"pass", "fail"} or wrapper_status in {"pass", "fail"}:
                final_status = "complete" if pipeline_status == "pass" or wrapper_status == "pass" else "failed"
            payload = {
                "schema": "wandb_gidm_inference_sidecar_status.v1",
                "status": final_status,
                "wandb_run_id": getattr(run, "id", None),
                "wandb_run_url": getattr(run, "url", None),
                "project": project,
                "entity_configured": bool(os.environ.get("WANDB_ENTITY")),
                "run_name": args.run_name,
                "loop_index": loop_index,
                "mcap": mcap,
                "planned_recordings": planned_recordings,
                "planned_outputs": planned_outputs,
                "manifest_chunked": manifest_chunked,
                "target_rows": target_rows,
                "process_running": process_running,
                "pipeline_summary_status": pipeline_status,
                "inference_summary_status": inference_summary.get("status") if inference_summary else None,
                "wrapper_summary_status": wrapper_status,
                "updated_at_epoch": time.time(),
                "claim_boundary": "W&B G-IDM inference sidecar status contains no secrets and is logging-only evidence.",
            }
            _write_json(Path(args.output), payload)
            if final_status in {"complete", "failed"}:
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
