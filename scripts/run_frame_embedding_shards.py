#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ShardSpec:
    index: int
    skip_rows: int
    max_rows: int
    output_path: Path
    summary_path: Path
    progress_path: Path
    log_path: Path
    cuda_visible_devices: str | None = None


def build_shard_plan(
    *,
    total_rows: int,
    shard_count: int,
    start_row: int,
    output_dir: Path,
    artifact_dir: Path,
    artifact_prefix: str,
    devices: Sequence[str],
) -> list[ShardSpec]:
    if total_rows <= 0:
        raise ValueError("--total-rows must be positive")
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if start_row < 0:
        raise ValueError("--start-row must be non-negative")
    base = total_rows // shard_count
    remainder = total_rows % shard_count
    specs: list[ShardSpec] = []
    cursor = start_row
    for index in range(shard_count):
        count = base + (1 if index < remainder else 0)
        if count <= 0:
            continue
        device = devices[index % len(devices)] if devices else None
        specs.append(
            ShardSpec(
                index=index,
                skip_rows=cursor,
                max_rows=count,
                output_path=output_dir / f"shard_{index:04d}.jsonl",
                summary_path=artifact_dir / f"{artifact_prefix}_shard{index}_summary.json",
                progress_path=artifact_dir / f"{artifact_prefix}_shard{index}_progress.json",
                log_path=artifact_dir / f"{artifact_prefix}_shard{index}.log",
                cuda_visible_devices=device,
            )
        )
        cursor += count
    return specs


def _parse_devices(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _float_value(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def summarize_gpu_monitor(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"available": False, "path": str(path), "samples": 0, "by_index": {}}
    by_index: dict[str, list[float]] = {}
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {"available": False, "path": str(path), "samples": 0, "by_index": {}}
        normalized = {name: name.strip().lower() for name in reader.fieldnames}
        index_key = next((name for name, norm in normalized.items() if norm == "index"), None)
        util_key = next((name for name, norm in normalized.items() if norm.startswith("utilization.gpu")), None)
        if index_key is None or util_key is None:
            return {"available": False, "path": str(path), "samples": 0, "by_index": {}, "fieldnames": reader.fieldnames}
        for row in reader:
            rows += 1
            index = str(row.get(index_key, "")).strip()
            util = _float_value(row.get(util_key))
            if index and util is not None:
                by_index.setdefault(index, []).append(util)
    return {
        "available": True,
        "path": str(path),
        "samples": rows,
        "by_index": {
            index: {
                "samples": len(values),
                "mean": sum(values) / len(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
            for index, values in sorted(by_index.items(), key=lambda item: item[0])
        },
    }


def _add_bool_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def build_materializer_command(args: argparse.Namespace, spec: ShardSpec) -> list[str]:
    cmd = [
        str(args.python_executable),
        str(args.materializer_script),
        "--input-path",
        str(args.input_path),
        "--output-path",
        str(spec.output_path),
        "--summary-out",
        str(spec.summary_path),
        "--progress-output",
        str(spec.progress_path),
        "--backend",
        args.backend,
        "--model-id",
        args.model_id,
        "--frame-offsets",
        args.frame_offsets,
        "--frame-source",
        args.frame_source,
        "--image-size",
        str(args.image_size),
        "--frame-fps",
        str(args.frame_fps),
        "--missing-frame-policy",
        args.missing_frame_policy,
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--embedding-pooling",
        args.embedding_pooling,
        "--hf-preprocess",
        args.hf_preprocess,
        "--max-rows",
        str(spec.max_rows),
        "--skip-rows",
        str(spec.skip_rows),
        "--round-digits",
        str(args.round_digits),
        "--progress-rows",
        str(args.progress_rows),
        "--source-label",
        f"{args.source_label}_shard{spec.index}",
    ]
    _add_bool_flag(cmd, "--no-normalize-embeddings", args.no_normalize_embeddings)
    _add_bool_flag(cmd, "--no-embedding-deltas", args.no_embedding_deltas)
    _add_bool_flag(cmd, "--no-summary-features", args.no_summary_features)
    if args.summary_feature_mode:
        cmd.extend(["--summary-feature-mode", args.summary_feature_mode])
    _add_bool_flag(cmd, "--trust-remote-code", args.trust_remote_code)
    for path_map in args.path_map:
        cmd.extend(["--path-map", path_map])
    return cmd


def _start_monitor(path: Path, interval_seconds: int) -> subprocess.Popen[str] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv",
        "-l",
        str(max(1, interval_seconds)),
    ]
    try:
        return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    except Exception:
        handle.close()
        raise


def _stop_monitor(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=5)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _concat_shards(specs: Sequence[ShardSpec], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_path.open("w", encoding="utf-8") as dst:
        for spec in specs:
            with spec.output_path.open("r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        rows += 1
                    dst.write(line)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch frozen frame-embedding materialization over contiguous row shards.")
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--combined-output-path", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--artifact-prefix", default="frame_embedding_shards")
    parser.add_argument("--total-rows", required=True, type=int)
    parser.add_argument("--start-row", default=0, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--devices", default="", help="Comma-separated CUDA_VISIBLE_DEVICES assignment cycled across shards.")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--materializer-script", type=Path, default=_repo_root() / "scripts" / "materialize_frame_embedding_features.py")
    parser.add_argument("--gpu-monitor-output", type=Path)
    parser.add_argument("--gpu-monitor-interval-seconds", type=int, default=30)
    parser.add_argument("--no-gpu-monitor", action="store_true")
    parser.add_argument("--backend", default="dummy-stat", choices=["dummy-stat", "hf-vision", "dinov2-torchhub"])
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--frame-offsets", default="0,2")
    parser.add_argument("--frame-source", default="video", choices=["video", "compact-luma"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frame-fps", type=int, default=20)
    parser.add_argument("--missing-frame-policy", default="zero", choices=["zero", "error"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-pooling", default="cls", choices=["cls", "mean", "pooler", "image"])
    parser.add_argument("--hf-preprocess", default="manual-imagenet", choices=["manual-imagenet", "auto"])
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument("--no-embedding-deltas", action="store_true")
    parser.add_argument("--no-summary-features", action="store_true")
    parser.add_argument("--summary-feature-mode", default="summary_compact_luma16_pair_shift_time_state_duration_prior_action")
    parser.add_argument("--round-digits", default="6")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--progress-rows", type=int, default=50_000)
    parser.add_argument("--source-label", default="g005_frozen_frame_embedding_materialization")
    args = parser.parse_args(argv)
    args.input_path = args.input_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.summary_out = args.summary_out.resolve()
    args.artifact_dir = (args.artifact_dir or args.summary_out.parent).resolve()
    args.materializer_script = args.materializer_script.resolve()
    if args.combined_output_path is not None:
        args.combined_output_path = args.combined_output_path.resolve()
    if args.gpu_monitor_output is None:
        args.gpu_monitor_output = args.artifact_dir / f"{args.artifact_prefix}_gpu_monitor.csv"
    else:
        args.gpu_monitor_output = args.gpu_monitor_output.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    devices = _parse_devices(args.devices)
    specs = build_shard_plan(
        total_rows=args.total_rows,
        shard_count=args.shard_count,
        start_row=args.start_row,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        artifact_prefix=args.artifact_prefix,
        devices=devices,
    )
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.time()
    monitor = None if args.no_gpu_monitor else _start_monitor(args.gpu_monitor_output, args.gpu_monitor_interval_seconds)
    processes: list[tuple[ShardSpec, subprocess.Popen[str], Any, list[str]]] = []
    try:
        for spec in specs:
            env = os.environ.copy()
            if spec.cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices
            env["FRAME_EMBEDDING_SHARD_INDEX"] = str(spec.index)
            env["FRAME_EMBEDDING_SHARD_SKIP_ROWS"] = str(spec.skip_rows)
            env["FRAME_EMBEDDING_SHARD_MAX_ROWS"] = str(spec.max_rows)
            cmd = build_materializer_command(args, spec)
            log_handle = spec.log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=_repo_root(), env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            processes.append((spec, proc, log_handle, cmd))
        shard_results: list[dict[str, Any]] = []
        for spec, proc, log_handle, cmd in processes:
            returncode = proc.wait()
            log_handle.close()
            summary = _load_json(spec.summary_path)
            shard_results.append(
                {
                    "index": spec.index,
                    "returncode": returncode,
                    "skip_rows": spec.skip_rows,
                    "max_rows": spec.max_rows,
                    "cuda_visible_devices": spec.cuda_visible_devices,
                    "output_path": str(spec.output_path),
                    "summary_path": str(spec.summary_path),
                    "progress_path": str(spec.progress_path),
                    "log_path": str(spec.log_path),
                    "rows_written": int((summary or {}).get("rows_written") or 0),
                    "summary_status": (summary or {}).get("status"),
                    "source_rows_scanned": (summary or {}).get("source_rows_scanned"),
                    "source_rows_skipped": (summary or {}).get("source_rows_skipped"),
                    "command": cmd,
                }
            )
    finally:
        for _spec, _proc, log_handle, _cmd in processes:
            try:
                log_handle.close()
            except Exception:
                pass
        _stop_monitor(monitor)
    rows_written = sum(int(item.get("rows_written") or 0) for item in shard_results)
    failed = [item for item in shard_results if item["returncode"] != 0 or item.get("summary_status") != "pass"]
    combined_rows: int | None = None
    combined_output_path = str(args.combined_output_path) if args.combined_output_path else None
    status = "pass" if not failed and rows_written == args.total_rows else "fail"
    if status == "pass" and args.combined_output_path is not None:
        combined_rows = _concat_shards(specs, args.combined_output_path)
        if combined_rows != args.total_rows:
            status = "fail"
    summary = {
        "schema": "frame_embedding_shard_launcher.v1",
        "status": status,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.time() - t0,
        "input_path": str(args.input_path),
        "output_dir": str(args.output_dir),
        "combined_output_path": combined_output_path,
        "combined_rows": combined_rows,
        "total_rows": args.total_rows,
        "start_row": args.start_row,
        "shard_count": len(specs),
        "requested_shard_count": args.shard_count,
        "rows_written": rows_written,
        "failed_shards": failed,
        "shards": shard_results,
        "gpu_monitor": summarize_gpu_monitor(args.gpu_monitor_output),
        "monitor_started_before_shards": not args.no_gpu_monitor,
        "devices": devices,
        "backend": args.backend,
        "model_id": args.model_id,
        "frame_source": args.frame_source,
        "frame_offsets": args.frame_offsets,
        "batch_size": args.batch_size,
        "device_arg": args.device,
        "claim_boundary": "Frame-embedding shard launcher/materialization evidence only; no trained IDM metric claim.",
    }
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "rows_written": rows_written, "summary_out": str(args.summary_out)}, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
