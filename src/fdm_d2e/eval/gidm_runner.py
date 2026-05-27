from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fdm_d2e.io_utils import ensure_dir, read_json, sha256_file, write_json

REMOVED_INLINE_DEPS = ("owa-cli @", "owa-env-gst @")


@dataclass(frozen=True)
class GidmRunPlan:
    index: int
    universe_row_id: str
    video_path: str
    prediction_mcap_path: str
    cuda_device: str
    log_path: str


@dataclass(frozen=True)
class GidmChunkRunPlan:
    index: int
    chunk_index: int
    universe_row_id: str
    video_path: str
    prediction_mcap_path: str
    cuda_device: str
    log_path: str
    start_time_seconds: float
    duration_seconds: float
    timestamp_offset_seconds: float


def prepare_desktop_minimal_inference_script(d2e_repo: str | Path) -> Path:
    """Create the smallest upstream D2E inference script needed for batch runs.

    The public D2E script includes CLI/GStreamer desktop environment packages
    that are not imported by inference.py but can fail to build in lean cluster
    images. We keep owa-env-desktop because owa-data imports VK constants.
    """

    repo = Path(d2e_repo)
    source = repo / "inference.py"
    if not source.exists():
        raise FileNotFoundError(f"missing upstream D2E inference.py: {source}")
    target = repo / "inference_desktop_minimal.py"
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if any(fragment in line for fragment in REMOVED_INLINE_DEPS):
            continue
        lines.append(line)
    text = "\n".join(lines) + "\n"
    replacements = {
        '#     "loguru==0.7.2",\n': (
            '#     "loguru==0.7.2",\n'
            '#     "imageio-ffmpeg==0.5.1",\n'
        ),
        "import subprocess\n": "import subprocess\nimport shutil\n",
        "def preprocess_video(input_path: str, output_path: str, duration: float) -> str:\n": (
            "def _ffmpeg_executable() -> str:\n"
            "    path = shutil.which(\"ffmpeg\")\n"
            "    if path:\n"
            "        return path\n"
            "    try:\n"
            "        import imageio_ffmpeg\n"
            "    except ImportError as exc:\n"
            "        raise FileNotFoundError(\"ffmpeg is not installed and imageio-ffmpeg is unavailable\") from exc\n"
            "    return imageio_ffmpeg.get_ffmpeg_exe()\n"
            "\n\n"
            "def preprocess_video(input_path: str, output_path: str, duration: float, start_time: float = 0.0) -> str:\n"
        ),
        '        "ffmpeg",\n': '        _ffmpeg_executable(),\n',
        '        "-y",\n        "-i",\n': (
            '        "-y",\n        *(["-ss", str(float(start_time))] if float(start_time) > 0 else []),\n        "-i",\n'
        ),
        "def create_mcap_from_video(video_path: str, mcap_path: str, fps: float = 20.0):\n": (
            "def create_mcap_from_video(\n"
            "    video_path: str,\n"
            "    mcap_path: str,\n"
            "    fps: float = 20.0,\n"
            "    timestamp_offset_seconds: float = 0.0,\n"
            "    duration_seconds: float | None = None,\n"
            "):\n"
        ),
        "    duration = get_video_duration(video_path)\n": (
            "    duration = float(duration_seconds) if duration_seconds is not None else get_video_duration(video_path)\n"
        ),
        "        for i in range(num_frames):\n            timestamp_ns = i * interval_ns\n            screen_msg = ScreenCaptured(\n                utc_ns=timestamp_ns,\n                media_ref={\"uri\": video_abs_path, \"pts_ns\": timestamp_ns},\n            )\n            writer.write_message(screen_msg, topic=\"screen\", timestamp=timestamp_ns)\n": (
            "        timestamp_offset_ns = int(float(timestamp_offset_seconds) * 1e9)\n"
            "        for i in range(num_frames):\n"
            "            local_pts_ns = i * interval_ns\n"
            "            timestamp_ns = timestamp_offset_ns + local_pts_ns\n"
            "            screen_msg = ScreenCaptured(\n"
            "                utc_ns=timestamp_ns,\n"
            "                media_ref={\"uri\": video_abs_path, \"pts_ns\": local_pts_ns},\n"
            "            )\n"
            "            writer.write_message(screen_msg, topic=\"screen\", timestamp=timestamp_ns)\n"
        ),
        '    parser.add_argument(\n        "--max-duration", type=float, default=None, help="Max video duration in seconds (default: no limit)"\n    )\n': (
            '    parser.add_argument(\n'
            '        "--max-duration", type=float, default=None, help="Max video duration in seconds (default: no limit)"\n'
            '    )\n'
            '    parser.add_argument("--start-time", type=float, default=0.0, help="Video start offset in seconds for chunked inference")\n'
            '    parser.add_argument(\n'
            '        "--timestamp-offset",\n'
            '        type=float,\n'
            '        default=None,\n'
            '        help="Timestamp offset in seconds to stamp the first preprocessed frame; defaults to start-time",\n'
            '    )\n'
        ),
        "        # Preprocess video if max_duration is specified\n        if args.max_duration is not None:\n            logger.info(f\"Preprocessing video (max {args.max_duration}s)...\")\n            processed_video = str(tmpdir / \"processed.mkv\")\n            preprocess_video(str(input_video), processed_video, args.max_duration)\n        else:\n            processed_video = str(input_video)\n": (
            "        # Preprocess video if max_duration or start_time is specified.\n"
            "        if args.max_duration is not None or float(args.start_time) > 0:\n"
            "            duration = args.max_duration if args.max_duration is not None else max(0.0, get_video_duration(str(input_video)) - float(args.start_time))\n"
            "            logger.info(f\"Preprocessing video (start {args.start_time}s, duration {duration}s)...\")\n"
            "            processed_video = str(tmpdir / \"processed.mkv\")\n"
            "            preprocess_video(str(input_video), processed_video, duration, start_time=float(args.start_time))\n"
            "        else:\n"
            "            processed_video = str(input_video)\n"
        ),
        "        create_mcap_from_video(processed_video, input_mcap)\n": (
            "        timestamp_offset = float(args.start_time if args.timestamp_offset is None else args.timestamp_offset)\n"
            "        create_mcap_from_video(\n"
            "            processed_video,\n"
            "            input_mcap,\n"
            "            timestamp_offset_seconds=timestamp_offset,\n"
            "            duration_seconds=duration if args.max_duration is not None or float(args.start_time) > 0 else None,\n"
            "        )\n"
        ),
    }
    for needle, replacement in replacements.items():
        if needle not in text:
            raise ValueError(f"upstream D2E inference.py no longer matches expected patch anchor: {needle[:80]!r}")
        text = text.replace(needle, replacement, 1)
    target.write_text(text, encoding="utf-8")
    return target


def build_gidm_run_plan(
    *,
    manifest_path: str | Path,
    cuda_devices: Sequence[str],
    log_dir: str | Path,
    max_recordings: int | None = None,
    recording_keys: Sequence[str] | None = None,
    resume: bool = True,
) -> list[GidmRunPlan]:
    if not cuda_devices:
        raise ValueError("at least one CUDA device is required")
    manifest = read_json(manifest_path)
    wanted = set(str(key) for key in (recording_keys or []))
    rows = []
    for row in manifest.get("recordings", []):
        if not isinstance(row, dict):
            continue
        key = str(row.get("universe_row_id") or "")
        if wanted and key not in wanted:
            continue
        pred = Path(str(row.get("prediction_mcap_path") or ""))
        if resume and pred.exists() and pred.stat().st_size > 0:
            continue
        rows.append(row)
        if max_recordings is not None and len(rows) >= int(max_recordings):
            break

    log_root = ensure_dir(log_dir)
    plans = []
    for index, row in enumerate(rows):
        key = str(row["universe_row_id"])
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key).strip("_")
        plans.append(
            GidmRunPlan(
                index=index,
                universe_row_id=key,
                video_path=str(row["video_path"]),
                prediction_mcap_path=str(row["prediction_mcap_path"]),
                cuda_device=str(cuda_devices[index % len(cuda_devices)]),
                log_path=str(log_root / f"{index:05d}_{safe}.log"),
            )
        )
    return plans


def _safe_plan_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "recording"


def _recording_timing(row: dict[str, Any], *, bin_ms: int) -> tuple[float, float, float]:
    if row.get("bin_index_min") is not None and row.get("bin_index_max") is not None:
        start_rel = max(0.0, float(int(row["bin_index_min"]) * int(bin_ms)) / 1000.0)
        end_rel = max(start_rel + int(bin_ms) / 1000.0, float((int(row["bin_index_max"]) + 1) * int(bin_ms)) / 1000.0)
        if row.get("timestamp_min_ns") is not None:
            base_timestamp_seconds = (
                float(int(row["timestamp_min_ns"]) - int(row["bin_index_min"]) * int(bin_ms) * 1_000_000) / 1e9
            )
        else:
            base_timestamp_seconds = 0.0
        return start_rel, end_rel, base_timestamp_seconds
    if (
        row.get("timestamp_min_ns") is not None
        and row.get("timestamp_max_ns") is not None
        and bool(row.get("timestamps_are_video_relative"))
    ):
        start_rel = max(0.0, float(int(row["timestamp_min_ns"])) / 1e9)
        end_rel = max(start_rel + int(bin_ms) / 1000.0, float(int(row["timestamp_max_ns"])) / 1e9 + int(bin_ms) / 1000.0)
        return start_rel, end_rel, 0.0
    row_count = int(row.get("row_count", 0) or 0)
    return 0.0, max(int(bin_ms) / 1000.0, row_count * int(bin_ms) / 1000.0), 0.0


def _chunk_output_path(prediction_mcap_path: str, *, universe_row_id: str, chunk_index: int, start_time: float, duration: float) -> Path:
    base = Path(prediction_mcap_path)
    safe = _safe_plan_name(universe_row_id)
    chunk_dir = base.parent / f"{base.stem}_chunks" / safe
    start_ms = int(round(float(start_time) * 1000.0))
    duration_ms = int(round(float(duration) * 1000.0))
    return chunk_dir / f"chunk_{chunk_index:04d}_start{start_ms:012d}_dur{duration_ms:09d}.mcap"


def _chunk_schedule_for_row(
    row: dict[str, Any],
    *,
    chunk_seconds: float,
    chunk_context_seconds: float,
    bin_ms: int,
) -> list[dict[str, float | int]]:
    start_rel, end_rel, base_timestamp_seconds = _recording_timing(row, bin_ms=bin_ms)
    first_start = max(0.0, start_rel - float(chunk_context_seconds))
    final_end = max(first_start + int(bin_ms) / 1000.0, end_rel + float(chunk_context_seconds))
    schedule: list[dict[str, float | int]] = []
    cursor = first_start
    chunk_index = 0
    while cursor < final_end - 1e-9:
        chunk_end = min(final_end, cursor + float(chunk_seconds))
        duration = max(int(bin_ms) / 1000.0, chunk_end - cursor)
        schedule.append(
            {
                "chunk_index": int(chunk_index),
                "start_time_seconds": float(cursor),
                "duration_seconds": float(duration),
                "timestamp_offset_seconds": float(base_timestamp_seconds + cursor),
                "timestamp_end_seconds": float(base_timestamp_seconds + cursor + duration),
            }
        )
        cursor += float(chunk_seconds)
        chunk_index += 1
    return schedule


def build_gidm_chunk_run_plan(
    *,
    manifest_path: str | Path,
    cuda_devices: Sequence[str],
    log_dir: str | Path,
    chunk_seconds: float,
    chunk_context_seconds: float = 1.0,
    bin_ms: int = 50,
    max_recordings: int | None = None,
    max_chunks: int | None = None,
    recording_keys: Sequence[str] | None = None,
    resume: bool = True,
) -> tuple[list[GidmChunkRunPlan], dict[str, list[str]]]:
    if not cuda_devices:
        raise ValueError("at least one CUDA device is required")
    if float(chunk_seconds) <= 0:
        raise ValueError("chunk_seconds must be positive")
    if float(chunk_context_seconds) < 0:
        raise ValueError("chunk_context_seconds must be non-negative")
    if max_chunks is not None and int(max_chunks) <= 0:
        raise ValueError("max_chunks must be positive when provided")
    manifest = read_json(manifest_path)
    wanted = set(str(key) for key in (recording_keys or []))
    rows = []
    for row in manifest.get("recordings", []):
        if not isinstance(row, dict):
            continue
        key = str(row.get("universe_row_id") or "")
        if wanted and key not in wanted:
            continue
        rows.append(row)
        if max_recordings is not None and len(rows) >= int(max_recordings):
            break

    log_root = ensure_dir(log_dir)
    plans: list[GidmChunkRunPlan] = []
    chunk_paths_by_key: dict[str, list[str]] = {}
    selected_chunk_count = 0
    for row_index, row in enumerate(rows):
        if max_chunks is not None and selected_chunk_count >= int(max_chunks):
            break
        key = str(row["universe_row_id"])
        schedule = _chunk_schedule_for_row(
            row,
            chunk_seconds=float(chunk_seconds),
            chunk_context_seconds=float(chunk_context_seconds),
            bin_ms=int(bin_ms),
        )
        chunk_paths: list[str] = []
        for scheduled in schedule:
            if max_chunks is not None and selected_chunk_count >= int(max_chunks):
                break
            chunk_index = int(scheduled["chunk_index"])
            chunk_start = float(scheduled["start_time_seconds"])
            duration = float(scheduled["duration_seconds"])
            output_path = _chunk_output_path(
                str(row["prediction_mcap_path"]),
                universe_row_id=key,
                chunk_index=chunk_index,
                start_time=chunk_start,
                duration=duration,
            )
            chunk_paths.append(str(output_path))
            selected_chunk_count += 1
            if resume and output_path.exists() and output_path.stat().st_size > 0:
                continue
            safe = _safe_plan_name(key)
            plans.append(
                GidmChunkRunPlan(
                    index=len(plans),
                    chunk_index=chunk_index,
                    universe_row_id=key,
                    video_path=str(row["video_path"]),
                    prediction_mcap_path=str(output_path),
                    cuda_device=str(cuda_devices[len(plans) % len(cuda_devices)]),
                    log_path=str(log_root / f"{row_index:05d}_{chunk_index:04d}_{safe}.log"),
                    start_time_seconds=float(chunk_start),
                    duration_seconds=float(duration),
                    timestamp_offset_seconds=float(scheduled["timestamp_offset_seconds"]),
                )
            )
        if chunk_paths:
            chunk_paths_by_key[key] = chunk_paths
    return plans, chunk_paths_by_key


def _run_one(
    plan: GidmRunPlan,
    *,
    script_path: Path,
    d2e_repo: Path,
    model: str,
    max_context_length: int,
    max_duration: float | None,
    uv_cache_dir: str | Path,
    hf_home: str | Path,
    start_time: float | None = None,
    timestamp_offset: float | None = None,
) -> dict[str, Any]:
    manifest_output_path = Path(plan.prediction_mcap_path)
    output_path = manifest_output_path if manifest_output_path.is_absolute() else (Path.cwd() / manifest_output_path).resolve()
    ensure_dir(output_path.parent)
    ensure_dir(Path(plan.log_path).parent)
    if output_path.exists() and output_path.stat().st_size > 0:
        return {
            "universe_row_id": plan.universe_row_id,
            "video_path": plan.video_path,
            "prediction_mcap_path": plan.prediction_mcap_path,
            "resolved_prediction_mcap_path": str(output_path),
            "temp_prediction_mcap_path": None,
            "cuda_device": plan.cuda_device,
            "log_path": plan.log_path,
            "exit_code": 0,
            "success": True,
            "skipped_existing_at_run": True,
            "elapsed_seconds": 0.0,
            "start_time_seconds": start_time,
            "timestamp_offset_seconds": timestamp_offset,
            "output_exists": True,
            "output_size": output_path.stat().st_size,
            "output_sha256": sha256_file(output_path),
        }
    temp_output_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}.{plan.index}")
    if temp_output_path.exists():
        temp_output_path.unlink()
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": plan.cuda_device,
            "UV_CACHE_DIR": str(Path(uv_cache_dir).resolve()),
            "HF_HOME": str(Path(hf_home).resolve()),
            "TRANSFORMERS_CACHE": str((Path(hf_home) / "transformers").resolve()),
        }
    )
    cmd = [
        "uv",
        "run",
        str(script_path.name),
        plan.video_path,
        str(temp_output_path),
        "--model",
        model,
        "--device",
        "cuda",
        "--max-context-length",
        str(int(max_context_length)),
    ]
    if max_duration is not None:
        cmd.extend(["--max-duration", str(float(max_duration))])
    if start_time is not None:
        cmd.extend(["--start-time", str(float(start_time))])
    if timestamp_offset is not None:
        cmd.extend(["--timestamp-offset", str(float(timestamp_offset))])
    started = time.time()
    with Path(plan.log_path).open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(cmd, cwd=d2e_repo, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    ended = time.time()
    if proc.returncode == 0 and temp_output_path.exists() and temp_output_path.stat().st_size > 0:
        shutil.move(str(temp_output_path), str(output_path))
    elif temp_output_path.exists() and proc.returncode != 0:
        temp_output_path.unlink()
    success = int(proc.returncode) == 0 and output_path.exists() and output_path.stat().st_size > 0
    return {
        "universe_row_id": plan.universe_row_id,
        "video_path": plan.video_path,
        "prediction_mcap_path": plan.prediction_mcap_path,
        "resolved_prediction_mcap_path": str(output_path),
        "temp_prediction_mcap_path": str(temp_output_path),
        "cuda_device": plan.cuda_device,
        "log_path": plan.log_path,
        "exit_code": int(proc.returncode),
        "success": bool(success),
        "elapsed_seconds": ended - started,
        "start_time_seconds": start_time,
        "timestamp_offset_seconds": timestamp_offset,
        "output_exists": output_path.exists(),
        "output_size": output_path.stat().st_size if output_path.exists() else 0,
        "output_sha256": sha256_file(output_path) if output_path.exists() and output_path.is_file() else None,
    }


def _run_chunk(
    plan: GidmChunkRunPlan,
    *,
    script_path: Path,
    d2e_repo: Path,
    model: str,
    max_context_length: int,
    uv_cache_dir: str | Path,
    hf_home: str | Path,
) -> dict[str, Any]:
    row = _run_one(
        GidmRunPlan(
            index=plan.index,
            universe_row_id=plan.universe_row_id,
            video_path=plan.video_path,
            prediction_mcap_path=plan.prediction_mcap_path,
            cuda_device=plan.cuda_device,
            log_path=plan.log_path,
        ),
        script_path=script_path,
        d2e_repo=d2e_repo,
        model=model,
        max_context_length=max_context_length,
        max_duration=plan.duration_seconds,
        start_time=plan.start_time_seconds,
        timestamp_offset=plan.timestamp_offset_seconds,
        uv_cache_dir=uv_cache_dir,
        hf_home=hf_home,
    )
    row.update(
        {
            "chunk_index": plan.chunk_index,
            "duration_seconds": plan.duration_seconds,
            "chunked": True,
        }
    )
    return row


def write_chunked_gidm_manifest(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    chunk_paths_by_key: dict[str, list[str]],
    chunk_seconds: float,
    chunk_context_seconds: float,
    bin_ms: int,
) -> dict[str, Any]:
    manifest = copy.deepcopy(read_json(manifest_path))
    updated = 0
    selected_rows: list[dict[str, Any]] = []
    for row in manifest.get("recordings", []):
        if not isinstance(row, dict):
            continue
        key = str(row.get("universe_row_id") or "")
        paths = chunk_paths_by_key.get(key)
        if not paths:
            continue
        schedule = _chunk_schedule_for_row(
            row,
            chunk_seconds=float(chunk_seconds),
            chunk_context_seconds=float(chunk_context_seconds),
            bin_ms=int(bin_ms),
        )
        chunk_rows = []
        for path, scheduled in zip(paths, schedule, strict=False):
            timestamp_offset = float(scheduled["timestamp_offset_seconds"])
            duration = float(scheduled["duration_seconds"])
            chunk_rows.append(
                {
                    "chunk_index": int(scheduled["chunk_index"]),
                    "prediction_mcap_path": str(path),
                    "start_time_seconds": float(scheduled["start_time_seconds"]),
                    "duration_seconds": duration,
                    "timestamp_offset_seconds": timestamp_offset,
                    "timestamp_start_ns": int(round(timestamp_offset * 1e9)),
                    "timestamp_end_ns_exclusive": int(round((timestamp_offset + duration) * 1e9)),
                }
            )
        row["prediction_mcap_paths"] = paths
        row["prediction_mcap_chunks"] = chunk_rows
        row["prediction_timestamps_aligned_to_ground_truth"] = True
        row["chunked_prediction"] = {
            "chunk_seconds": float(chunk_seconds),
            "chunk_context_seconds": float(chunk_context_seconds),
            "bin_ms": int(bin_ms),
            "timestamp_mode": "ground_truth_aligned",
            "chunk_count": len(chunk_rows),
        }
        updated += 1
        selected_rows.append(row)
    manifest["recordings"] = selected_rows
    manifest["recording_count"] = len(selected_rows)
    manifest["target_rows"] = sum(int(row.get("row_count", 0) or 0) for row in selected_rows)
    manifest["schema"] = str(manifest.get("schema", "gidm_inference_manifest.v1")) + "+chunked"
    manifest["chunked_prediction_count"] = updated
    manifest["source_manifest_path"] = str(manifest_path)
    manifest["claim_boundary"] = (
        "Chunked released G-IDM inference manifest; timestamps are aligned to ground-truth recording time. "
        "This is baseline/teacher infrastructure, not our-IDM metric success."
    )
    write_json(output_path, manifest)
    return manifest


def run_gidm_manifest_inference(
    *,
    manifest_path: str | Path,
    d2e_repo: str | Path,
    output_summary: str | Path,
    cuda_devices: Sequence[str],
    workers: int,
    model: str = "open-world-agents/Generalist-IDM-1B",
    max_recordings: int | None = None,
    recording_keys: Sequence[str] | None = None,
    max_context_length: int = 2048,
    max_duration: float | None = None,
    chunk_seconds: float | None = None,
    chunk_context_seconds: float = 1.0,
    chunk_manifest_output: str | Path | None = None,
    bin_ms: int = 50,
    max_chunks: int | None = None,
    uv_cache_dir: str | Path = "outputs/external/uv-cache-desktop-minimal",
    hf_home: str | Path = "outputs/external/hf-home",
    log_dir: str | Path = "artifacts/eval/gidm_manifest_inference_logs",
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo = Path(d2e_repo)
    script = repo / "inference_desktop_minimal.py" if dry_run else prepare_desktop_minimal_inference_script(repo)
    chunk_paths_by_key: dict[str, list[str]] = {}
    if chunk_seconds is not None:
        plans, chunk_paths_by_key = build_gidm_chunk_run_plan(
            manifest_path=manifest_path,
            cuda_devices=cuda_devices,
            log_dir=log_dir,
            max_recordings=max_recordings,
            max_chunks=max_chunks,
            recording_keys=recording_keys,
            chunk_seconds=float(chunk_seconds),
            chunk_context_seconds=float(chunk_context_seconds),
            bin_ms=int(bin_ms),
            resume=resume,
        )
    else:
        plans = build_gidm_run_plan(
            manifest_path=manifest_path,
            cuda_devices=cuda_devices,
            log_dir=log_dir,
            max_recordings=max_recordings,
            recording_keys=recording_keys,
            resume=resume,
        )
    if dry_run:
        rows = [plan.__dict__ for plan in plans]
    else:
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            if chunk_seconds is not None:
                futures = [
                    executor.submit(
                        _run_chunk,
                        plan,
                        script_path=script,
                        d2e_repo=repo,
                        model=model,
                        max_context_length=max_context_length,
                        uv_cache_dir=uv_cache_dir,
                        hf_home=hf_home,
                    )
                    for plan in plans
                ]
            else:
                futures = [
                    executor.submit(
                        _run_one,
                        plan,
                        script_path=script,
                        d2e_repo=repo,
                        model=model,
                        max_context_length=max_context_length,
                        max_duration=max_duration,
                        uv_cache_dir=uv_cache_dir,
                        hf_home=hf_home,
                    )
                    for plan in plans
                ]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda row: str(row.get("universe_row_id", "")))
    written_chunk_manifest = None
    if chunk_seconds is not None and chunk_manifest_output is not None:
        written_chunk_manifest = write_chunked_gidm_manifest(
            manifest_path=manifest_path,
            output_path=chunk_manifest_output,
            chunk_paths_by_key=chunk_paths_by_key,
            chunk_seconds=float(chunk_seconds),
            chunk_context_seconds=float(chunk_context_seconds),
            bin_ms=int(bin_ms),
        )
    if chunk_seconds is not None:
        planned_recordings = len(chunk_paths_by_key)
        planned_chunks = sum(len(paths) for paths in chunk_paths_by_key.values())
        if dry_run:
            completed_recordings = 0
            failed_recordings = 0
            completed_chunks = 0
            failed_chunks = 0
        else:
            successful_paths = {
                str(row.get("prediction_mcap_path") or row.get("resolved_prediction_mcap_path") or "")
                for row in rows
                if bool(row.get("success", int(row.get("exit_code", 0)) == 0 and row.get("output_exists")))
            }

            def chunk_path_done(path: str) -> bool:
                output = Path(path)
                return path in successful_paths or (output.exists() and output.stat().st_size > 0)

            completed_chunks = sum(1 for paths in chunk_paths_by_key.values() for path in paths if chunk_path_done(path))
            failed_chunks = max(0, planned_chunks - completed_chunks)
            completed_recordings = sum(1 for paths in chunk_paths_by_key.values() if paths and all(chunk_path_done(path) for path in paths))
            failed_recordings = max(0, planned_recordings - completed_recordings)
    else:
        planned_recordings = len(plans)
        planned_chunks = None
        completed_chunks = None
        failed_chunks = None
        completed_recordings = (
            sum(1 for row in rows if bool(row.get("success", int(row.get("exit_code", 0)) == 0 and row.get("output_exists"))))
            if not dry_run
            else 0
        )
        failed_recordings = (
            sum(1 for row in rows if not bool(row.get("success", int(row.get("exit_code", 0)) == 0 and row.get("output_exists"))))
            if not dry_run
            else 0
        )

    payload = {
        "schema": "gidm_manifest_inference_run.v1",
        "manifest_path": str(manifest_path),
        "chunk_manifest_output": str(chunk_manifest_output) if chunk_manifest_output is not None else None,
        "d2e_repo": str(repo),
        "inference_script": str(script),
        "model": model,
        "workers": int(workers),
        "cuda_devices": [str(device) for device in cuda_devices],
        "max_recordings": max_recordings,
        "recording_keys": [str(key) for key in (recording_keys or [])],
        "max_context_length": int(max_context_length),
        "max_duration": max_duration,
        "max_chunks": int(max_chunks) if max_chunks is not None else None,
        "chunk_seconds": float(chunk_seconds) if chunk_seconds is not None else None,
        "chunk_context_seconds": float(chunk_context_seconds) if chunk_seconds is not None else None,
        "bin_ms": int(bin_ms),
        "resume": bool(resume),
        "dry_run": bool(dry_run),
        "planned_recordings": planned_recordings,
        "planned_chunks": planned_chunks,
        "chunked_recording_count": len(chunk_paths_by_key) if chunk_seconds is not None else None,
        "completed_recordings": completed_recordings,
        "failed_recordings": failed_recordings,
        "completed_chunks": completed_chunks,
        "failed_chunks": failed_chunks,
        "rows": rows,
        "chunk_paths_by_key": chunk_paths_by_key,
        "chunk_manifest_written": bool(written_chunk_manifest) if chunk_seconds is not None else False,
        "claim_boundary": "Released G-IDM inference execution evidence only; metric claims require conversion plus paper-metric artifacts.",
    }
    write_json(output_summary, payload)
    return payload
