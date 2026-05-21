from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from fdm_d2e.data.full_corpus import included_universe_rows, universe_row_id
from fdm_d2e.io_utils import read_json, write_json


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def _iter_log_json(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "decoded" in row:
                rows.append(row)
    return rows


def _expected_by_shard(data_universe_path: Path, *, num_shards: int) -> dict[int, list[str]]:
    if not data_universe_path.exists():
        return {idx: [] for idx in range(num_shards)}
    rows = included_universe_rows(read_json(data_universe_path))
    out = {idx: [] for idx in range(num_shards)}
    for idx, row in enumerate(rows):
        out[idx % num_shards].append(universe_row_id(row))
    return out


def _recording_summary_count(shard_dir: Path) -> int:
    if not shard_dir.exists():
        return 0
    return sum(1 for _ in shard_dir.glob("by_recording/*/*/*/decode_summary.json"))


def _safe_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_mtime


def _shard_report(
    *,
    index: int,
    expected_ids: list[str],
    shard_root: Path,
    log_dir: Path,
    now: float,
    stale_seconds: float,
) -> dict[str, Any]:
    shard_dir = shard_root / f"shard_{index}"
    log_path = log_dir / f"d2e_full_corpus_shard_{index}.log"
    summary_path = shard_dir / "decode_summary.json"
    log_rows = _iter_log_json(log_path)
    last_row = log_rows[-1] if log_rows else None
    log_mtime = _safe_mtime(log_path)
    summary = read_json(summary_path) if summary_path.exists() else None
    summary_count = _recording_summary_count(shard_dir)
    expected = len(expected_ids)
    decoded = max(summary_count, int(last_row.get("decoded", 0)) if last_row else 0)
    seconds_since_update = None if log_mtime is None else max(0.0, now - log_mtime)
    if summary is not None:
        status = "complete" if int(summary.get("selected_recording_variants", 0)) == expected else "summary_mismatch"
    elif decoded <= 0 and seconds_since_update is not None and seconds_since_update >= stale_seconds:
        status = "no_progress_stale"
    elif seconds_since_update is not None and seconds_since_update >= stale_seconds:
        status = "stale_log"
    else:
        status = "running_or_pending"
    return {
        "shard_index": index,
        "status": status,
        "expected_variants": expected,
        "decoded_variants": decoded,
        "decoded_log_count": int(last_row.get("decoded", 0)) if last_row else 0,
        "recording_summary_count": summary_count,
        "summary_exists": summary is not None,
        "log_path": str(log_path),
        "log_exists": log_path.exists(),
        "seconds_since_log_update": seconds_since_update,
        "last_universe_row_id": last_row.get("universe_row_id") if last_row else None,
        "last_log_row": last_row,
        "completion_ratio": decoded / expected if expected else None,
    }


def build_g003_progress_report(
    *,
    shard_root: str | Path = "outputs/data/d2e_full_corpus_shards",
    log_dir: str | Path = "artifacts/sources",
    data_universe: str | Path = "artifacts/sources/d2e_full_data_universe_manifest.json",
    output_dir: str | Path = "outputs/data/d2e_full_corpus",
    idm_output_dir: str | Path = "outputs/idm_streaming_d2e_full_compact",
    pid_file: str | Path = "outputs/cluster/g003_full_compact_parallel.pid",
    num_shards: int = 16,
    stale_seconds: float = 3600.0,
    now: float | None = None,
) -> dict[str, Any]:
    now_value = time.time() if now is None else float(now)
    shard_root_path = Path(shard_root)
    log_dir_path = Path(log_dir)
    expected = _expected_by_shard(Path(data_universe), num_shards=int(num_shards))
    shards = [
        _shard_report(index=idx, expected_ids=expected.get(idx, []), shard_root=shard_root_path, log_dir=log_dir_path, now=now_value, stale_seconds=float(stale_seconds))
        for idx in range(int(num_shards))
    ]
    total_expected = sum(row["expected_variants"] for row in shards)
    decoded = sum(row["decoded_variants"] for row in shards)
    complete_shards = sum(1 for row in shards if row["status"] == "complete")
    stale_shards = [row for row in shards if row["status"] in {"stale_log", "no_progress_stale"}]
    no_progress_shards = [row for row in shards if row["decoded_variants"] == 0 and not row["summary_exists"]]
    pid = _read_pid(Path(pid_file))
    running = _pid_running(pid) if pid is not None else False
    merged_summary_exists = (Path(output_dir) / "train_core.jsonl").exists() and (Path(output_dir) / "target_all_eval.jsonl").exists()
    idm_metrics_exists = (Path(idm_output_dir) / "metrics.json").exists()
    if complete_shards == int(num_shards) and merged_summary_exists and idm_metrics_exists:
        status = "complete"
    elif stale_shards:
        status = "review_stale_shards"
    elif running:
        status = "running"
    elif decoded > 0:
        status = "not_running_partial"
    else:
        status = "not_started_or_unknown"
    return {
        "schema": "g003_progress_report.v1",
        "status": status,
        "pid_file": str(pid_file),
        "pid": pid,
        "pid_running": running,
        "num_shards": int(num_shards),
        "complete_shards": complete_shards,
        "stale_shards": [row["shard_index"] for row in stale_shards],
        "no_progress_shards": [row["shard_index"] for row in no_progress_shards],
        "decoded_recording_variants": decoded,
        "expected_recording_variants": total_expected,
        "completion_ratio": decoded / total_expected if total_expected else None,
        "merged_train_eval_exists": merged_summary_exists,
        "idm_metrics_exists": idm_metrics_exists,
        "stale_seconds_threshold": float(stale_seconds),
        "shards": shards,
        "claim_boundary": "Progress report only; it does not prove G003 completion until all shards, merge outputs, IDM checkpoint/metrics, label-quality, and statistical comparison artifacts exist.",
    }


def write_g003_progress_report(output: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_g003_progress_report(**kwargs)
    write_json(output, payload)
    return payload


def _shell_join(parts: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in parts)


def _extract_command(
    *,
    shard_index: int,
    num_shards: int,
    shard_root: str,
    cache_dir: str,
    uv_bin: str,
    bin_ms: int,
    frame_fps: int,
    image_size: int,
    video_mode: str,
) -> dict[str, Any]:
    shard_dir = f"{shard_root}/shard_{shard_index}"
    log_path = f"artifacts/sources/d2e_full_corpus_shard_{shard_index}.log"
    argv = [
        uv_bin,
        "run",
        "python",
        "scripts/extract_d2e_full_corpus.py",
        "--config",
        "configs/data/d2e_full_corpus.yaml",
        "--data-universe",
        "artifacts/sources/d2e_full_data_universe_manifest.json",
        "--split-contract",
        "artifacts/sources/d2e_full_split_contract.json",
        "--output-dir",
        shard_dir,
        "--summary-out",
        f"{shard_dir}/decode_summary.json",
        "--cache-dir",
        cache_dir,
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
        "--bin-ms",
        str(bin_ms),
        "--frame-fps",
        str(frame_fps),
        "--image-size",
        str(image_size),
        "--video-mode",
        video_mode,
    ]
    return {
        "shard_index": shard_index,
        "shard_dir": shard_dir,
        "log_path": log_path,
        "argv": argv,
        "shell": f"mkdir -p {shard_dir} artifacts/sources && {_shell_join(argv)} > {log_path} 2>&1",
    }


def build_g003_resume_plan(
    *,
    progress_report: dict[str, Any] | None = None,
    allow_active_parent: bool = False,
    shard_root: str | Path = "outputs/data/d2e_full_corpus_shards",
    log_dir: str | Path = "artifacts/sources",
    data_universe: str | Path = "artifacts/sources/d2e_full_data_universe_manifest.json",
    output_dir: str | Path = "outputs/data/d2e_full_corpus",
    idm_output_dir: str | Path = "outputs/idm_streaming_d2e_full_compact",
    pid_file: str | Path = "outputs/cluster/g003_full_compact_parallel.pid",
    num_shards: int = 16,
    stale_seconds: float = 3600.0,
    cache_dir: str = "/root/work/data/d2e/cache",
    uv_bin: str = "uv",
    bin_ms: int = 50,
    frame_fps: int = 20,
    image_size: int = 64,
    video_mode: str = "download",
) -> dict[str, Any]:
    report = progress_report or build_g003_progress_report(
        shard_root=shard_root,
        log_dir=log_dir,
        data_universe=data_universe,
        output_dir=output_dir,
        idm_output_dir=idm_output_dir,
        pid_file=pid_file,
        num_shards=num_shards,
        stale_seconds=stale_seconds,
    )
    incomplete_shards = [row["shard_index"] for row in report.get("shards", []) if row.get("status") != "complete"]
    commands = [
        _extract_command(
            shard_index=int(index),
            num_shards=int(report.get("num_shards", num_shards)),
            shard_root=str(shard_root),
            cache_dir=cache_dir,
            uv_bin=uv_bin,
            bin_ms=bin_ms,
            frame_fps=frame_fps,
            image_size=image_size,
            video_mode=video_mode,
        )
        for index in incomplete_shards
    ]
    merge_command = [
        uv_bin,
        "run",
        "python",
        "scripts/merge_d2e_full_corpus_shards.py",
        "--shard-root",
        str(shard_root),
        "--output-dir",
        str(output_dir),
        "--summary-out",
        "artifacts/sources/d2e_full_corpus_decode_summary.json",
    ]
    train_command = [uv_bin, "run", "python", "scripts/train_idm_streaming.py", "--config", "configs/model/idm_streaming_d2e_full_compact.yaml", "--require-torch"]
    pid_running = bool(report.get("pid_running"))
    if not incomplete_shards:
        status = "no_resume_needed"
        runnable = False
    elif pid_running and not allow_active_parent:
        status = "defer_active_parent"
        runnable = False
    elif report.get("stale_shards") and pid_running:
        status = "review_stale_active_parent"
        runnable = bool(allow_active_parent)
    else:
        status = "ready_to_resume"
        runnable = True
    return {
        "schema": "g003_resume_plan.v1",
        "status": status,
        "runnable": runnable,
        "allow_active_parent": bool(allow_active_parent),
        "reason": "Do not run shard resume commands while the original parent PID is active unless an operator intentionally sets allow_active_parent.",
        "progress_status": report.get("status"),
        "pid_running": pid_running,
        "decoded_recording_variants": report.get("decoded_recording_variants"),
        "expected_recording_variants": report.get("expected_recording_variants"),
        "complete_shards": report.get("complete_shards"),
        "num_shards": report.get("num_shards"),
        "incomplete_shards": incomplete_shards,
        "stale_shards": report.get("stale_shards", []),
        "no_progress_shards": report.get("no_progress_shards", []),
        "shard_commands": commands,
        "followup_commands_after_all_shards_complete": {
            "merge": {"argv": merge_command, "shell": _shell_join(merge_command)},
            "train_idm": {"argv": train_command, "shell": "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} " + _shell_join(train_command)},
        },
        "claim_boundary": "Resume plan only; it does not execute commands and does not prove G003 completion.",
    }


def write_g003_resume_plan(output: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_g003_resume_plan(**kwargs)
    write_json(output, payload)
    return payload
