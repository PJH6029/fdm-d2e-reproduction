from __future__ import annotations

import json
import os
import shlex
import time
from collections import Counter
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


def _proc_cmdline_parts(entry: Path) -> list[str]:
    try:
        raw = (entry / "cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return []
    if not raw:
        return []
    return [part.decode("utf-8", errors="ignore") for part in raw.split(b"\0") if part]


def _proc_ppid(entry: Path) -> int | None:
    try:
        stat = (entry / "stat").read_text(encoding="utf-8", errors="ignore")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    # /proc/<pid>/stat uses "pid (comm) state ppid ..." where comm may contain
    # spaces. Split after the final ")" to avoid corrupting the ppid field.
    try:
        tail = stat.rsplit(")", 1)[1].strip().split()
        return int(tail[1]) if len(tail) > 1 else None
    except (IndexError, ValueError):
        return None


def _arg_value(parts: list[str], flag: str) -> str | None:
    if flag not in parts:
        return None
    try:
        return parts[parts.index(flag) + 1]
    except IndexError:
        return None


def _classify_g003_process(parts: list[str]) -> tuple[str | None, int | None]:
    joined = " ".join(parts)
    if "scripts/extract_d2e_full_corpus.py" in joined or "extract_d2e_full_corpus.py" in joined:
        shard_raw = _arg_value(parts, "--shard-index")
        try:
            return "extractor", int(shard_raw) if shard_raw is not None else None
        except ValueError:
            return "extractor", None
    if "run_g003_d2e_full_idm_parallel.sh" in joined:
        return "parent", None
    if "scripts/watch_g003_then_finalize.py" in joined or "watch_g003_then_finalize.py" in joined:
        return "postrun_watcher", None
    if "scripts/attach_g003_gpu_monitor.py" in joined or "attach_g003_gpu_monitor.py" in joined:
        return "gpu_monitor", None
    if "scripts/merge_d2e_full_corpus_shards.py" in joined or "merge_d2e_full_corpus_shards.py" in joined:
        return "merge", None
    if "scripts/train_idm_streaming.py" in joined or "train_idm_streaming.py" in joined or "torchrun" in joined:
        return "idm_train", None
    if "scripts/finalize_g003_integrated_run.py" in joined or "finalize_g003_integrated_run.py" in joined:
        return "finalizer", None
    return None, None


def _process_summary(row: dict[str, Any]) -> dict[str, Any]:
    parts_raw = row.get("cmdline", [])
    if isinstance(parts_raw, str):
        parts = shlex.split(parts_raw)
        cmdline = parts_raw
    else:
        parts = [str(part) for part in parts_raw]
        cmdline = " ".join(parts)
    role = row.get("role")
    shard_index = row.get("shard_index")
    if role is None:
        role, parsed_shard = _classify_g003_process(parts)
        shard_index = parsed_shard if shard_index is None else shard_index
    try:
        shard_index = int(shard_index) if shard_index is not None else None
    except (TypeError, ValueError):
        shard_index = None
    return {
        "pid": int(row["pid"]),
        "ppid": int(row["ppid"]) if row.get("ppid") is not None else None,
        "role": role,
        "shard_index": shard_index,
        "cmdline": cmdline,
    }


def _detect_g003_processes() -> list[dict[str, Any]]:
    """Best-effort Linux /proc scan for live G003-related processes."""

    proc = Path("/proc")
    if not proc.exists():
        return []
    processes: list[dict[str, Any]] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        parts = _proc_cmdline_parts(entry)
        if not parts:
            continue
        role, shard_index = _classify_g003_process(parts)
        if role is None:
            continue
        processes.append(
            {
                "pid": int(entry.name),
                "ppid": _proc_ppid(entry),
                "role": role,
                "shard_index": shard_index,
                "cmdline": " ".join(parts),
            }
        )
    return sorted(processes, key=lambda row: (str(row.get("role")), int(row.get("pid", 0))))


def _detect_active_shard_processes() -> set[int]:
    """Best-effort Linux /proc scan for active full-corpus extraction shards."""

    return {
        int(row["shard_index"])
        for row in _detect_g003_processes()
        if row.get("role") == "extractor" and row.get("shard_index") is not None
    }


def _pid_running_in_snapshot(processes: list[dict[str, Any]], pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    return any(int(row.get("pid", -1)) == pid for row in processes)


def _pid_file_status(path: Path, processes: list[dict[str, Any]], *, from_snapshot: bool) -> dict[str, Any]:
    pid = _read_pid(path)
    running = _pid_running_in_snapshot(processes, pid) if from_snapshot else _pid_running(pid) if pid is not None else False
    return {"pid_file": str(path), "pid": pid, "running": bool(running)}


def _processes_by_role(processes: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [row for row in processes if row.get("role") == role]


def build_g003_live_health_report(
    *,
    shard_root: str | Path = "outputs/data/d2e_full_corpus_shards",
    log_dir: str | Path = "artifacts/sources",
    data_universe: str | Path = "artifacts/sources/d2e_full_data_universe_manifest.json",
    output_dir: str | Path = "outputs/data/d2e_full_corpus",
    idm_output_dir: str | Path = "outputs/idm_streaming_d2e_full_compact",
    pid_file: str | Path = "outputs/cluster/g003_full_compact_parallel.pid",
    watcher_pid_file: str | Path = "outputs/cluster/g003_postrun_watcher.pid",
    gpu_monitor_pid_file: str | Path = "outputs/cluster/g003_attached_gpu_monitor.pid",
    num_shards: int = 16,
    stale_seconds: float = 3600.0,
    min_active_extractors: int | None = None,
    now: float | None = None,
    process_snapshot: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a non-mutating live process-topology report for a G003 integrated run."""

    now_value = time.time() if now is None else float(now)
    from_snapshot = process_snapshot is not None
    processes = [_process_summary(row) for row in process_snapshot] if process_snapshot is not None else _detect_g003_processes()
    extractors = _processes_by_role(processes, "extractor")
    active_shards = sorted({int(row["shard_index"]) for row in extractors if row.get("shard_index") is not None})
    progress = build_g003_progress_report(
        shard_root=shard_root,
        log_dir=log_dir,
        data_universe=data_universe,
        output_dir=output_dir,
        idm_output_dir=idm_output_dir,
        pid_file=pid_file,
        num_shards=num_shards,
        stale_seconds=stale_seconds,
        now=now_value,
        active_shard_processes=set(active_shards),
    )
    parent = _pid_file_status(Path(pid_file), processes, from_snapshot=from_snapshot)
    watcher = _pid_file_status(Path(watcher_pid_file), processes, from_snapshot=from_snapshot)
    gpu_monitor = _pid_file_status(Path(gpu_monitor_pid_file), processes, from_snapshot=from_snapshot)
    active_roles = sorted({str(row["role"]) for row in processes if row.get("role")})
    train_active = bool(_processes_by_role(processes, "idm_train"))
    merge_active = bool(_processes_by_role(processes, "merge"))
    finalizer_active = bool(_processes_by_role(processes, "finalizer"))
    incomplete_shards = [int(row["shard_index"]) for row in progress.get("shards", []) if row.get("status") != "complete"]
    inactive_incomplete_shards = sorted(set(incomplete_shards) - set(active_shards))
    shard_counts = Counter(int(row["shard_index"]) for row in extractors if row.get("shard_index") is not None)
    duplicate_active_shards = sorted(index for index, count in shard_counts.items() if count > 1)
    expected_active = min_active_extractors
    if expected_active is None:
        expected_active = len(incomplete_shards) if incomplete_shards and not (train_active or merge_active or finalizer_active) else 0
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    if parent["running"] and progress.get("status") == "review_stale_shards":
        errors.append({"code": "stale_shards", "shards": progress.get("stale_shards", [])})
    if parent["running"] and expected_active > 0 and len(active_shards) < int(expected_active):
        warnings.append(
            {
                "code": "low_active_extractor_count",
                "active_count": len(active_shards),
                "expected_active_extractors": int(expected_active),
                "inactive_incomplete_shards": inactive_incomplete_shards,
            }
        )
    if parent["running"] and not watcher["running"]:
        warnings.append({"code": "postrun_watcher_not_running", "pid_file": watcher["pid_file"]})
    if parent["running"] and not gpu_monitor["running"]:
        warnings.append({"code": "gpu_monitor_not_running", "pid_file": gpu_monitor["pid_file"]})
    if duplicate_active_shards:
        observations.append(
            {
                "code": "duplicate_extractor_processes",
                "shards": duplicate_active_shards,
                "note": "Usually expected when both uv wrapper and child Python processes expose the shard commandline.",
            }
        )
    if progress.get("status") == "complete":
        stage = "complete_pending_audit"
        status = "complete_pending_audit"
    elif errors:
        stage = "needs_operator_review"
        status = "blocked_live_health"
    elif train_active:
        stage = "idm_training"
        status = "healthy_running" if parent["running"] else "parent_not_running"
    elif merge_active:
        stage = "merge"
        status = "healthy_running" if parent["running"] else "parent_not_running"
    elif finalizer_active:
        stage = "postrun_finalization"
        status = "postrun_finalizing"
    elif extractors:
        stage = "extracting"
        status = "warn_live_health" if warnings else "healthy_running"
    elif parent["running"]:
        stage = "parent_running_no_known_worker"
        status = "warn_live_health"
        if not warnings:
            warnings.append({"code": "parent_running_no_known_worker", "active_roles": active_roles})
    elif progress.get("decoded_recording_variants", 0) > 0:
        stage = "interrupted_or_between_stages"
        status = "parent_not_running_partial"
    else:
        stage = "not_started_or_unknown"
        status = "not_started_or_unknown"
    return {
        "schema": "g003_live_health_report.v1",
        "generated_at_unix": now_value,
        "status": status,
        "stage": stage,
        "progress_status": progress.get("status"),
        "parent": parent,
        "postrun_watcher": watcher,
        "gpu_monitor": gpu_monitor,
        "process_roles": active_roles,
        "processes": processes,
        "active_extractor_shards": active_shards,
        "duplicate_active_shards": duplicate_active_shards,
        "inactive_incomplete_shards": inactive_incomplete_shards,
        "expected_active_extractors": int(expected_active),
        "warnings": warnings,
        "errors": errors,
        "observations": observations,
        "progress": {
            "decoded_recording_variants": progress.get("decoded_recording_variants"),
            "expected_recording_variants": progress.get("expected_recording_variants"),
            "completion_ratio": progress.get("completion_ratio"),
            "complete_shards": progress.get("complete_shards"),
            "num_shards": progress.get("num_shards"),
            "stale_shards": progress.get("stale_shards", []),
            "long_running_shards": progress.get("long_running_shards", []),
            "no_progress_shards": progress.get("no_progress_shards", []),
            "decoded_recording_variants_per_hour": progress.get("decoded_recording_variants_per_hour"),
            "eta_seconds_at_current_rate": progress.get("eta_seconds_at_current_rate"),
            "quiet_active_shards": progress.get("quiet_active_shards", []),
            "recommendation": progress.get("recommendation"),
        },
        "claim_boundary": "Live health report only; it does not mutate the G003 run and does not prove G003 completion or quality-gate passage.",
    }


def write_g003_live_health_report(output: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_g003_live_health_report(**kwargs)
    write_json(output, payload)
    return payload


def _shard_report(
    *,
    index: int,
    expected_ids: list[str],
    shard_root: Path,
    log_dir: Path,
    now: float,
    stale_seconds: float,
    active_shard_processes: set[int],
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
    process_active = index in active_shard_processes
    if summary is not None:
        status = "complete" if int(summary.get("selected_recording_variants", 0)) == expected else "summary_mismatch"
    elif process_active and seconds_since_update is not None and seconds_since_update >= stale_seconds:
        status = "running_long_recording"
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
        "process_active": process_active,
        "log_path": str(log_path),
        "log_exists": log_path.exists(),
        "log_mtime": log_mtime,
        "seconds_since_log_update": seconds_since_update,
        "last_universe_row_id": last_row.get("universe_row_id") if last_row else None,
        "last_log_row": last_row,
        "completion_ratio": decoded / expected if expected else None,
    }


def _quiet_active_shard_details(shards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for row in shards:
        if row.get("status") != "running_long_recording":
            continue
        details.append(
            {
                "shard_index": row["shard_index"],
                "seconds_since_log_update": row.get("seconds_since_log_update"),
                "decoded_variants": row.get("decoded_variants"),
                "expected_variants": row.get("expected_variants"),
                "last_universe_row_id": row.get("last_universe_row_id"),
            }
        )
    return sorted(
        details,
        key=lambda row: float(row.get("seconds_since_log_update") or 0.0),
        reverse=True,
    )


def _progress_recommendation(
    *,
    status: str,
    running: bool,
    stale_shards: list[dict[str, Any]],
    long_running_shards: list[dict[str, Any]],
    no_progress_shards: list[dict[str, Any]],
    complete_shards: int,
    num_shards: int,
    merged_summary_exists: bool,
    idm_metrics_exists: bool,
) -> dict[str, Any]:
    if status == "complete":
        return {
            "code": "run_completion_audit",
            "severity": "info",
            "next_actions": [
                "Run scripts/validate_g003_full_idm_completion.py before any OMX completion checkpoint.",
                "Do not claim G003 complete from the progress monitor alone.",
            ],
        }
    if stale_shards:
        return {
            "code": "inspect_stale_inactive_shards",
            "severity": "error",
            "shards": [row["shard_index"] for row in stale_shards],
            "next_actions": [
                "Inspect shard logs and live process topology before relaunching anything.",
                "Generate scripts/plan_g003_resume.py only after confirming the original parent process is not active.",
            ],
        }
    if running and long_running_shards:
        return {
            "code": "continue_monitor_long_recordings",
            "severity": "info",
            "shards": [row["shard_index"] for row in long_running_shards],
            "next_actions": [
                "Keep the active parent/extractor processes running; original-resolution recordings can stay quiet for long periods.",
                "Re-run scripts/audit_g003_live_health.py instead of restarting shards while extractor processes remain active.",
            ],
        }
    if running:
        return {
            "code": "continue_waiting",
            "severity": "info",
            "next_actions": [
                "Continue periodic non-mutating progress and live-health monitoring.",
                "Wait for shard summaries before merge/training/finalization evidence.",
            ],
        }
    if complete_shards == num_shards and (not merged_summary_exists or not idm_metrics_exists):
        return {
            "code": "run_merge_or_training_followup",
            "severity": "warning",
            "next_actions": [
                "All shard summaries exist, but merge outputs or IDM metrics are missing.",
                "Inspect the parent/postrun watcher logs before launching follow-up commands manually.",
            ],
        }
    if status == "not_running_partial":
        return {
            "code": "plan_resume_if_parent_exited",
            "severity": "warning",
            "next_actions": [
                "The run has partial artifacts but no live parent/extractor evidence.",
                "Generate a read-only resume plan before restarting any extraction shards.",
            ],
        }
    if no_progress_shards:
        return {
            "code": "not_started_or_missing_logs",
            "severity": "warning",
            "shards": [row["shard_index"] for row in no_progress_shards],
            "next_actions": [
                "Confirm the data universe path and shard log directory before treating this as a failed run.",
                "If the parent is not active, use scripts/plan_g003_resume.py for exact resume commands.",
            ],
        }
    return {
        "code": "plan_resume_if_parent_exited",
        "severity": "warning",
        "next_actions": [
            "The run is not complete and no live parent/extractor evidence was found.",
            "Generate a read-only resume plan before restarting any extraction shards.",
        ],
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
    active_shard_processes: set[int] | list[int] | None = None,
) -> dict[str, Any]:
    now_value = time.time() if now is None else float(now)
    shard_root_path = Path(shard_root)
    log_dir_path = Path(log_dir)
    expected = _expected_by_shard(Path(data_universe), num_shards=int(num_shards))
    active_processes = set(int(idx) for idx in active_shard_processes) if active_shard_processes is not None else _detect_active_shard_processes()
    shards = [
        _shard_report(
            index=idx,
            expected_ids=expected.get(idx, []),
            shard_root=shard_root_path,
            log_dir=log_dir_path,
            now=now_value,
            stale_seconds=float(stale_seconds),
            active_shard_processes=active_processes,
        )
        for idx in range(int(num_shards))
    ]
    total_expected = sum(row["expected_variants"] for row in shards)
    decoded = sum(row["decoded_variants"] for row in shards)
    complete_shards = sum(1 for row in shards if row["status"] == "complete")
    stale_shards = [row for row in shards if row["status"] in {"stale_log", "no_progress_stale"}]
    long_running_shards = [row for row in shards if row["status"] == "running_long_recording"]
    no_progress_shards = [row for row in shards if row["decoded_variants"] == 0 and not row["summary_exists"] and not row["process_active"]]
    log_mtimes = [float(row["log_mtime"]) for row in shards if row.get("log_mtime") is not None]
    elapsed_seconds = max(0.0, now_value - min(log_mtimes)) if log_mtimes else None
    max_seconds_since_log_update = max((float(row["seconds_since_log_update"]) for row in shards if row.get("seconds_since_log_update") is not None), default=None)
    decoded_per_hour = None
    eta_seconds = None
    if elapsed_seconds and elapsed_seconds > 0 and decoded > 0:
        decoded_per_hour = decoded / (elapsed_seconds / 3600.0)
        remaining = max(0, total_expected - decoded)
        eta_seconds = remaining / (decoded / elapsed_seconds) if decoded else None
    pid = _read_pid(Path(pid_file))
    running = (_pid_running(pid) if pid is not None else False) or bool(active_processes)
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
    recommendation = _progress_recommendation(
        status=status,
        running=running,
        stale_shards=stale_shards,
        long_running_shards=long_running_shards,
        no_progress_shards=no_progress_shards,
        complete_shards=complete_shards,
        num_shards=int(num_shards),
        merged_summary_exists=merged_summary_exists,
        idm_metrics_exists=idm_metrics_exists,
    )
    return {
        "schema": "g003_progress_report.v1",
        "generated_at_unix": now_value,
        "status": status,
        "pid_file": str(pid_file),
        "pid": pid,
        "pid_running": running,
        "num_shards": int(num_shards),
        "complete_shards": complete_shards,
        "stale_shards": [row["shard_index"] for row in stale_shards],
        "long_running_shards": [row["shard_index"] for row in long_running_shards],
        "no_progress_shards": [row["shard_index"] for row in no_progress_shards],
        "active_shard_processes": sorted(active_processes),
        "decoded_recording_variants": decoded,
        "expected_recording_variants": total_expected,
        "completion_ratio": decoded / total_expected if total_expected else None,
        "elapsed_seconds_since_first_log": elapsed_seconds,
        "max_seconds_since_log_update": max_seconds_since_log_update,
        "decoded_recording_variants_per_hour": decoded_per_hour,
        "eta_seconds_at_current_rate": eta_seconds,
        "quiet_active_shards": _quiet_active_shard_details(shards),
        "recommendation": recommendation,
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
