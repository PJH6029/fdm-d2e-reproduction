#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json


DEFAULT_OUTPUT = "artifacts/aux/g005_aux_materialization_progress.json"
DEFAULT_AUX_CANDIDATES = "artifacts/sources/aux_game_action_dataset_candidates.json"
DEFAULT_NAMESPACE_ROOT = "outputs/aux"
DEFAULT_PID_FILE = "outputs/cluster/g005_aux_materialization.pid"
DEFAULT_MATERIALIZATION_SUMMARY = "artifacts/aux/g005_aux_materialization_execute_summary.json"
DEFAULT_WATCHER_SUMMARY = "artifacts/aux/g005_aux_materialization_watcher_summary.json"
DEFAULT_SPLITS = ("train", "val", "test")


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"schema": "invalid_json", "error": str(exc)}
    return payload if isinstance(payload, dict) else {"schema": "unexpected_json", "payload_type": type(payload).__name__}


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
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
    return not _pid_is_zombie(pid)


def _pid_is_zombie(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="ignore")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return False
    try:
        tail = stat.rsplit(")", 1)[1].strip().split()
    except IndexError:
        return False
    return bool(tail and tail[0] == "Z")


def _selected_candidates(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if payload is None:
        return {}
    return {
        str(row["id"]): row
        for row in payload.get("candidates", [])
        if isinstance(row, dict) and row.get("id") and row.get("selection_status") == "selected_candidate"
    }


def _iter_files(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    files: list[Path] = []
    for item in path.rglob("*"):
        try:
            if item.is_file():
                files.append(item)
        except OSError:
            # Download managers can rename/remove temporary files while the
            # monitor walks the tree. Treat those as transient churn, not as a
            # monitor failure.
            continue
    return sorted(files)


def _safe_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except (FileNotFoundError, OSError):
        return None


def _split_status(namespace: Path, split: str) -> dict[str, Any]:
    split_dir = namespace / split
    manifest = split_dir / "manifest.json"
    paths = _iter_files(split_dir)
    sizes = [_safe_file_size(path) for path in paths]
    existing_sizes = [int(size) for size in sizes if size is not None]
    return {
        "split": split,
        "dir_exists": split_dir.exists() and split_dir.is_dir(),
        "manifest_exists": manifest.exists() and manifest.is_file(),
        "file_count": len(existing_sizes),
        "bytes": sum(existing_sizes),
        "transient_missing_file_count": len(sizes) - len(existing_sizes),
    }


def _expected_size_bytes(candidate: dict[str, Any]) -> int | None:
    raw = candidate.get("size_bytes")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    gib = candidate.get("estimated_size_gib")
    if gib is not None:
        try:
            return int(float(gib) * 1024 * 1024 * 1024)
        except (TypeError, ValueError):
            return None
    return None


def _source_row(source_id: str, candidate: dict[str, Any], namespace_root: Path, splits: tuple[str, ...], max_files: int) -> dict[str, Any]:
    namespace = namespace_root / source_id
    raw_dir = namespace / "raw"
    raw_files = _iter_files(raw_dir)
    raw_rows = []
    raw_visible_sizes: list[int] = []
    raw_transient_missing = 0
    for path in raw_files:
        size = _safe_file_size(path)
        if size is None:
            raw_transient_missing += 1
            continue
        raw_visible_sizes.append(int(size))
        if len(raw_rows) < max_files:
            raw_rows.append({"path": str(path.relative_to(namespace)), "bytes": int(size)})
    split_rows = [_split_status(namespace, split) for split in splits]
    raw_total_bytes = sum(raw_visible_sizes)
    expected_size_bytes = _expected_size_bytes(candidate)
    remaining_expected_bytes = max(0, expected_size_bytes - raw_total_bytes) if expected_size_bytes is not None else None
    return {
        "id": source_id,
        "namespace": str(namespace),
        "source_url": candidate.get("source_url"),
        "license_id": candidate.get("license_id"),
        "expected_size_bytes": expected_size_bytes,
        "namespace_exists": namespace.exists() and namespace.is_dir(),
        "raw_dir_exists": raw_dir.exists() and raw_dir.is_dir(),
        "raw_file_count": len(raw_visible_sizes),
        "raw_total_bytes": raw_total_bytes,
        "raw_completion_ratio": raw_total_bytes / expected_size_bytes if expected_size_bytes else None,
        "raw_remaining_expected_bytes": remaining_expected_bytes,
        "raw_files": raw_rows,
        "raw_files_truncated": len(raw_visible_sizes) > max_files,
        "raw_transient_missing_file_count": raw_transient_missing,
        "split_manifests_ready": all(row["manifest_exists"] for row in split_rows),
        "splits": split_rows,
    }


def _recommendation(*, running: bool, materialization_summary: dict[str, Any] | None, errors: list[dict[str, Any]], partial_sources: list[str], missing_sources: list[str]) -> dict[str, Any]:
    if errors:
        return {
            "code": "inspect_materialization_errors",
            "severity": "error",
            "next_actions": ["Inspect materialization summary/logs before rerunning downloads.", "Do not launch G005 training from blocked materialization evidence."],
        }
    if running:
        return {
            "code": "continue_materialization",
            "severity": "info",
            "next_actions": ["Keep the active materializer and watcher running.", "Use raw_completion_ratio and source lists only as download progress telemetry."],
        }
    if materialization_summary and materialization_summary.get("status") == "pass":
        return {
            "code": "run_integrity_and_namespace_gates",
            "severity": "info",
            "next_actions": ["Run/inspect the G005 materialization watcher outputs for integrity, source evidence, examples, runtime env, and namespace manifest.", "Do not claim D2E+aux model quality before G003/G004 hard gates and G005 completion audit pass."],
        }
    if partial_sources or missing_sources:
        return {
            "code": "plan_or_resume_materialization",
            "severity": "warning",
            "partial_source_ids": partial_sources,
            "missing_source_ids": missing_sources,
            "next_actions": ["If no materializer PID is active, resume selected-source materialization before building G005 evidence artifacts."],
        }
    return {
        "code": "start_materialization",
        "severity": "warning",
        "next_actions": ["No active materializer or passing summary was found; start the approved G005 aux materialization flow if still needed."],
    }


def build_progress(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    candidates = _selected_candidates(_load_json(_path(root, args.aux_candidates)))
    if args.source_id:
        missing = sorted(set(args.source_id) - set(candidates))
        if missing:
            raise SystemExit(f"requested source ids are not selected candidates: {', '.join(missing)}")
        candidates = {key: candidates[key] for key in args.source_id}
    namespace_root = _path(root, args.namespace_root)
    pid_file = _path(root, args.pid_file)
    pid = _read_pid(pid_file)
    running = _pid_running(pid)
    materialization_summary = _load_json(_path(root, args.materialization_summary))
    watcher_summary = _load_json(_path(root, args.watcher_summary))
    splits = tuple(args.splits or DEFAULT_SPLITS)
    sources = [_source_row(source_id, candidate, namespace_root, splits, int(args.max_files)) for source_id, candidate in sorted(candidates.items())]

    total_raw_bytes = sum(int(row.get("raw_total_bytes") or 0) for row in sources)
    total_expected_raw_bytes = sum(int(row.get("expected_size_bytes") or 0) for row in sources)
    raw_remaining_expected_bytes = max(0, total_expected_raw_bytes - total_raw_bytes) if total_expected_raw_bytes else None
    completed_sources = [row["id"] for row in sources if row.get("raw_file_count", 0) > 0 and row.get("split_manifests_ready")]
    partial_sources = [row["id"] for row in sources if row.get("raw_file_count", 0) > 0 and not row.get("split_manifests_ready")]
    missing_sources = [row["id"] for row in sources if not row.get("raw_file_count", 0)]
    findings: list[dict[str, Any]] = []
    if not candidates:
        findings.append({"severity": "error", "code": "no_selected_aux_candidates"})
    if materialization_summary is not None and materialization_summary.get("status") not in {"pass", "planned"}:
        findings.append({"severity": "error", "code": "materialization_summary_not_pass", "status": materialization_summary.get("status")})
    if not running and materialization_summary is None:
        findings.append({"severity": "warning", "code": "materializer_not_running_without_summary", "pid": pid})
    status = "running" if running else ("pass" if materialization_summary and materialization_summary.get("status") == "pass" else "blocked")
    errors = [item for item in findings if item.get("severity") == "error"]
    recommendation = _recommendation(
        running=running,
        materialization_summary=materialization_summary,
        errors=errors,
        partial_sources=sorted(partial_sources),
        missing_sources=sorted(missing_sources),
    )
    return {
        "schema": "g005_aux_materialization_progress.v1",
        "status": "blocked" if errors else status,
        "root": str(root),
        "pid_file": str(args.pid_file),
        "pid": pid,
        "pid_running": running,
        "materialization_summary_status": materialization_summary.get("status") if isinstance(materialization_summary, dict) else None,
        "watcher_summary_status": watcher_summary.get("status") if isinstance(watcher_summary, dict) else None,
        "selected_source_ids": sorted(candidates),
        "completed_source_ids": sorted(completed_sources),
        "partial_source_ids": sorted(partial_sources),
        "missing_source_ids": sorted(missing_sources),
        "raw_total_bytes": total_raw_bytes,
        "expected_raw_total_bytes": total_expected_raw_bytes,
        "raw_completion_ratio": total_raw_bytes / total_expected_raw_bytes if total_expected_raw_bytes else None,
        "raw_remaining_expected_bytes": raw_remaining_expected_bytes,
        "recommendation": recommendation,
        "aux_sources": sources,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "Progress telemetry only; it does not prove source completeness, start G005 training, checkpoint G005, or support D2E+aux model-quality claims.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor selected G005 auxiliary source materialization progress without claiming readiness.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--aux-candidates", default=DEFAULT_AUX_CANDIDATES)
    parser.add_argument("--namespace-root", default=DEFAULT_NAMESPACE_ROOT)
    parser.add_argument("--source-id", action="append", help="Selected source id to inspect; repeatable. Defaults to all selected candidates.")
    parser.add_argument("--pid-file", default=DEFAULT_PID_FILE)
    parser.add_argument("--materialization-summary", default=DEFAULT_MATERIALIZATION_SUMMARY)
    parser.add_argument("--watcher-summary", default=DEFAULT_WATCHER_SUMMARY)
    parser.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_progress(args)
    write_json(Path(args.root).resolve() / args.output, payload)
    print(
        "g005 aux materialization progress: "
        f"status={payload['status']} raw_bytes={payload['raw_total_bytes']} "
        f"raw_ratio={payload['raw_completion_ratio']} "
        f"partial={payload['partial_source_ids']} complete={payload['completed_source_ids']} output={args.output}"
    )
    return 0 if payload["status"] in {"running", "pass"} or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
