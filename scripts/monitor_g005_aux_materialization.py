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
    return True


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
    return sorted(item for item in path.rglob("*") if item.is_file())


def _split_status(namespace: Path, split: str) -> dict[str, Any]:
    split_dir = namespace / split
    manifest = split_dir / "manifest.json"
    paths = _iter_files(split_dir)
    return {
        "split": split,
        "dir_exists": split_dir.exists() and split_dir.is_dir(),
        "manifest_exists": manifest.exists() and manifest.is_file(),
        "file_count": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
    }


def _source_row(source_id: str, candidate: dict[str, Any], namespace_root: Path, splits: tuple[str, ...], max_files: int) -> dict[str, Any]:
    namespace = namespace_root / source_id
    raw_dir = namespace / "raw"
    raw_files = _iter_files(raw_dir)
    raw_rows = [
        {
            "path": str(path.relative_to(namespace)),
            "bytes": path.stat().st_size,
        }
        for path in raw_files[:max_files]
    ]
    split_rows = [_split_status(namespace, split) for split in splits]
    return {
        "id": source_id,
        "namespace": str(namespace),
        "source_url": candidate.get("source_url"),
        "license_id": candidate.get("license_id"),
        "expected_size_bytes": candidate.get("size_bytes") or candidate.get("estimated_size_gib"),
        "namespace_exists": namespace.exists() and namespace.is_dir(),
        "raw_dir_exists": raw_dir.exists() and raw_dir.is_dir(),
        "raw_file_count": len(raw_files),
        "raw_total_bytes": sum(path.stat().st_size for path in raw_files),
        "raw_files": raw_rows,
        "raw_files_truncated": len(raw_files) > max_files,
        "split_manifests_ready": all(row["manifest_exists"] for row in split_rows),
        "splits": split_rows,
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
        f"partial={payload['partial_source_ids']} complete={payload['completed_source_ids']} output={args.output}"
    )
    return 0 if payload["status"] in {"running", "pass"} or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
