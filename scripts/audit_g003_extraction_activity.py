#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import write_json


def _dir_file_stats(path: Path) -> dict[str, Any]:
    file_count = 0
    total_bytes = 0
    newest_mtime: float | None = None
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        file_count += 1
        try:
            stat = child.stat()
        except OSError:
            continue
        total_bytes += stat.st_size
        newest_mtime = stat.st_mtime if newest_mtime is None else max(newest_mtime, stat.st_mtime)
    return {"file_count": file_count, "total_bytes": total_bytes, "newest_file_mtime_unix": newest_mtime}


def _latest_recording_dirs(shard_dir: Path, *, now: float, limit: int) -> list[dict[str, Any]]:
    roots = list(shard_dir.glob("by_recording/*/*/*"))
    rows: list[dict[str, Any]] = []
    for root in roots:
        try:
            stat = root.stat()
        except OSError:
            continue
        stats = _dir_file_stats(root)
        latest_mtime = max([value for value in [stat.st_mtime, stats["newest_file_mtime_unix"]] if value is not None])
        rows.append(
            {
                "recording_dir": str(root),
                "seconds_since_activity": max(0.0, now - float(latest_mtime)),
                "dir_mtime_unix": stat.st_mtime,
                "newest_file_mtime_unix": stats["newest_file_mtime_unix"],
                "file_count": stats["file_count"],
                "total_bytes": stats["total_bytes"],
                "has_decode_summary": (root / "decode_summary.json").exists(),
            }
        )
    return sorted(rows, key=lambda row: row["seconds_since_activity"])[:limit]


def _log_tail(path: Path, *, tail: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "decoded" in payload:
            rows.append(payload)
    return rows[-tail:]


def build_activity_report(
    *,
    shard_root: str | Path,
    log_dir: str | Path,
    num_shards: int,
    output: str | Path,
    max_latest_dirs: int = 3,
    log_tail: int = 3,
) -> dict[str, Any]:
    now = time.time()
    shard_root_path = Path(shard_root)
    log_dir_path = Path(log_dir)
    shard_rows: list[dict[str, Any]] = []
    for shard_index in range(int(num_shards)):
        shard_dir = shard_root_path / f"shard_{shard_index}"
        log_path = log_dir_path / f"d2e_full_corpus_shard_{shard_index}.log"
        summaries = list(shard_dir.glob("by_recording/*/*/*/decode_summary.json"))
        shard_summary = shard_dir / "decode_summary.json"
        latest_dirs = _latest_recording_dirs(shard_dir, now=now, limit=int(max_latest_dirs)) if shard_dir.exists() else []
        shard_rows.append(
            {
                "shard_index": shard_index,
                "shard_dir": str(shard_dir),
                "log_path": str(log_path),
                "recording_summary_count": len(summaries),
                "shard_summary_exists": shard_summary.exists(),
                "latest_recording_dirs": latest_dirs,
                "last_log_rows": _log_tail(log_path, tail=int(log_tail)),
            }
        )
    payload = {
        "schema": "g003_extraction_activity.v1",
        "generated_at_unix": now,
        "shard_root": str(shard_root_path),
        "log_dir": str(log_dir_path),
        "num_shards": int(num_shards),
        "shards": shard_rows,
        "claim_boundary": "Filesystem activity evidence only; this does not prove extraction, IDM training, or G003 completion.",
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Write non-mutating filesystem activity evidence for G003 shard extraction.")
    parser.add_argument("--shard-root", default="outputs/data/d2e_full_corpus_shards")
    parser.add_argument("--log-dir", default="artifacts/sources")
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--max-latest-dirs", type=int, default=3)
    parser.add_argument("--log-tail", type=int, default=3)
    parser.add_argument("--output", default="artifacts/idm/g003_extraction_activity.json")
    args = parser.parse_args()
    report = build_activity_report(
        shard_root=args.shard_root,
        log_dir=args.log_dir,
        num_shards=args.num_shards,
        output=args.output,
        max_latest_dirs=args.max_latest_dirs,
        log_tail=args.log_tail,
    )
    active_dirs = sum(1 for shard in report["shards"] if shard["latest_recording_dirs"])
    print(
        "g003 extraction activity: "
        f"shards_with_activity={active_dirs}/{report['num_shards']} "
        f"shard_root={report['shard_root']} log_dir={report['log_dir']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
