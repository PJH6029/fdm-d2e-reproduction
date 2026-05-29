#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _iter_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _loads(line: str, *, path: Path, line_no: int) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except Exception as exc:
        raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected object row at {path}:{line_no}")
    return payload


def _row_groups(row: dict[str, Any], key: str, allowed: set[str] | None) -> list[str]:
    value = row.get(key)
    if isinstance(value, list):
        groups = [str(item) for item in value]
    elif value is None:
        groups = []
    else:
        groups = [str(value)]
    if allowed is not None:
        groups = [group for group in groups if group in allowed]
    return sorted(dict.fromkeys(groups))


def _row_id(row: dict[str, Any]) -> str:
    for key in ("sequence_id", "row_id", "id"):
        value = row.get(key)
        if value:
            return str(value)
    return json.dumps(
        {
            "recording_id": row.get("recording_id"),
            "timestamp_ns": row.get("timestamp_ns"),
            "bin_index": row.get("bin_index"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def materialize_balanced_prefix(
    *,
    input_patterns: list[str],
    output: Path,
    summary_out: Path,
    balance_key: str,
    max_rows: int,
    group_values: list[str] | None,
    per_group_rows: int | None,
    max_per_group: int | None,
    source_label: str,
) -> dict[str, Any]:
    started = time.time()
    paths = _iter_paths(input_patterns)
    if not paths:
        raise ValueError("no input paths")
    allowed = set(str(item) for item in group_values) if group_values else None
    if group_values and per_group_rows is None:
        per_group_rows = int(math.ceil(max_rows / max(1, len(group_values))))
    if per_group_rows is None and max_per_group is None:
        raise ValueError("provide --per-group-rows, --max-per-group, or --group-values with --max-rows")
    target_groups = list(group_values or [])
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned_rows = 0
    skipped_duplicate_rows = 0
    skipped_missing_group_rows = 0
    skipped_saturated_rows = 0
    source_rows_by_path: dict[str, int] = {}
    selected_rows_by_path: dict[str, int] = {}

    def group_has_capacity(groups: list[str]) -> bool:
        if not groups:
            return False
        if per_group_rows is not None:
            return any(counts[group] < per_group_rows for group in groups)
        assert max_per_group is not None
        return any(counts[group] < max_per_group for group in groups)

    for path in paths:
        if len(selected) >= max_rows:
            break
        path_rows = 0
        selected_for_path = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if len(selected) >= max_rows:
                    break
                if not line.strip():
                    continue
                row = _loads(line, path=path, line_no=line_no)
                scanned_rows += 1
                path_rows += 1
                groups = _row_groups(row, balance_key, allowed)
                if not groups:
                    skipped_missing_group_rows += 1
                    continue
                rid = _row_id(row)
                if rid in seen:
                    skipped_duplicate_rows += 1
                    continue
                if not group_has_capacity(groups):
                    skipped_saturated_rows += 1
                    continue
                selected.append(row)
                seen.add(rid)
                selected_for_path += 1
                for group in groups:
                    counts[group] += 1
                if target_groups and all(counts[group] >= int(per_group_rows or 0) for group in target_groups):
                    if len(selected) >= max_rows or per_group_rows is not None:
                        break
        source_rows_by_path[str(path)] = path_rows
        selected_rows_by_path[str(path)] = selected_for_path
        if target_groups and all(counts[group] >= int(per_group_rows or 0) for group in target_groups):
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    status = "pass"
    errors: list[str] = []
    if not selected:
        status = "fail"
        errors.append("no rows selected")
    if target_groups:
        missing = {group: int(per_group_rows or 0) - counts[group] for group in target_groups if counts[group] < int(per_group_rows or 0)}
        if missing:
            status = "partial" if selected else "fail"
            errors.append(f"group quotas not met: {missing}")
    summary = {
        "schema": "balanced_prefix_summary.v1",
        "status": status,
        "errors": errors,
        "source_label": source_label,
        "input_patterns": input_patterns,
        "input_paths": [str(path) for path in paths],
        "output": str(output),
        "balance_key": balance_key,
        "group_values": target_groups,
        "per_group_rows": per_group_rows,
        "max_per_group": max_per_group,
        "max_rows": max_rows,
        "rows": len(selected),
        "scanned_rows": scanned_rows,
        "group_counts": dict(sorted(counts.items())),
        "source_rows_by_path": source_rows_by_path,
        "selected_rows_by_path": selected_rows_by_path,
        "skipped_duplicate_rows": skipped_duplicate_rows,
        "skipped_missing_group_rows": skipped_missing_group_rows,
        "skipped_saturated_rows": skipped_saturated_rows,
        "elapsed_seconds": time.time() - started,
        "claim_boundary": "Balanced prefix materialization only; no model metric claim.",
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a bounded JSONL prefix balanced by recording/game/eval split tags.")
    parser.add_argument("--input", action="append", required=True, help="Input JSONL path or glob. Can be repeated.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--balance-key", default="recording_id")
    parser.add_argument("--group-value", action="append", default=None)
    parser.add_argument("--per-group-rows", type=int)
    parser.add_argument("--max-per-group", type=int)
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--source-label", default="balanced_prefix")
    args = parser.parse_args()
    summary = materialize_balanced_prefix(
        input_patterns=[str(item) for item in args.input],
        output=Path(args.output),
        summary_out=Path(args.summary_out),
        balance_key=str(args.balance_key),
        max_rows=int(args.max_rows),
        group_values=[str(item) for item in args.group_value] if args.group_value else None,
        per_group_rows=args.per_group_rows,
        max_per_group=args.max_per_group,
        source_label=str(args.source_label),
    )
    print(f"balanced prefix: status={summary['status']} rows={summary['rows']} output={summary['output']}")
    return 0 if summary["status"] in {"pass", "partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
