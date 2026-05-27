#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} did not decode to an object")
            yield row


def _row_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("recording_id", "")),
        int(row.get("timestamp_ns", 0) or 0),
        str(row.get("sequence_id", "")),
    )


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _order_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    last_ts: dict[str, int] = {}
    violations = 0
    examples: list[dict[str, Any]] = []
    recording_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        recording = str(row.get("recording_id", ""))
        timestamp = int(row.get("timestamp_ns", 0) or 0)
        recording_counts[recording] += 1
        for tag in row.get("eval_split_tags") or []:
            split_counts[str(tag)] += 1
        prev = last_ts.get(recording)
        if prev is not None and timestamp < prev:
            violations += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "row_index": idx,
                        "recording_id": recording,
                        "previous_timestamp_ns": prev,
                        "timestamp_ns": timestamp,
                        "sequence_id": row.get("sequence_id"),
                    }
                )
        last_ts[recording] = timestamp
    return {
        "rows": len(rows),
        "recordings": len(recording_counts),
        "per_recording_timestamp_violations": violations,
        "violation_examples": examples,
        "top_recordings": recording_counts.most_common(10),
        "split_tag_counts": dict(sorted(split_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a bounded JSONL prefix sorted by recording/timestamp for closed-loop inference.")
    parser.add_argument("--input", required=True, help="Input JSONL path or glob. Quote globs in the shell.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--source-label", default="")
    args = parser.parse_args()

    if args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    input_paths = [Path(p) for p in sorted(glob.glob(args.input))]
    if not input_paths:
        candidate = Path(args.input)
        if candidate.exists():
            input_paths = [candidate]
    if not input_paths:
        raise FileNotFoundError(args.input)

    started = time.time()
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for path in input_paths:
        for row in _iter_jsonl(path):
            if len(rows) >= args.max_rows:
                break
            row = dict(row)
            row.setdefault("chronological_prefix_source_path", str(path))
            rows.append(row)
            source_counts[str(path)] += 1
        if len(rows) >= args.max_rows:
            break
    input_diag = _order_diagnostics(rows)
    rows.sort(key=_row_sort_key)
    output_diag = _order_diagnostics(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "schema": "chronological_prefix_materialization.v1",
        "status": "pass",
        "source_label": args.source_label,
        "input": args.input,
        "input_paths": [str(p) for p in input_paths],
        "output": str(output_path),
        "max_rows": args.max_rows,
        "rows": len(rows),
        "source_counts": dict(source_counts),
        "input_order": input_diag,
        "output_order": output_diag,
        "output_sha256": _sha256(output_path),
        "wall_clock_seconds": time.time() - started,
        "claim_boundary": "Ordering materialization only; no model-quality claim.",
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "rows": len(rows), "output": str(output_path), "output_order": output_diag}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
