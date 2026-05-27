#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _dumps(row: dict[str, Any]) -> str:
    if orjson is not None:
        return orjson.dumps(row, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _expand(pattern: str) -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths and Path(pattern).exists():
        paths = [Path(pattern)]
    if not paths:
        raise FileNotFoundError(pattern)
    return paths


def _iter_prefix(pattern: str, max_rows: int) -> Iterable[dict[str, Any]]:
    emitted = 0
    for path in _expand(pattern):
        with path.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
            for line_no, line in enumerate(handle, 1):
                if emitted >= max_rows:
                    return
                if not line.strip():
                    continue
                try:
                    yield _loads(line)
                except Exception as exc:
                    raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc
                emitted += 1


def _luma(row: dict[str, Any], *, luma_key: str, expected_len: int) -> list[float] | None:
    frame = row.get("frame", {})
    values = frame.get(luma_key, []) if isinstance(frame, dict) else []
    if not isinstance(values, list) or len(values) != expected_len:
        return None
    return [float(value) for value in values]


def _attach_windows(rows: list[dict[str, Any]], *, offsets: tuple[int, ...], luma_key: str, expected_len: int) -> list[dict[str, Any]]:
    by_recording: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_recording[str(row.get("recording_id", ""))].append(index)
    out = [dict(row) for row in rows]
    zero = [0.0 for _ in range(expected_len)]
    for indices in by_recording.values():
        # Preserve source order inside each recording. Prefix diagnostics only use sources that are already chronological per recording.
        lumas = [_luma(rows[index], luma_key=luma_key, expected_len=expected_len) for index in indices]
        for local_index, row_index in enumerate(indices):
            window: list[list[float]] = []
            mask: list[float] = []
            for offset in offsets:
                neighbor = local_index + int(offset)
                if 0 <= neighbor < len(indices) and lumas[neighbor] is not None:
                    window.append(list(lumas[neighbor] or zero))
                    mask.append(1.0)
                else:
                    window.append(list(zero))
                    mask.append(0.0)
            out[row_index]["compact_luma_window_schema"] = "d2e_compact_luma16_window.prefix.v1"
            out[row_index]["compact_luma_window_offsets"] = list(offsets)
            out[row_index]["compact_luma_window"] = window
            out[row_index]["compact_luma_window_mask"] = mask
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        for row in rows:
            handle.write(_dumps(row) + "\n")
    tmp.replace(path)
    return {"path": str(path), "rows": len(rows), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _parse_offsets(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("offset list must not be empty")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("offset list must not contain duplicates")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize bounded D2E luma-window train/target prefixes for G005 NEP-style diagnostics.")
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--target-input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--target-output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-train-rows", type=int, default=320000)
    parser.add_argument("--max-target-rows", type=int, default=320000)
    parser.add_argument("--offsets", type=_parse_offsets, default="0,2,4,6,8")
    parser.add_argument("--luma-size", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    luma_key = f"luma{int(args.luma_size)}"
    expected_len = int(args.luma_size) * int(args.luma_size)
    train_rows = list(_iter_prefix(args.train_input, int(args.max_train_rows)))
    target_rows = list(_iter_prefix(args.target_input, int(args.max_target_rows)))
    # Keep train/target windows independent. The prefix probe intentionally
    # exposes future visual context within each split, but must not let train
    # rows borrow target/eval frames from the same recording.
    train_windowed = _attach_windows(train_rows, offsets=tuple(args.offsets), luma_key=luma_key, expected_len=expected_len)
    target_windowed = _attach_windows(target_rows, offsets=tuple(args.offsets), luma_key=luma_key, expected_len=expected_len)
    train_out = _write(Path(args.train_output), train_windowed)
    target_out = _write(Path(args.target_output), target_windowed)
    present_counts = [0 for _ in args.offsets]
    for row in train_windowed + target_windowed:
        for idx, value in enumerate(row.get("compact_luma_window_mask", [])):
            present_counts[idx] += int(float(value) > 0.0)
    summary = {
        "schema": "d2e_luma_window_prefix_materialization.v1",
        "status": "pass",
        "train_input": args.train_input,
        "target_input": args.target_input,
        "train_output": train_out,
        "target_output": target_out,
        "max_train_rows": int(args.max_train_rows),
        "max_target_rows": int(args.max_target_rows),
        "offsets": list(args.offsets),
        "luma_key": luma_key,
        "offset_present_counts": {str(offset): count for offset, count in zip(args.offsets, present_counts)},
        "wall_clock_seconds": time.time() - started,
        "claim_boundary": "Prefix materialization only; not G005 completion evidence.",
    }
    write_json(args.summary, summary)
    print(json.dumps({"status": summary["status"], "train_rows": train_out["rows"], "target_rows": target_out["rows"], "offsets": summary["offsets"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
