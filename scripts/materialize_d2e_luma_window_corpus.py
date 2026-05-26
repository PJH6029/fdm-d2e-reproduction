#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json

try:  # pragma: no cover - exercised on cluster images when present.
    import orjson  # type: ignore
except Exception:  # pragma: no cover - fallback is covered.
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


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield _loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _output_path(input_path: Path, *, input_root: Path, output_root: Path) -> Path:
    try:
        rel = input_path.relative_to(input_root)
    except ValueError:
        rel = Path(input_path.name)
    return output_root / rel


def _luma(row: dict[str, Any], *, luma_key: str, expected_len: int) -> list[float] | None:
    frame = row.get("frame", {})
    values = frame.get(luma_key, []) if isinstance(frame, dict) else []
    if not isinstance(values, list) or len(values) != expected_len:
        return None
    return [float(value) for value in values]


def _attach_windows(
    rows: list[dict[str, Any]],
    *,
    offsets: tuple[int, ...],
    luma_key: str,
    expected_len: int,
) -> list[dict[str, Any]]:
    by_recording: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_recording[str(row.get("recording_id", ""))].append(index)

    out_rows = [dict(row) for row in rows]
    zero = [0.0 for _ in range(expected_len)]
    for indices in by_recording.values():
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
            out_rows[row_index]["compact_luma_window_schema"] = "d2e_compact_luma16_window.v1"
            out_rows[row_index]["compact_luma_window_offsets"] = list(offsets)
            out_rows[row_index]["compact_luma_window"] = window
            out_rows[row_index]["compact_luma_window_mask"] = mask
    return out_rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        for row in rows:
            handle.write(_dumps(row) + "\n")
    tmp_path.replace(path)
    return {"output_path": str(path), "rows": len(rows), "sha256": _sha256_file(path)}


def _materialize_pair(payload: dict[str, Any]) -> dict[str, Any]:
    train_input = Path(payload["train_input"])
    target_input = Path(payload["target_input"])
    train_output = Path(payload["train_output"])
    target_output = Path(payload["target_output"])
    offsets = tuple(int(value) for value in payload["offsets"])
    luma_key = str(payload["luma_key"])
    expected_len = int(payload["expected_len"])

    train_rows = list(_iter_jsonl(train_input))
    target_rows = list(_iter_jsonl(target_input))
    combined = _attach_windows(train_rows + target_rows, offsets=offsets, luma_key=luma_key, expected_len=expected_len)
    train_out = _write_rows(train_output, combined[: len(train_rows)])
    target_out = _write_rows(target_output, combined[len(train_rows) :])
    present_counts = [0 for _ in offsets]
    for row in combined:
        for idx, value in enumerate(row.get("compact_luma_window_mask", [])):
            present_counts[idx] += int(float(value) > 0.0)
    return {
        "pair_index": int(payload["pair_index"]),
        "train_output": {**train_out, "input_path": str(train_input)},
        "target_output": {**target_out, "input_path": str(target_input)},
        "combined_rows": len(combined),
        "offset_present_counts": dict(zip([str(value) for value in offsets], present_counts)),
    }


def materialize_luma_window_corpus(
    *,
    train_inputs: list[Path],
    target_inputs: list[Path],
    input_root: Path,
    output_root: Path,
    summary_path: Path,
    offsets: tuple[int, ...] = (-2, -1, 0, 1, 2),
    luma_size: int = 16,
    workers: int = 1,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    if len(train_inputs) != len(target_inputs):
        raise ValueError("luma-window materialization expects matching train/target shard counts")
    luma_key = f"luma{int(luma_size)}"
    expected_len = int(luma_size) * int(luma_size)
    tasks = [
        {
            "pair_index": idx,
            "train_input": str(train_path),
            "target_input": str(target_path),
            "train_output": str(_output_path(train_path, input_root=input_root, output_root=output_root)),
            "target_output": str(_output_path(target_path, input_root=input_root, output_root=output_root)),
            "offsets": list(offsets),
            "luma_key": luma_key,
            "expected_len": expected_len,
        }
        for idx, (train_path, target_path) in enumerate(zip(train_inputs, target_inputs))
    ]
    if progress_path:
        write_json(
            progress_path,
            {
                "schema": "d2e_luma_window_materialization_progress.v1",
                "status": "running",
                "completed_pairs": 0,
                "total_pairs": len(tasks),
            },
        )

    completed: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks))) as pool:
            futures = [pool.submit(_materialize_pair, task) for task in tasks]
            for future in as_completed(futures):
                completed.append(future.result())
                if progress_path:
                    write_json(
                        progress_path,
                        {
                            "schema": "d2e_luma_window_materialization_progress.v1",
                            "status": "running",
                            "completed_pairs": len(completed),
                            "total_pairs": len(tasks),
                        },
                    )
    else:
        for task in tasks:
            completed.append(_materialize_pair(task))
            if progress_path:
                write_json(
                    progress_path,
                    {
                        "schema": "d2e_luma_window_materialization_progress.v1",
                        "status": "running",
                        "completed_pairs": len(completed),
                        "total_pairs": len(tasks),
                    },
                )

    completed = sorted(completed, key=lambda item: int(item["pair_index"]))
    train_rows = sum(int(row["train_output"]["rows"]) for row in completed)
    target_rows = sum(int(row["target_output"]["rows"]) for row in completed)
    present_counts: dict[str, int] = {str(offset): 0 for offset in offsets}
    for row in completed:
        for key, value in row.get("offset_present_counts", {}).items():
            present_counts[str(key)] = present_counts.get(str(key), 0) + int(value)
    payload = {
        "schema": "d2e_luma_window_materialization.v1",
        "status": "pass",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "offsets": list(offsets),
        "luma_size": int(luma_size),
        "train_rows": train_rows,
        "target_rows": target_rows,
        "pair_count": len(completed),
        "offset_present_counts": present_counts,
        "train_outputs": [row["train_output"] for row in completed],
        "target_outputs": [row["target_output"] for row in completed],
        "wall_clock_seconds": time.time() - started,
        "claim_boundary": "Adds non-label compact luma windows for NEP-style IDM context; D2E event-token targets are preserved unchanged.",
    }
    write_json(summary_path, payload)
    if progress_path:
        write_json(
            progress_path,
            {
                "schema": "d2e_luma_window_materialization_progress.v1",
                "status": "pass",
                "completed_pairs": len(completed),
                "total_pairs": len(tasks),
                "train_rows": train_rows,
                "target_rows": target_rows,
            },
        )
    return payload


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _parse_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not offsets:
        raise argparse.ArgumentTypeError("offset list must not be empty")
    if len(offsets) != len(set(offsets)):
        raise argparse.ArgumentTypeError("offset list must not contain duplicates")
    return offsets


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize D2E compact luma windows for NEP-style IDM context.")
    parser.add_argument("--train-input", action="append", required=True)
    parser.add_argument("--target-input", action="append", required=True)
    parser.add_argument("--input-root", default="outputs/data/d2e_full_corpus_shards_accel64")
    parser.add_argument("--output-root", default="outputs/data/d2e_luma_window5_corpus_shards_accel64")
    parser.add_argument("--summary", default="artifacts/idm/g005_idm_luma_window5_materialization_summary.json")
    parser.add_argument("--progress-output", default="artifacts/idm/g005_idm_luma_window5_materialization_progress.json")
    parser.add_argument("--offsets", type=_parse_offsets, default="-2,-1,0,1,2")
    parser.add_argument("--luma-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    payload = materialize_luma_window_corpus(
        train_inputs=_expand(args.train_input),
        target_inputs=_expand(args.target_input),
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        summary_path=Path(args.summary),
        offsets=tuple(args.offsets),
        luma_size=int(args.luma_size),
        workers=max(1, int(args.workers)),
        progress_path=Path(args.progress_output) if args.progress_output else None,
    )
    print(json.dumps({"status": payload["status"], "train_rows": payload["train_rows"], "target_rows": payload["target_rows"]}, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
