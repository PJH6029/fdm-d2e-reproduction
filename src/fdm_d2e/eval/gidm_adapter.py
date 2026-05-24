from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from fdm_d2e.data.d2e_real import decode_mcap_events
from fdm_d2e.io_utils import ensure_dir, read_json, stable_hash_json, write_json
from fdm_d2e.tokenization.actions import tokenize_event

_JSON_DECODER = json.JSONDecoder()


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            yield row


def _extract_value(line: str, key: str) -> Any:
    needle = f'"{key}":'
    idx = line.find(needle)
    if idx < 0:
        return None
    start = idx + len(needle)
    while start < len(line) and line[start].isspace():
        start += 1
    try:
        value, _end = _JSON_DECODER.raw_decode(line, start)
    except json.JSONDecodeError:
        return None
    return value


def _iter_jsonl_fields(path: str | Path, fields: Sequence[str]) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = {field: _extract_value(line, field) for field in fields}
            if row.get("sequence_id") is None and "sequence_id" in fields:
                raise ValueError(f"missing sequence_id in target JSONL at {path}:{line_no}")
            yield row


def _record_key(row: dict[str, Any]) -> str:
    return str(row.get("universe_row_id") or f"{row.get('source_id')}:{row.get('cross_resolution_key')}")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "recording"


def _split_tags(row: dict[str, Any]) -> list[str]:
    value = row.get("eval_split_tags")
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    tags = []
    for key in ("split_temporal", "split_heldout_recording", "split_heldout_game"):
        text = str(row.get(key, ""))
        if text.startswith("heldout"):
            tags.append(key.removeprefix("split_"))
    return tags


def _decode_summary_records(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("recordings", [])
    if not isinstance(rows, list):
        raise ValueError(f"decode summary has no recordings list: {path}")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        keys = [
            str(row.get("universe_row_id") or ""),
            f"{row.get('source_id')}:{row.get('cross_resolution_key')}",
            f"{row.get('source_id')}:{row.get('game')}/{row.get('recording_id')}",
        ]
        for key in keys:
            if key and "None" not in key:
                indexed[key] = row
    return indexed


def build_gidm_inference_manifest(
    *,
    target_record_paths: Sequence[str | Path],
    decode_summary_path: str | Path,
    output_dir: str | Path,
    model: str = "open-world-agents/Generalist-IDM-1B",
    split_tags: Sequence[str] | None = None,
    max_recordings: int | None = None,
) -> dict[str, Any]:
    """Build a per-recording manifest for upstream D2E Generalist-IDM inference."""

    wanted_tags = {str(tag) for tag in split_tags or []}
    by_recording: dict[str, dict[str, Any]] = {}
    fields = [
        "sequence_id",
        "source_id",
        "universe_row_id",
        "cross_resolution_key",
        "source_recording_id",
        "recording_id",
        "game",
        "timestamp_ns",
        "bin_index",
        "eval_split_tags",
        "split_temporal",
        "split_heldout_recording",
        "split_heldout_game",
    ]
    for path in target_record_paths:
        for row in _iter_jsonl_fields(path, fields):
            tags = set(_split_tags(row))
            if wanted_tags and not (tags & wanted_tags):
                continue
            key = _record_key(row)
            bucket = by_recording.setdefault(
                key,
                {
                    "universe_row_id": key,
                    "source_id": row.get("source_id"),
                    "cross_resolution_key": row.get("cross_resolution_key"),
                    "game": row.get("game"),
                    "recording_id": row.get("source_recording_id") or row.get("recording_id"),
                    "row_count": 0,
                    "split_tags": set(),
                    "timestamp_min_ns": None,
                    "timestamp_max_ns": None,
                    "bin_index_min": None,
                    "bin_index_max": None,
                },
            )
            timestamp_ns = int(row.get("timestamp_ns", 0))
            bin_index = int(row.get("bin_index", 0))
            bucket["row_count"] += 1
            bucket["split_tags"].update(tags)
            bucket["timestamp_min_ns"] = timestamp_ns if bucket["timestamp_min_ns"] is None else min(int(bucket["timestamp_min_ns"]), timestamp_ns)
            bucket["timestamp_max_ns"] = timestamp_ns if bucket["timestamp_max_ns"] is None else max(int(bucket["timestamp_max_ns"]), timestamp_ns)
            bucket["bin_index_min"] = bin_index if bucket["bin_index_min"] is None else min(int(bucket["bin_index_min"]), bin_index)
            bucket["bin_index_max"] = bin_index if bucket["bin_index_max"] is None else max(int(bucket["bin_index_max"]), bin_index)

    decode_records = _decode_summary_records(decode_summary_path)
    output_root = Path(output_dir)
    manifest_rows: list[dict[str, Any]] = []
    missing_decode_rows: list[str] = []
    for key, row in sorted(by_recording.items(), key=lambda item: item[0]):
        decoded = decode_records.get(key)
        if decoded is None:
            missing_decode_rows.append(key)
            continue
        safe = _safe_name(key)
        pred_mcap = output_root / "predicted_mcap" / f"{safe}.mcap"
        row_out = {
            **{k: v for k, v in row.items() if k != "split_tags"},
            "split_tags": sorted(row["split_tags"]),
            "video_path": decoded.get("video_source") or decoded.get("video_path"),
            "ground_truth_mcap_path": decoded.get("mcap_path"),
            "prediction_mcap_path": str(pred_mcap),
            "upstream_model": model,
            "inference_command": [
                "uv",
                "run",
                "python",
                "inference.py",
                str(decoded.get("video_source") or decoded.get("video_path")),
                str(pred_mcap),
                "--model",
                model,
                "--device",
                "cuda",
                "--max-context-length",
                "2048",
            ],
        }
        manifest_rows.append(row_out)
        if max_recordings is not None and len(manifest_rows) >= int(max_recordings):
            break

    payload = {
        "schema": "gidm_inference_manifest.v1",
        "model": model,
        "target_record_paths": [str(path) for path in target_record_paths],
        "decode_summary_path": str(decode_summary_path),
        "output_dir": str(output_root),
        "recordings": manifest_rows,
        "recording_count": len(manifest_rows),
        "target_rows": sum(int(row["row_count"]) for row in manifest_rows),
        "missing_decode_rows": missing_decode_rows,
        "manifest_fingerprint": stable_hash_json(
            {
                "model": model,
                "target_record_paths": [str(path) for path in target_record_paths],
                "recordings": [
                    {
                        "universe_row_id": row["universe_row_id"],
                        "row_count": row["row_count"],
                        "video_path": row["video_path"],
                        "prediction_mcap_path": row["prediction_mcap_path"],
                    }
                    for row in manifest_rows
                ],
            }
        ),
        "claim_boundary": "This manifest plans released Generalist-IDM inference over local D2E heldout recordings. It is not training evidence and not a metric result.",
    }
    return payload


def write_gidm_inference_manifest(
    *,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_gidm_inference_manifest(**kwargs)
    write_json(output_path, payload)
    return payload


def _tokens_by_bin(
    events: Sequence[dict[str, Any]],
    *,
    target_timestamps: Sequence[int],
    bin_ms: int,
    timestamp_shift_ns: int = 0,
) -> dict[int, list[str]]:
    if not target_timestamps:
        return {}
    bin_ns = int(bin_ms) * 1_000_000
    starts = list(target_timestamps)
    last_end = starts[-1] + bin_ns
    bins: dict[int, list[str]] = defaultdict(list)
    for event in events:
        timestamp = int(event.get("timestamp_ns", 0)) + int(timestamp_shift_ns)
        if timestamp < starts[0] or timestamp >= last_end:
            continue
        idx = bisect_right(starts, timestamp) - 1
        if idx < 0 or idx >= len(starts):
            continue
        if timestamp >= starts[idx] + bin_ns:
            continue
        for token in tokenize_event(event):
            if token != "NOOP":
                bins[idx].append(token)
    return bins


def convert_gidm_mcap_predictions(
    *,
    manifest_path: str | Path,
    target_record_paths: Sequence[str | Path],
    output_path: str | Path,
    summary_out: str | Path | None = None,
    bin_ms: int = 50,
    timestamp_shift_ns: int = 0,
    allow_missing: bool = False,
    decode_fn: Callable[..., list[dict[str, Any]]] = decode_mcap_events,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    manifest_rows = manifest.get("recordings", [])
    by_key = {str(row["universe_row_id"]): row for row in manifest_rows if isinstance(row, dict)}
    available = {
        key: row
        for key, row in by_key.items()
        if row.get("prediction_mcap_path") and Path(str(row["prediction_mcap_path"])).exists()
    }
    if not allow_missing:
        missing = sorted(set(by_key) - set(available))
        if missing:
            raise FileNotFoundError(f"missing predicted MCAP files for {len(missing)} recording(s), first={missing[0]}")

    targets_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fields = [
        "sequence_id",
        "source_id",
        "universe_row_id",
        "cross_resolution_key",
        "recording_id",
        "game",
        "timestamp_ns",
    ]
    for path in target_record_paths:
        for row in _iter_jsonl_fields(path, fields):
            key = _record_key(row)
            if key in available:
                targets_by_key[key].append(row)

    decoded_counts: dict[str, int] = {}
    token_bins_by_key: dict[str, dict[int, list[str]]] = {}
    for key, target_rows in targets_by_key.items():
        mcap_path = str(available[key]["prediction_mcap_path"])
        events = decode_fn(mcap_path, topics=["keyboard", "mouse/raw", "mouse"])
        decoded_counts[key] = len(events)
        target_rows.sort(key=lambda row: int(row.get("timestamp_ns", 0)))
        token_bins_by_key[key] = _tokens_by_bin(
            events,
            target_timestamps=[int(row["timestamp_ns"]) for row in target_rows],
            bin_ms=bin_ms,
            timestamp_shift_ns=timestamp_shift_ns,
        )

    rows_written = 0
    out = Path(output_path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        for key in sorted(targets_by_key):
            target_rows = targets_by_key[key]
            bins = token_bins_by_key.get(key, {})
            for idx, row in enumerate(target_rows):
                pred = {
                    "schema": "gidm_mcap_prediction.v1",
                    "sequence_id": row["sequence_id"],
                    "recording_id": row.get("recording_id"),
                    "source_id": row.get("source_id"),
                    "universe_row_id": key,
                    "cross_resolution_key": row.get("cross_resolution_key"),
                    "game": row.get("game"),
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "predicted_tokens": bins.get(idx, []),
                }
                handle.write(json.dumps(pred, sort_keys=True) + "\n")
                rows_written += 1

    payload = {
        "schema": "gidm_mcap_conversion_summary.v1",
        "manifest_path": str(manifest_path),
        "target_record_paths": [str(path) for path in target_record_paths],
        "predictions_path": str(output_path),
        "recording_count": len(targets_by_key),
        "rows_written": rows_written,
        "decoded_event_counts": decoded_counts,
        "missing_prediction_count": len(set(by_key) - set(available)),
        "allow_missing": bool(allow_missing),
        "bin_ms": int(bin_ms),
        "timestamp_shift_ns": int(timestamp_shift_ns),
        "claim_boundary": "This converts released G-IDM MCAP predictions to the local predictions.jsonl metric boundary; metric claims require a separate paper-metric/audit artifact.",
    }
    if summary_out:
        write_json(summary_out, payload)
    return payload
