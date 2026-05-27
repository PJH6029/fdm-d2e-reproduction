from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from fdm_d2e.data.d2e_real import decode_mcap_events
from fdm_d2e.eval.gidm_targets import prediction_windows_ns
from fdm_d2e.io_utils import ensure_dir, read_json, stable_hash_json, write_json
from fdm_d2e.tokenization.actions import tokenize_event

_JSON_DECODER = json.JSONDecoder()
_TARGET_SPLIT_COUNT_KEYS = {
    "temporal": "target_temporal",
    "heldout_recording": "target_heldout_recording",
    "heldout_game": "target_heldout_game",
}


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


def _decode_summary_record_list(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("recordings", [])
    if not isinstance(rows, list):
        raise ValueError(f"decode summary has no recordings list: {path}")
    return [row for row in rows if isinstance(row, dict)]


def _decode_summary_records(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = _decode_summary_record_list(path)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        keys = [
            str(row.get("universe_row_id") or ""),
            f"{row.get('source_id')}:{row.get('cross_resolution_key')}",
            f"{row.get('source_id')}:{row.get('game')}/{row.get('recording_id')}",
        ]
        for key in keys:
            if key and "None" not in key:
                indexed[key] = row
    return indexed


def _target_tags_and_count_from_split_counts(
    split_counts: dict[str, Any],
    *,
    wanted_tags: set[str],
) -> tuple[list[str], int]:
    selected: list[tuple[str, int]] = []
    for tag, key in _TARGET_SPLIT_COUNT_KEYS.items():
        if wanted_tags and tag not in wanted_tags:
            continue
        count = int(split_counts.get(key, 0) or 0)
        if count > 0:
            selected.append((tag, count))
    if not selected:
        return [], 0
    # target_all_eval is the union of target_* split memberships. For heldout
    # recordings/games, the full-record split contains the temporal tail too.
    return [tag for tag, _count in selected], max(count for _tag, count in selected)


def _manifest_row_from_decode_record(
    *,
    key: str,
    decoded: dict[str, Any],
    row_count: int,
    split_tags: Sequence[str],
    output_root: Path,
    model: str,
    timestamp_min_ns: int | None = None,
    timestamp_max_ns: int | None = None,
    bin_index_min: int | None = None,
    bin_index_max: int | None = None,
) -> dict[str, Any]:
    safe = _safe_name(key)
    pred_mcap = output_root / "predicted_mcap" / f"{safe}.mcap"
    return {
        "universe_row_id": key,
        "source_id": decoded.get("source_id"),
        "cross_resolution_key": decoded.get("cross_resolution_key"),
        "game": decoded.get("game"),
        "recording_id": decoded.get("source_recording_id") or decoded.get("recording_id"),
        "row_count": int(row_count),
        "split_tags": sorted(str(tag) for tag in split_tags),
        "timestamp_min_ns": timestamp_min_ns,
        "timestamp_max_ns": timestamp_max_ns,
        "bin_index_min": bin_index_min,
        "bin_index_max": bin_index_max,
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


def build_gidm_inference_manifest(
    *,
    target_record_paths: Sequence[str | Path] | None = None,
    decode_summary_path: str | Path,
    output_dir: str | Path,
    model: str = "open-world-agents/Generalist-IDM-1B",
    split_tags: Sequence[str] | None = None,
    max_recordings: int | None = None,
    use_decode_summary_counts: bool = False,
) -> dict[str, Any]:
    """Build a per-recording manifest for upstream D2E Generalist-IDM inference."""

    wanted_tags = {str(tag) for tag in split_tags or []}
    output_root = Path(output_dir)
    manifest_rows: list[dict[str, Any]] = []
    missing_decode_rows: list[str] = []
    target_record_path_strings = [str(path) for path in target_record_paths or []]

    if use_decode_summary_counts:
        for decoded in sorted(
            _decode_summary_record_list(decode_summary_path),
            key=lambda row: str(row.get("universe_row_id") or ""),
        ):
            split_count_tags, row_count = _target_tags_and_count_from_split_counts(
                decoded.get("split_counts", {}) if isinstance(decoded.get("split_counts"), dict) else {},
                wanted_tags=wanted_tags,
            )
            if row_count <= 0:
                continue
            key = str(decoded.get("universe_row_id") or f"{decoded.get('source_id')}:{decoded.get('cross_resolution_key')}")
            manifest_rows.append(
                _manifest_row_from_decode_record(
                    key=key,
                    decoded=decoded,
                    row_count=row_count,
                    split_tags=split_count_tags,
                    output_root=output_root,
                    model=model,
                )
            )
            if max_recordings is not None and len(manifest_rows) >= int(max_recordings):
                break
    else:
        if not target_record_paths:
            raise ValueError("target_record_paths is required unless use_decode_summary_counts=True")
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
        for key, row in sorted(by_recording.items(), key=lambda item: item[0]):
            decoded = decode_records.get(key)
            if decoded is None:
                missing_decode_rows.append(key)
                continue
            manifest_rows.append(
                _manifest_row_from_decode_record(
                    key=key,
                    decoded={
                        **decoded,
                        **{
                            k: v
                            for k, v in row.items()
                            if k in ("source_id", "cross_resolution_key", "game", "recording_id")
                        },
                    },
                    row_count=int(row["row_count"]),
                    split_tags=sorted(row["split_tags"]),
                    output_root=output_root,
                    model=model,
                    timestamp_min_ns=row["timestamp_min_ns"],
                    timestamp_max_ns=row["timestamp_max_ns"],
                    bin_index_min=row["bin_index_min"],
                    bin_index_max=row["bin_index_max"],
                )
            )
            if max_recordings is not None and len(manifest_rows) >= int(max_recordings):
                break

    payload = {
        "schema": "gidm_inference_manifest.v1",
        "model": model,
        "target_record_paths": target_record_path_strings,
        "decode_summary_path": str(decode_summary_path),
        "output_dir": str(output_root),
        "source_mode": "decode_summary_split_counts" if use_decode_summary_counts else "target_record_scan",
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


def _first_screen_timestamp_ns(
    *,
    mcap_path: str | Path,
    decode_fn: Callable[..., list[dict[str, Any]]],
) -> int:
    rows = decode_fn(str(mcap_path), topics=["screen"], limit=1)
    for row in rows:
        if row.get("type") == "screen" and row.get("timestamp_ns") is not None:
            return int(row["timestamp_ns"])
    raise ValueError(f"no screen timestamp found in MCAP: {mcap_path}")


def _prediction_paths(row: dict[str, Any]) -> list[str]:
    paths = row.get("prediction_mcap_paths")
    if isinstance(paths, list) and paths:
        return [str(path) for path in paths]
    path = row.get("prediction_mcap_path")
    return [str(path)] if path else []


def _prediction_paths_exist(row: dict[str, Any]) -> bool:
    paths = _prediction_paths(row)
    return bool(paths) and all(Path(path).exists() and Path(path).stat().st_size > 0 for path in paths)


def convert_gidm_mcap_predictions(
    *,
    manifest_path: str | Path,
    target_record_paths: Sequence[str | Path],
    output_path: str | Path,
    summary_out: str | Path | None = None,
    bin_ms: int = 50,
    timestamp_shift_ns: int = 0,
    auto_timestamp_shift_from_screen: bool = False,
    filter_targets_to_prediction_windows: bool = False,
    allow_missing: bool = False,
    decode_fn: Callable[..., list[dict[str, Any]]] = decode_mcap_events,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    manifest_rows = manifest.get("recordings", [])
    by_key = {str(row["universe_row_id"]): row for row in manifest_rows if isinstance(row, dict)}
    available = {
        key: row
        for key, row in by_key.items()
        if _prediction_paths_exist(row)
    }
    if not allow_missing:
        missing = sorted(set(by_key) - set(available))
        if missing:
            raise FileNotFoundError(f"missing predicted MCAP files for {len(missing)} recording(s), first={missing[0]}")
    windows_by_key: dict[str, list[tuple[int, int]]] = {}
    if filter_targets_to_prediction_windows:
        missing_windows = []
        for key, row in available.items():
            windows = prediction_windows_ns(row, bin_ms=bin_ms)
            if not windows:
                missing_windows.append(key)
            else:
                windows_by_key[key] = windows
        if missing_windows:
            raise ValueError(
                "filter_targets_to_prediction_windows requested but prediction windows are missing "
                f"for {len(missing_windows)} recording(s), first={missing_windows[0]}"
            )

    targets_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    filtered_out_by_key: dict[str, int] = defaultdict(int)
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
                if filter_targets_to_prediction_windows:
                    try:
                        timestamp_ns = int(row.get("timestamp_ns", 0))
                    except (TypeError, ValueError):
                        filtered_out_by_key[key] += 1
                        continue
                    if not any(start <= timestamp_ns < end for start, end in windows_by_key.get(key, [])):
                        filtered_out_by_key[key] += 1
                        continue
                targets_by_key[key].append(row)

    decoded_counts: dict[str, int] = {}
    decoded_counts_by_path: dict[str, dict[str, int]] = {}
    timestamp_shifts_by_key: dict[str, int] = {}
    auto_shift_skipped_by_key: dict[str, bool] = {}
    token_bins_by_key: dict[str, dict[int, list[str]]] = {}
    for key, target_rows in targets_by_key.items():
        prediction_paths = _prediction_paths(available[key])
        effective_shift_ns = int(timestamp_shift_ns)
        timestamps_aligned = bool(available[key].get("prediction_timestamps_aligned_to_ground_truth"))
        if auto_timestamp_shift_from_screen and not timestamps_aligned:
            gt_mcap_path = available[key].get("ground_truth_mcap_path")
            if not gt_mcap_path:
                raise ValueError(f"manifest row lacks ground_truth_mcap_path for auto timestamp shift: {key}")
            effective_shift_ns += _first_screen_timestamp_ns(
                mcap_path=str(gt_mcap_path),
                decode_fn=decode_fn,
            ) - _first_screen_timestamp_ns(
                mcap_path=prediction_paths[0],
                decode_fn=decode_fn,
            )
        auto_shift_skipped_by_key[key] = bool(auto_timestamp_shift_from_screen and timestamps_aligned)
        timestamp_shifts_by_key[key] = effective_shift_ns
        events: list[dict[str, Any]] = []
        decoded_counts_by_path[key] = {}
        for mcap_path in prediction_paths:
            decoded = decode_fn(mcap_path, topics=["keyboard", "mouse/raw", "mouse"])
            decoded_counts_by_path[key][mcap_path] = len(decoded)
            events.extend(decoded)
        decoded_counts[key] = len(events)
        target_rows.sort(key=lambda row: int(row.get("timestamp_ns", 0)))
        token_bins_by_key[key] = _tokens_by_bin(
            events,
            target_timestamps=[int(row["timestamp_ns"]) for row in target_rows],
            bin_ms=bin_ms,
            timestamp_shift_ns=effective_shift_ns,
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
        "decoded_event_counts_by_path": decoded_counts_by_path,
        "missing_prediction_count": len(set(by_key) - set(available)),
        "allow_missing": bool(allow_missing),
        "bin_ms": int(bin_ms),
        "timestamp_shift_ns": int(timestamp_shift_ns),
        "auto_timestamp_shift_from_screen": bool(auto_timestamp_shift_from_screen),
        "filter_targets_to_prediction_windows": bool(filter_targets_to_prediction_windows),
        "prediction_window_counts": {key: len(value) for key, value in windows_by_key.items()},
        "filtered_out_target_rows_by_key": dict(filtered_out_by_key),
        "timestamp_shifts_by_key": timestamp_shifts_by_key,
        "auto_shift_skipped_by_key": auto_shift_skipped_by_key,
        "claim_boundary": "This converts released G-IDM MCAP predictions to the local predictions.jsonl metric boundary; metric claims require a separate paper-metric/audit artifact.",
    }
    if summary_out:
        write_json(summary_out, payload)
    return payload
