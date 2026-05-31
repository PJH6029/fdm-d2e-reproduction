from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.io_utils import ensure_dir, sha256_file, stable_hash_json, write_json, write_jsonl
from fdm_d2e.tokenization.fdm1_actions import ActionSlotTokenizer, MouseMoveBinner, next_click_position_targets, summarize_slot_overflow


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _row_start_ns(row: dict[str, Any], *, bin_ms: int) -> int:
    if row.get("start_ns") is not None:
        return _as_int(row.get("start_ns"))
    if row.get("timestamp_ns") is not None:
        return _as_int(row.get("timestamp_ns"))
    return _as_int(row.get("bin_index")) * int(bin_ms) * 1_000_000


def _events_for_click_targets(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"events": list(row.get("events", []) or [])} for row in records]


def _features_available(frame: dict[str, Any]) -> bool:
    return bool(frame.get("features") or frame.get("grid8") or frame.get("luma16"))


def _video_bin(row: dict[str, Any], *, frame_fps: int, bin_ms: int) -> dict[str, Any]:
    frame = row.get("frame") if isinstance(row.get("frame"), dict) else {}
    return {
        "schema": "fdm1_video_bin_reference.v1",
        "sample_policy": "center_frame_per_50ms_bin",
        "frame_fps": int(frame_fps),
        "bin_ms": int(bin_ms),
        "frame_index": _as_int(frame.get("index", row.get("bin_index", 0))),
        "frame_path": frame.get("path"),
        "features_available": _features_available(frame),
        "feature_fields": [key for key in ("features", "grid8", "luma16") if frame.get(key)],
    }


def materialize_action_slot_records(
    records: Sequence[dict[str, Any]],
    *,
    tokenizer: ActionSlotTokenizer | None = None,
    bin_ms: int = 50,
    frame_fps: int = 20,
    click_horizon_seconds: float = 1.0,
    click_grid: tuple[int, int] = (32, 18),
    screen_size: tuple[int, int] = (854, 480),
) -> list[dict[str, Any]]:
    """Convert D2E 50ms window records into FDM-1-style fixed action slots."""

    tokenizer = tokenizer or ActionSlotTokenizer(k_event_slots=8, bin_ms=bin_ms)
    horizon_bins = max(0, int(round(float(click_horizon_seconds) * 1000.0 / int(bin_ms))))
    click_targets = next_click_position_targets(
        _events_for_click_targets(records),
        horizon_bins=horizon_bins,
        grid_width=int(click_grid[0]),
        grid_height=int(click_grid[1]),
        screen_width=int(screen_size[0]),
        screen_height=int(screen_size[1]),
    )
    out: list[dict[str, Any]] = []
    bin_ns = int(bin_ms) * 1_000_000
    for idx, row in enumerate(records):
        start_ns = _row_start_ns(row, bin_ms=bin_ms)
        serialized = tokenizer.serialize_bin(list(row.get("events", []) or []))
        event_slots = list(serialized["event_slots"])
        packed = {
            "schema": "fdm1_action_slot_record.v1",
            "canonical_roadmap": "ROADMAP.md",
            "source_schema": row.get("schema"),
            "sequence_id": row.get("sequence_id", f"unknown#{idx:06d}"),
            "recording_id": row.get("recording_id"),
            "game": row.get("game"),
            "split": row.get("split"),
            "eval_split_tags": list(row.get("eval_split_tags", []) or []),
            "timestamp_ns": start_ns,
            "bin_index": _as_int(row.get("bin_index", idx)),
            "timebase": {"timestamp_unit": "nanoseconds", "bin_ms": int(bin_ms), "frame_fps": int(frame_fps), "start_ns": start_ns, "end_ns": start_ns + bin_ns},
            "video_bin": _video_bin(row, frame_fps=frame_fps, bin_ms=bin_ms),
            "action_tokens": list(serialized["action_tokens"]),
            "movement_token_count": int(serialized["movement_token_count"]),
            "event_slots": event_slots,
            "idm_masked_action_tokens": tokenizer.mask_for_idm(serialized),
            "click_position_target": click_targets[idx] if idx < len(click_targets) else "NO_CLICK_WITHIN_H",
            "mouse_dx_sum": serialized["mouse_dx_sum"],
            "mouse_dy_sum": serialized["mouse_dy_sum"],
            "source_event_count": len(row.get("events", []) or []),
            "discrete_event_count": int(serialized["discrete_event_count"]),
            "overflow_count": int(serialized["overflow_count"]),
            "source": "fdm1_action_slot_materializer",
        }
        for key in ("source_id", "resolution_tier", "source_recording_key", "cross_resolution_key", "universe_row_id"):
            if key in row:
                packed[key] = row[key]
        out.append(packed)
    return out


def split_action_slot_records(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "all": list(records),
        "train_core": [row for row in records if row.get("split") == "train_core"],
        "target_temporal": [row for row in records if "temporal" in (row.get("eval_split_tags") or [])],
        "target_heldout_recording": [row for row in records if "heldout_recording" in (row.get("eval_split_tags") or [])],
        "target_heldout_game": [row for row in records if "heldout_game" in (row.get("eval_split_tags") or [])],
    }
    buckets["target_all_eval"] = [row for row in records if row.get("eval_split_tags")]
    return buckets


def build_alignment_summary(records: Sequence[dict[str, Any]], *, bin_ms: int = 50, frame_fps: int = 20) -> dict[str, Any]:
    bin_ns = int(bin_ms) * 1_000_000
    by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_outside = 0
    events_checked = 0
    frame_index_missing = 0
    frame_features_missing = 0
    for row in records:
        rid = str(row.get("recording_id", "UNKNOWN"))
        by_recording[rid].append(row)
        start = _row_start_ns(row, bin_ms=bin_ms)
        end = start + bin_ns
        frame = row.get("frame") if isinstance(row.get("frame"), dict) else {}
        if "index" not in frame:
            frame_index_missing += 1
        if not _features_available(frame):
            frame_features_missing += 1
        for event in row.get("events", []) or []:
            if event.get("type") == "screen" or event.get("timestamp_ns") is None:
                continue
            events_checked += 1
            ts = _as_int(event.get("timestamp_ns"), start)
            if ts < start or ts >= end:
                event_outside += 1
    bad_spacing = 0
    non_monotonic = 0
    for rows in by_recording.values():
        ordered = sorted(rows, key=lambda item: (_as_int(item.get("bin_index", 0)), _row_start_ns(item, bin_ms=bin_ms)))
        previous_start: int | None = None
        previous_bin: int | None = None
        for row in ordered:
            start = _row_start_ns(row, bin_ms=bin_ms)
            bin_index = _as_int(row.get("bin_index", 0))
            if previous_start is not None:
                if start <= previous_start:
                    non_monotonic += 1
                expected = previous_start + (bin_index - (previous_bin or 0)) * bin_ns
                if bin_index == (previous_bin or 0) + 1 and start != expected:
                    bad_spacing += 1
            previous_start = start
            previous_bin = bin_index
    errors = []
    if event_outside:
        errors.append(f"{event_outside} source events fell outside their 50ms bins")
    if non_monotonic:
        errors.append(f"{non_monotonic} non-monotonic per-recording timestamps")
    if bad_spacing:
        errors.append(f"{bad_spacing} adjacent bins did not advance by {bin_ms}ms")
    return {
        "schema": "fdm1_action_slot_alignment_summary.v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "record_count": len(records),
        "recording_count": len(by_recording),
        "bin_ms": int(bin_ms),
        "frame_fps": int(frame_fps),
        "expected_frame_interval_ms": 1000.0 / float(frame_fps) if frame_fps else None,
        "events_checked": events_checked,
        "event_outside_bin_count": event_outside,
        "non_monotonic_count": non_monotonic,
        "bad_bin_spacing_count": bad_spacing,
        "frame_index_missing_count": frame_index_missing,
        "frame_features_missing_count": frame_features_missing,
    }


def build_action_dataset_summary(
    records: Sequence[dict[str, Any]],
    *,
    source_paths: Sequence[str],
    output_paths: dict[str, str],
    tokenization_config_path: str | None,
    bin_ms: int,
    frame_fps: int,
    k_event_slots: int,
) -> dict[str, Any]:
    token_counter = Counter(token for row in records for token in row.get("action_tokens", []))
    split_counts = {name: len(rows) for name, rows in split_action_slot_records(records).items()}
    return {
        "schema": "fdm1_action_slot_dataset_summary.v1",
        "canonical_roadmap": "ROADMAP.md",
        "source_paths": list(source_paths),
        "source_hashes": {str(path): sha256_file(path) for path in source_paths if Path(path).exists()},
        "tokenization_config": tokenization_config_path,
        "tokenization_config_sha256": sha256_file(tokenization_config_path) if tokenization_config_path and Path(tokenization_config_path).exists() else None,
        "timebase": {"bin_ms": int(bin_ms), "frame_fps": int(frame_fps)},
        "k_event_slots": int(k_event_slots),
        "records": len(records),
        "split_counts": split_counts,
        "games": sorted({str(row.get("game")) for row in records if row.get("game") is not None}),
        "token_count": sum(token_counter.values()),
        "unique_token_count": len(token_counter),
        "top_tokens": token_counter.most_common(20),
        "output_paths": output_paths,
        "dataset_fingerprint": stable_hash_json(
            {
                "source_hashes": {str(path): sha256_file(path) for path in source_paths if Path(path).exists()},
                "records": len(records),
                "split_counts": split_counts,
                "first_sequences": [row.get("sequence_id") for row in records[:16]],
                "last_sequences": [row.get("sequence_id") for row in records[-16:]],
                "tokenization": {"bin_ms": int(bin_ms), "frame_fps": int(frame_fps), "k_event_slots": int(k_event_slots)},
            }
        ),
    }


def write_action_slot_dataset(
    records: Sequence[dict[str, Any]],
    *,
    output_dir: str | Path,
    source_paths: Sequence[str] = (),
    tokenization_config_path: str | None = None,
    tokenizer: ActionSlotTokenizer | None = None,
    bin_ms: int = 50,
    frame_fps: int = 20,
    click_horizon_seconds: float = 1.0,
    click_grid: tuple[int, int] = (32, 18),
    screen_size: tuple[int, int] = (854, 480),
) -> dict[str, Any]:
    tokenizer = tokenizer or ActionSlotTokenizer(k_event_slots=8, bin_ms=bin_ms)
    output_root = ensure_dir(output_dir)
    action_records = materialize_action_slot_records(
        records,
        tokenizer=tokenizer,
        bin_ms=bin_ms,
        frame_fps=frame_fps,
        click_horizon_seconds=click_horizon_seconds,
        click_grid=click_grid,
        screen_size=screen_size,
    )
    split_rows = split_action_slot_records(action_records)
    split_dir = ensure_dir(output_root / "splits")
    output_paths = {"all": str(output_root / "action_slots.jsonl")}
    write_jsonl(output_paths["all"], action_records)
    for name, rows in split_rows.items():
        if name == "all":
            continue
        path = split_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        output_paths[name] = str(path)
    alignment = build_alignment_summary(records, bin_ms=bin_ms, frame_fps=frame_fps)
    overflow = summarize_slot_overflow(action_records, by_game=[str(row.get("game", "UNKNOWN")) for row in action_records])
    summary = build_action_dataset_summary(
        action_records,
        source_paths=source_paths,
        output_paths=output_paths,
        tokenization_config_path=tokenization_config_path,
        bin_ms=bin_ms,
        frame_fps=frame_fps,
        k_event_slots=tokenizer.k_event_slots,
    )
    sequence_pack = {
        "schema": "fdm1_action_sequence_pack.v1",
        "canonical_roadmap": "ROADMAP.md",
        "dataset_fingerprint": summary["dataset_fingerprint"],
        "timebase": summary["timebase"],
        "tokenization": {
            "config_path": tokenization_config_path,
            "k_event_slots": tokenizer.k_event_slots,
            "mouse_boundaries": list(tokenizer.mouse_binner.boundaries),
            "mouse_compound": tokenizer.mouse_binner.compound,
        },
        "counts": summary["split_counts"],
        "paths": output_paths,
        "sample_records": action_records[: min(3, len(action_records))],
    }
    write_json(output_root / "alignment_summary.json", alignment)
    write_json(output_root / "overflow_summary.json", overflow)
    write_json(output_root / "dataset_summary.json", summary)
    write_json(output_root / "sequence_pack.json", sequence_pack)
    return {"records": action_records, "alignment": alignment, "overflow": overflow, "summary": summary, "sequence_pack": sequence_pack}


__all__ = [
    "build_alignment_summary",
    "materialize_action_slot_records",
    "split_action_slot_records",
    "write_action_slot_dataset",
]
