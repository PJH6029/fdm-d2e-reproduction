from __future__ import annotations

import hashlib
import json
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



def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _empty_output_paths(output_root: Path) -> dict[str, str]:
    split_dir = ensure_dir(output_root / "splits")
    paths = {"all": str(output_root / "action_slots.jsonl")}
    for name in ("train_core", "target_temporal", "target_heldout_recording", "target_heldout_game", "target_all_eval"):
        paths[name] = str(split_dir / f"{name}.jsonl")
    for path in paths.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("", encoding="utf-8")
    return paths


def _merge_alignment(total: dict[str, Any], batch: dict[str, Any]) -> None:
    total["record_count"] += int(batch.get("record_count", 0))
    total["recording_count"] += int(batch.get("recording_count", 0))
    total["events_checked"] += int(batch.get("events_checked", 0))
    total["event_outside_bin_count"] += int(batch.get("event_outside_bin_count", 0))
    total["non_monotonic_count"] += int(batch.get("non_monotonic_count", 0))
    total["bad_bin_spacing_count"] += int(batch.get("bad_bin_spacing_count", 0))
    total["frame_index_missing_count"] += int(batch.get("frame_index_missing_count", 0))
    total["frame_features_missing_count"] += int(batch.get("frame_features_missing_count", 0))
    total["errors"].extend(batch.get("errors", []))


def _iter_jsonl_rows(path: str | Path, *, digest: Any | None = None) -> Iterable[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            if digest is not None:
                digest.update(raw_line)
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            yield row


def _iter_recording_batches(input_paths: Sequence[str | Path], *, source_digests: dict[str, Any] | None = None) -> Iterable[list[dict[str, Any]]]:
    """Yield consecutive recording groups without loading a full corpus JSONL."""

    current_key: str | None = None
    current_rows: list[dict[str, Any]] = []
    for path in input_paths:
        digest = source_digests.get(str(path)) if source_digests is not None else None
        for row in _iter_jsonl_rows(path, digest=digest):
            key = str(row.get("recording_id", row.get("source_recording_key", "UNKNOWN")))
            if current_key is None:
                current_key = key
            if key != current_key and current_rows:
                yield current_rows
                current_rows = []
                current_key = key
            current_rows.append(row)
    if current_rows:
        yield current_rows


def write_action_slot_dataset_streaming_from_jsonl(
    input_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    tokenization_config_path: str | None = None,
    tokenizer: ActionSlotTokenizer | None = None,
    bin_ms: int = 50,
    frame_fps: int = 20,
    click_horizon_seconds: float = 1.0,
    click_grid: tuple[int, int] = (32, 18),
    screen_size: tuple[int, int] = (854, 480),
    max_records: int | None = None,
) -> dict[str, Any]:
    """Stream materialization from decoded D2E JSONL grouped by recording.

    Full D2E-480p materialization can contain tens of millions of 50ms rows.
    This writer only holds one consecutive recording group at a time, preserving
    next-click horizon targets within each recording while avoiding all-corpus
    memory pressure.
    """

    tokenizer = tokenizer or ActionSlotTokenizer(k_event_slots=8, bin_ms=bin_ms)
    output_root = ensure_dir(output_dir)
    output_paths = _empty_output_paths(output_root)
    token_counter: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    games: set[str] = set()
    first_samples: list[dict[str, Any]] = []
    first_sequences: list[Any] = []
    last_sequences: list[Any] = []
    source_paths = [str(path) for path in input_paths]
    source_digests = {str(path): hashlib.sha256() for path in source_paths if Path(path).exists()}
    alignment_total: dict[str, Any] = {
        "schema": "fdm1_action_slot_alignment_summary.v1",
        "status": "pass",
        "errors": [],
        "record_count": 0,
        "recording_count": 0,
        "bin_ms": int(bin_ms),
        "frame_fps": int(frame_fps),
        "expected_frame_interval_ms": 1000.0 / float(frame_fps) if frame_fps else None,
        "events_checked": 0,
        "event_outside_bin_count": 0,
        "non_monotonic_count": 0,
        "bad_bin_spacing_count": 0,
        "frame_index_missing_count": 0,
        "frame_features_missing_count": 0,
    }
    overflow_bins = 0
    overflow_events = 0
    per_game_overflow: dict[str, Counter[str]] = defaultdict(Counter)
    records_written = 0
    batches = 0
    stop = False
    for batch in _iter_recording_batches(input_paths, source_digests=source_digests):
        if max_records is not None:
            remaining = int(max_records) - records_written
            if remaining <= 0:
                break
            if len(batch) > remaining:
                batch = batch[:remaining]
                stop = True
        batches += 1
        action_records = materialize_action_slot_records(
            batch,
            tokenizer=tokenizer,
            bin_ms=bin_ms,
            frame_fps=frame_fps,
            click_horizon_seconds=click_horizon_seconds,
            click_grid=click_grid,
            screen_size=screen_size,
        )
        split_rows = split_action_slot_records(action_records)
        _append_jsonl(Path(output_paths["all"]), action_records)
        for name, rows in split_rows.items():
            split_counts[name] += len(rows)
            if name != "all":
                _append_jsonl(Path(output_paths[name]), rows)
        for row in action_records:
            token_counter.update(row.get("action_tokens", []))
            if row.get("game") is not None:
                games.add(str(row.get("game")))
            if len(first_samples) < 3:
                first_samples.append(row)
            if len(first_sequences) < 16:
                first_sequences.append(row.get("sequence_id"))
            last_sequences.append(row.get("sequence_id"))
            if len(last_sequences) > 16:
                last_sequences.pop(0)
        alignment_batch = build_alignment_summary(batch, bin_ms=bin_ms, frame_fps=frame_fps)
        _merge_alignment(alignment_total, alignment_batch)
        overflow_batch = summarize_slot_overflow(action_records, by_game=[str(row.get("game", "UNKNOWN")) for row in action_records])
        overflow_bins += int(overflow_batch.get("overflow_bins", 0))
        overflow_events += int(overflow_batch.get("overflow_events", 0))
        for game, counts in overflow_batch.get("per_game", {}).items():
            for key, value in counts.items():
                per_game_overflow[str(game)][str(key)] += int(value)
        records_written += len(action_records)
        if stop:
            break
    alignment_total["status"] = "pass" if not alignment_total["errors"] else "fail"
    overflow = {
        "schema": "fdm1_action_slot_overflow_summary.v1",
        "bins": records_written,
        "overflow_bins": overflow_bins,
        "overflow_events": overflow_events,
        "overflow_rate": (overflow_bins / records_written) if records_written else 0.0,
        "per_game": {game: dict(counter) for game, counter in sorted(per_game_overflow.items())},
        "recommended_threshold": 0.001,
        "threshold_exceeded": (overflow_bins / records_written) > 0.001 if records_written else False,
    }
    split_counts.setdefault("all", records_written)
    source_hashes = {path: digest.hexdigest() for path, digest in source_digests.items()} if max_records is None else {}
    source_prefix_hashes: dict[str, str] = {}
    dataset_fingerprint = stable_hash_json(
        {
            "source_hashes": source_hashes,
            "source_prefix_hashes": source_prefix_hashes,
            "records": records_written,
            "split_counts": dict(split_counts),
            "first_sequences": first_sequences,
            "last_sequences": last_sequences,
            "tokenization": {"bin_ms": int(bin_ms), "frame_fps": int(frame_fps), "k_event_slots": int(tokenizer.k_event_slots)},
        }
    )
    summary = {
        "schema": "fdm1_action_slot_dataset_summary.v1",
        "canonical_roadmap": "ROADMAP.md",
        "streaming": True,
        "streaming_policy": "consecutive_recording_groups",
        "source_paths": source_paths,
        "source_hashes": source_hashes,
        "source_prefix_hashes": source_prefix_hashes,
        "source_hash_policy": "sha256_while_streaming_full_input" if max_records is None else "omitted_because_max_records_was_set",
        "tokenization_config": tokenization_config_path,
        "tokenization_config_sha256": sha256_file(tokenization_config_path) if tokenization_config_path and Path(tokenization_config_path).exists() else None,
        "timebase": {"bin_ms": int(bin_ms), "frame_fps": int(frame_fps)},
        "k_event_slots": int(tokenizer.k_event_slots),
        "records": records_written,
        "recording_batches": batches,
        "split_counts": dict(split_counts),
        "games": sorted(games),
        "token_count": sum(token_counter.values()),
        "unique_token_count": len(token_counter),
        "top_tokens": token_counter.most_common(20),
        "output_paths": output_paths,
        "dataset_fingerprint": dataset_fingerprint,
    }
    sequence_pack = {
        "schema": "fdm1_action_sequence_pack.v1",
        "canonical_roadmap": "ROADMAP.md",
        "dataset_fingerprint": dataset_fingerprint,
        "timebase": summary["timebase"],
        "streaming": True,
        "tokenization": {
            "config_path": tokenization_config_path,
            "k_event_slots": tokenizer.k_event_slots,
            "mouse_boundaries": list(tokenizer.mouse_binner.boundaries),
            "mouse_compound": tokenizer.mouse_binner.compound,
        },
        "counts": dict(split_counts),
        "paths": output_paths,
        "sample_records": first_samples,
    }
    write_json(output_root / "alignment_summary.json", alignment_total)
    write_json(output_root / "overflow_summary.json", overflow)
    write_json(output_root / "dataset_summary.json", summary)
    write_json(output_root / "sequence_pack.json", sequence_pack)
    return {"alignment": alignment_total, "overflow": overflow, "summary": summary, "sequence_pack": sequence_pack}


__all__ = [
    "build_alignment_summary",
    "materialize_action_slot_records",
    "split_action_slot_records",
    "write_action_slot_dataset",
    "write_action_slot_dataset_streaming_from_jsonl",
]
