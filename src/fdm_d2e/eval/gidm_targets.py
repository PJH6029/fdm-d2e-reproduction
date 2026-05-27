from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Sequence

from fdm_d2e.io_utils import ensure_dir, read_json, sha256_file, stable_hash_json, write_json

TARGET_SPLIT_TAGS = ("temporal", "heldout_recording", "heldout_game")
TARGET_FIELDS = (
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
    "ground_truth_tokens",
)


def _iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_no}")
            yield payload


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


def _record_key(row: dict[str, Any]) -> str:
    return str(row.get("universe_row_id") or f"{row.get('source_id')}:{row.get('cross_resolution_key')}")


def _expand_roots(patterns: Sequence[str | Path]) -> list[Path]:
    roots: list[Path] = []
    for pattern in patterns:
        text = str(pattern)
        matches = sorted(glob.glob(text))
        if matches:
            roots.extend(Path(match) for match in matches)
            continue
        path = Path(text)
        if path.exists():
            roots.append(path)
    return roots


def _prediction_exists(row: dict[str, Any]) -> bool:
    paths = row.get("prediction_mcap_paths")
    if isinstance(paths, list) and paths:
        outputs = [Path(str(path)) for path in paths]
        return all(output.exists() and output.stat().st_size > 0 for output in outputs)
    path = row.get("prediction_mcap_path")
    if not path:
        return False
    output = Path(str(path))
    return output.exists() and output.stat().st_size > 0


def _records_path_for_row(row: dict[str, Any], roots: Sequence[Path]) -> Path | None:
    source_id = str(row.get("source_id") or "")
    game = str(row.get("game") or "")
    recording_id = str(row.get("recording_id") or row.get("source_recording_id") or "")
    if not source_id or not game or not recording_id:
        return None
    suffix = Path(source_id) / game / recording_id / "all_records.jsonl"
    for root in roots:
        candidate = root / suffix
        if candidate.exists():
            return candidate
    return None


def extract_gidm_target_records(
    *,
    manifest_path: str | Path,
    by_recording_roots: Sequence[str | Path],
    output_path: str | Path,
    summary_out: str | Path | None = None,
    recording_keys: Sequence[str] | None = None,
    split_tags: Sequence[str] = TARGET_SPLIT_TAGS,
    only_existing_predictions: bool = False,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    wanted = {str(key) for key in (recording_keys or [])}
    wanted_tags = {str(tag) for tag in split_tags}
    roots = _expand_roots(by_recording_roots)
    if not roots:
        raise FileNotFoundError(f"no by_recording roots matched: {[str(path) for path in by_recording_roots]}")

    manifest_rows = []
    for row in manifest.get("recordings", []):
        if not isinstance(row, dict):
            continue
        key = str(row.get("universe_row_id") or "")
        if wanted and key not in wanted:
            continue
        if only_existing_predictions and not _prediction_exists(row):
            continue
        manifest_rows.append(row)
    manifest_rows.sort(key=lambda item: str(item.get("universe_row_id") or ""))

    output = Path(output_path)
    ensure_dir(output.parent)
    findings: list[dict[str, Any]] = []
    per_recording: list[dict[str, Any]] = []
    rows_written = 0
    with output.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        for manifest_row in manifest_rows:
            key = str(manifest_row.get("universe_row_id") or "")
            records_path = _records_path_for_row(manifest_row, roots)
            if records_path is None:
                findings.append({"severity": "error", "code": "missing_by_recording_records", "universe_row_id": key})
                per_recording.append({"universe_row_id": key, "status": "missing_records", "rows_written": 0})
                continue
            selected = []
            for row in _iter_jsonl(records_path):
                tags = _split_tags(row)
                if wanted_tags and not (wanted_tags & set(tags)):
                    continue
                compact = {field: row.get(field) for field in TARGET_FIELDS if field in row}
                compact["eval_split_tags"] = tags
                selected.append(compact)
            selected.sort(key=lambda item: (int(item.get("timestamp_ns", 0)), str(item.get("sequence_id") or "")))
            for compact in selected:
                handle.write(json.dumps(compact, sort_keys=True) + "\n")
                rows_written += 1
            expected = int(manifest_row.get("row_count", 0) or 0)
            if expected and expected != len(selected):
                findings.append(
                    {
                        "severity": "error",
                        "code": "manifest_row_count_mismatch",
                        "universe_row_id": key,
                        "expected": expected,
                        "actual": len(selected),
                    }
                )
            per_recording.append(
                {
                    "universe_row_id": key,
                    "status": "pass" if expected == 0 or expected == len(selected) else "row_count_mismatch",
                    "records_path": str(records_path),
                    "expected_rows": expected,
                    "rows_written": len(selected),
                    "split_tags": sorted({tag for row in selected for tag in _split_tags(row)}),
                }
            )

    errors = [item for item in findings if item.get("severity") == "error"]
    payload = {
        "schema": "gidm_target_records_extraction.v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "manifest_path": str(manifest_path),
        "by_recording_roots": [str(path) for path in by_recording_roots],
        "expanded_root_count": len(roots),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output) if output.exists() else None,
        "recording_count": len(manifest_rows),
        "rows_written": rows_written,
        "split_tags": sorted(wanted_tags),
        "only_existing_predictions": bool(only_existing_predictions),
        "recording_keys": [str(key) for key in (recording_keys or [])],
        "per_recording": per_recording,
        "findings": findings,
        "manifest_fingerprint": stable_hash_json(
            {
                "manifest_path": str(manifest_path),
                "recordings": [
                    {
                        "universe_row_id": row.get("universe_row_id"),
                        "row_count": row.get("row_count"),
                        "prediction_mcap_path": row.get("prediction_mcap_path"),
                    }
                    for row in manifest_rows
                ],
                "split_tags": sorted(wanted_tags),
            }
        ),
        "claim_boundary": "This extracts exact target rows for released G-IDM conversion/metric evaluation; it is not a model-quality claim by itself.",
    }
    if summary_out:
        write_json(summary_out, payload)
    return payload


def enrich_gidm_manifest_with_target_timing(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    by_recording_roots: Sequence[str | Path] = (),
    target_record_paths: Sequence[str | Path] = (),
    summary_out: str | Path | None = None,
    recording_keys: Sequence[str] | None = None,
    split_tags: Sequence[str] = TARGET_SPLIT_TAGS,
    max_recordings: int | None = None,
) -> dict[str, Any]:
    """Add target split timing ranges to a released G-IDM inference manifest.

    Decode-summary-derived manifests know how many target rows a recording has,
    but not where heldout rows occur in the original video. Chunked G-IDM
    inference needs `bin_index_min/max` plus `timestamp_min/max` so chunks start
    near the actual heldout tail instead of replaying from video time zero.
    """

    manifest = read_json(manifest_path)
    wanted = {str(key) for key in (recording_keys or [])}
    wanted_tags = {str(tag) for tag in split_tags}
    roots = _expand_roots(by_recording_roots)
    if not roots and not target_record_paths:
        raise FileNotFoundError(
            "either by_recording_roots must match at least one root or target_record_paths must be provided"
        )

    target_rows_by_key: dict[str, list[dict[str, Any]]] = {}
    if target_record_paths:
        for path in target_record_paths:
            for target_row in _iter_jsonl(path):
                target_rows_by_key.setdefault(_record_key(target_row), []).append(target_row)

    findings: list[dict[str, Any]] = []
    updated_recordings: list[dict[str, Any]] = []
    selected_count = 0
    for raw_row in manifest.get("recordings", []):
        if not isinstance(raw_row, dict):
            continue
        row = dict(raw_row)
        key = str(row.get("universe_row_id") or "")
        if wanted and key not in wanted:
            continue
        if max_recordings is not None and selected_count >= int(max_recordings):
            continue
        selected_count += 1
        records_path = _records_path_for_row(row, roots) if roots else None
        target_iterable: Sequence[dict[str, Any]]
        timing_source: str
        if key in target_rows_by_key:
            target_iterable = target_rows_by_key[key]
            timing_source = "target_record_scan"
        elif records_path is not None:
            target_iterable = list(_iter_jsonl(records_path))
            timing_source = "by_recording_scan"
        else:
            findings.append({"severity": "error", "code": "missing_by_recording_records", "universe_row_id": key})
            updated_recordings.append(row)
            continue
        row_count = 0
        timestamp_min_ns: int | None = None
        timestamp_max_ns: int | None = None
        bin_index_min: int | None = None
        bin_index_max: int | None = None
        tags_seen: set[str] = set()
        for target in target_iterable:
            tags = set(_split_tags(target))
            if wanted_tags and not (wanted_tags & tags):
                continue
            try:
                timestamp_ns = int(target.get("timestamp_ns", 0))
                bin_index = int(target.get("bin_index", 0))
            except (TypeError, ValueError):
                findings.append({"severity": "warning", "code": "invalid_target_timing_row", "universe_row_id": key})
                continue
            row_count += 1
            tags_seen.update(tags)
            timestamp_min_ns = timestamp_ns if timestamp_min_ns is None else min(timestamp_min_ns, timestamp_ns)
            timestamp_max_ns = timestamp_ns if timestamp_max_ns is None else max(timestamp_max_ns, timestamp_ns)
            bin_index_min = bin_index if bin_index_min is None else min(bin_index_min, bin_index)
            bin_index_max = bin_index if bin_index_max is None else max(bin_index_max, bin_index)
        if row_count <= 0:
            findings.append({"severity": "error", "code": "no_target_rows_for_timing", "universe_row_id": key})
        else:
            row.update(
                {
                    "row_count": row_count,
                    "timestamp_min_ns": timestamp_min_ns,
                    "timestamp_max_ns": timestamp_max_ns,
                    "bin_index_min": bin_index_min,
                    "bin_index_max": bin_index_max,
                    "split_tags": sorted(tags_seen & wanted_tags) if wanted_tags else sorted(tags_seen),
                    "target_timing_records_path": str(records_path) if records_path is not None else None,
                    "target_timing_source": timing_source,
                }
            )
        updated_recordings.append(row)

    output_manifest = dict(manifest)
    output_manifest["recordings"] = updated_recordings
    output_manifest["recording_count"] = len(updated_recordings)
    output_manifest["target_rows"] = sum(int(row.get("row_count", 0) or 0) for row in updated_recordings)
    output_manifest["source_manifest_path"] = str(manifest_path)
    output_manifest["target_timing_enriched"] = True
    output_manifest["target_timing_split_tags"] = sorted(wanted_tags)
    output_manifest["manifest_fingerprint"] = stable_hash_json(
        {
            "source_manifest_path": str(manifest_path),
            "recordings": [
                {
                    "universe_row_id": row.get("universe_row_id"),
                    "row_count": row.get("row_count"),
                    "timestamp_min_ns": row.get("timestamp_min_ns"),
                    "timestamp_max_ns": row.get("timestamp_max_ns"),
                    "bin_index_min": row.get("bin_index_min"),
                    "bin_index_max": row.get("bin_index_max"),
                    "prediction_mcap_path": row.get("prediction_mcap_path"),
                }
                for row in updated_recordings
            ],
            "split_tags": sorted(wanted_tags),
        }
    )
    output_manifest["claim_boundary"] = (
        "Timing-enriched released G-IDM inference manifest; supports chunk scheduling only and is not a model-quality claim."
    )
    write_json(output_path, output_manifest)

    errors = [item for item in findings if item.get("severity") == "error"]
    payload = {
        "schema": "gidm_manifest_target_timing_enrichment.v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "warning_count": sum(1 for item in findings if item.get("severity") == "warning"),
        "manifest_path": str(manifest_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "by_recording_roots": [str(path) for path in by_recording_roots],
        "target_record_paths": [str(path) for path in target_record_paths],
        "expanded_root_count": len(roots),
        "recording_count": len(updated_recordings),
        "target_rows": output_manifest["target_rows"],
        "split_tags": sorted(wanted_tags),
        "recording_keys": [str(key) for key in (recording_keys or [])],
        "max_recordings": max_recordings,
        "per_recording": [
            {
                "universe_row_id": row.get("universe_row_id"),
                "row_count": row.get("row_count"),
                "timestamp_min_ns": row.get("timestamp_min_ns"),
                "timestamp_max_ns": row.get("timestamp_max_ns"),
                "bin_index_min": row.get("bin_index_min"),
                "bin_index_max": row.get("bin_index_max"),
                "split_tags": row.get("split_tags"),
                "target_timing_records_path": row.get("target_timing_records_path"),
            }
            for row in updated_recordings
        ],
        "findings": findings,
        "claim_boundary": "Target timing enrichment evidence only; metric claims require inference/conversion/metrics artifacts.",
    }
    if summary_out:
        write_json(summary_out, payload)
    return payload
