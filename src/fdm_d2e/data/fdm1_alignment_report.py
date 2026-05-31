from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.io_utils import read_jsonl, write_json


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def sample_action_slot_rows(
    rows: Sequence[dict[str, Any]],
    *,
    max_rows: int = 24,
    recording_id: str | None = None,
    game: str | None = None,
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        if recording_id is not None and str(row.get("recording_id")) != str(recording_id):
            continue
        if game is not None and str(row.get("game")) != str(game):
            continue
        filtered.append(row)
        if len(filtered) >= max_rows:
            break
    return filtered


def audit_action_slot_alignment(rows: Sequence[dict[str, Any]], *, expected_bin_ms: int = 50) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    expected_ns = int(expected_bin_ms) * 1_000_000
    previous_ts: int | None = None
    previous_bin: int | None = None
    for idx, row in enumerate(rows):
        tokens = list(row.get("action_tokens", []) or [])
        mask = list(row.get("idm_masked_action_tokens", []) or [])
        movement_count = _as_int(row.get("movement_token_count", 1), 1)
        event_slots = list(row.get("event_slots", []) or [])
        if len(mask) != len(tokens):
            errors.append(f"row {idx} mask/token length mismatch: {len(mask)} != {len(tokens)}")
        if tokens[:movement_count] != mask[:movement_count]:
            errors.append(f"row {idx} IDM mask does not preserve movement token prefix")
        if len(tokens) != movement_count + len(event_slots):
            errors.append(f"row {idx} movement/event slot length mismatch")
        ts = _as_int(row.get("timestamp_ns", 0))
        bin_index = _as_int(row.get("bin_index", idx))
        if previous_ts is not None and previous_bin is not None:
            if ts <= previous_ts:
                errors.append(f"row {idx} timestamp is non-monotonic")
            if bin_index == previous_bin + 1 and ts - previous_ts != expected_ns:
                errors.append(f"row {idx} adjacent timestamp gap is {ts - previous_ts}ns, expected {expected_ns}ns")
        if not row.get("video_bin", {}).get("frame_path"):
            warnings.append(f"row {idx} lacks a frame path reference")
        previous_ts = ts
        previous_bin = bin_index
    return {
        "schema": "fdm1_action_alignment_visual_audit.v1",
        "status": "pass" if not errors else "fail",
        "row_count": len(rows),
        "expected_bin_ms": int(expected_bin_ms),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings[:50],
    }


def _token_summary(row: dict[str, Any]) -> str:
    tokens = list(row.get("action_tokens", []) or [])
    if not tokens:
        return ""
    movement = tokens[0]
    events = [token for token in row.get("event_slots", []) if token != "NO_ACTION"]
    return movement + ("; " + ", ".join(events) if events else "; NO_ACTION")


def render_alignment_markdown(rows: Sequence[dict[str, Any]], audit: dict[str, Any]) -> str:
    lines = [
        "# FDM-1 action-slot alignment visual check",
        "",
        f"**Status:** `{audit['status']}`  ",
        f"**Rows inspected:** `{audit['row_count']}`  ",
        f"**Expected bin size:** `{audit['expected_bin_ms']}ms`",
        "",
        "This report is a human-inspectable timeline of sampled 50ms bins. It is intended to verify that the representative video bin, action-token slots, IDM mask view, click auxiliary target, and overflow markers line up before expensive IDM/FDM training.",
        "",
        "## Timeline",
        "",
        "| # | sequence | t(ms) | frame | tokens | click target | overflow |",
        "| ---: | --- | ---: | --- | --- | --- | ---: |",
    ]
    if not rows:
        lines.append("| - | _no sampled rows_ | - | - | - | - | - |")
    base_ts = _as_int(rows[0].get("timestamp_ns", 0)) if rows else 0
    for idx, row in enumerate(rows):
        video = row.get("video_bin", {}) if isinstance(row.get("video_bin"), dict) else {}
        frame = f"{video.get('frame_index')} `{video.get('frame_path')}`"
        t_ms = (_as_int(row.get("timestamp_ns", 0)) - base_ts) / 1_000_000.0
        lines.append(
            f"| {idx} | `{row.get('sequence_id')}` | {t_ms:.1f} | {frame} | `{_token_summary(row)}` | `{row.get('click_position_target')}` | {row.get('overflow_count', 0)} |"
        )
    if audit.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {item}" for item in audit["errors"])
    if audit.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in audit["warnings"])
    lines.extend(["", "## Claim boundary", "", "This visual check validates sampled action-slot alignment only. It is not model training evidence and does not prove metric wins or FDM-1 parity."])
    return "\n".join(lines) + "\n"


def build_alignment_report(
    rows: Sequence[dict[str, Any]],
    *,
    markdown_path: str | Path,
    audit_path: str | Path,
    expected_bin_ms: int = 50,
) -> dict[str, Any]:
    audit = audit_action_slot_alignment(rows, expected_bin_ms=expected_bin_ms)
    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text(render_alignment_markdown(rows, audit), encoding="utf-8")
    write_json(audit_path, audit)
    return audit


def build_alignment_report_from_jsonl(
    action_slots_path: str | Path,
    *,
    markdown_path: str | Path,
    audit_path: str | Path,
    expected_bin_ms: int = 50,
    max_rows: int = 24,
    recording_id: str | None = None,
    game: str | None = None,
) -> dict[str, Any]:
    rows = sample_action_slot_rows(read_jsonl(action_slots_path), max_rows=max_rows, recording_id=recording_id, game=game)
    return build_alignment_report(rows, markdown_path=markdown_path, audit_path=audit_path, expected_bin_ms=expected_bin_ms)


__all__ = [
    "audit_action_slot_alignment",
    "build_alignment_report",
    "build_alignment_report_from_jsonl",
    "render_alignment_markdown",
    "sample_action_slot_rows",
]
