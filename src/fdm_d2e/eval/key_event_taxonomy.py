from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.eval.paper_idm_metrics import _PaperMetricAccumulator
from fdm_d2e.eval.state_transition_diagnostics import (
    merge_motion_and_categorical,
    previous_motion_tokens,
    prior_state_sets,
    state_delta_tokens,
)
from fdm_d2e.io_utils import write_json

try:  # pragma: no cover
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _expand_paths(patterns: Sequence[str | Path]) -> list[Path]:
    out: list[Path] = []
    for item in patterns:
        matches = sorted(glob.glob(str(item)))
        if matches:
            out.extend(Path(match) for match in matches)
            continue
        path = Path(item)
        if path.exists():
            out.append(path)
    return out


def _iter_rows(paths: Sequence[str | Path], *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    rows = 0
    for path in _expand_paths(paths):
        with path.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
            for line_no, line in enumerate(handle, 1):
                if max_rows is not None and rows >= max_rows:
                    return
                if not line.strip():
                    continue
                try:
                    yield _loads(line)
                except Exception as exc:
                    raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc
                rows += 1


def _iter_rows_with_next(paths: Sequence[str | Path], *, max_rows: int | None = None) -> Iterable[tuple[dict[str, Any], dict[str, Any] | None]]:
    iterator = iter(_iter_rows(paths, max_rows=max_rows))
    try:
        current = next(iterator)
    except StopIteration:
        return
    for nxt in iterator:
        yield current, nxt
        current = nxt
    yield current, None


def _tokens(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None and key == "ground_truth_tokens":
        value = row.get("target_tokens")
    return [str(token) for token in value] if isinstance(value, list) else []


def _recording_key(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    for key in ("source_recording_key", "recording_id", "source_recording_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    sequence_id = row.get("sequence_id")
    if isinstance(sequence_id, str) and "#" in sequence_id:
        return sequence_id.rsplit("#", 1)[0]
    return str(sequence_id) if isinstance(sequence_id, str) and sequence_id else None


def _split_tags(row: dict[str, Any]) -> list[str]:
    value = row.get("eval_split_tags") or row.get("split_tags")
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    value = row.get("eval_split") or row.get("split") or row.get("split_name")
    return [str(value)] if isinstance(value, str) and value else []


def _key_event(token: str) -> tuple[str, str] | None:
    if token.startswith("KEY_PRESS_"):
        return ("press", token.removeprefix("KEY_PRESS_"))
    if token.startswith("KEY_RELEASE_"):
        return ("release", token.removeprefix("KEY_RELEASE_"))
    return None


def _key_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(token) for token in tokens if str(token).startswith("KEY_")]


def _paired_codes(tokens: Sequence[str]) -> set[str]:
    seen: dict[str, set[str]] = defaultdict(set)
    for token in _key_tokens(tokens):
        parsed = _key_event(token)
        if parsed is None:
            continue
        kind, code = parsed
        seen[code].add(kind)
    return {code for code, kinds in seen.items() if {"press", "release"}.issubset(kinds)}


def classify_key_token(row: dict[str, Any], next_row: dict[str, Any] | None, token: str) -> str:
    parsed = _key_event(token)
    if parsed is None:
        return "other_key_token"
    kind, code = parsed
    if next_row is None or _recording_key(row) != _recording_key(next_row):
        return "recording_boundary_unclassified"
    prior_keys, _prior_buttons = prior_state_sets(row)
    next_keys, _next_buttons = prior_state_sets(next_row)
    prior_has = code in prior_keys
    next_has = code in next_keys
    paired = code in _paired_codes(_tokens(row, "ground_truth_tokens"))
    if kind == "press":
        if not prior_has and next_has:
            return "visible_new_press"
        if not prior_has and not next_has and paired:
            return "same_bin_tap_press"
        if prior_has and next_has:
            return "held_repeat_press"
        if prior_has and not next_has and paired:
            return "press_then_release_while_held"
        return "press_other"
    if kind == "release":
        if prior_has and not next_has:
            return "visible_release"
        if not prior_has and not next_has and paired:
            return "same_bin_tap_release"
        if prior_has and next_has and paired:
            return "release_then_press_while_held"
        return "release_other"
    return "other_key_token"


def same_bin_extra_key_tokens(row: dict[str, Any], next_row: dict[str, Any] | None) -> list[str]:
    """Return current-row key tokens hidden from prior->next state differencing.

    This is an oracle-only helper: it inspects ground-truth key tokens from the
    current row to expose same-bin taps/repeats that leave little or no held-state
    footprint.  It must not be used as model evidence.
    """

    delta = Counter(state_delta_tokens(row, next_row))
    extras: list[str] = []
    for token in _key_tokens(_tokens(row, "ground_truth_tokens")):
        if delta[token] > 0:
            delta[token] -= 1
        else:
            extras.append(token)
    return extras


def _policy_metrics(accumulators: dict[str, _PaperMetricAccumulator]) -> dict[str, Any]:
    return {name: {"all": acc.metrics()} for name, acc in sorted(accumulators.items())}


def build_key_event_taxonomy(
    *,
    target_paths: Sequence[str | Path],
    max_rows: int | None = None,
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    split_category_counts: dict[str, Counter[str]] = {tag: Counter() for tag in split_tags}
    key_code_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    accs = {
        "previous_motion_only": _PaperMetricAccumulator(empty_bins_as_correct=False),
        "state_delta_only": _PaperMetricAccumulator(empty_bins_as_correct=False),
        "state_delta_plus_previous_motion": _PaperMetricAccumulator(empty_bins_as_correct=False),
        "state_delta_plus_same_bin_key_oracle_plus_previous_motion": _PaperMetricAccumulator(empty_bins_as_correct=False),
    }
    rows = 0
    key_rows = 0
    total_key_tokens = 0
    rows_with_same_bin_key_pair = 0
    rows_with_events = 0
    for row, next_row in _iter_rows_with_next(target_paths, max_rows=max_rows):
        rows += 1
        gt = _tokens(row, "ground_truth_tokens")
        key_tokens = _key_tokens(gt)
        tags = _split_tags(row)
        if row.get("events"):
            rows_with_events += 1
        if key_tokens:
            key_rows += 1
        paired = _paired_codes(gt)
        if paired:
            rows_with_same_bin_key_pair += 1
        state_delta = state_delta_tokens(row, next_row)
        motion = previous_motion_tokens(row)
        oracle_tokens = merge_motion_and_categorical(motion, state_delta + same_bin_extra_key_tokens(row, next_row))
        accs["previous_motion_only"].update(motion, gt)
        accs["state_delta_only"].update(state_delta, gt)
        accs["state_delta_plus_previous_motion"].update(merge_motion_and_categorical(motion, state_delta), gt)
        accs["state_delta_plus_same_bin_key_oracle_plus_previous_motion"].update(oracle_tokens, gt)
        for token in key_tokens:
            total_key_tokens += 1
            parsed = _key_event(token)
            if parsed:
                key_code_counts[parsed[1]] += 1
            category = classify_key_token(row, next_row, token)
            category_counts[category] += 1
            for tag in tags:
                if tag in split_category_counts:
                    split_category_counts[tag][category] += 1
            if len(examples[category]) < 3:
                examples[category].append(
                    {
                        "sequence_id": row.get("sequence_id"),
                        "recording_id": _recording_key(row),
                        "timestamp_ns": row.get("timestamp_ns"),
                        "token": token,
                        "ground_truth_tokens": gt,
                        "previous_event_tokens": row.get("previous_event_tokens"),
                        "prior_action_tokens": row.get("prior_action_tokens"),
                        "next_prior_action_tokens": (next_row or {}).get("prior_action_tokens"),
                        "eval_split_tags": tags,
                    }
                )
        row_counts.update({tag: 1 for tag in tags if tag in split_category_counts})
    visible_categories = {"visible_new_press", "visible_release"}
    hidden_categories = {
        "same_bin_tap_press",
        "same_bin_tap_release",
        "held_repeat_press",
        "press_then_release_while_held",
        "release_then_press_while_held",
    }
    visible = sum(category_counts[name] for name in visible_categories)
    hidden = sum(category_counts[name] for name in hidden_categories)
    return {
        "schema": "g005_key_event_taxonomy.v1",
        "status": "pass",
        "rows": rows,
        "key_rows": key_rows,
        "rows_with_events": rows_with_events,
        "rows_with_same_bin_key_pair": rows_with_same_bin_key_pair,
        "total_key_tokens": total_key_tokens,
        "state_transition_visible_key_tokens": visible,
        "hidden_or_repeat_key_tokens": hidden,
        "state_transition_visible_fraction": visible / total_key_tokens if total_key_tokens else None,
        "hidden_or_repeat_fraction": hidden / total_key_tokens if total_key_tokens else None,
        "category_counts": dict(category_counts),
        "split_category_counts": {tag: dict(counter) for tag, counter in split_category_counts.items()},
        "top_key_codes": key_code_counts.most_common(20),
        "policy_metrics": _policy_metrics(accs),
        "examples": dict(examples),
        "target_paths": [str(path) for path in target_paths],
        "max_rows": max_rows,
        "claim_boundary": "Diagnostic only. same_bin_key_oracle policies inspect current-row ground-truth key tokens and are not valid model evidence or G005 completion evidence.",
    }


def write_key_event_taxonomy(
    *,
    target_paths: Sequence[str | Path],
    output_path: str | Path,
    max_rows: int | None = None,
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
) -> dict[str, Any]:
    payload = build_key_event_taxonomy(target_paths=target_paths, max_rows=max_rows, split_tags=split_tags)
    write_json(output_path, payload)
    return payload
