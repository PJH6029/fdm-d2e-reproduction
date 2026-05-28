from __future__ import annotations

import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.io_utils import write_json

try:  # pragma: no cover
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

_CONTEXTS = ("code_phase", "code_hold_phase", "code_holdbucket_phase", "code_holdmod_phase", "global_hold_phase")
_DEFAULT_PERIODS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40)
_DEFAULT_THRESHOLDS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.65)


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _expand_paths(patterns: Sequence[str | Path] | str | Path) -> list[Path]:
    items = [patterns] if isinstance(patterns, (str, Path)) else list(patterns)
    paths: list[Path] = []
    for item in items:
        matches = sorted(glob.glob(str(item)))
        if matches:
            paths.extend(Path(match) for match in matches)
            continue
        path = Path(item)
        if path.exists():
            paths.append(path)
    return paths


def _iter_rows(patterns: Sequence[str | Path] | str | Path, *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    rows = 0
    for path in _expand_paths(patterns):
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


def sequence_bin_index(row: dict[str, Any]) -> int:
    seq = str(row.get("sequence_id", ""))
    match = re.search(r"#(\d+)$", seq)
    if match:
        return int(match.group(1))
    try:
        return int(row.get("timestamp_ns") or 0) // 50_000_000
    except (TypeError, ValueError):
        return 0


def _holds(row: dict[str, Any]) -> dict[str, int]:
    value = row.get("prior_key_hold_bins")
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return out


def key_counter(tokens: Sequence[str]) -> Counter[str]:
    return Counter(str(token) for token in tokens if str(token).startswith("KEY_"))


def key_accuracy_counts(predicted: Counter[str], ground_truth: Counter[str]) -> tuple[int, int]:
    correct = 0
    total = 0
    for token in set(predicted) | set(ground_truth):
        total += 1
        correct += int(predicted.get(token, 0) == ground_truth.get(token, 0))
    return correct, total


def _contexts(row: dict[str, Any], code: str, hold: int, period: int) -> dict[str, tuple[Any, ...]]:
    phase = sequence_bin_index(row) % int(period)
    return {
        "code_phase": (code, phase),
        "code_hold_phase": (code, min(hold, 256), phase),
        "code_holdbucket_phase": (code, min(hold // 4, 64), phase),
        "code_holdmod_phase": (code, min(hold // 10, 32), hold % 10, phase),
        "global_hold_phase": (min(hold // 4, 64), phase),
    }


def build_key_phase_repeat_diagnostic(
    *,
    train_paths: Sequence[str | Path] | str | Path,
    target_paths: Sequence[str | Path] | str | Path,
    base_prediction_paths: Sequence[str | Path] | str | Path,
    max_train_rows: int | None = 320_000,
    max_target_rows: int | None = 50_000,
    periods: Sequence[int] = _DEFAULT_PERIODS,
    thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
    min_support: int = 5,
) -> dict[str, Any]:
    model_counts: dict[tuple[str, int], defaultdict[tuple[Any, ...], list[int]]] = {
        (name, int(period)): defaultdict(lambda: [0, 0]) for name in _CONTEXTS for period in periods
    }
    train_rows = 0
    for row in _iter_rows(train_paths, max_rows=max_train_rows):
        train_rows += 1
        gt = set(str(token) for token in row.get("ground_truth_tokens", []) or [])
        for code, hold in _holds(row).items():
            positive = int(f"KEY_PRESS_{code}" in gt)
            for period in periods:
                contexts = _contexts(row, code, hold, int(period))
                for name in _CONTEXTS:
                    rec = model_counts[(name, int(period))][contexts[name]]
                    rec[0] += 1
                    rec[1] += positive
    probs = {
        key: {context: positive / count for context, (count, positive) in counts.items() if count >= int(min_support)}
        for key, counts in model_counts.items()
    }
    policies = [(name, int(period), float(threshold)) for name in _CONTEXTS for period in periods for threshold in thresholds]
    correct = {policy: 0 for policy in policies}
    total = {policy: 0 for policy in policies}
    base_correct = 0
    base_total = 0
    rows = 0
    mismatches = 0
    examples: list[dict[str, Any]] = []
    base_iter = iter(_iter_rows(base_prediction_paths, max_rows=max_target_rows))
    for row in _iter_rows(target_paths, max_rows=max_target_rows):
        try:
            base = next(base_iter)
        except StopIteration:
            base = {}
        rows += 1
        if base.get("sequence_id") is not None and row.get("sequence_id") is not None and base.get("sequence_id") != row.get("sequence_id"):
            mismatches += 1
            if len(examples) < 20:
                examples.append({"prediction_sequence_id": base.get("sequence_id"), "target_sequence_id": row.get("sequence_id")})
        gt_keys = key_counter(row.get("ground_truth_tokens", []) or [])
        base_keys = key_counter(base.get("predicted_tokens", []) or [])
        c, t = key_accuracy_counts(base_keys, gt_keys)
        base_correct += c
        base_total += t
        holds = _holds(row)
        context_cache: dict[tuple[str, int, str, int], tuple[Any, ...]] = {}
        for policy in policies:
            name, period, threshold = policy
            pred = Counter(base_keys)
            for code, hold in holds.items():
                cache_key = (code, hold, name, period)
                context = context_cache.get(cache_key)
                if context is None:
                    context = _contexts(row, code, hold, period)[name]
                    context_cache[cache_key] = context
                if probs[(name, period)].get(context, 0.0) >= threshold:
                    pred[f"KEY_PRESS_{code}"] += 1
            c, t = key_accuracy_counts(pred, gt_keys)
            correct[policy] += c
            total[policy] += t
    ranked = sorted(
        (
            {
                "policy": f"{name}_period{period}_threshold{threshold:g}",
                "context": name,
                "period": period,
                "threshold": threshold,
                "keyboard_accuracy": (correct[(name, period, threshold)] / total[(name, period, threshold)] if total[(name, period, threshold)] else None),
                "sample_count": total[(name, period, threshold)],
            }
            for name, period, threshold in policies
        ),
        key=lambda row: row.get("keyboard_accuracy") if row.get("keyboard_accuracy") is not None else -1.0,
        reverse=True,
    )
    base_accuracy = base_correct / base_total if base_total else None
    best = ranked[0] if ranked else None
    return {
        "schema": "g005_key_phase_repeat_diagnostic.v1",
        "status": "pass",
        "rows": rows,
        "train_rows": train_rows,
        "max_train_rows": max_train_rows,
        "max_target_rows": max_target_rows,
        "periods": [int(v) for v in periods],
        "thresholds": [float(v) for v in thresholds],
        "min_support": int(min_support),
        "alignment": {"sequence_id_mismatches": mismatches, "examples": examples},
        "base": {"keyboard_accuracy": base_accuracy, "sample_count": base_total},
        "best_policy": best,
        "ranked_policies": ranked,
        "context_counts": {f"{name}_period{period}": len(table) for (name, period), table in probs.items()},
        "claim_boundary": "CPU prefix diagnostic only. It adds phase-aware held-key press heuristics to an existing aligned base stream; it is not trained-model or G005 completion evidence.",
    }


def write_key_phase_repeat_diagnostic(*, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_key_phase_repeat_diagnostic(**kwargs)
    write_json(output_path, payload)
    return payload
