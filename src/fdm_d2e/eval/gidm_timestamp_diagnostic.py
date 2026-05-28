from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.eval.idm_alignment_shifts import _FastMetricAccumulator, _features
from fdm_d2e.eval.paper_idm_metrics import _split_tags, _tokens
from fdm_d2e.io_utils import read_json, write_json

_PAPER_TARGET_DEFAULTS = {
    "keyboard_accuracy": 0.7301,
    "mouse_button_accuracy": 0.957283,
    "pearson_x": 0.79585,
    "pearson_y": 0.782817,
    "scale_ratio_x": 1.231667,
    "scale_ratio_y": 1.315,
}


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_no}")
            yield payload


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _paper_targets(path: str | Path | None) -> dict[str, float]:
    if path is None or not Path(path).exists():
        return dict(_PAPER_TARGET_DEFAULTS)
    payload = read_json(path)
    aggregate = payload.get("paper_reported_targets", {}).get("aggregate")
    if not isinstance(aggregate, dict):
        aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        return dict(_PAPER_TARGET_DEFAULTS)
    targets = dict(_PAPER_TARGET_DEFAULTS)
    for key in targets:
        value = aggregate.get(key)
        if isinstance(value, (int, float)):
            targets[key] = float(value)
    return targets


def _group_accumulators(split_tags: Sequence[str]) -> dict[str, _FastMetricAccumulator]:
    groups = {"all": _FastMetricAccumulator(empty_bins_as_correct=False)}
    for tag in split_tags:
        groups[f"eval_split:{tag}"] = _FastMetricAccumulator(empty_bins_as_correct=False)
    return groups


def _update_groups(
    groups: dict[str, _FastMetricAccumulator],
    *,
    predicted_tokens: Sequence[str],
    ground_truth_tokens: Sequence[str],
    target_split_tags: Sequence[str],
) -> None:
    predicted = _features(predicted_tokens)
    target = _features(ground_truth_tokens)
    groups["all"].update_features(predicted, target)
    active_tags = set(str(tag) for tag in target_split_tags)
    for tag in active_tags:
        group = groups.get(f"eval_split:{tag}")
        if group is not None:
            group.update_features(predicted, target)


def _shift_indices(prediction_count: int, target_count: int, shift_rows: int) -> Iterable[tuple[int, int]]:
    """Yield `(prediction_index, target_index)` pairs for a row shift.

    The sign convention matches the existing warmup pilot sweep artifact:
    `shift_rows=+k` compares prediction row `i+k` against target row `i`.
    This is the diagnostic equivalent of asking whether the converted
    prediction stream should be shifted `+k * bin_ms` relative to the target
    rows.
    """

    if shift_rows >= 0:
        upper = min(target_count, prediction_count - shift_rows)
        for target_index in range(max(0, upper)):
            yield target_index + shift_rows, target_index
    else:
        start = -shift_rows
        upper = min(target_count, prediction_count - shift_rows)
        for target_index in range(start, max(start, upper)):
            yield target_index + shift_rows, target_index


def _all_metrics(groups: dict[str, _FastMetricAccumulator]) -> dict[str, Any]:
    return {key: accumulator.metrics() for key, accumulator in sorted(groups.items())}


def _extract_summary(all_group: dict[str, Any]) -> dict[str, Any]:
    paper = all_group.get("paper_compatible", {})
    move = paper.get("mouse_move", {})
    keyboard = paper.get("keyboard", {})
    button = paper.get("mouse_button", {})
    strict_button = (all_group.get("strict_local", {}).get("mouse_button", {}))
    return {
        "keyboard_accuracy": keyboard.get("key_accuracy"),
        "keyboard_sample_count": keyboard.get("sample_count"),
        "mouse_button_accuracy": button.get("button_accuracy"),
        "mouse_button_sample_count": button.get("sample_count"),
        "pearson_x": move.get("pearson_x"),
        "pearson_y": move.get("pearson_y"),
        "scale_ratio_x": move.get("scale_ratio_x"),
        "scale_ratio_y": move.get("scale_ratio_y"),
        "mouse_move_sample_count": move.get("sample_count"),
        "strict_mouse_button_f1": strict_button.get("f1"),
        "no_button_fpr": strict_button.get("no_button_false_positive_rate"),
    }


def _paper_target_results(summary: dict[str, Any], targets: dict[str, float]) -> dict[str, Any]:
    checks = {
        "keyboard_accuracy": "min",
        "mouse_button_accuracy": "min",
        "pearson_x": "min",
        "pearson_y": "min",
        "scale_ratio_x": "max",
        "scale_ratio_y": "max",
    }
    results: dict[str, Any] = {}
    for metric, direction in checks.items():
        observed = summary.get(metric)
        target = targets.get(metric)
        if observed is None or target is None:
            passed = False
        elif direction == "min":
            passed = float(observed) >= float(target)
        else:
            passed = float(observed) <= float(target)
        results[metric] = {
            "observed": observed,
            "target": target,
            "direction": direction,
            "passed": passed,
        }
    return results


def _meets_all_targets(summary: dict[str, Any], targets: dict[str, float]) -> bool:
    return all(item["passed"] for item in _paper_target_results(summary, targets).values())


def _base_offset_rows(manifest: dict[str, Any], *, bin_ms: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bin_ns = int(bin_ms) * 1_000_000
    for row in manifest.get("recordings", []):
        if not isinstance(row, dict):
            continue
        if row.get("timestamp_min_ns") is None or row.get("bin_index_min") is None:
            continue
        base_ns = int(row["timestamp_min_ns"]) - int(row["bin_index_min"]) * bin_ns
        rows.append(
            {
                "universe_row_id": row.get("universe_row_id"),
                "timestamp_min_ns": int(row["timestamp_min_ns"]),
                "bin_index_min": int(row["bin_index_min"]),
                "base_timestamp_ns": int(base_ns),
                "base_timestamp_seconds": base_ns / 1e9,
                "base_shift_rows_float": base_ns / float(bin_ns),
                "base_shift_rows_floor": math.floor(base_ns / float(bin_ns)),
                "base_shift_rows_round": int(round(base_ns / float(bin_ns))),
                "base_shift_rows_ceil": math.ceil(base_ns / float(bin_ns)),
            }
        )
    return rows


def _candidate_shifts(base_rows: Sequence[dict[str, Any]], extra_shifts: Sequence[int]) -> list[int]:
    shifts = {0}
    for item in base_rows:
        for key in ("base_shift_rows_floor", "base_shift_rows_round", "base_shift_rows_ceil"):
            value = item.get(key)
            if isinstance(value, int):
                shifts.add(value)
                shifts.add(-value)
    shifts.update(int(shift) for shift in extra_shifts)
    return sorted(shifts)


def build_gidm_base_offset_shift_diagnostic(
    *,
    manifest_path: str | Path,
    prediction_path: str | Path,
    target_path: str | Path,
    output_path: str | Path | None = None,
    baseline_contract_path: str | Path | None = "artifacts/eval/g003_gidm_baseline_contract.json",
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
    bin_ms: int = 50,
    extra_shifts: Sequence[int] = (),
    max_rows: int | None = None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    targets = _load_rows(target_path)
    predictions = _load_rows(prediction_path)
    if max_rows is not None:
        targets = targets[: int(max_rows)]
        predictions = predictions[: int(max_rows)]

    paper_targets = _paper_targets(baseline_contract_path)
    base_rows = _base_offset_rows(manifest, bin_ms=bin_ms)
    shifts = _candidate_shifts(base_rows, extra_shifts=extra_shifts)
    rows: list[dict[str, Any]] = []
    for shift_rows in shifts:
        groups = _group_accumulators(split_tags)
        examples = []
        overlap_rows = 0
        for pred_index, target_index in _shift_indices(len(predictions), len(targets), shift_rows):
            pred = predictions[pred_index]
            target = targets[target_index]
            overlap_rows += 1
            if len(examples) < 3:
                examples.append(
                    {
                        "prediction_sequence_id": pred.get("sequence_id"),
                        "target_sequence_id": target.get("sequence_id"),
                    }
                )
            _update_groups(
                groups,
                predicted_tokens=_tokens(pred, "predicted_tokens"),
                ground_truth_tokens=_tokens(target, "ground_truth_tokens"),
                target_split_tags=_split_tags(target),
            )
        metrics = _all_metrics(groups)
        summary = _extract_summary(metrics["all"])
        row = {
            "row_shift": int(shift_rows),
            "timestamp_shift_seconds": float(shift_rows) * float(bin_ms) / 1000.0,
            "overlap_rows": int(overlap_rows),
            **summary,
            "paper_target_results": _paper_target_results(summary, paper_targets),
            "paper_targets_pass": _meets_all_targets(summary, paper_targets),
            "metrics": metrics,
            "examples": examples,
        }
        rows.append(row)

    def _value(row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    best_keyboard = sorted(rows, key=lambda item: _value(item, "keyboard_accuracy"), reverse=True)[:5]
    best_mouse = sorted(
        rows,
        key=lambda item: _value(item, "pearson_x") + _value(item, "pearson_y"),
        reverse=True,
    )[:5]
    passing = [row for row in rows if row["paper_targets_pass"]]
    findings: list[dict[str, Any]] = []
    if not base_rows:
        findings.append({"severity": "error", "code": "no_manifest_base_timestamp_rows"})
    if len(predictions) != len(targets):
        findings.append(
            {
                "severity": "warning",
                "code": "prediction_target_row_count_differs",
                "prediction_rows": len(predictions),
                "target_rows": len(targets),
            }
        )
    if not passing:
        findings.append({"severity": "info", "code": "no_base_offset_shift_meets_paper_targets"})
    payload = {
        "schema": "gidm_base_offset_shift_diagnostic.v1",
        "status": "pass" if not any(item.get("severity") == "error" for item in findings) else "fail",
        "created_at_epoch": int(time.time()),
        "goal_id": "G005-g014-idm-full-paper-target",
        "manifest_path": str(manifest_path),
        "prediction_path": str(prediction_path),
        "target_path": str(target_path),
        "baseline_contract_path": str(baseline_contract_path) if baseline_contract_path else None,
        "bin_ms": int(bin_ms),
        "prediction_rows": len(predictions),
        "target_rows": len(targets),
        "base_offsets": base_rows,
        "candidate_shifts": shifts,
        "paper_targets": paper_targets,
        "best_by_keyboard": best_keyboard,
        "best_by_mouse_pearson_sum": best_mouse,
        "passing_shifts": passing,
        "rows": rows,
        "findings": findings,
        "decision": "rejected_no_base_offset_shift_meets_paper_targets" if not passing else "diagnostic_shift_meets_paper_targets",
        "claim_boundary": (
            "Released G-IDM base-offset timing diagnostic only. It is not our-IDM evidence, "
            "not a G005 completion claim, and not an FDM-1 parity claim."
        ),
    }
    if output_path is not None:
        write_json(output_path, payload)
    return payload
