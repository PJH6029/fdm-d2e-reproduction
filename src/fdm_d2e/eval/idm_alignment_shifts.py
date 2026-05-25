from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.eval.paper_idm_metrics import (
    _axis_total,
    _button_tokens,
    _expand_paths,
    _iter_jsonl,
    _key_tokens,
    _split_tags,
    _tokens,
)
from fdm_d2e.io_utils import write_json


@dataclass(slots=True)
class _TokenFeatures:
    keys: dict[str, int]
    buttons: dict[str, int]
    dx: float
    dy: float
    strict_keys: tuple[str, ...]
    strict_buttons: tuple[str, ...]


class _FastMetricAccumulator:
    def __init__(self, *, empty_bins_as_correct: bool = False) -> None:
        self.empty_bins_as_correct = empty_bins_as_correct
        self.rows = 0
        self.key_samples = 0
        self.key_correct = 0
        self.button_samples = 0
        self.button_correct = 0
        self.move_samples = 0
        self.scale_samples = 0
        self.sum_pred_x = 0.0
        self.sum_gt_x = 0.0
        self.sum_pred_x_sq = 0.0
        self.sum_gt_x_sq = 0.0
        self.sum_x_cross = 0.0
        self.sum_pred_y = 0.0
        self.sum_gt_y = 0.0
        self.sum_pred_y_sq = 0.0
        self.sum_gt_y_sq = 0.0
        self.sum_y_cross = 0.0
        self.sum_abs_pred_x = 0.0
        self.sum_abs_gt_x = 0.0
        self.sum_abs_pred_y = 0.0
        self.sum_abs_gt_y = 0.0
        self.strict_keyboard_total = 0
        self.strict_keyboard_correct = 0
        self.strict_button_total = 0
        self.strict_button_correct = 0
        self.strict_button_predicted_total = 0
        self.strict_button_tp = 0
        self.strict_button_fp = 0
        self.strict_button_fn = 0
        self.strict_no_button_total = 0
        self.strict_no_button_fp = 0

    @staticmethod
    def _corr(n: int, sum_pred: float, sum_gt: float, sum_pred_sq: float, sum_gt_sq: float, sum_cross: float) -> float | None:
        if n < 2:
            return None
        numerator = sum_cross - (sum_pred * sum_gt / float(n))
        pred_var = sum_pred_sq - (sum_pred * sum_pred / float(n))
        gt_var = sum_gt_sq - (sum_gt * sum_gt / float(n))
        denominator = (pred_var * gt_var) ** 0.5 if pred_var > 0 and gt_var > 0 else 0.0
        return numerator / denominator if denominator else None

    @staticmethod
    def _scale_ratio(sum_abs_pred: float, sum_abs_gt: float, n: int) -> float | None:
        if n <= 0:
            return None
        pred_mean = sum_abs_pred / float(n)
        gt_mean = sum_abs_gt / float(n)
        if pred_mean <= 0 or gt_mean <= 0:
            return None
        ratio = gt_mean / pred_mean
        return ratio if ratio >= 1.0 else 1.0 / ratio

    def update_features(self, predicted: _TokenFeatures, ground_truth: _TokenFeatures) -> None:
        self.rows += 1
        for token in set(predicted.keys) | set(ground_truth.keys):
            self.key_samples += 1
            self.key_correct += int(predicted.keys.get(token, 0) == ground_truth.keys.get(token, 0))
        if not predicted.keys and not ground_truth.keys and self.empty_bins_as_correct:
            self.key_samples += 1
            self.key_correct += 1

        for token in set(predicted.buttons) | set(ground_truth.buttons):
            self.button_samples += 1
            self.button_correct += int(predicted.buttons.get(token, 0) == ground_truth.buttons.get(token, 0))
        if not predicted.buttons and not ground_truth.buttons and self.empty_bins_as_correct:
            self.button_samples += 1
            self.button_correct += 1

        if predicted.dx != 0.0 or ground_truth.dx != 0.0 or predicted.dy != 0.0 or ground_truth.dy != 0.0:
            self.move_samples += 1
            self.sum_pred_x += predicted.dx
            self.sum_gt_x += ground_truth.dx
            self.sum_pred_x_sq += predicted.dx * predicted.dx
            self.sum_gt_x_sq += ground_truth.dx * ground_truth.dx
            self.sum_x_cross += predicted.dx * ground_truth.dx
            self.sum_pred_y += predicted.dy
            self.sum_gt_y += ground_truth.dy
            self.sum_pred_y_sq += predicted.dy * predicted.dy
            self.sum_gt_y_sq += ground_truth.dy * ground_truth.dy
            self.sum_y_cross += predicted.dy * ground_truth.dy
        self.scale_samples += 1
        self.sum_abs_pred_x += abs(predicted.dx)
        self.sum_abs_gt_x += abs(ground_truth.dx)
        self.sum_abs_pred_y += abs(predicted.dy)
        self.sum_abs_gt_y += abs(ground_truth.dy)

        if ground_truth.strict_keys:
            self.strict_keyboard_total += 1
            self.strict_keyboard_correct += int(predicted.strict_keys == ground_truth.strict_keys)
        if predicted.strict_buttons:
            self.strict_button_predicted_total += 1
        if ground_truth.strict_buttons:
            self.strict_button_total += 1
            if predicted.strict_buttons == ground_truth.strict_buttons:
                self.strict_button_correct += 1
                self.strict_button_tp += 1
            else:
                self.strict_button_fn += 1
                if predicted.strict_buttons:
                    self.strict_button_fp += 1
        else:
            self.strict_no_button_total += 1
            if predicted.strict_buttons:
                self.strict_button_fp += 1
                self.strict_no_button_fp += 1

    def metrics(self) -> dict[str, Any]:
        button_denom = (2 * self.strict_button_tp) + self.strict_button_fp + self.strict_button_fn
        return {
            "rows": self.rows,
            "paper_compatible": {
                "empty_bins_as_correct": self.empty_bins_as_correct,
                "mouse_move": {
                    "pearson_x": self._corr(
                        self.move_samples,
                        self.sum_pred_x,
                        self.sum_gt_x,
                        self.sum_pred_x_sq,
                        self.sum_gt_x_sq,
                        self.sum_x_cross,
                    ),
                    "pearson_y": self._corr(
                        self.move_samples,
                        self.sum_pred_y,
                        self.sum_gt_y,
                        self.sum_pred_y_sq,
                        self.sum_gt_y_sq,
                        self.sum_y_cross,
                    ),
                    "scale_ratio_x": self._scale_ratio(self.sum_abs_pred_x, self.sum_abs_gt_x, self.scale_samples),
                    "scale_ratio_y": self._scale_ratio(self.sum_abs_pred_y, self.sum_abs_gt_y, self.scale_samples),
                    "sample_count": self.move_samples,
                    "scale_sample_count": self.scale_samples,
                },
                "keyboard": {
                    "key_accuracy": self.key_correct / self.key_samples if self.key_samples else None,
                    "sample_count": self.key_samples,
                },
                "mouse_button": {
                    "button_accuracy": self.button_correct / self.button_samples if self.button_samples else None,
                    "sample_count": self.button_samples,
                },
            },
            "strict_local": {
                "keyboard": {
                    "accuracy": self.strict_keyboard_correct / self.strict_keyboard_total if self.strict_keyboard_total else None,
                    "num_examples": self.strict_keyboard_total,
                },
                "mouse_button": {
                    "accuracy": self.strict_button_correct / self.strict_button_total if self.strict_button_total else None,
                    "num_examples": self.strict_button_total,
                    "predicted_examples": self.strict_button_predicted_total,
                    "exact_true_positive_examples": self.strict_button_tp,
                    "false_positive_examples": self.strict_button_fp,
                    "false_negative_examples": self.strict_button_fn,
                    "precision": (
                        self.strict_button_tp / (self.strict_button_tp + self.strict_button_fp)
                        if (self.strict_button_tp + self.strict_button_fp)
                        else None
                    ),
                    "recall": (
                        self.strict_button_tp / (self.strict_button_tp + self.strict_button_fn)
                        if (self.strict_button_tp + self.strict_button_fn)
                        else None
                    ),
                    "f1": (2 * self.strict_button_tp) / button_denom if button_denom else None,
                    "no_button_examples": self.strict_no_button_total,
                    "no_button_false_positive_examples": self.strict_no_button_fp,
                    "no_button_false_positive_rate": (
                        self.strict_no_button_fp / self.strict_no_button_total if self.strict_no_button_total else None
                    ),
                },
            },
        }


@dataclass(slots=True)
class _AlignmentRow:
    sequence_id: str | None
    recording_id: str
    predicted_features: _TokenFeatures
    ground_truth_features: _TokenFeatures
    split_tags: list[str]


def _recording_id(row: dict[str, Any]) -> str:
    value = row.get("recording_id")
    if isinstance(value, str) and value:
        return value
    sequence_id = row.get("sequence_id")
    if isinstance(sequence_id, str) and "#" in sequence_id:
        return sequence_id.rsplit("#", 1)[0]
    if isinstance(sequence_id, str) and sequence_id:
        return sequence_id
    return "__unknown_recording__"


def _features(tokens: Sequence[str]) -> _TokenFeatures:
    token_list = [str(token) for token in tokens]
    strict_buttons = tuple(
        sorted(token for token in token_list if token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")))
    )
    return _TokenFeatures(
        keys=dict(_key_tokens(token_list)),
        buttons=dict(_button_tokens(token_list)),
        dx=float(_axis_total(token_list, "MOUSE_DX_")),
        dy=float(_axis_total(token_list, "MOUSE_DY_")),
        strict_keys=tuple(sorted(token for token in token_list if token.startswith("KEY_"))),
        strict_buttons=strict_buttons,
    )


def _group_accumulators(split_tags: Sequence[str], *, empty_bins_as_correct: bool) -> dict[str, _FastMetricAccumulator]:
    groups = {"all": _FastMetricAccumulator(empty_bins_as_correct=empty_bins_as_correct)}
    for tag in split_tags:
        groups[f"eval_split:{tag}"] = _FastMetricAccumulator(empty_bins_as_correct=empty_bins_as_correct)
    return groups


def _update_groups(
    groups: dict[str, _FastMetricAccumulator],
    *,
    predicted_features: _TokenFeatures,
    ground_truth_features: _TokenFeatures,
    target_split_tags: Sequence[str],
) -> None:
    groups["all"].update_features(predicted_features, ground_truth_features)
    active = set(target_split_tags)
    for key, accumulator in groups.items():
        if not key.startswith("eval_split:"):
            continue
        tag = key.split(":", 1)[1]
        if tag in active:
            accumulator.update_features(predicted_features, ground_truth_features)


def _metrics(groups: dict[str, _FastMetricAccumulator]) -> dict[str, Any]:
    return {key: value.metrics() for key, value in sorted(groups.items())}


def _shift_indices(length: int, shift: int) -> range:
    if shift >= 0:
        return range(0, max(0, length - shift))
    return range(-shift, length)


def _process_block(
    block: Sequence[_AlignmentRow],
    *,
    shifts: Sequence[int],
    split_tags: Sequence[str],
    empty_bins_as_correct: bool,
    include_model_shift: bool,
) -> tuple[dict[int, dict[str, _FastMetricAccumulator]], dict[int, dict[str, _FastMetricAccumulator]], dict[int, int]]:
    target_autocorr = {
        shift: _group_accumulators(split_tags, empty_bins_as_correct=empty_bins_as_correct)
        for shift in shifts
    }
    model_vs_shifted_target = {
        shift: _group_accumulators(split_tags, empty_bins_as_correct=empty_bins_as_correct)
        for shift in shifts
    }
    pair_counts = {shift: 0 for shift in shifts}
    for shift in shifts:
        for base_idx in _shift_indices(len(block), shift):
            target_idx = base_idx + shift
            base = block[base_idx]
            shifted_target = block[target_idx]
            pair_counts[shift] += 1
            _update_groups(
                target_autocorr[shift],
                predicted_features=base.ground_truth_features,
                ground_truth_features=shifted_target.ground_truth_features,
                target_split_tags=shifted_target.split_tags,
            )
            if include_model_shift:
                _update_groups(
                    model_vs_shifted_target[shift],
                    predicted_features=base.predicted_features,
                    ground_truth_features=shifted_target.ground_truth_features,
                    target_split_tags=shifted_target.split_tags,
                )
    return target_autocorr, model_vs_shifted_target, pair_counts


def _merge_accumulators(dest: _FastMetricAccumulator, src: _FastMetricAccumulator) -> None:
    for key, value in src.__dict__.items():
        if isinstance(value, bool):
            continue
        setattr(dest, key, getattr(dest, key) + value)


def _merge_group_accumulators(
    dest: dict[int, dict[str, _FastMetricAccumulator]],
    src: dict[int, dict[str, _FastMetricAccumulator]],
) -> None:
    for shift, src_groups in src.items():
        for group_name, src_accumulator in src_groups.items():
            _merge_accumulators(dest[shift][group_name], src_accumulator)


def _iter_alignment_rows(
    *,
    prediction_paths: Sequence[Path],
    target_paths: Sequence[Path],
    max_rows: int | None,
    alignment: dict[str, Any],
) -> Iterable[_AlignmentRow]:
    pred_fields = ["sequence_id", "predicted_tokens"]
    target_fields = ["sequence_id", "recording_id", "eval_split_tags", "ground_truth_tokens"]
    if prediction_paths:
        row_iter = zip_longest(
            _iter_jsonl(prediction_paths, fields=pred_fields),
            _iter_jsonl(target_paths, fields=target_fields),
        )
    else:
        row_iter = ((None, target) for target in _iter_jsonl(target_paths, fields=target_fields))
    for pred, target in row_iter:
        if max_rows is not None and alignment["rows_seen"] >= max_rows:
            break
        if pred is None and prediction_paths:
            alignment["missing_prediction_rows"] += 1
            continue
        if target is None:
            alignment["missing_target_rows"] += 1
            continue
        alignment["rows_seen"] += 1
        pred_sequence_id = pred.get("sequence_id") if pred else None
        target_sequence_id = target.get("sequence_id")
        if pred_sequence_id is not None and target_sequence_id is not None and pred_sequence_id != target_sequence_id:
            alignment["sequence_id_mismatches"] += 1
            if len(alignment["examples"]) < 20:
                alignment["examples"].append(
                    {"pred_sequence_id": pred_sequence_id, "target_sequence_id": target_sequence_id}
                )
        yield _AlignmentRow(
            sequence_id=str(target_sequence_id) if target_sequence_id is not None else None,
            recording_id=_recording_id(target),
            predicted_features=_features(_tokens(pred or {}, "predicted_tokens")),
            ground_truth_features=_features(_tokens(target, "ground_truth_tokens")),
            split_tags=_split_tags(target),
        )


def build_idm_alignment_shift_diagnostics(
    *,
    target_paths: Sequence[str | Path],
    prediction_paths: Sequence[str | Path] = (),
    shifts: Sequence[int] = (-3, -2, -1, 0, 1, 2, 3),
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
    model_name: str = "model",
    max_rows: int | None = None,
    empty_bins_as_correct: bool = False,
    progress_output_path: str | Path | None = None,
    progress_rows: int = 1_000_000,
) -> dict[str, Any]:
    predictions = _expand_paths(prediction_paths)
    targets = _expand_paths(target_paths)
    findings: list[dict[str, Any]] = []
    if prediction_paths and not predictions:
        findings.append(
            {"severity": "error", "code": "missing_prediction_paths", "patterns": [str(path) for path in prediction_paths]}
        )
    if not targets:
        findings.append({"severity": "error", "code": "missing_target_paths", "patterns": [str(path) for path in target_paths]})
    normalized_shifts = sorted(set(int(shift) for shift in shifts))
    include_model_shift = bool(predictions)
    target_autocorr = {
        shift: _group_accumulators(split_tags, empty_bins_as_correct=empty_bins_as_correct)
        for shift in normalized_shifts
    }
    model_vs_shifted_target = {
        shift: _group_accumulators(split_tags, empty_bins_as_correct=empty_bins_as_correct)
        for shift in normalized_shifts
    }
    pair_counts = {shift: 0 for shift in normalized_shifts}
    alignment: dict[str, Any] = {
        "rows_seen": 0,
        "sequence_id_mismatches": 0,
        "missing_prediction_rows": 0,
        "missing_target_rows": 0,
        "examples": [],
    }
    block_stats: dict[str, Any] = {
        "recording_fragments": 0,
        "max_fragment_rows": 0,
        "short_fragments": 0,
    }
    block: list[_AlignmentRow] = []
    active_recording: str | None = None

    def flush_block() -> None:
        nonlocal block
        if not block:
            return
        block_stats["recording_fragments"] += 1
        block_stats["max_fragment_rows"] = max(block_stats["max_fragment_rows"], len(block))
        if len(block) <= max((abs(shift) for shift in normalized_shifts), default=0):
            block_stats["short_fragments"] += 1
        block_target, block_model, block_counts = _process_block(
            block,
            shifts=normalized_shifts,
            split_tags=split_tags,
            empty_bins_as_correct=empty_bins_as_correct,
            include_model_shift=include_model_shift,
        )
        _merge_group_accumulators(target_autocorr, block_target)
        if include_model_shift:
            _merge_group_accumulators(model_vs_shifted_target, block_model)
        for shift, count in block_counts.items():
            pair_counts[shift] += count
        block = []

    if targets and (not prediction_paths or predictions):
        for row in _iter_alignment_rows(
            prediction_paths=predictions,
            target_paths=targets,
            max_rows=max_rows,
            alignment=alignment,
        ):
            if active_recording is not None and row.recording_id != active_recording:
                flush_block()
            active_recording = row.recording_id
            block.append(row)
            if progress_output_path and progress_rows > 0 and alignment["rows_seen"] % progress_rows == 0:
                write_json(
                    progress_output_path,
                    {
                        "schema": "idm_alignment_shift_diagnostics_progress.v1",
                        "status": "running",
                        "model_name": model_name,
                        "rows_seen": alignment["rows_seen"],
                        "recording_fragments": block_stats["recording_fragments"],
                        "sequence_id_mismatches": alignment["sequence_id_mismatches"],
                    },
                )
        flush_block()
    if alignment["sequence_id_mismatches"]:
        findings.append(
            {
                "severity": "error",
                "code": "sequence_id_mismatches_detected",
                "count": alignment["sequence_id_mismatches"],
                "examples": alignment["examples"],
            }
        )
    if alignment["missing_prediction_rows"] or alignment["missing_target_rows"]:
        findings.append(
            {
                "severity": "error",
                "code": "prediction_target_row_count_mismatch",
                "missing_prediction_rows": alignment["missing_prediction_rows"],
                "missing_target_rows": alignment["missing_target_rows"],
            }
        )
    errors = [item for item in findings if item.get("severity") == "error"]
    diagnostics: dict[str, Any] = {
        "target_autocorr": {str(shift): _metrics(groups) for shift, groups in sorted(target_autocorr.items())},
        "pair_counts": {str(shift): count for shift, count in sorted(pair_counts.items())},
    }
    if include_model_shift:
        diagnostics["model_vs_shifted_target"] = {
            str(shift): _metrics(groups) for shift, groups in sorted(model_vs_shifted_target.items())
        }
    return {
        "schema": "idm_alignment_shift_diagnostics.v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "model_name": model_name,
        "prediction_paths": [str(path) for path in predictions],
        "target_paths": [str(path) for path in targets],
        "split_tags": list(split_tags),
        "shifts": normalized_shifts,
        "max_rows": max_rows,
        "alignment": alignment,
        "block_stats": block_stats,
        "diagnostics": diagnostics,
        "findings": findings,
        "interpretation": {
            "shift_definition": "For model_vs_shifted_target, shift=k compares prediction row i against target row i+k within the same recording fragment.",
            "target_autocorr_definition": "target_autocorr compares ground-truth tokens at row i against ground-truth tokens at row i+k within the same recording fragment.",
        },
    }


def write_idm_alignment_shift_diagnostics(
    *,
    target_paths: Sequence[str | Path],
    output_path: str | Path,
    prediction_paths: Sequence[str | Path] = (),
    shifts: Sequence[int] = (-3, -2, -1, 0, 1, 2, 3),
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
    model_name: str = "model",
    max_rows: int | None = None,
    empty_bins_as_correct: bool = False,
    progress_output_path: str | Path | None = None,
    progress_rows: int = 1_000_000,
) -> dict[str, Any]:
    payload = build_idm_alignment_shift_diagnostics(
        target_paths=target_paths,
        prediction_paths=prediction_paths,
        shifts=shifts,
        split_tags=split_tags,
        model_name=model_name,
        max_rows=max_rows,
        empty_bins_as_correct=empty_bins_as_correct,
        progress_output_path=progress_output_path,
        progress_rows=progress_rows,
    )
    write_json(output_path, payload)
    return payload
