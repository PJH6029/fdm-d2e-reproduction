from __future__ import annotations

import glob
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.io_utils import write_json
from fdm_d2e.tokenization.actions import token_to_delta_class

try:  # pragma: no cover - exercised only when optional fast parser is present.
    import orjson  # type: ignore
except Exception:  # pragma: no cover - fallback is covered.
    orjson = None


_JSON_DECODER = json.JSONDecoder()


def _loads(line: str) -> dict[str, Any]:
    if orjson is not None:
        payload = orjson.loads(line)
    else:
        payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


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


def _extract_fields(line: str, fields: Sequence[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in fields:
        value = _extract_value(line, field)
        if value is not None:
            row[field] = value
    return row


def _expand_paths(patterns: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        text = str(pattern)
        matches = sorted(glob.glob(text))
        if matches:
            paths.extend(Path(match) for match in matches)
            continue
        path = Path(text)
        if path.exists():
            paths.append(path)
    return paths


def _iter_jsonl(paths: Sequence[Path], *, fields: Sequence[str] | None = None) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield _extract_fields(line, fields) if fields else _loads(line)
                except Exception as exc:
                    raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc


def _tokens(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None and key == "ground_truth_tokens":
        value = row.get("target_tokens")
    if value is None and key == "predicted_tokens":
        value = row.get("tokens")
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _split_tags(row: dict[str, Any]) -> list[str]:
    for key in ("eval_split_tags", "split_tags"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
    for key in ("eval_split", "split", "split_name"):
        value = row.get(key)
        if isinstance(value, str):
            return [value]
    return []


def _key_tokens(tokens: Sequence[str]) -> Counter[str]:
    return Counter(token for token in tokens if token.startswith("KEY_"))


def _button_tokens(tokens: Sequence[str]) -> Counter[str]:
    prefixes = ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")
    return Counter(token for token in tokens if token.startswith(prefixes))


def _axis_total(tokens: Sequence[str], prefix: str) -> float:
    total = 0.0
    for token in tokens:
        if not token.startswith(prefix):
            continue
        value = token_to_delta_class(token)
        if value is not None:
            total += float(value)
    return total


class _PaperMetricAccumulator:
    """Streaming approximation of the public D2E evaluate.py binned metrics."""

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

    @staticmethod
    def _strict_category(tokens: Sequence[str], prefixes: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(token for token in tokens if token.startswith(prefixes)))

    def update(self, predicted_tokens: Sequence[str], ground_truth_tokens: Sequence[str]) -> None:
        self.rows += 1
        pred = [str(token) for token in predicted_tokens]
        gt = [str(token) for token in ground_truth_tokens]

        pred_keys = _key_tokens(pred)
        gt_keys = _key_tokens(gt)
        for token in sorted(set(pred_keys) | set(gt_keys)):
            self.key_samples += 1
            self.key_correct += int(pred_keys.get(token, 0) == gt_keys.get(token, 0))
        if not pred_keys and not gt_keys and self.empty_bins_as_correct:
            self.key_samples += 1
            self.key_correct += 1

        pred_buttons = _button_tokens(pred)
        gt_buttons = _button_tokens(gt)
        for token in sorted(set(pred_buttons) | set(gt_buttons)):
            self.button_samples += 1
            self.button_correct += int(pred_buttons.get(token, 0) == gt_buttons.get(token, 0))
        if not pred_buttons and not gt_buttons and self.empty_bins_as_correct:
            self.button_samples += 1
            self.button_correct += 1

        pred_x = _axis_total(pred, "MOUSE_DX_")
        gt_x = _axis_total(gt, "MOUSE_DX_")
        pred_y = _axis_total(pred, "MOUSE_DY_")
        gt_y = _axis_total(gt, "MOUSE_DY_")
        if pred_x != 0.0 or gt_x != 0.0 or pred_y != 0.0 or gt_y != 0.0:
            self.move_samples += 1
            self.sum_pred_x += pred_x
            self.sum_gt_x += gt_x
            self.sum_pred_x_sq += pred_x * pred_x
            self.sum_gt_x_sq += gt_x * gt_x
            self.sum_x_cross += pred_x * gt_x
            self.sum_pred_y += pred_y
            self.sum_gt_y += gt_y
            self.sum_pred_y_sq += pred_y * pred_y
            self.sum_gt_y_sq += gt_y * gt_y
            self.sum_y_cross += pred_y * gt_y
        self.scale_samples += 1
        self.sum_abs_pred_x += abs(pred_x)
        self.sum_abs_gt_x += abs(gt_x)
        self.sum_abs_pred_y += abs(pred_y)
        self.sum_abs_gt_y += abs(gt_y)

        strict_pred_keys = self._strict_category(pred, ("KEY_",))
        strict_gt_keys = self._strict_category(gt, ("KEY_",))
        if strict_gt_keys:
            self.strict_keyboard_total += 1
            self.strict_keyboard_correct += int(strict_pred_keys == strict_gt_keys)
        strict_pred_buttons = self._strict_category(pred, ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
        strict_gt_buttons = self._strict_category(gt, ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
        if strict_pred_buttons:
            self.strict_button_predicted_total += 1
        if strict_gt_buttons:
            self.strict_button_total += 1
            if strict_pred_buttons == strict_gt_buttons:
                self.strict_button_correct += 1
                self.strict_button_tp += 1
            else:
                self.strict_button_fn += 1
                if strict_pred_buttons:
                    self.strict_button_fp += 1
        else:
            self.strict_no_button_total += 1
            if strict_pred_buttons:
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


def build_paper_idm_metrics(
    *,
    prediction_paths: Sequence[str | Path],
    target_paths: Sequence[str | Path],
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
    model_name: str = "model",
    max_rows: int | None = None,
    progress_output_path: str | Path | None = None,
    progress_rows: int = 1_000_000,
    empty_bins_as_correct: bool = False,
) -> dict[str, Any]:
    predictions = _expand_paths(prediction_paths)
    targets = _expand_paths(target_paths)
    findings: list[dict[str, Any]] = []
    if not predictions:
        findings.append({"severity": "error", "code": "missing_prediction_paths", "patterns": [str(path) for path in prediction_paths]})
    if not targets:
        findings.append({"severity": "error", "code": "missing_target_paths", "patterns": [str(path) for path in target_paths]})

    groups: dict[str, _PaperMetricAccumulator] = {"all": _PaperMetricAccumulator(empty_bins_as_correct=empty_bins_as_correct)}
    for tag in split_tags:
        groups[f"eval_split:{tag}"] = _PaperMetricAccumulator(empty_bins_as_correct=empty_bins_as_correct)
    alignment = {
        "rows_seen": 0,
        "sequence_id_mismatches": 0,
        "missing_prediction_rows": 0,
        "missing_target_rows": 0,
        "examples": [],
    }
    if predictions and targets:
        pred_fields = ["sequence_id", "predicted_tokens"]
        target_fields = ["sequence_id", "eval_split_tags", "ground_truth_tokens"]
        for pred, target in zip_longest(
            _iter_jsonl(predictions, fields=pred_fields),
            _iter_jsonl(targets, fields=target_fields),
        ):
            if max_rows is not None and alignment["rows_seen"] >= max_rows:
                break
            if pred is None:
                alignment["missing_prediction_rows"] += 1
                continue
            if target is None:
                alignment["missing_target_rows"] += 1
                continue
            alignment["rows_seen"] += 1
            pred_id = pred.get("sequence_id")
            target_id = target.get("sequence_id")
            if pred_id is not None and target_id is not None and pred_id != target_id:
                alignment["sequence_id_mismatches"] += 1
                if len(alignment["examples"]) < 20:
                    alignment["examples"].append({"pred_sequence_id": pred_id, "target_sequence_id": target_id})
            predicted_tokens = _tokens(pred, "predicted_tokens")
            ground_truth_tokens = _tokens(target, "ground_truth_tokens")
            groups["all"].update(predicted_tokens, ground_truth_tokens)
            active_tags = set(_split_tags(target))
            for tag in split_tags:
                if tag in active_tags:
                    groups[f"eval_split:{tag}"].update(predicted_tokens, ground_truth_tokens)
            if progress_output_path and progress_rows > 0 and alignment["rows_seen"] % progress_rows == 0:
                write_json(
                    progress_output_path,
                    {
                        "schema": "paper_idm_metrics_progress.v1",
                        "status": "running",
                        "model_name": model_name,
                        "rows_seen": alignment["rows_seen"],
                        "sequence_id_mismatches": alignment["sequence_id_mismatches"],
                        "prediction_paths": [str(path) for path in predictions],
                        "target_paths": [str(path) for path in targets],
                    },
                )
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
    return {
        "schema": "paper_idm_metrics.v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "model_name": model_name,
        "prediction_paths": [str(path) for path in predictions],
        "target_paths": [str(path) for path in targets],
        "split_tags": list(split_tags),
        "max_rows": max_rows,
        "alignment": alignment,
        "groups": {key: value.metrics() for key, value in sorted(groups.items())},
        "findings": findings,
        "claim_boundary": "Paper-compatible metrics approximate public D2E evaluate.py semantics over pre-binned JSONL tokens; exact released G-IDM comparison remains G006.",
    }


def write_paper_idm_metrics(
    *,
    prediction_paths: Sequence[str | Path],
    target_paths: Sequence[str | Path],
    output_path: str | Path,
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
    model_name: str = "model",
    max_rows: int | None = None,
    progress_output_path: str | Path | None = None,
    progress_rows: int = 1_000_000,
    empty_bins_as_correct: bool = False,
) -> dict[str, Any]:
    payload = build_paper_idm_metrics(
        prediction_paths=prediction_paths,
        target_paths=target_paths,
        split_tags=split_tags,
        model_name=model_name,
        max_rows=max_rows,
        progress_output_path=progress_output_path,
        progress_rows=progress_rows,
        empty_bins_as_correct=empty_bins_as_correct,
    )
    write_json(output_path, payload)
    return payload
