from __future__ import annotations

import glob
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

from fdm_d2e.io_utils import write_json
from fdm_d2e.tokenization.actions import token_to_delta_class

try:  # pragma: no cover - exercised only when optional fast parser is present.
    import orjson  # type: ignore
except Exception:  # pragma: no cover - fallback is covered.
    orjson = None


def _loads(line: str) -> dict[str, Any]:
    if orjson is not None:
        payload = orjson.loads(line)
    else:
        payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


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


def _iter_jsonl(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
            for line in handle:
                if line.strip():
                    yield _loads(line)


def _category(tokens: list[str], prefixes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(token for token in tokens if token.startswith(prefixes)))


def _axis_values(tokens: list[str], axis_prefix: str) -> list[float]:
    values = [token_to_delta_class(token) for token in tokens if token.startswith(axis_prefix)]
    return [float(value) for value in values if value is not None]


class ActionAccumulator:
    def __init__(self) -> None:
        self.rows = 0
        self.keyboard_total = 0
        self.keyboard_correct = 0
        self.button_total = 0
        self.button_correct = 0
        self.button_predicted_total = 0
        self.button_exact_tp = 0
        self.button_false_positive = 0
        self.button_false_negative = 0
        self.no_button_total = 0
        self.no_button_false_positive = 0
        self.mouse_n = 0
        self.mouse_sum_pred = 0.0
        self.mouse_sum_gt = 0.0
        self.mouse_sum_pred_sq = 0.0
        self.mouse_sum_gt_sq = 0.0
        self.mouse_sum_cross = 0.0
        self.mouse_sum_abs_pred = 0.0
        self.mouse_sum_abs_gt = 0.0
        self.failures = 0
        self.predicted_tokens: Counter[str] = Counter()
        self.ground_truth_tokens: Counter[str] = Counter()

    def update(self, predicted_tokens: list[str], ground_truth_tokens: list[str]) -> None:
        self.rows += 1
        ptokens = [str(token) for token in predicted_tokens]
        gtokens = [str(token) for token in ground_truth_tokens]
        self.predicted_tokens.update(ptokens or ["NOOP"])
        self.ground_truth_tokens.update(gtokens or ["NOOP"])
        pk = _category(ptokens, ("KEY_",))
        gk = _category(gtokens, ("KEY_",))
        if gk:
            self.keyboard_total += 1
            self.keyboard_correct += int(pk == gk)
        pb = _category(ptokens, ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
        gb = _category(gtokens, ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
        if pb:
            self.button_predicted_total += 1
        if gb:
            self.button_total += 1
            if pb == gb:
                self.button_correct += 1
                self.button_exact_tp += 1
            else:
                self.button_false_negative += 1
                if pb:
                    self.button_false_positive += 1
        else:
            self.no_button_total += 1
            if pb:
                self.button_false_positive += 1
                self.no_button_false_positive += 1
        for axis_prefix in ("MOUSE_DX_", "MOUSE_DY_"):
            pred_values = _axis_values(ptokens, axis_prefix)
            gt_values = _axis_values(gtokens, axis_prefix)
            if pred_values and gt_values:
                pred_value = sum(pred_values) / len(pred_values)
                gt_value = sum(gt_values) / len(gt_values)
                self.mouse_n += 1
                self.mouse_sum_pred += pred_value
                self.mouse_sum_gt += gt_value
                self.mouse_sum_pred_sq += pred_value * pred_value
                self.mouse_sum_gt_sq += gt_value * gt_value
                self.mouse_sum_cross += pred_value * gt_value
                self.mouse_sum_abs_pred += abs(pred_value)
                self.mouse_sum_abs_gt += abs(gt_value)
        if ptokens != gtokens:
            self.failures += 1

    def metrics(self, *, top_k: int = 20) -> dict[str, Any]:
        denom = (2 * self.button_exact_tp) + self.button_false_positive + self.button_false_negative
        mouse_status = "computed" if self.mouse_n else "absent"
        pearson = None
        scale_ratio = None
        if self.mouse_n >= 2:
            n = float(self.mouse_n)
            numerator = self.mouse_sum_cross - (self.mouse_sum_pred * self.mouse_sum_gt / n)
            pred_var = self.mouse_sum_pred_sq - (self.mouse_sum_pred * self.mouse_sum_pred / n)
            gt_var = self.mouse_sum_gt_sq - (self.mouse_sum_gt * self.mouse_sum_gt / n)
            denominator = (pred_var * gt_var) ** 0.5 if pred_var > 0 and gt_var > 0 else 0.0
            pearson = numerator / denominator if denominator else None
        if self.mouse_n:
            pred_abs = self.mouse_sum_abs_pred / self.mouse_n
            gt_abs = self.mouse_sum_abs_gt / self.mouse_n
            if pred_abs > 0 and gt_abs > 0:
                scale_ratio = max(pred_abs, gt_abs) / min(pred_abs, gt_abs)
        return {
            "rows": self.rows,
            "keyboard": {
                "accuracy": self.keyboard_correct / self.keyboard_total if self.keyboard_total else None,
                "num_examples": self.keyboard_total,
            },
            "mouse_button": {
                "accuracy": self.button_correct / self.button_total if self.button_total else None,
                "num_examples": self.button_total,
                "predicted_examples": self.button_predicted_total,
                "exact_true_positive_examples": self.button_exact_tp,
                "false_positive_examples": self.button_false_positive,
                "false_negative_examples": self.button_false_negative,
                "precision": (
                    self.button_exact_tp / (self.button_exact_tp + self.button_false_positive)
                    if (self.button_exact_tp + self.button_false_positive)
                    else None
                ),
                "recall": (
                    self.button_exact_tp / (self.button_exact_tp + self.button_false_negative)
                    if (self.button_exact_tp + self.button_false_negative)
                    else None
                ),
                "f1": (2 * self.button_exact_tp) / denom if denom else None,
                "no_button_examples": self.no_button_total,
                "no_button_false_positive_examples": self.no_button_false_positive,
                "no_button_false_positive_rate": (
                    self.no_button_false_positive / self.no_button_total if self.no_button_total else None
                ),
            },
            "mouse_move": {
                "status": mouse_status,
                "pearson": pearson,
                "scale_ratio": scale_ratio,
                "num_values": self.mouse_n,
            },
            "failure_count": self.failures,
            "top_predicted_tokens": self.predicted_tokens.most_common(top_k),
            "top_ground_truth_tokens": self.ground_truth_tokens.most_common(top_k),
        }


def _row_tokens(row: dict[str, Any], key: str) -> list[str]:
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


def _game(pred: dict[str, Any], target: dict[str, Any]) -> str:
    value = target.get("game") or pred.get("game")
    return str(value) if value else "unknown_game"


def build_streaming_action_diagnostics(
    *,
    prediction_paths: list[str | Path],
    target_paths: list[str | Path],
    max_rows: int | None = None,
    top_k: int = 20,
    progress_output_path: str | Path | None = None,
    progress_rows: int = 1_000_000,
) -> dict[str, Any]:
    predictions = _expand_paths(prediction_paths)
    targets = _expand_paths(target_paths)
    findings: list[dict[str, Any]] = []
    if not predictions:
        findings.append({"severity": "error", "code": "missing_prediction_paths", "patterns": [str(path) for path in prediction_paths]})
    if not targets:
        findings.append({"severity": "error", "code": "missing_target_paths", "patterns": [str(path) for path in target_paths]})
    groups: dict[str, ActionAccumulator] = {"all": ActionAccumulator()}
    alignment = {
        "rows_seen": 0,
        "sequence_id_mismatches": 0,
        "missing_prediction_rows": 0,
        "missing_target_rows": 0,
        "examples": [],
    }
    if predictions and targets:
        for pred, target in zip_longest(_iter_jsonl(predictions), _iter_jsonl(targets)):
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
            predicted_tokens = _row_tokens(pred, "predicted_tokens")
            ground_truth_tokens = _row_tokens(target, "ground_truth_tokens")
            group_keys = ["all", f"game:{_game(pred, target)}"]
            for tag in _split_tags(target):
                group_keys.append(f"eval_split:{tag}")
            for key in group_keys:
                groups.setdefault(key, ActionAccumulator()).update(predicted_tokens, ground_truth_tokens)
            if progress_output_path and progress_rows > 0 and alignment["rows_seen"] % progress_rows == 0:
                write_json(
                    progress_output_path,
                    {
                        "schema": "g002_streaming_action_diagnostics_progress.v1",
                        "status": "running",
                        "rows_seen": alignment["rows_seen"],
                        "sequence_id_mismatches": alignment["sequence_id_mismatches"],
                        "group_count": len(groups),
                        "prediction_paths": [str(path) for path in predictions],
                        "target_paths": [str(path) for path in targets],
                    },
                )
    if alignment["sequence_id_mismatches"]:
        findings.append(
            {
                "severity": "warning",
                "code": "sequence_id_mismatches_detected",
                "count": alignment["sequence_id_mismatches"],
                "examples": alignment["examples"],
            }
        )
    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g002_streaming_action_diagnostics.v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "prediction_paths": [str(path) for path in predictions],
        "target_paths": [str(path) for path in targets],
        "max_rows": max_rows,
        "alignment": alignment,
        "groups": {key: value.metrics(top_k=top_k) for key, value in sorted(groups.items())},
        "findings": findings,
    }


def write_streaming_action_diagnostics(
    *,
    prediction_paths: list[str | Path],
    target_paths: list[str | Path],
    output_path: str | Path,
    max_rows: int | None = None,
    top_k: int = 20,
    progress_output_path: str | Path | None = None,
    progress_rows: int = 1_000_000,
) -> dict[str, Any]:
    payload = build_streaming_action_diagnostics(
        prediction_paths=prediction_paths,
        target_paths=target_paths,
        max_rows=max_rows,
        top_k=top_k,
        progress_output_path=progress_output_path,
        progress_rows=progress_rows,
    )
    write_json(output_path, payload)
    return payload
