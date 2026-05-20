from __future__ import annotations

import math
from typing import Any

from fdm_d2e.schema import validate_named
from fdm_d2e.tokenization.actions import token_to_delta_class


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def _scale_ratio(xs: list[float], ys: list[float]) -> float | None:
    if not xs or not ys:
        return None
    ax = sum(abs(x) for x in xs) / len(xs)
    ay = sum(abs(y) for y in ys) / len(ys)
    if ax == 0 or ay == 0:
        return None
    ratio = max(ax, ay) / min(ax, ay)
    return ratio


def _category_tokens(tokens: list[str], prefix: tuple[str, ...]) -> list[str]:
    return sorted(t for t in tokens if t.startswith(prefix))


def _axis_mean(tokens: list[str], axis_prefix: str) -> float | None:
    values = [token_to_delta_class(t) for t in tokens if t.startswith(axis_prefix)]
    numeric = [float(v) for v in values if v is not None]
    return sum(numeric) / len(numeric) if numeric else None


def compute_metrics(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    gt_by_id = {row['sequence_id']: row for row in ground_truth}
    keyboard_total = keyboard_correct = 0
    button_total = button_correct = 0
    button_predicted_total = button_exact_tp = button_false_positive = button_false_negative = 0
    button_no_ground_truth_total = button_no_ground_truth_false_positive = 0
    pred_mouse: list[float] = []
    gt_mouse: list[float] = []
    failures: list[dict[str, Any]] = []
    matched_examples = 0
    for pred in predictions:
        gt = gt_by_id.get(pred['sequence_id'])
        if not gt:
            continue
        matched_examples += 1
        ptokens = list(pred.get('predicted_tokens', []))
        gtokens = list(gt.get('ground_truth_tokens', []))
        pk, gk = _category_tokens(ptokens, ('KEY_',)), _category_tokens(gtokens, ('KEY_',))
        if gk:
            keyboard_total += 1
            keyboard_correct += int(pk == gk)
        pb = _category_tokens(ptokens, ('MOUSE_LEFT_', 'MOUSE_RIGHT_', 'MOUSE_MIDDLE_'))
        gb = _category_tokens(gtokens, ('MOUSE_LEFT_', 'MOUSE_RIGHT_', 'MOUSE_MIDDLE_'))
        if pb:
            button_predicted_total += 1
        if gb:
            button_total += 1
            button_correct += int(pb == gb)
            if pb == gb:
                button_exact_tp += 1
            else:
                button_false_negative += 1
                if pb:
                    button_false_positive += 1
        else:
            button_no_ground_truth_total += 1
            if pb:
                button_false_positive += 1
                button_no_ground_truth_false_positive += 1
        for axis_prefix in ('MOUSE_DX_', 'MOUSE_DY_'):
            pred_axis = _axis_mean(ptokens, axis_prefix)
            gt_axis = _axis_mean(gtokens, axis_prefix)
            if pred_axis is not None and gt_axis is not None:
                pred_mouse.append(pred_axis)
                gt_mouse.append(gt_axis)
        if ptokens != gtokens:
            failures.append({'sequence_id': pred['sequence_id'], 'predicted_tokens': ptokens, 'ground_truth_tokens': gtokens})
    mouse_status = 'computed' if pred_mouse and gt_mouse else 'absent'
    metrics = {
        'schema': 'metrics.v1',
        'stage': 'fdm_eval',
        'num_examples': matched_examples,
        'keyboard': {
            'status': 'computed' if keyboard_total else 'absent',
            'accuracy': keyboard_correct / keyboard_total if keyboard_total else None,
            'num_examples': keyboard_total,
        },
        'mouse_button': {
            'status': 'computed' if button_total else 'absent',
            'accuracy': button_correct / button_total if button_total else None,
            'num_examples': button_total,
            'predicted_examples': button_predicted_total,
            'exact_true_positive_examples': button_exact_tp,
            'false_positive_examples': button_false_positive,
            'false_negative_examples': button_false_negative,
            'precision': button_exact_tp / (button_exact_tp + button_false_positive) if (button_exact_tp + button_false_positive) else None,
            'recall': button_exact_tp / (button_exact_tp + button_false_negative) if (button_exact_tp + button_false_negative) else None,
            'f1': (
                (2 * button_exact_tp) / ((2 * button_exact_tp) + button_false_positive + button_false_negative)
                if ((2 * button_exact_tp) + button_false_positive + button_false_negative)
                else None
            ),
            'no_button_examples': button_no_ground_truth_total,
            'no_button_false_positive_examples': button_no_ground_truth_false_positive,
            'no_button_false_positive_rate': (
                button_no_ground_truth_false_positive / button_no_ground_truth_total
                if button_no_ground_truth_total
                else None
            ),
        },
        'mouse_move': {
            'status': mouse_status,
            'pearson': _pearson(pred_mouse, gt_mouse) if mouse_status == 'computed' else None,
            'scale_ratio': _scale_ratio(pred_mouse, gt_mouse) if mouse_status == 'computed' else None,
            'num_values': len(pred_mouse),
        },
        'failure_count': len(failures),
    }
    validate_named(metrics, 'metrics.schema.json')
    return metrics
