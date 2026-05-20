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


def compute_metrics(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    gt_by_id = {row['sequence_id']: row for row in ground_truth}
    keyboard_total = keyboard_correct = 0
    button_total = button_correct = 0
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
        if gb:
            button_total += 1
            button_correct += int(pb == gb)
        for p_tok in ptokens:
            pd = token_to_delta_class(p_tok)
            if pd is None:
                continue
            axis_prefix = 'MOUSE_DX_' if p_tok.startswith('MOUSE_DX_') else 'MOUSE_DY_'
            matching = [token_to_delta_class(t) for t in gtokens if t.startswith(axis_prefix)]
            if matching and matching[0] is not None:
                pred_mouse.append(float(pd))
                gt_mouse.append(float(matching[0]))
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
