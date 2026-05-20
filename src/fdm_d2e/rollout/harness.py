from __future__ import annotations

from typing import Any

from fdm_d2e.schema import validate_named


def tokens_to_rollout_actions(predictions: list[dict[str, Any]], mode: str = 'stub') -> dict[str, Any]:
    actions = []
    for step, pred in enumerate(predictions[:8]):
        action = {
            'schema': 'rollout_action.v1',
            'step': step,
            'timestamp_ns': int(pred['timestamp_ns']),
            'tokens': list(pred.get('predicted_tokens', [])),
            'mode': mode,
            'valid': True,
        }
        validate_named(action, 'rollout_action.schema.json')
        actions.append(action)
    return {
        'schema': 'rollout_smoke.v1',
        'mode': mode,
        'status': 'stubbed' if mode == 'stub' else 'smoke-ran',
        'num_actions': len(actions),
        'actions': actions,
        'latency_ms_p50': 0.0 if mode == 'stub' else 1.0,
        'notes': 'Bounded smoke harness; not production VM rollout infrastructure.',
    }
