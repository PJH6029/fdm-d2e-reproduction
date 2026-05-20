from __future__ import annotations

from typing import Any

from fdm_d2e.schema import validate_named


def _clean_key(key: str) -> str:
    return ''.join(ch for ch in key.upper() if ch.isalnum() or ch == '_') or 'UNKNOWN'


def bin_delta(delta: int | float) -> str:
    value = float(delta)
    if value == 0:
        return 'Z0'
    sign = 'P' if value > 0 else 'N'
    mag = abs(value)
    if mag <= 1:
        bucket = '1'
    elif mag <= 3:
        bucket = '2'
    elif mag <= 8:
        bucket = '3'
    elif mag <= 16:
        bucket = '4'
    else:
        bucket = '5'
    return f'{sign}{bucket}'


def token_to_delta_class(token: str) -> int | None:
    if not token.startswith(('MOUSE_DX_', 'MOUSE_DY_')):
        return None
    suffix = token.rsplit('_', 1)[-1]
    if suffix == 'Z0':
        return 0
    sign = 1 if suffix.startswith('P') else -1
    bucket = int(suffix[1:])
    representative = {1: 1, 2: 2, 3: 6, 4: 12, 5: 24}[bucket]
    return sign * representative


def tokenize_event(event: dict[str, Any]) -> list[str]:
    etype = event.get('type')
    if etype == 'keyboard':
        action = 'PRESS' if event.get('event_type') == 'press' else 'RELEASE'
        return [f'KEY_{action}_{_clean_key(str(event.get("key", "UNKNOWN")))}']
    if etype == 'mouse_button':
        button = str(event.get('button', 'unknown')).upper()
        action = 'DOWN' if event.get('event_type') == 'press' else 'UP'
        return [f'MOUSE_{button}_{action}']
    if etype == 'mouse_move':
        return [f'MOUSE_DX_{bin_delta(event.get("dx", 0))}', f'MOUSE_DY_{bin_delta(event.get("dy", 0))}']
    if etype == 'scroll':
        dy = float(event.get('dy', 0))
        if dy == 0:
            return ['SCROLL_Z0']
        return ['SCROLL_UP' if dy > 0 else 'SCROLL_DOWN']
    return ['NOOP']


def tokenize_record(record: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for event in record.get('events', []):
        tokens.extend(tokenize_event(event))
    return tokens or ['NOOP']


def add_tokens(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row['ground_truth_tokens'] = tokenize_record(row)
        enriched.append(row)
    return enriched


def build_vocab(records: list[dict[str, Any]]) -> dict[str, Any]:
    token_set = sorted({token for r in records for token in r.get('ground_truth_tokens', tokenize_record(r))})
    categories = {
        'keyboard': [t for t in token_set if t.startswith('KEY_')],
        'mouse_button': [t for t in token_set if t.startswith('MOUSE_') and not t.startswith(('MOUSE_DX_', 'MOUSE_DY_'))],
        'mouse_move': [t for t in token_set if t.startswith(('MOUSE_DX_', 'MOUSE_DY_'))],
        'scroll': [t for t in token_set if t.startswith('SCROLL_')],
    }
    vocab = {'schema': 'action_vocab.v1', 'tokens': token_set, 'categories': categories}
    validate_named(vocab, 'action_vocab.schema.json')
    return vocab
