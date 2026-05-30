from __future__ import annotations

from typing import Any, Iterable

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


def _decomposed_axis_tokens(value: float, *, prefix: str, max_tokens: int) -> list[str]:
    max_tokens = max(1, int(max_tokens))
    magnitude = int(round(abs(float(value))))
    if magnitude <= 0:
        return [f"{prefix}Z0"]
    sign = "P" if value >= 0 else "N"
    buckets = ((5, 24), (4, 12), (3, 6), (2, 2), (1, 1))
    tokens: list[str] = []
    remaining = magnitude
    while remaining > 0 and len(tokens) < max_tokens:
        for suffix, bucket_value in buckets:
            if bucket_value <= remaining or suffix == 1:
                tokens.append(f"{prefix}{sign}{suffix}")
                remaining -= bucket_value
                break
    return tokens


def tokens_from_delta(dx: float, dy: float, *, emit_mode: str = "single", max_tokens_per_axis: int = 8) -> list[str]:
    normalized = emit_mode.replace("-", "_").lower()
    if normalized in {"single", "bin", "binned"}:
        return [f"MOUSE_DX_{bin_delta(dx)}", f"MOUSE_DY_{bin_delta(dy)}"]
    if normalized in {"decompose", "decomposed", "sum_decompose"}:
        return _decomposed_axis_tokens(dx, prefix="MOUSE_DX_", max_tokens=max_tokens_per_axis) + _decomposed_axis_tokens(
            dy,
            prefix="MOUSE_DY_",
            max_tokens=max_tokens_per_axis,
        )
    raise ValueError("mouse token emit mode must be one of: single, decompose")


def _clean_state_key(token: str, prefix: str) -> str:
    return _clean_key(token[len(prefix) :])


def state_tokens_from_event_tokens(
    tokens: Iterable[str],
    *,
    pressed_keys: set[str] | None = None,
    pressed_buttons: set[str] | None = None,
    mouse_emit_mode: str = "decompose",
    mouse_max_tokens_per_axis: int = 32,
) -> tuple[list[str], set[str], set[str]]:
    """Convert sparse press/release event tokens into a 50 ms control state.

    The public D2E metric is reported over fixed 50 ms bins.  Raw ocap bins can
    contain many mouse events plus sparse keyboard/button transitions; this
    helper turns those transitions into the held key/button state at the end of
    the bin while preserving summed mouse motion.
    """

    keys = set(pressed_keys or set())
    buttons = set(pressed_buttons or set())
    dx = 0.0
    dy = 0.0
    for raw_token in tokens:
        token = str(raw_token)
        if token.startswith("KEY_PRESS_"):
            keys.add(_clean_state_key(token, "KEY_PRESS_"))
            continue
        if token.startswith("KEY_RELEASE_"):
            keys.discard(_clean_state_key(token, "KEY_RELEASE_"))
            continue
        if token.startswith("KEY_DOWN_"):
            keys.add(_clean_state_key(token, "KEY_DOWN_"))
            continue
        if token.startswith("MOUSE_") and token.endswith("_DOWN") and not token.startswith(("MOUSE_DX_", "MOUSE_DY_")):
            buttons.add(token[len("MOUSE_") : -len("_DOWN")])
            continue
        if token.startswith("MOUSE_") and token.endswith("_UP") and not token.startswith(("MOUSE_DX_", "MOUSE_DY_")):
            buttons.discard(token[len("MOUSE_") : -len("_UP")])
            continue
        value = token_to_delta_class(token)
        if value is None:
            continue
        if token.startswith("MOUSE_DX_"):
            dx += float(value)
        elif token.startswith("MOUSE_DY_"):
            dy += float(value)
    state_tokens = tokens_from_delta(dx, dy, emit_mode=mouse_emit_mode, max_tokens_per_axis=mouse_max_tokens_per_axis)
    state_tokens.extend(f"KEY_DOWN_{key}" for key in sorted(keys))
    state_tokens.extend(f"MOUSE_{button}_DOWN" for button in sorted(buttons))
    return state_tokens or ["NOOP"], keys, buttons


def held_keys_from_state_tokens(tokens: Iterable[str]) -> set[str]:
    """Extract held-key state from D2E/FDM action-state tokens."""

    keys: set[str] = set()
    for raw in tokens:
        token = str(raw)
        if token.startswith("KEY_DOWN_"):
            keys.add(_clean_state_key(token, "KEY_DOWN_"))
        elif token.startswith("KEY_PRESS_"):
            keys.add(_clean_state_key(token, "KEY_PRESS_"))
        elif token.startswith("KEY_RELEASE_"):
            keys.discard(_clean_state_key(token, "KEY_RELEASE_"))
    return keys


def held_buttons_from_state_tokens(tokens: Iterable[str]) -> set[str]:
    """Extract held mouse-button state from D2E/FDM action-state tokens."""

    buttons: set[str] = set()
    for raw in tokens:
        token = str(raw)
        if token.startswith("MOUSE_") and token.endswith("_DOWN") and not token.startswith(("MOUSE_DX_", "MOUSE_DY_")):
            buttons.add(token[len("MOUSE_") : -len("_DOWN")])
        elif token.startswith("MOUSE_") and token.endswith("_UP") and not token.startswith(("MOUSE_DX_", "MOUSE_DY_")):
            buttons.discard(token[len("MOUSE_") : -len("_UP")])
    return buttons


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
