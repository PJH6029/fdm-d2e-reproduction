from __future__ import annotations

from typing import Any


def deterministic_split(records: list[dict[str, Any]], train_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda r: (r.get('recording_id', ''), int(r['timestamp_ns'])))
    train, heldout = ordered[:train_count], ordered[train_count:]
    for row in train:
        row['split'] = 'train'
    for row in heldout:
        row['split'] = 'heldout'
    return train, heldout
