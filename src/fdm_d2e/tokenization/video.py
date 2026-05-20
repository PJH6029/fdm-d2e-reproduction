from __future__ import annotations

from typing import Any

from fdm_d2e.schema import validate_named


def build_sequence_pack(records: list[dict[str, Any]], context_length: int = 4) -> dict[str, Any]:
    sequences = []
    for row in records:
        frame = row.get('frame', {})
        sequences.append({
            'sequence_id': row['sequence_id'],
            'timestamp_ns': int(row['timestamp_ns']),
            'frame_features': list(frame.get('features', [])),
            'frame_ref': {'path': frame.get('path', ''), 'index': int(frame.get('index', 0))},
            'ground_truth_tokens': list(row.get('ground_truth_tokens', [])),
            'split': row.get('split', 'unknown'),
        })
    pack = {'schema': 'sequence_pack.v1', 'context_length': context_length, 'sequences': sequences}
    validate_named(pack, 'sequence_pack.schema.json')
    return pack
