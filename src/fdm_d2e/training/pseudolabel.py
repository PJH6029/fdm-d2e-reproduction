from __future__ import annotations

from typing import Any

from fdm_d2e.io_utils import stable_hash_json
from fdm_d2e.models.idm import NearestFeatureIDM
from fdm_d2e.schema import validate_named


def generate_pseudolabels(train_records: list[dict[str, Any]], target_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model = NearestFeatureIDM().fit(train_records)
    rows: list[dict[str, Any]] = []
    train_hash = stable_hash_json([{'id': r['sequence_id'], 'tokens': r.get('ground_truth_tokens', [])} for r in train_records])
    for record in target_records:
        tokens, confidence = model.predict(record)
        row = {
            'schema': 'idm_pseudolabel.v1',
            'sequence_id': record['sequence_id'],
            'timestamp_ns': int(record['timestamp_ns']),
            'predicted_tokens': tokens,
            'label_source': 'idm_generated',
            'confidence': confidence,
            'model': 'nearest_feature_signature_idm_smoke',
            'training_split_hash': train_hash,
            'input_window': {'frame_ref': record.get('frame', {}).get('path', ''), 'frame_index': int(record.get('frame', {}).get('index', 0))},
        }
        validate_named(row, 'idm_pseudolabel.schema.json')
        rows.append(row)
    return rows
