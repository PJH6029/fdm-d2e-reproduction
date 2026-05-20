from __future__ import annotations

from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from fdm_d2e.models.fdm import FrequencyFDM
from fdm_d2e.schema import validate_named


def run_fdm_smoke(config: dict[str, Any], labels_path: str | Path) -> dict[str, Any]:
    labels_path = Path(labels_path)
    labels = read_jsonl(labels_path)
    required_source = config.get('label_source_required', 'idm_generated')
    for row in labels:
        validate_named(row, 'idm_pseudolabel.schema.json')
        if row.get('label_source') != required_source:
            raise ValueError(f"FDM canonical smoke requires label_source={required_source}; got {row.get('label_source')}")
    model = FrequencyFDM().fit(labels)
    label_hash = sha256_file(labels_path)
    predictions = []
    for row in labels:
        pred = {
            'schema': 'fdm_prediction.v1',
            'sequence_id': row['sequence_id'],
            'timestamp_ns': int(row['timestamp_ns']),
            'predicted_tokens': model.predict(row),
            'model': 'frequency_fdm_smoke',
            'source_label_artifact': str(labels_path),
            'source_label_sha256': label_hash,
        }
        predictions.append(pred)
    pred_path = Path(config.get('predictions_path', 'outputs/fdm/predictions.jsonl'))
    write_jsonl(pred_path, predictions)
    checkpoint = {
        'schema': 'fdm_checkpoint_metadata.v1',
        'model': 'frequency_fdm_smoke',
        'label_source': 'idm_pseudolabel',
        'source_label_artifact': str(labels_path),
        'source_label_sha256': label_hash,
        'predictions_path': str(pred_path),
        'num_training_examples': len(labels),
        'oracle_ground_truth_control': False,
    }
    validate_named(checkpoint, 'fdm_checkpoint_metadata.schema.json')
    checkpoint_path = Path(config.get('checkpoint_metadata_path', 'outputs/fdm/checkpoint_metadata.json'))
    write_json(checkpoint_path, checkpoint)
    train_log = {
        'schema': 'fdm_train_log.v1',
        'consumed_idm_pseudolabels': True,
        'source_label_artifact': str(labels_path),
        'source_label_sha256': label_hash,
        'checkpoint_metadata_path': str(checkpoint_path),
    }
    write_json(config.get('train_log_path', 'outputs/fdm/train_log.json'), train_log)
    return checkpoint
