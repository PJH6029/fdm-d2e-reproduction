from __future__ import annotations

from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import read_jsonl, write_json, write_jsonl
from fdm_d2e.training.pseudolabel import generate_pseudolabels


def run_idm_smoke(config: dict[str, Any]) -> dict[str, Any]:
    train = read_jsonl(config['train_split'])
    targets = read_jsonl(config.get('all_records') or config['target_split'])
    labels = generate_pseudolabels(train, targets)
    pseudo_path = Path(config.get('pseudo_label_path', 'outputs/idm/pseudolabels.jsonl'))
    write_jsonl(pseudo_path, labels)
    by_id = {row['sequence_id']: row for row in targets}
    correct = 0
    total = 0
    for label in labels:
        gt = by_id[label['sequence_id']].get('ground_truth_tokens', [])
        if label['predicted_tokens'] == gt:
            correct += 1
        total += 1
    metrics = {
        'schema': 'metrics.v1',
        'stage': 'idm',
        'num_examples': total,
        'exact_token_sequence_accuracy': correct / total if total else 0.0,
        'pseudo_label_path': str(pseudo_path),
        'label_source': 'idm_generated',
    }
    write_json(config.get('metrics_path', 'outputs/idm/metrics.json'), metrics)
    return metrics
