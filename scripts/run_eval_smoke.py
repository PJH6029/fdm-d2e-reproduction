#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse

from fdm_d2e.eval.action_metrics import compute_metrics
from fdm_d2e.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--ground-truth', required=True)
    parser.add_argument('--metrics-out', default='outputs/eval/metrics.json')
    parser.add_argument('--failures-out', default='outputs/eval/failure_examples.jsonl')
    args = parser.parse_args()
    predictions = read_jsonl(args.predictions)
    ground_truth = read_jsonl(args.ground_truth)
    metrics = compute_metrics(predictions, ground_truth)
    ensure_dir('outputs/eval')
    write_json(args.metrics_out, metrics)
    by_id = {row['sequence_id']: row for row in ground_truth}
    failures = []
    for pred in predictions:
        gt = by_id.get(pred['sequence_id'])
        if gt and pred.get('predicted_tokens') != gt.get('ground_truth_tokens'):
            failures.append({'sequence_id': pred['sequence_id'], 'predicted_tokens': pred.get('predicted_tokens'), 'ground_truth_tokens': gt.get('ground_truth_tokens')})
    write_jsonl(args.failures_out, failures)
    print(f"eval metrics: {args.metrics_out} examples={metrics['num_examples']} failures={metrics['failure_count']}")


if __name__ == '__main__':
    main()
