#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse

from fdm_d2e.io_utils import read_jsonl, write_json
from fdm_d2e.evidence import write_evidence_index
from fdm_d2e.rollout.harness import tokens_to_rollout_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['stub', 'replay', 'live'], default='stub')
    parser.add_argument('--predictions', default='outputs/fdm/predictions.jsonl')
    parser.add_argument('--output', default='outputs/rollout/rollout_smoke.json')
    args = parser.parse_args()
    predictions = read_jsonl(args.predictions)
    smoke = tokens_to_rollout_actions(predictions, args.mode)
    write_json(args.output, smoke)
    write_evidence_index()
    print(f"rollout smoke: {args.output} mode={args.mode} actions={smoke['num_actions']}")


if __name__ == '__main__':
    main()
