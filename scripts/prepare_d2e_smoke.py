#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse

from fdm_d2e.config import load_config
from fdm_d2e.data.d2e_reader import prepare_smoke_dataset
from fdm_d2e.io_utils import ensure_dir, write_json, write_jsonl
from fdm_d2e.tokenization.actions import add_tokens, build_vocab
from fdm_d2e.tokenization.video import build_sequence_pack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data/d2e_smoke.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    prepared = prepare_smoke_dataset(config)
    output_dir = config.get('output_dir', 'outputs')
    records = add_tokens(prepared['records'])
    train = [r for r in records if r['split'] == 'train']
    heldout = [r for r in records if r['split'] == 'heldout']
    write_jsonl(f'{output_dir}/data/all_records.jsonl', records)
    write_jsonl(f'{output_dir}/data/train.jsonl', train)
    write_jsonl(f'{output_dir}/data/heldout.jsonl', heldout)
    tok_dir = ensure_dir(f'{output_dir}/tokenization')
    vocab = build_vocab(records)
    pack = build_sequence_pack(records)
    write_json(tok_dir / 'action_vocab.json', vocab)
    write_json(tok_dir / 'sample_sequence_pack.json', pack)
    write_json(f'{output_dir}/run_manifest.json', {
        'schema': 'run_manifest.v1',
        'stage': 'prepare_d2e_smoke',
        'data_manifest': f'{output_dir}/data/manifest.json',
        'action_vocab': f'{output_dir}/tokenization/action_vocab.json',
        'sequence_pack': f'{output_dir}/tokenization/sample_sequence_pack.json',
        'non_parity_notice': 'recipe-faithful scaled reproduction; not FDM-1 parity',
    })
    print(f'prepared {len(records)} records; train={len(train)} heldout={len(heldout)}')


if __name__ == '__main__':
    main()
