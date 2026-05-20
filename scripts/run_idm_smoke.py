#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse

from fdm_d2e.config import load_config
from fdm_d2e.training.train_idm import run_idm_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/model/idm_smoke.yaml')
    args = parser.parse_args()
    metrics = run_idm_smoke(load_config(args.config))
    print(f"idm pseudo-labels: {metrics['pseudo_label_path']} accuracy={metrics['exact_token_sequence_accuracy']:.3f}")


if __name__ == '__main__':
    main()
