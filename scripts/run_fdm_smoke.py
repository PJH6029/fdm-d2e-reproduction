#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse

from fdm_d2e.config import load_config
from fdm_d2e.training.train_fdm import run_fdm_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/model/fdm_smoke.yaml')
    parser.add_argument('--labels', required=True)
    args = parser.parse_args()
    checkpoint = run_fdm_smoke(load_config(args.config), args.labels)
    print(f"fdm checkpoint metadata: {checkpoint['predictions_path']} labels_sha256={checkpoint['source_label_sha256'][:12]}")


if __name__ == '__main__':
    main()
