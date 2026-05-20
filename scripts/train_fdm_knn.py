#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.knn_fdm import train_knn_fdm


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a KNN FDM from IDM pseudo-labels.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = train_knn_fdm(load_config(args.config))
    print(
        "knn fdm checkpoint: "
        f"model={summary['checkpoint']['model']} train={summary['checkpoint']['num_training_examples']} "
        f"target={summary['checkpoint']['target_examples']} predictions={summary['predictions_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
