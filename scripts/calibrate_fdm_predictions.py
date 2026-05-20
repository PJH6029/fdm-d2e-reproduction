#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.calibrated_fdm import calibrate_fdm_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a trained FDM prediction stream using train-only IDM pseudo-label scale.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = calibrate_fdm_predictions(load_config(args.config))
    checkpoint = summary["checkpoint"]
    print(
        "calibrated fdm predictions: "
        f"model={checkpoint['model']} train={checkpoint['num_training_examples']} "
        f"target={checkpoint['target_examples']} predictions={checkpoint['predictions_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
