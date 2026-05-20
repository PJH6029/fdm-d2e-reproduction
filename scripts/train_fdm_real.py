#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.train_fdm import train_fdm_real


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a real FDM model from IDM pseudo-labels.")
    parser.add_argument("--config", default="configs/model/fdm_shooter64_surface_motion.yaml")
    args = parser.parse_args()
    summary = train_fdm_real(load_config(args.config))
    checkpoint = summary["checkpoint"]
    print(
        "fdm real checkpoint: "
        f"model={checkpoint['model']} train={checkpoint['num_training_examples']} "
        f"target={checkpoint['target_examples']} predictions={checkpoint['predictions_path']}"
    )


if __name__ == "__main__":
    main()
