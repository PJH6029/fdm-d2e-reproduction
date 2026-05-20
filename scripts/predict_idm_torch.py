#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.torch_idm import predict_torch_idm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained Torch IDM checkpoint over arbitrary D2E records.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = predict_torch_idm(load_config(args.config))
    print(
        "idm predictions: "
        f"records={summary['num_records']} pseudolabels={summary['pseudolabels_path']} "
        f"device={summary['device']}"
    )


if __name__ == "__main__":
    main()
