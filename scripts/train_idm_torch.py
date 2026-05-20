#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.torch_idm import torch_available, train_torch_idm


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Torch MLP IDM on decoded real D2E records.")
    parser.add_argument("--config", default="configs/model/idm_torch_apex8.yaml")
    parser.add_argument("--require-torch", action="store_true", help="Exit non-zero if the train extra / torch is unavailable.")
    args = parser.parse_args()
    if not torch_available():
        msg = "torch unavailable; run `uv sync --extra train` or execute inside the MLXP training image"
        if args.require_torch:
            raise SystemExit(msg)
        print(msg)
        return 0
    summary = train_torch_idm(load_config(args.config))
    print(
        "trained torch IDM: "
        f"model={summary['metadata']['model']} device={summary['device']} "
        f"target={summary['metadata']['target_records']} metrics={summary['metadata']['metrics_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
