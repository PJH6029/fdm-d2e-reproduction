#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.streaming_idm import train_streaming_idm
from fdm_d2e.training.torch_idm import torch_available


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a streaming compact-feature IDM without loading full D2E into GPU memory.")
    parser.add_argument("--config", default="configs/model/idm_streaming_d2e_full_compact.yaml")
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    if not torch_available():
        msg = "torch unavailable; run `uv sync --extra train` or execute inside the MLXP training image"
        if args.require_torch:
            raise SystemExit(msg)
        print(msg)
        return 0
    summary = train_streaming_idm(load_config(args.config))
    print(
        "trained streaming IDM: "
        f"model={summary['metadata']['model']} device={summary['device']} "
        f"train={summary['metadata']['train_records']} target={summary['metadata']['target_records']} "
        f"metrics={summary['metadata']['metrics_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
