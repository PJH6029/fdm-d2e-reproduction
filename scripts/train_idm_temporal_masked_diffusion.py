#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.temporal_masked_diffusion_idm_trainer import torch_available, train_temporal_masked_diffusion_idm


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an FDM-1-recipe-shaped temporal masked-diffusion IDM on D2E rows.")
    parser.add_argument("--config", default="configs/model/idm_temporal_masked_diffusion_d2e_luma_window5_prefix80k.yaml")
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    if not torch_available():
        msg = "torch unavailable; run `uv sync --extra train` or execute inside the MLXP training image"
        if args.require_torch:
            raise SystemExit(msg)
        print(msg)
        return 0
    config = load_config(args.config)
    config.setdefault("config_path", args.config)
    summary = train_temporal_masked_diffusion_idm(config)
    print(
        "trained temporal masked-diffusion IDM: "
        f"model={summary['model_name']} train={summary['train_rows']} target={summary['target_rows']} "
        f"device={summary['device']} metrics={summary['metrics_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
