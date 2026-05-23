#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.torch_idm import torch_available
from fdm_d2e.training.video_idm import train_video_idm


def _summary_message(summary: dict) -> str:
    if "metadata" not in summary:
        return (
            "video IDM worker complete: "
            f"rank={summary.get('rank')} world_size={summary.get('world_size')} "
            f"status={summary.get('status')}"
        )
    return (
        "trained video IDM: "
        f"model={summary['metadata']['model']} device={summary['device']} "
        f"train={summary['metadata']['train_records']} target={summary['metadata']['target_records']} "
        f"metrics={summary['metadata']['metrics_path']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a raw frame-pair video IDM from precomputed D2E tensor caches.")
    parser.add_argument("--config", required=True)
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
    summary = train_video_idm(config)
    print(_summary_message(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
