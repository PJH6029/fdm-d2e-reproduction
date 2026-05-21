#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.streaming_fdm import train_streaming_fdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a streaming FDM from IDM pseudo-labels.")
    parser.add_argument("--config", default="configs/model/fdm_streaming_d2e_full_compact.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    config.setdefault("config_path", args.config)
    summary = train_streaming_fdm(config)
    if summary.get("schema") == "streaming_fdm_worker_summary.v1":
        print(
            "streaming fdm worker complete: "
            f"rank={summary.get('rank')} world_size={summary.get('world_size')}"
        )
        return
    checkpoint = summary["checkpoint"]
    print(
        "streaming fdm checkpoint: "
        f"model={checkpoint['model']} train={checkpoint['num_training_examples']} "
        f"target={checkpoint['target_examples']} predictions={checkpoint['predictions_path']}"
    )


if __name__ == "__main__":
    main()
