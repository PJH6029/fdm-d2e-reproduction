#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.video_idm import precompute_video_idm_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute raw frame-pair tensor caches for video IDM training.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config.setdefault("config_path", args.config)
    summary = precompute_video_idm_cache(config)
    print(
        "video IDM cache precompute: "
        f"status={summary['status']} train_rows={summary['train_cache']['rows']} "
        f"target_rows={summary['target_cache']['rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
