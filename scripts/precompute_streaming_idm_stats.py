#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import read_json, write_json
from fdm_d2e.training.streaming_idm import scan_streaming_idm_stats_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute streaming IDM stats before a distributed torchrun.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs/idm_streaming_full"))
    stats_path = output_dir / "streaming_stats.json"
    if stats_path.exists() and not args.force and not bool(config.get("rescan_stats", False)):
        stats = read_json(stats_path)
        print(
            "streaming IDM stats already exist: "
            f"path={stats_path} examples={stats.get('num_examples')} input_dim={stats.get('input_dim')}"
        )
        return 0

    stats = scan_streaming_idm_stats_from_config(config)
    write_json(stats_path, stats)
    print(
        "precomputed streaming IDM stats: "
        f"path={stats_path} examples={stats.get('num_examples')} input_dim={stats.get('input_dim')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
