#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.reporting.idm_exploration import write_idm_exploration_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the renewed G004 IDM exploration summary.")
    parser.add_argument("--config", default="configs/eval/g004_idm_exploration.yaml")
    parser.add_argument("--output", help="Override config output path")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    payload = write_idm_exploration_summary(load_config(args.config), root=args.root, output_path=args.output)
    print(
        "G004 IDM exploration: "
        f"status={payload['status']} errors={payload['error_count']} "
        f"ranked={len(payload['ranked_candidates'])} output={args.output or load_config(args.config).get('output_path')}"
    )
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
