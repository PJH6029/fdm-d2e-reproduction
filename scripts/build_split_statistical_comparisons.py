#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.eval.split_statistics import write_split_statistical_comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description="Build split-specific statistical comparisons for G006 from predictions and D2E target records.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = write_split_statistical_comparisons(load_config(args.config), root=args.root)
    print(f"split statistical comparisons: status={payload['status']} outputs={len(payload['outputs'])}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
