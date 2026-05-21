#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.reporting.quality_gates import write_final_quality_gate_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate final G001-G009 ultragoal evidence gates without mutating goal state.")
    parser.add_argument("--config", default="configs/eval/final_quality_gates.yaml")
    parser.add_argument("--output", help="Override config output path")
    parser.add_argument("--root", default=".")
    parser.add_argument("--allow-fail", action="store_true", help="Exit 0 even when the audit fails; useful for recording current incomplete state.")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = write_final_quality_gate_audit(config, root=args.root, output_path=args.output)
    print(f"final quality gates: status={payload['status']} errors={payload['error_count']} output={args.output or config.get('output_path')}")
    if payload["status"] == "pass" or args.allow_fail:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
