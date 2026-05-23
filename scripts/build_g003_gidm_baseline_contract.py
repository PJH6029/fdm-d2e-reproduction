#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.reporting.gidm_baseline_contract import write_gidm_baseline_contract


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the released Generalist-IDM baseline compatibility contract.")
    parser.add_argument("--config", default="configs/eval/g003_gidm_baseline_contract.yaml")
    parser.add_argument("--output", help="Override config output path")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = write_gidm_baseline_contract(config, root=args.root, output_path=args.output)
    print(
        "G003 G-IDM baseline contract: "
        f"status={payload['status']} errors={payload['error_count']} "
        f"row_count={payload['paper_reported_targets']['row_count']} "
        f"output={args.output or config.get('output_path')}"
    )
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
