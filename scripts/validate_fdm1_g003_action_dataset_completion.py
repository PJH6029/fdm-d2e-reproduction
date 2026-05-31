#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.reporting.fdm1_g003_completion import write_fdm1_g003_action_dataset_completion_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate reset G003 FDM-1 action dataset completion evidence.")
    parser.add_argument("--config", default="configs/eval/fdm1_g003_action_dataset_completion.yaml")
    parser.add_argument("--output")
    parser.add_argument("--root", default=".")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = write_fdm1_g003_action_dataset_completion_audit(config, root=args.root, output_path=args.output)
    print(f"fdm1 g003 action dataset completion: status={payload['status']} errors={payload['error_count']} output={args.output or config.get('output_path')}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
