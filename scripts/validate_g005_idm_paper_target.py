#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.reporting.g005_idm_paper_target import write_g005_idm_paper_target_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate G005 full-corpus IDM paper-target completion evidence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = write_g005_idm_paper_target_audit(config, root=args.root, output_path=args.output)
    output = args.output or config.get("output_path", "artifacts/idm/g005_idm_paper_target_audit.json")
    print(f"g005 paper-target audit: status={payload['status']} errors={payload['error_count']} output={output}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
