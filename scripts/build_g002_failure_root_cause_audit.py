#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.reporting.failure_root_cause import write_failure_root_cause_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the renewed G002 failure root-cause audit.")
    parser.add_argument("--config", default="configs/eval/g002_failure_root_cause.yaml")
    parser.add_argument("--output", help="Override config output path")
    parser.add_argument("--root", default=".")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = write_failure_root_cause_audit(config, root=args.root, output_path=args.output)
    top = payload["ranked_root_causes"][0]["id"] if payload.get("ranked_root_causes") else "none"
    print(
        "g002 failure root-cause audit: "
        f"status={payload['status']} errors={payload['error_count']} top={top} output={args.output or config.get('output_path')}"
    )
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
