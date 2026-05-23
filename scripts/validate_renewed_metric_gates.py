#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.reporting.renewed_gates import write_renewed_metric_gate_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate renewed FDM-D2E metric gates against current evidence.")
    parser.add_argument("--config", default="configs/eval/renewed_metric_gates.yaml")
    parser.add_argument("--output", help="Override config output path")
    parser.add_argument("--root", default=".")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = write_renewed_metric_gate_audit(config, root=args.root, output_path=args.output)
    print(
        "renewed metric gates: "
        f"status={payload['status']} gate_status={payload['gate_status']} "
        f"gate_errors={payload['gate_error_count']} output={args.output or config.get('output_path')}"
    )
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
