#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.final_eval import build_g006_final_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final G006 endpoint/failure/claim artifacts from completed split-aware G003/G004 evidence.")
    parser.add_argument("--config", default="configs/eval/g006_final_artifacts.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--summary-out", default="artifacts/eval/g006_final_artifact_build_summary.json")
    parser.add_argument("--allow-fail", action="store_true", help="Exit 0 even when final artifacts are still failing/pending.")
    args = parser.parse_args()
    payload = build_g006_final_artifacts(load_config(args.config), root=args.root)
    write_json(Path(args.root) / args.summary_out, payload)
    print(f"g006 final artifacts: status={payload['status']} statuses={payload['statuses']} summary={args.summary_out}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
