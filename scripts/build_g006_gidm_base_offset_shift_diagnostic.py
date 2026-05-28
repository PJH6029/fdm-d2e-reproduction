#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.gidm_timestamp_diagnostic import build_gidm_base_offset_shift_diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a released-GIDM base-offset row-shift diagnostic from an existing pilot JSONL pair."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-contract", default="artifacts/eval/g003_gidm_baseline_contract.json")
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--extra-shift", action="append", type=int, default=[])
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    payload = build_gidm_base_offset_shift_diagnostic(
        manifest_path=args.manifest,
        prediction_path=args.predictions,
        target_path=args.targets,
        output_path=args.output,
        baseline_contract_path=args.baseline_contract,
        bin_ms=args.bin_ms,
        extra_shifts=args.extra_shift,
        max_rows=args.max_rows,
    )
    print(json.dumps({k: payload[k] for k in ("status", "decision", "candidate_shifts")}, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
