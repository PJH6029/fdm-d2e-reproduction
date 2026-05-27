#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.state_transition_diagnostics import write_state_transition_diagnostics
from fdm_d2e.io_utils import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 IDM state-transition/repeat-prior diagnostic artifacts.")
    parser.add_argument("--train-records", nargs="+", required=True)
    parser.add_argument("--target-records", nargs="+", required=True)
    parser.add_argument("--output-dir", default="artifacts/idm")
    parser.add_argument("--summary-output", default="artifacts/idm/g005_state_transition_diagnostics_summary.json")
    parser.add_argument("--max-train-rows", type=int, default=320_000)
    parser.add_argument("--max-target-rows", type=int, default=320_000)
    parser.add_argument("--prefix", default="g005_idm")
    args = parser.parse_args()
    summary = write_state_transition_diagnostics(
        train_paths=args.train_records,
        target_paths=args.target_records,
        output_dir=args.output_dir,
        max_train_rows=args.max_train_rows,
        max_target_rows=args.max_target_rows,
        prefix=args.prefix,
    )
    write_json(args.summary_output, summary)
    print(f"g005 state-transition diagnostics: status={summary['status']} summary={args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
