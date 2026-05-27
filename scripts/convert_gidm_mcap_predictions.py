#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.gidm_adapter import convert_gidm_mcap_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert released Generalist-IDM predicted MCAP files to local predictions.jsonl.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-records", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--timestamp-shift-ns", type=int, default=0)
    parser.add_argument(
        "--auto-timestamp-shift-from-screen",
        action="store_true",
        help="For each recording, align predicted MCAP timestamps to ground-truth MCAP first-screen timestamp.",
    )
    parser.add_argument(
        "--filter-targets-to-prediction-windows",
        action="store_true",
        help="For chunked/partial G-IDM pilots, convert only target rows covered by predicted chunk timestamp windows.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    payload = convert_gidm_mcap_predictions(
        manifest_path=args.manifest,
        target_record_paths=args.target_records,
        output_path=args.output,
        summary_out=args.summary_out,
        bin_ms=args.bin_ms,
        timestamp_shift_ns=args.timestamp_shift_ns,
        auto_timestamp_shift_from_screen=args.auto_timestamp_shift_from_screen,
        filter_targets_to_prediction_windows=args.filter_targets_to_prediction_windows,
        allow_missing=args.allow_missing,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
