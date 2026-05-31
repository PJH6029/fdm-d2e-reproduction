#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.data.fdm1_mouse_bins import build_fitted_mouse_bins


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit global FDM-1 mouse movement bins from D2E training 50ms rows.")
    parser.add_argument("--input-records", action="append", required=True)
    parser.add_argument("--base-tokenization-config", default="configs/tokenization/fdm1_action_slots.json")
    parser.add_argument("--bins-output", default="artifacts/sources/fdm1_g003_fitted_mouse_bins.json")
    parser.add_argument("--fitted-config-output", default="artifacts/sources/fdm1_action_slots_fitted_config.json")
    parser.add_argument("--split", default="train_core")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    result = build_fitted_mouse_bins(
        args.input_records,
        base_tokenization_config=args.base_tokenization_config,
        bins_output_path=args.bins_output,
        fitted_config_path=args.fitted_config_output,
        split=args.split,
        max_records=args.max_records,
    )
    summary = result["summary"]
    print(
        "fit FDM-1 mouse bins: "
        f"status={summary['status']} records_used={summary['records_used']} mouse_events={summary['mouse_events']} output={args.bins_output}"
    )
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
