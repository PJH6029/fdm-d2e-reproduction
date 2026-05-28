#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.key_phase_repeat_diagnostic import write_key_phase_repeat_diagnostic


def _ints(values: list[str]) -> list[int]:
    return [int(v) for item in values for v in str(item).split(",") if str(v).strip()]


def _floats(values: list[str]) -> list[float]:
    return [float(v) for item in values for v in str(item).split(",") if str(v).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 phase-aware key-repeat prefix diagnostic.")
    parser.add_argument("--train-records", nargs="+", required=True)
    parser.add_argument("--target-records", nargs="+", required=True)
    parser.add_argument("--base-predictions", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/idm/g005_idm_key_phase_repeat_diagnostic_prefix50k.json")
    parser.add_argument("--max-train-rows", type=int, default=320_000)
    parser.add_argument("--max-target-rows", type=int, default=50_000)
    parser.add_argument("--periods", nargs="+", default=["2,3,4,5,6,8,10,12,16,20,24,32,40"])
    parser.add_argument("--thresholds", nargs="+", default=["0.05,0.1,0.2,0.35,0.5,0.65"])
    parser.add_argument("--min-support", type=int, default=5)
    args = parser.parse_args()
    payload = write_key_phase_repeat_diagnostic(
        train_paths=args.train_records,
        target_paths=args.target_records,
        base_prediction_paths=args.base_predictions,
        output_path=args.output,
        max_train_rows=args.max_train_rows,
        max_target_rows=args.max_target_rows,
        periods=_ints(args.periods),
        thresholds=_floats(args.thresholds),
        min_support=args.min_support,
    )
    best = payload.get("best_policy") or {}
    print(
        "g005 key-phase repeat diagnostic: "
        f"status={payload['status']} rows={payload['rows']} base={payload['base'].get('keyboard_accuracy')} "
        f"best={best.get('policy')} keyboard={best.get('keyboard_accuracy')} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
