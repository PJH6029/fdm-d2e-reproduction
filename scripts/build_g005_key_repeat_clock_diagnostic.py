#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.key_repeat_clock_diagnostic import write_key_repeat_clock_diagnostic


def _floats(values: list[str]) -> list[float]:
    return [float(value) for item in values for value in str(item).split(",") if value.strip()]


def _ints(values: list[str]) -> list[int]:
    return [int(value) for item in values for value in str(item).split(",") if value.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 per-key repeat-clock prefix diagnostic.")
    parser.add_argument("--train-records", nargs="+", required=True)
    parser.add_argument("--target-records", nargs="+", required=True)
    parser.add_argument("--base-predictions", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/idm/g005_idm_key_repeat_clock_diagnostic_prefix50k.json")
    parser.add_argument("--output-predictions", default="")
    parser.add_argument("--max-train-rows", type=int, default=320_000)
    parser.add_argument("--max-target-rows", type=int, default=50_000)
    parser.add_argument("--candidate-key-count", type=int, default=0)
    parser.add_argument("--thresholds", nargs="+", default=["0.05,0.1,0.2,0.35,0.5"])
    parser.add_argument("--min-supports", nargs="+", default=["1,3,8"])
    parser.add_argument("--clock-modes", nargs="+", default=["predicted", "teacher_forced"])
    args = parser.parse_args()
    payload = write_key_repeat_clock_diagnostic(
        train_paths=args.train_records,
        target_paths=args.target_records,
        base_prediction_paths=args.base_predictions,
        output_path=args.output,
        output_prediction_path=args.output_predictions or None,
        max_train_rows=args.max_train_rows,
        max_target_rows=args.max_target_rows,
        candidate_key_count=args.candidate_key_count,
        thresholds=_floats(args.thresholds),
        min_supports=_ints(args.min_supports),
        clock_modes=args.clock_modes,
    )
    best = payload["ranked_policies"][0] if payload.get("ranked_policies") else {}
    print(
        "g005 key-repeat clock diagnostic: "
        f"status={payload['status']} rows={payload['rows']} train_rows={payload['train_rows']} "
        f"best={best.get('policy')} keyboard={best.get('keyboard_accuracy')} "
        f"button={best.get('mouse_button_accuracy')} pearson_x={best.get('pearson_x')} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
