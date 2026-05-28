#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.key_repeat_specialist import write_key_repeat_specialist_matrix


def _floats(values: list[str]) -> list[float]:
    return [float(v) for item in values for v in str(item).split(",") if str(v).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 key-repeat specialist prefix diagnostic matrix.")
    parser.add_argument("--train-records", nargs="+", required=True)
    parser.add_argument("--target-records", nargs="+", required=True)
    parser.add_argument("--base-predictions", nargs="+", default=[])
    parser.add_argument("--output", default="artifacts/idm/g005_idm_key_repeat_specialist_matrix_prefix320k.json")
    parser.add_argument("--max-train-rows", type=int, default=320_000)
    parser.add_argument("--max-target-rows", type=int, default=320_000)
    parser.add_argument("--press-thresholds", nargs="+", default=["0.05,0.1,0.2,0.35,0.5"])
    parser.add_argument("--release-thresholds", nargs="+", default=["0.05,0.1,0.2,0.35,0.5"])
    parser.add_argument("--min-support", type=int, default=5)
    args = parser.parse_args()
    payload = write_key_repeat_specialist_matrix(
        train_paths=args.train_records,
        target_paths=args.target_records,
        base_prediction_paths=args.base_predictions,
        output_path=args.output,
        max_train_rows=args.max_train_rows,
        max_target_rows=args.max_target_rows,
        press_thresholds=_floats(args.press_thresholds),
        release_thresholds=_floats(args.release_thresholds),
        min_support=args.min_support,
    )
    best = payload["ranked_policies"][0] if payload.get("ranked_policies") else {}
    print(
        "g005 key-repeat specialist matrix: "
        f"status={payload['status']} rows={payload['rows']} best={best.get('policy')} "
        f"keyboard={best.get('keyboard_accuracy')} button={best.get('mouse_button_accuracy')} "
        f"pearson_x={best.get('pearson_x')} pearson_y={best.get('pearson_y')} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
