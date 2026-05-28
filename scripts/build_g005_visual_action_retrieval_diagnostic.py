#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.visual_action_retrieval_diagnostic import write_visual_action_retrieval_diagnostic


def _floats(values: list[str]) -> list[float]:
    return [float(v) for item in values for v in str(item).split(",") if str(v).strip()]


def _ints(values: list[str]) -> list[int]:
    return [int(v) for item in values for v in str(item).split(",") if str(v).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 visual/state action-memory prefix diagnostic.")
    parser.add_argument("--train-records", nargs="+", required=True)
    parser.add_argument("--target-records", nargs="+", required=True)
    parser.add_argument("--base-predictions", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/idm/g005_idm_visual_action_retrieval_diagnostic_prefix50k.json")
    parser.add_argument("--output-predictions", default="")
    parser.add_argument("--max-train-rows", type=int, default=100_000)
    parser.add_argument("--max-target-rows", type=int, default=50_000)
    parser.add_argument("--context-names", nargs="+", default=[])
    parser.add_argument("--thresholds", nargs="+", default=["0.2,0.5"])
    parser.add_argument("--min-supports", nargs="+", default=["1,3"])
    args = parser.parse_args()
    payload = write_visual_action_retrieval_diagnostic(
        train_paths=args.train_records,
        target_paths=args.target_records,
        base_prediction_paths=args.base_predictions,
        output_path=args.output,
        output_prediction_path=args.output_predictions or None,
        max_train_rows=args.max_train_rows,
        max_target_rows=args.max_target_rows,
        context_names=args.context_names or ("state_visual_grid4", "state_visual_features", "visual_grid4", "visual_features", "state_only"),
        thresholds=_floats(args.thresholds),
        min_supports=_ints(args.min_supports),
    )
    best = payload["ranked_policies"][0] if payload.get("ranked_policies") else {}
    print(
        "g005 visual-action retrieval diagnostic: "
        f"status={payload['status']} rows={payload['rows']} policies={payload['policy_count']} "
        f"best={best.get('policy')} keyboard={best.get('keyboard_accuracy')} "
        f"button={best.get('mouse_button_accuracy')} pearson_x={best.get('pearson_x')} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
