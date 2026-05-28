#!/usr/bin/env python3
from __future__ import annotations

import argparse

from fdm_d2e.eval.button_hash_sequence_diagnostic import write_button_hash_sequence_diagnostic


def _floats(values: list[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        out.extend(float(part) for part in str(value).split(",") if part)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a CPU prefix mouse-button hash specialist diagnostic for G005.")
    parser.add_argument("--train-records", nargs="+", required=True)
    parser.add_argument("--target-records", nargs="+", required=True)
    parser.add_argument("--base-predictions", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-predictions")
    parser.add_argument("--max-train-rows", type=int, default=320_000)
    parser.add_argument("--max-target-rows", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dim", type=int, default=1 << 18)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--down-weight", type=float, default=8.0)
    parser.add_argument("--up-weight", type=float, default=8.0)
    parser.add_argument("--down-thresholds", nargs="+", default=["0.35,0.5,0.65,0.8,0.9,0.95"])
    parser.add_argument("--up-thresholds", nargs="+", default=["0.35,0.5,0.65,0.8,0.9,0.95"])
    args = parser.parse_args()
    payload = write_button_hash_sequence_diagnostic(
        output_path=args.output,
        train_paths=args.train_records,
        target_paths=args.target_records,
        base_prediction_paths=args.base_predictions,
        output_prediction_path=args.output_predictions,
        max_train_rows=args.max_train_rows,
        max_target_rows=args.max_target_rows,
        epochs=args.epochs,
        dim=args.dim,
        learning_rate=args.learning_rate,
        down_weight=args.down_weight,
        up_weight=args.up_weight,
        down_thresholds=_floats(args.down_thresholds),
        up_thresholds=_floats(args.up_thresholds),
    )
    best = payload["ranked_policies"][0] if payload.get("ranked_policies") else {}
    print(
        "g005 button-hash sequence diagnostic: "
        f"status={payload.get('status')} rows={payload.get('rows')} "
        f"best={best.get('policy')} button={best.get('mouse_button_accuracy')} "
        f"f1={best.get('strict_mouse_button_f1')} no_button_fpr={best.get('strict_no_button_fpr')} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
