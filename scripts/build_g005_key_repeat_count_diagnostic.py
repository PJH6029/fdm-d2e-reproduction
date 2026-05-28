#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.state_transition_diagnostics import build_key_repeat_count_prior_metrics
from fdm_d2e.io_utils import write_json


def _floats(values: list[str]) -> list[float]:
    return [float(value) for item in values for value in str(item).split(",") if value.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 count-aware held-key repeat diagnostic.")
    parser.add_argument("--train-records", nargs="+", required=True)
    parser.add_argument("--target-records", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/idm/g005_idm_key_repeat_count_prior_prefix320k_metrics.json")
    parser.add_argument("--max-train-rows", type=int, default=320_000)
    parser.add_argument("--max-target-rows", type=int, default=320_000)
    parser.add_argument("--thresholds", nargs="+", default=["0.05,0.1,0.2,0.35,0.5,0.65"])
    parser.add_argument("--min-support", type=int, default=3)
    args = parser.parse_args()

    payload = build_key_repeat_count_prior_metrics(
        train_paths=args.train_records,
        target_paths=args.target_records,
        max_train_rows=args.max_train_rows,
        max_target_rows=args.max_target_rows,
        thresholds=_floats(args.thresholds),
        min_support=args.min_support,
    )
    write_json(args.output, payload)

    ranked: list[dict[str, object]] = []
    for name, policy in payload["policies"].items():
        group = policy["all"]
        paper = group["paper_compatible"]
        ranked.append(
            {
                "policy": name,
                "keyboard_accuracy": paper["keyboard"].get("key_accuracy"),
                "mouse_button_accuracy": paper["mouse_button"].get("button_accuracy"),
                "pearson_x": paper["mouse_move"].get("pearson_x"),
                "pearson_y": paper["mouse_move"].get("pearson_y"),
            }
        )
    ranked.sort(
        key=lambda item: (
            item["keyboard_accuracy"] if item["keyboard_accuracy"] is not None else -1.0,
            item["mouse_button_accuracy"] if item["mouse_button_accuracy"] is not None else -1.0,
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else {}
    print(
        "g005 key-repeat count diagnostic: "
        f"status={payload['schema']} rows={payload['rows']} "
        f"best={best.get('policy')} keyboard={best.get('keyboard_accuracy')} "
        f"button={best.get('mouse_button_accuracy')} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
