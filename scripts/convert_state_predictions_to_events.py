#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.state_prediction_events import convert_state_prediction_file
from fdm_d2e.io_utils import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert held-state IDM prediction tokens into D2E press/release event tokens.")
    parser.add_argument("--prediction-path", action="append", required=True, help="State prediction JSONL path or glob. Repeatable.")
    parser.add_argument(
        "--seed-prior-target-path",
        action="append",
        default=[],
        help=(
            "Optional event-state target JSONL/glob aligned with prediction rows. "
            "Only the first row per recording seeds the closed-loop held-state tracker from prior_action_tokens; "
            "subsequent state comes from predictions."
        ),
    )
    parser.add_argument("--output", required=True, help="Converted prediction JSONL output.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--key-press-rows", type=int, default=1)
    parser.add_argument("--key-release-rows", type=int, default=1)
    parser.add_argument("--button-press-rows", type=int, default=1)
    parser.add_argument("--button-release-rows", type=int, default=1)
    parser.add_argument("--drop-mouse-motion", action="store_true")
    parser.add_argument("--include-source-state-tokens", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--progress-output")
    parser.add_argument("--progress-rows", type=int, default=1_000_000)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = convert_state_prediction_file(
        prediction_paths=[str(path) for path in args.prediction_path],
        output_path=args.output,
        seed_prior_paths=[str(path) for path in args.seed_prior_target_path] or None,
        key_press_rows=args.key_press_rows,
        key_release_rows=args.key_release_rows,
        button_press_rows=args.button_press_rows,
        button_release_rows=args.button_release_rows,
        include_mouse_motion=not args.drop_mouse_motion,
        include_state_prediction_tokens=args.include_source_state_tokens,
        max_rows=args.max_rows,
        progress_output_path=args.progress_output,
        progress_rows=args.progress_rows,
    )
    write_json(args.summary, payload)
    print(json.dumps({"status": payload["status"], "rows": payload["rows"], "output": args.output}, sort_keys=True))
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
