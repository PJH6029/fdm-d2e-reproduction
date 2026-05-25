#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.idm_alignment_shifts import write_idm_alignment_shift_diagnostics


def _paths(values: list[str] | None) -> list[str]:
    return [str(value) for value in values or []]


def _shifts(value: str) -> list[int]:
    shifts: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            shifts.append(int(item))
    if not shifts:
        raise argparse.ArgumentTypeError("at least one shift is required")
    return shifts


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose IDM prediction/target row-shift alignment over token JSONL rows.")
    parser.add_argument("--target-path", action="append", required=True, help="Target JSONL path or glob. Repeatable.")
    parser.add_argument("--prediction-path", action="append", default=[], help="Prediction JSONL path or glob. Repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--shifts", type=_shifts, default=_shifts("-3,-2,-1,0,1,2,3"))
    parser.add_argument("--split-tag", action="append", default=["temporal", "heldout_recording", "heldout_game"])
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--empty-bins-as-correct", action="store_true")
    parser.add_argument("--progress-output")
    parser.add_argument("--progress-rows", type=int, default=1_000_000)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = write_idm_alignment_shift_diagnostics(
        target_paths=_paths(args.target_path),
        prediction_paths=_paths(args.prediction_path),
        output_path=args.output,
        shifts=args.shifts,
        split_tags=[str(tag) for tag in args.split_tag],
        model_name=args.model_name,
        max_rows=args.max_rows,
        empty_bins_as_correct=bool(args.empty_bins_as_correct),
        progress_output_path=args.progress_output,
        progress_rows=args.progress_rows,
    )
    print(
        "idm alignment shifts: "
        f"status={payload['status']} rows={payload['alignment']['rows_seen']} "
        f"fragments={payload['block_stats']['recording_fragments']} output={args.output}"
    )
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
