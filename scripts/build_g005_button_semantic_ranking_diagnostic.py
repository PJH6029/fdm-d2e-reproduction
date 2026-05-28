#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.button_semantic_ranking_diagnostic import write_button_semantic_ranking_diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 mouse-button semantic/ranking diagnostics for IDM predictions.")
    parser.add_argument("--prediction", action="append", required=True, help="Prediction JSONL path/glob. Can be repeated.")
    parser.add_argument("--target", action="append", required=True, help="Target JSONL path/glob. Can be repeated.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--offset-radius", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()
    offsets = list(range(-abs(args.offset_radius), abs(args.offset_radius) + 1))
    payload = write_button_semantic_ranking_diagnostic(
        prediction_paths=args.prediction,
        target_paths=args.target,
        output_path=args.output,
        max_rows=args.max_rows,
        offsets=offsets,
        max_examples=args.max_examples,
    )
    base = payload["base"]
    print(
        "g005 button semantic diagnostic: "
        f"status={payload['status']} rows={payload['alignment']['rows_seen']} "
        f"pred={base['predicted_examples']} gt={base['ground_truth_examples']} "
        f"exact_tp={base['exact_true_positive_examples']} semantic_overlap={base['semantic_any_button_overlap_examples']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
