#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.streaming_action_diagnostics import write_streaming_action_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream FDM predictions and target rows into G002 action/error diagnostics.")
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=["outputs/fdm_streaming_d2e_full_compact/torch_model/predictions.jsonl"],
        help="Prediction JSONL paths or glob patterns.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["outputs/fdm_streaming_d2e_full_compact/fdm_target_shards/shard_*.jsonl"],
        help="Target JSONL paths or glob patterns in prediction order.",
    )
    parser.add_argument("--output", default="artifacts/eval/g002_pvc_action_error_diagnostics.json")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = write_streaming_action_diagnostics(
        prediction_paths=args.predictions,
        target_paths=args.targets,
        output_path=args.output,
        max_rows=args.max_rows,
        top_k=args.top_k,
    )
    print(
        "g002 pvc action diagnostics: "
        f"status={payload['status']} rows={payload['alignment']['rows_seen']} "
        f"mismatches={payload['alignment']['sequence_id_mismatches']} output={args.output}"
    )
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
