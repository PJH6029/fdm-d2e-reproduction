#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.gidm_adapter import write_gidm_inference_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a released Generalist-IDM inference manifest for local D2E heldout rows.")
    parser.add_argument("--target-records", action="append", default=[])
    parser.add_argument("--decode-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="open-world-agents/Generalist-IDM-1B")
    parser.add_argument("--split-tag", action="append", default=[])
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument(
        "--use-decode-summary-counts",
        action="store_true",
        help="Build from per-recording decode_summary split_counts instead of scanning large target JSONL rows.",
    )
    args = parser.parse_args()
    if not args.target_records and not args.use_decode_summary_counts:
        parser.error("--target-records is required unless --use-decode-summary-counts is set")
    payload = write_gidm_inference_manifest(
        target_record_paths=args.target_records,
        decode_summary_path=args.decode_summary,
        output_dir=args.output_dir,
        output_path=args.output,
        model=args.model,
        split_tags=args.split_tag,
        max_recordings=args.max_recordings,
        use_decode_summary_counts=args.use_decode_summary_counts,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "recording_count": payload["recording_count"],
                "target_rows": payload["target_rows"],
                "missing_decode_rows": len(payload["missing_decode_rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
