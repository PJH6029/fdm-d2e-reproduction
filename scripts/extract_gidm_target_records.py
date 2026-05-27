#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.gidm_targets import TARGET_SPLIT_TAGS, extract_gidm_target_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract exact D2E target JSONL rows for released G-IDM manifest recordings.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--by-recording-root", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--recording-key", action="append", default=[])
    parser.add_argument("--split-tag", action="append", default=[])
    parser.add_argument("--only-existing-predictions", action="store_true")
    parser.add_argument(
        "--filter-to-prediction-windows",
        action="store_true",
        help="For chunked/partial G-IDM pilots, write only target rows covered by predicted chunk timestamp windows.",
    )
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    roots = args.by_recording_root or ["outputs/data/d2e_full_corpus_shards_accel64/shard_*/by_recording"]
    payload = extract_gidm_target_records(
        manifest_path=args.manifest,
        by_recording_roots=roots,
        output_path=args.output,
        summary_out=args.summary_out,
        recording_keys=args.recording_key,
        split_tags=args.split_tag or TARGET_SPLIT_TAGS,
        only_existing_predictions=args.only_existing_predictions,
        filter_to_prediction_windows=args.filter_to_prediction_windows,
        bin_ms=args.bin_ms,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
