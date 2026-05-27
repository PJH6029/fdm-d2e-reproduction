#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.gidm_targets import TARGET_SPLIT_TAGS, enrich_gidm_manifest_with_target_timing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enrich released G-IDM inference manifest rows with target timestamp/bin ranges for chunk scheduling."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--by-recording-root", action="append", default=[])
    parser.add_argument("--target-records", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--recording-key", action="append", default=[])
    parser.add_argument("--split-tag", action="append", default=[])
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    roots = args.by_recording_root or ([] if args.target_records else ["outputs/data/d2e_full_corpus_shards_accel64/shard_*/by_recording"])
    payload = enrich_gidm_manifest_with_target_timing(
        manifest_path=args.manifest,
        by_recording_roots=roots,
        target_record_paths=args.target_records,
        output_path=args.output,
        summary_out=args.summary_out,
        recording_keys=args.recording_key,
        split_tags=args.split_tag or TARGET_SPLIT_TAGS,
        max_recordings=args.max_recordings,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
