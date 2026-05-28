#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.key_event_taxonomy import write_key_event_taxonomy


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 key-event taxonomy and oracle diagnostics.")
    parser.add_argument("--target-records", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/idm/g005_idm_key_event_taxonomy_prefix320k.json")
    parser.add_argument("--max-rows", type=int, default=320_000)
    parser.add_argument("--split-tags", nargs="+", default=["temporal", "heldout_recording", "heldout_game"])
    args = parser.parse_args()
    payload = write_key_event_taxonomy(
        target_paths=args.target_records,
        output_path=args.output,
        max_rows=args.max_rows,
        split_tags=args.split_tags,
    )
    print(
        "g005 key-event taxonomy: "
        f"status={payload['status']} rows={payload['rows']} key_tokens={payload['total_key_tokens']} "
        f"visible_fraction={payload['state_transition_visible_fraction']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
