#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.data.fdm1_alignment_report import build_alignment_report_from_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human-inspectable FDM-1 action-slot timeline alignment report.")
    parser.add_argument("--action-slots", default="outputs/data/fdm1_action_slots/action_slots.jsonl")
    parser.add_argument("--markdown-out", default="artifacts/reports/fdm1_g003_action_alignment_visual_check.md")
    parser.add_argument("--audit-out", default="artifacts/sources/fdm1_g003_action_alignment_visual_check.json")
    parser.add_argument("--expected-bin-ms", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=24)
    parser.add_argument("--recording-id")
    parser.add_argument("--game")
    args = parser.parse_args()

    audit = build_alignment_report_from_jsonl(
        args.action_slots,
        markdown_path=args.markdown_out,
        audit_path=args.audit_out,
        expected_bin_ms=args.expected_bin_ms,
        max_rows=args.max_rows,
        recording_id=args.recording_id,
        game=args.game,
    )
    print(f"built FDM-1 action alignment visual check: status={audit['status']} rows={audit['row_count']} errors={audit['error_count']}")
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
