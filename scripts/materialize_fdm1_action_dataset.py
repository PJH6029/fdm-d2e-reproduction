#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.data.fdm1_action_dataset import write_action_slot_dataset
from fdm_d2e.io_utils import read_json, read_jsonl
from fdm_d2e.tokenization.fdm1_actions import ActionSlotTokenizer, MouseMoveBinner


def _load_records(paths: list[str], *, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            rows.append(row)
            if max_records is not None and len(rows) >= max_records:
                return rows
    return rows


def _tokenizer_from_config(path: str | None, *, k_event_slots: int | None, bin_ms: int) -> ActionSlotTokenizer:
    boundaries = None
    compound = True
    configured_k = None
    if path and Path(path).exists():
        config = read_json(path)
        configured_k = int(config.get("k_event_slots_default", 8))
        mouse = config.get("mouse_move", {})
        if mouse.get("positive_boundaries_default"):
            boundaries = tuple(float(value) for value in mouse["positive_boundaries_default"])
        compound = str(mouse.get("default", "compound")).lower() == "compound"
    return ActionSlotTokenizer(
        k_event_slots=int(k_event_slots if k_event_slots is not None else configured_k if configured_k is not None else 8),
        mouse_binner=MouseMoveBinner(boundaries=boundaries or MouseMoveBinner().boundaries, compound=compound),
        bin_ms=int(bin_ms),
    )


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args()
    defaults: dict[str, Any] = {}
    if known.config:
        defaults = load_config(known.config)

    parser = argparse.ArgumentParser(description="Materialize FDM-1-style fixed action-slot JSONL from D2E 50ms window records.", parents=[pre])
    parser.add_argument("--input-records", action="append", default=defaults.get("input_records"), required=not bool(defaults.get("input_records")))
    parser.add_argument("--output-dir", default=defaults.get("output_dir", "outputs/data/fdm1_action_slots"))
    parser.add_argument("--tokenization-config", default=defaults.get("tokenization_config", "configs/tokenization/fdm1_action_slots.json"))
    parser.add_argument("--bin-ms", type=int, default=int(defaults.get("bin_ms", 50)))
    parser.add_argument("--frame-fps", type=int, default=int(defaults.get("frame_fps", 20)))
    parser.add_argument("--k-event-slots", type=int, default=defaults.get("k_event_slots"))
    parser.add_argument("--click-horizon-seconds", type=float, default=float(defaults.get("click_horizon_seconds", 1.0)))
    parser.add_argument("--click-grid-width", type=int, default=int(defaults.get("click_grid_width", 32)))
    parser.add_argument("--click-grid-height", type=int, default=int(defaults.get("click_grid_height", 18)))
    parser.add_argument("--screen-width", type=int, default=int(defaults.get("screen_width", 854)))
    parser.add_argument("--screen-height", type=int, default=int(defaults.get("screen_height", 480)))
    parser.add_argument("--max-records", type=int, default=defaults.get("max_records"))
    args = parser.parse_args()

    input_records = args.input_records if isinstance(args.input_records, list) else [args.input_records]
    rows = _load_records([str(path) for path in input_records], max_records=args.max_records)
    if not rows:
        raise SystemExit("no input records loaded")
    tokenizer = _tokenizer_from_config(args.tokenization_config, k_event_slots=args.k_event_slots, bin_ms=args.bin_ms)
    result = write_action_slot_dataset(
        rows,
        output_dir=args.output_dir,
        source_paths=[str(path) for path in input_records],
        tokenization_config_path=args.tokenization_config,
        tokenizer=tokenizer,
        bin_ms=args.bin_ms,
        frame_fps=args.frame_fps,
        click_horizon_seconds=args.click_horizon_seconds,
        click_grid=(args.click_grid_width, args.click_grid_height),
        screen_size=(args.screen_width, args.screen_height),
    )
    summary = result["summary"]
    alignment = result["alignment"]
    print(
        "materialized FDM-1 action slots: "
        f"records={summary['records']} unique_tokens={summary['unique_token_count']} "
        f"alignment={alignment['status']} output_dir={args.output_dir}"
    )
    if alignment["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
