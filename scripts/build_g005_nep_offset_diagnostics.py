#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.idm_alignment_shifts import build_idm_alignment_shift_diagnostics
from fdm_d2e.io_utils import read_json, write_json
from fdm_d2e.reporting.g005_nep_offset_diagnostics import build_g005_nep_offset_summary


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


def _read_optional_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return read_json(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build G005 NEP/temporal-offset diagnostics over target JSONL rows. "
            "This is target-autocorrelation evidence only, not trained-model evidence."
        )
    )
    parser.add_argument("--target-path", action="append", required=True, help="Target JSONL path or glob. Repeatable.")
    parser.add_argument("--output", required=True, help="Compact G005 diagnostic summary JSON.")
    parser.add_argument("--raw-output", help="Optional raw idm_alignment_shift_diagnostics JSON.")
    parser.add_argument("--contract", default="artifacts/eval/g003_gidm_baseline_contract.json")
    parser.add_argument(
        "--baseline-metrics",
        default="artifacts/idm/g005_idm_endpoint_mixture_matrix_event_all_paper_metrics.json",
        help="Optional candidate paper-metrics JSON used only for diagnostic comparison.",
    )
    parser.add_argument("--source-label", default="target_autocorr")
    parser.add_argument("--expected-nep-shift", type=int, default=2, help="50ms-bin shift for NEP tau=100ms.")
    parser.add_argument("--shifts", type=_shifts, default=_shifts("-4,-3,-2,-1,0,1,2,3,4"))
    parser.add_argument("--split-tag", action="append", default=["temporal", "heldout_recording", "heldout_game"])
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--empty-bins-as-correct", action="store_true")
    parser.add_argument("--progress-output")
    parser.add_argument("--progress-rows", type=int, default=1_000_000)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    diagnostics = build_idm_alignment_shift_diagnostics(
        target_paths=_paths(args.target_path),
        prediction_paths=[],
        shifts=args.shifts,
        split_tags=[str(tag) for tag in args.split_tag],
        model_name="target_autocorr",
        max_rows=args.max_rows,
        empty_bins_as_correct=bool(args.empty_bins_as_correct),
        progress_output_path=args.progress_output,
        progress_rows=args.progress_rows,
    )
    if args.raw_output:
        write_json(args.raw_output, diagnostics)
    summary = build_g005_nep_offset_summary(
        diagnostics_payload=diagnostics,
        contract_payload=read_json(args.contract),
        baseline_metrics_payload=_read_optional_json(args.baseline_metrics),
        expected_nep_shift=args.expected_nep_shift,
        source_label=args.source_label,
    )
    write_json(args.output, summary)
    expected = summary["expected_shift"]
    metrics = expected["metrics"]
    print(
        "g005 NEP offset diagnostics: "
        f"status={summary['status']} rows={summary['alignment'].get('rows_seen')} "
        f"expected_shift={args.expected_nep_shift} expected_passes={expected['paper_target_passes']} "
        f"keyboard={metrics.get('keyboard_accuracy')} button={metrics.get('mouse_button_accuracy')} "
        f"pearson=({metrics.get('pearson_x')},{metrics.get('pearson_y')}) output={args.output}"
    )
    return 0 if summary["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
