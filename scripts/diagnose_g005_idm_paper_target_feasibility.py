#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.idm_paper_feasibility import write_idm_paper_target_feasibility


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether causal D2E row fields can support G005 paper-target IDM metrics "
            "before launching another full 4xH200 training run."
        )
    )
    parser.add_argument("--train-path", action="append", required=True, help="Train JSONL path/glob. Repeatable.")
    parser.add_argument("--target-path", action="append", required=True, help="Target JSONL path/glob. Repeatable.")
    parser.add_argument("--output", default="artifacts/idm/g005_idm_paper_target_feasibility.json")
    parser.add_argument("--model-name", default="g005_idm_paper_target_feasibility")
    parser.add_argument("--split-tag", action="append", default=["temporal", "heldout_recording", "heldout_game"])
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-target-rows", type=int)
    parser.add_argument("--max-train-rows-per-path", type=int)
    parser.add_argument("--max-target-rows-per-path", type=int)
    parser.add_argument("--min-feature-count", type=int, default=3)
    parser.add_argument("--paper-contract", default="artifacts/eval/g003_gidm_baseline_contract.json")
    parser.add_argument(
        "--baseline-metrics",
        default="artifacts/idm/g005_idm_event_state_duration_context_paper_metrics.json",
        help="Current strongest full-target model metrics for scale/no-scale comparison.",
    )
    parser.add_argument("--progress-output", default="artifacts/idm/g005_idm_paper_target_feasibility_progress.json")
    parser.add_argument("--progress-rows", type=int, default=1_000_000)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    payload = write_idm_paper_target_feasibility(
        train_paths=args.train_path,
        target_paths=args.target_path,
        output_path=args.output,
        split_tags=[str(tag) for tag in args.split_tag],
        model_name=args.model_name,
        max_train_rows=args.max_train_rows,
        max_target_rows=args.max_target_rows,
        max_train_rows_per_path=args.max_train_rows_per_path,
        max_target_rows_per_path=args.max_target_rows_per_path,
        min_feature_count=max(1, int(args.min_feature_count)),
        paper_contract_path=args.paper_contract,
        baseline_metrics_path=args.baseline_metrics,
        progress_output_path=args.progress_output,
        progress_rows=max(1, int(args.progress_rows)),
    )
    best = payload.get("predictor_summaries", [{}])[0] if payload.get("predictor_summaries") else {}
    values = best.get("values", {}) if isinstance(best, dict) else {}
    print(
        "g005 idm feasibility: "
        f"status={payload['status']} train_rows={payload.get('table_cardinality', {}).get('global', {}).get('rows')} "
        f"target_rows={payload['alignment'].get('target_rows')} best={best.get('name')} "
        f"keyboard={values.get('keyboard_accuracy')} button={values.get('mouse_button_accuracy')} "
        f"scale_new_full_gpu_run={payload['recommendation']['scale_new_full_gpu_run']} output={args.output}"
    )
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())

