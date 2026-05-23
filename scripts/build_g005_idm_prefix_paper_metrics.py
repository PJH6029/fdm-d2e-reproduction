#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.eval.paper_idm_metrics import _PaperMetricAccumulator
from fdm_d2e.io_utils import read_json, write_json
from fdm_d2e.training.streaming_idm import _chunk_sequence, _record_paths_from_config


def _iter_jsonl(paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL row must be an object at {path}:{line_no}")
                yield payload


def _tokens(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None and key == "ground_truth_tokens":
        value = row.get("target_tokens")
    if value is None and key == "predicted_tokens":
        value = row.get("tokens")
    return [str(item) for item in value] if isinstance(value, list) else []


def _split_tags(row: dict[str, Any]) -> list[str]:
    value = row.get("eval_split_tags") or row.get("split_tags")
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    value = row.get("eval_split") or row.get("split") or row.get("split_name")
    return [str(value)] if isinstance(value, str) else []


def _target_rows(contract_path: Path | None) -> dict[str, Any] | None:
    if contract_path is None or not contract_path.exists():
        return None
    contract = read_json(contract_path)
    targets = (((contract or {}).get("target_sequence") or {}).get("phase_1") or {}).get("primary_targets")
    return dict(targets) if isinstance(targets, dict) else None


def build_prefix_metrics(
    *,
    config: dict[str, Any],
    prediction_part_paths: Sequence[Path],
    rows_per_part: int,
    split_tags: Sequence[str],
    empty_bins_as_correct: bool,
    baseline_contract_path: Path | None = None,
) -> dict[str, Any]:
    target_paths = _record_paths_from_config(
        config,
        primary_key="target_records",
        paths_key="target_record_paths",
        glob_key="target_records_glob",
    )
    chunks = _chunk_sequence(target_paths, len(prediction_part_paths))
    groups: dict[str, _PaperMetricAccumulator] = {"all": _PaperMetricAccumulator(empty_bins_as_correct=empty_bins_as_correct)}
    for tag in split_tags:
        groups[f"eval_split:{tag}"] = _PaperMetricAccumulator(empty_bins_as_correct=empty_bins_as_correct)
    alignment = {
        "rows_seen": 0,
        "rows_per_part": int(rows_per_part),
        "sequence_id_mismatches": 0,
        "missing_prediction_rows": 0,
        "missing_target_rows": 0,
        "examples": [],
    }
    part_rows: list[dict[str, Any]] = []
    for part_index, (prediction_path, target_chunk) in enumerate(zip(prediction_part_paths, chunks)):
        rows = 0
        for pred, target in zip_longest(_iter_jsonl([prediction_path]), _iter_jsonl(target_chunk)):
            if rows >= rows_per_part:
                break
            if pred is None:
                alignment["missing_prediction_rows"] += 1
                continue
            if target is None:
                alignment["missing_target_rows"] += 1
                continue
            rows += 1
            alignment["rows_seen"] += 1
            if pred.get("sequence_id") != target.get("sequence_id"):
                alignment["sequence_id_mismatches"] += 1
                if len(alignment["examples"]) < 20:
                    alignment["examples"].append(
                        {
                            "part_index": part_index,
                            "pred_sequence_id": pred.get("sequence_id"),
                            "target_sequence_id": target.get("sequence_id"),
                        }
                    )
            predicted_tokens = _tokens(pred, "predicted_tokens")
            ground_truth_tokens = _tokens(target, "ground_truth_tokens")
            groups["all"].update(predicted_tokens, ground_truth_tokens)
            active_tags = set(_split_tags(target))
            for tag in split_tags:
                if tag in active_tags:
                    groups[f"eval_split:{tag}"].update(predicted_tokens, ground_truth_tokens)
        part_rows.append(
            {
                "part_index": part_index,
                "prediction_path": str(prediction_path),
                "target_paths": [str(path) for path in target_chunk],
                "rows": rows,
            }
        )
    findings = []
    if alignment["sequence_id_mismatches"]:
        findings.append({"severity": "error", "code": "sequence_id_mismatches_detected", "count": alignment["sequence_id_mismatches"]})
    if alignment["missing_prediction_rows"] or alignment["missing_target_rows"]:
        findings.append(
            {
                "severity": "error",
                "code": "prediction_target_row_count_mismatch",
                "missing_prediction_rows": alignment["missing_prediction_rows"],
                "missing_target_rows": alignment["missing_target_rows"],
            }
        )
    return {
        "schema": "g005_idm_candidate_prefix_paper_metrics.v1",
        "status": "pass" if not findings else "fail",
        "error_count": len([item for item in findings if item.get("severity") == "error"]),
        "model_name": str(config.get("model_name", "model")),
        "sample_method": f"first {rows_per_part} prediction rows per parallel prediction part aligned with corresponding target shard chunks",
        "prediction_part_paths": [str(path) for path in prediction_part_paths],
        "target_record_paths": [str(path) for path in target_paths],
        "parts": part_rows,
        "alignment": alignment,
        "paper_targets": _target_rows(baseline_contract_path),
        "groups": {key: value.metrics() for key, value in sorted(groups.items())},
        "findings": findings,
        "claim_boundary": "Prefix metrics are early-stop diagnostics for candidate selection only; they are not G005 completion evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 IDM paper-compatible metrics from partial parallel prediction outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prediction-parts-glob", required=True)
    parser.add_argument("--rows-per-part", type=int, default=50_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-contract", default="artifacts/eval/g003_gidm_baseline_contract.json")
    parser.add_argument("--split-tags", nargs="*", default=["temporal", "heldout_recording", "heldout_game"])
    parser.add_argument("--empty-bins-as-correct", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    prediction_part_paths = [Path(path) for path in sorted(glob.glob(args.prediction_parts_glob))]
    if not prediction_part_paths:
        raise FileNotFoundError(f"no prediction part files matched: {args.prediction_parts_glob}")
    payload = build_prefix_metrics(
        config=config,
        prediction_part_paths=prediction_part_paths,
        rows_per_part=int(args.rows_per_part),
        split_tags=[str(tag) for tag in args.split_tags],
        empty_bins_as_correct=bool(args.empty_bins_as_correct),
        baseline_contract_path=Path(args.baseline_contract) if args.baseline_contract else None,
    )
    write_json(args.output, payload)
    print(f"g005 prefix paper metrics: status={payload['status']} rows={payload['alignment']['rows_seen']} output={args.output}")
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
