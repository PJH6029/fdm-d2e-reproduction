#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.eval.split_statistics import write_split_statistical_comparisons
from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.g004_completion import write_g004_full_fdm_completion_audit
from fdm_d2e.training.streaming_fdm import ensure_fdm_canonical_records


DEFAULT_SUMMARY_OUT = "artifacts/fdm/g004_d2e_full_fdm_finalization_summary.json"


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _exists_nonempty(root: Path, value: str | Path) -> bool:
    p = _path(root, value)
    return p.exists() and p.is_file() and p.stat().st_size > 0


def _status_from_payload(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    return str(payload.get("status")) if payload.get("status") is not None else None


def _split_stats_ready(root: Path, config_path: str | Path) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    config = load_config(_path(root, config_path))
    required = {
        "predictions_path": str(config["predictions_path"]),
        "ground_truth_path": str(config["ground_truth_path"]),
        "train_records_path": str(config.get("train_records_path", "")),
    }
    exists = {key: _exists_nonempty(root, rel) if rel else True for key, rel in required.items()}
    return all(exists.values()), config, {"required_paths": required, "exists": exists}


def _maybe_build_split_stats(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    ready, config, inputs = _split_stats_ready(root, args.split_stats_config)
    summary_path = _path(root, config.get("summary_out", args.split_stats_summary))
    if args.skip_split_stats:
        return {"status": "skipped", "reason": "skip_split_stats", "summary_path": str(summary_path), **inputs}
    if summary_path.exists() and not args.force_split_stats:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {"status": "fail", "reason": "invalid_existing_summary", "summary_path": str(summary_path), "error": str(exc), **inputs}
        return {"status": _status_from_payload(payload) or "unknown", "reason": "existing_summary", "summary_path": str(summary_path), "payload": payload, **inputs}
    if not ready:
        return {"status": "blocked_missing_inputs", "summary_path": str(summary_path), **inputs}
    payload = write_split_statistical_comparisons(config, root=root)
    return {"status": _status_from_payload(payload) or "unknown", "reason": "built", "summary_path": str(summary_path), "payload": payload, **inputs}


def _maybe_build_canonical_records(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    split_summary_path = _path(root, args.split_summary)
    if not split_summary_path.exists():
        return {"status": "skipped", "reason": "missing_split_summary", "split_summary": args.split_summary}
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    canonical = ensure_fdm_canonical_records(split_summary, force=args.force_canonical_records)
    return {"status": canonical.get("status", "unknown"), "split_summary": args.split_summary, "canonical": canonical}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    findings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    run_summary_path = _path(root, args.run_summary)
    run_summary = _load_json(run_summary_path)
    if run_summary is None:
        findings.append({"severity": "error", "code": "missing_run_summary", "path": args.run_summary})
        status = "blocked_missing_run_summary"
        split_stats = None
        audit = None
    else:
        actions.append({"name": "run_summary", "path": args.run_summary, "exit_code": run_summary.get("exit_code")})
        if run_summary.get("exit_code") != 0:
            findings.append({"severity": "error", "code": "run_summary_exit_nonzero", "actual": run_summary.get("exit_code")})
        canonical = _maybe_build_canonical_records(args, root)
        actions.append({"name": "canonical_records", **canonical})
        split_stats = _maybe_build_split_stats(args, root)
        actions.append({"name": "split_stats", **{k: v for k, v in split_stats.items() if k != "payload"}})
        audit = write_g004_full_fdm_completion_audit(load_config(_path(root, args.g004_completion_config)), root=root, output_path=args.g004_audit_output)
        actions.append({"name": "g004_completion_audit", "output": args.g004_audit_output, "status": audit.get("status"), "error_count": audit.get("error_count")})
        if split_stats.get("status") != "pass":
            findings.append({"severity": "error", "code": "split_stats_not_pass", "status": split_stats.get("status"), "reason": split_stats.get("reason")})
        if audit.get("status") != "pass":
            findings.append({"severity": "error", "code": "g004_completion_audit_not_pass", "error_count": audit.get("error_count")})
        status = "pass" if not findings else "fail"
    payload = {
        "schema": "g004_d2e_full_fdm_finalization.v1",
        "status": status,
        "root": str(root),
        "run_summary": run_summary,
        "actions": actions,
        "split_stats": split_stats,
        "g004_audit_status": audit.get("status") if audit else None,
        "g004_audit_error_count": audit.get("error_count") if audit else None,
        "findings": findings,
        "claim_boundary": "Finalizes artifacts after the G004 D2E-only FDM 4xH200 run; it does not checkpoint G004 and does not prove completion unless status is pass and the OMX checkpoint is subsequently recorded.",
    }
    write_json(_path(root, args.summary_out), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize/audit artifacts after a G004 D2E-only FDM 4xH200 run.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--skip-split-stats", action="store_true")
    parser.add_argument("--force-split-stats", action="store_true")
    parser.add_argument("--split-stats-config", default="configs/eval/g004_split_statistics.yaml")
    parser.add_argument("--split-stats-summary", default="artifacts/eval/g004_split_statistical_comparisons_summary.json")
    parser.add_argument("--split-summary", default="outputs/fdm_streaming_d2e_full_compact/fdm_streaming_split_summary.json")
    parser.add_argument("--force-canonical-records", action="store_true")
    parser.add_argument("--g004-completion-config", default="configs/eval/g004_full_fdm_completion.yaml")
    parser.add_argument("--g004-audit-output", default="artifacts/fdm/g004_full_fdm_completion_audit.json")
    parser.add_argument("--run-summary", default="artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json")
    args = parser.parse_args()
    payload = finalize(args)
    print(f"g004 finalization: status={payload['status']} findings={len(payload['findings'])} output={args.summary_out}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
