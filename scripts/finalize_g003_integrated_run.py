#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdm_d2e.cluster.g003_monitor import build_g003_progress_report
from fdm_d2e.config import load_config
from fdm_d2e.eval.split_statistics import write_split_statistical_comparisons
from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.g003_completion import validate_g003_full_idm_completion, write_g003_full_idm_completion_audit
from build_g003_attached_train_run_summary import build_summary as build_attached_train_summary


DEFAULT_SUMMARY_OUT = "artifacts/idm/g003_integrated_finalization_summary.json"


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


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


def _build_train_summary(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    train_args = Namespace(
        integrated_run_evidence=str(_path(root, args.integrated_run_evidence)),
        idm_summary=str(_path(root, args.idm_summary)),
        checkpoint_metadata=str(_path(root, args.checkpoint_metadata)),
        metrics=str(_path(root, args.metrics)),
        gpu_monitor=str(_path(root, args.gpu_monitor)),
        attached_monitor_metadata=str(_path(root, args.attached_monitor_metadata)),
        nproc_per_node=int(args.nproc_per_node),
        expected_gpus=int(args.expected_gpus),
    )
    payload = build_attached_train_summary(train_args)
    out = _path(root, args.train_run_summary)
    write_json(out, payload)
    return payload


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    progress = build_g003_progress_report(
        shard_root=_path(root, args.shard_root),
        log_dir=_path(root, args.log_dir),
        data_universe=_path(root, args.data_universe),
        output_dir=_path(root, args.data_output_dir),
        idm_output_dir=_path(root, args.idm_output_dir),
        pid_file=_path(root, args.pid_file),
        repair_pid_glob=str(_path(root, args.repair_pid_glob)) if getattr(args, "repair_pid_glob", None) else None,
        num_shards=int(args.num_shards),
        stale_seconds=float(args.stale_seconds),
    )
    findings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    if progress.get("pid_running") and not args.allow_active_parent:
        findings.append(
            {
                "severity": "error",
                "code": "parent_still_running",
                "pid": progress.get("pid"),
                "status": progress.get("status"),
                "decoded_recording_variants": progress.get("decoded_recording_variants"),
                "expected_recording_variants": progress.get("expected_recording_variants"),
            }
        )
        status = "blocked_active_parent"
        train_summary = None
        split_stats = None
        audit = None
    else:
        split_stats = _maybe_build_split_stats(args, root)
        actions.append({"name": "split_stats", **{k: v for k, v in split_stats.items() if k != "payload"}})
        train_summary = _build_train_summary(args, root)
        actions.append({"name": "attached_train_summary", "output": args.train_run_summary, "exit_code": train_summary.get("exit_code"), "findings": train_summary.get("findings", [])})
        audit = write_g003_full_idm_completion_audit(load_config(_path(root, args.g003_completion_config)), root=root, output_path=args.g003_audit_output)
        actions.append({"name": "g003_completion_audit", "output": args.g003_audit_output, "status": audit.get("status"), "error_count": audit.get("error_count")})
        if split_stats.get("status") != "pass":
            findings.append({"severity": "error", "code": "split_stats_not_pass", "status": split_stats.get("status"), "reason": split_stats.get("reason")})
        if train_summary.get("exit_code") != 0:
            findings.append({"severity": "error", "code": "attached_train_summary_failed", "exit_code": train_summary.get("exit_code")})
        if audit.get("status") != "pass":
            findings.append({"severity": "error", "code": "g003_completion_audit_not_pass", "error_count": audit.get("error_count")})
        status = "pass" if not findings else "fail"
    payload = {
        "schema": "g003_integrated_finalization.v1",
        "status": status,
        "root": str(root),
        "progress": {
            "status": progress.get("status"),
            "pid": progress.get("pid"),
            "pid_running": progress.get("pid_running"),
            "log_dir": progress.get("log_dir"),
            "repair_pid_glob": progress.get("repair_pid_glob"),
            "decoded_recording_variants": progress.get("decoded_recording_variants"),
            "expected_recording_variants": progress.get("expected_recording_variants"),
            "complete_shards": progress.get("complete_shards"),
            "num_shards": progress.get("num_shards"),
            "stale_shards": progress.get("stale_shards"),
            "no_progress_shards": progress.get("no_progress_shards"),
            "merged_train_eval_exists": progress.get("merged_train_eval_exists"),
            "idm_metrics_exists": progress.get("idm_metrics_exists"),
        },
        "actions": actions,
        "split_stats": split_stats,
        "attached_train_summary_exit_code": train_summary.get("exit_code") if train_summary else None,
        "g003_audit_status": audit.get("status") if audit else None,
        "g003_audit_error_count": audit.get("error_count") if audit else None,
        "findings": findings,
        "claim_boundary": "Finalizes artifacts after an integrated G003 run exits; it does not checkpoint G003 and does not prove completion unless status is pass and the OMX checkpoint is subsequently recorded.",
    }
    write_json(_path(root, args.summary_out), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize/audit artifacts after an integrated G003 full-corpus run exits.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--allow-active-parent", action="store_true", help="Do not fail early when the integrated parent is still running; useful only for diagnostics.")
    parser.add_argument("--skip-split-stats", action="store_true")
    parser.add_argument("--force-split-stats", action="store_true")
    parser.add_argument("--split-stats-config", default="configs/eval/g003_split_statistics.yaml")
    parser.add_argument("--split-stats-summary", default="artifacts/eval/g003_split_statistical_comparisons_summary.json")
    parser.add_argument("--g003-completion-config", default="configs/eval/g003_full_idm_completion.yaml")
    parser.add_argument("--g003-audit-output", default="artifacts/idm/g003_full_idm_completion_audit.json")
    parser.add_argument("--integrated-run-evidence", default="artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json")
    parser.add_argument("--idm-summary", default="artifacts/idm/idm_streaming_d2e_full_compact_summary.json")
    parser.add_argument("--checkpoint-metadata", default="outputs/idm_streaming_d2e_full_compact/checkpoint_metadata.json")
    parser.add_argument("--metrics", default="outputs/idm_streaming_d2e_full_compact/metrics.json")
    parser.add_argument("--gpu-monitor", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv")
    parser.add_argument("--attached-monitor-metadata", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json")
    parser.add_argument("--train-run-summary", default="artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--shard-root", default="outputs/data/d2e_full_corpus_shards")
    parser.add_argument("--log-dir", default="artifacts/sources")
    parser.add_argument("--data-universe", default="artifacts/sources/d2e_full_data_universe_manifest.json")
    parser.add_argument("--data-output-dir", default="outputs/data/d2e_full_corpus")
    parser.add_argument("--idm-output-dir", default="outputs/idm_streaming_d2e_full_compact")
    parser.add_argument("--pid-file", default="outputs/cluster/g003_full_compact_parallel.pid")
    parser.add_argument(
        "--repair-pid-glob",
        default=None,
        help="Optional glob for isolated shard repair pid files; defaults to a lane-scoped pattern derived from --pid-file.",
    )
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--stale-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    payload = finalize(args)
    print(f"g003 finalization: status={payload['status']} findings={len(payload['findings'])} output={args.summary_out}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
