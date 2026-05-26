#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.eval.paper_idm_metrics import write_paper_idm_metrics
from fdm_d2e.eval.split_statistics import write_split_statistical_comparisons
from fdm_d2e.reporting.g005_idm_paper_target import write_g005_idm_paper_target_audit


def _rooted(root: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_output(args: list[str], *, root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _gpu_monitor_status(path: Path | None, expected_gpus: int) -> dict[str, Any]:
    status: dict[str, Any] = {
        "rows": 0,
        "sample_count": 0,
        "unique_gpu_indices": [],
        "expected_gpus": expected_gpus,
        "covers_expected_gpus": False,
    }
    if path is None or not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return status
    index_col: int | None = None
    sample_col: int | None = None
    samples: set[str] = set()
    gpu_indices: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for raw_row in csv.reader(handle):
            row = [cell.strip() for cell in raw_row]
            if not row:
                continue
            lowered = [cell.lower() for cell in row]
            if "index" in lowered:
                index_col = lowered.index("index")
                sample_col = lowered.index("sample_unix") if "sample_unix" in lowered else None
                continue
            if index_col is None:
                index_col = 1 if len(row) > 1 else 0
            if index_col < len(row):
                gpu_indices.add(row[index_col])
            if sample_col is not None and sample_col < len(row):
                samples.add(row[sample_col])
            status["rows"] += 1
    status["unique_gpu_indices"] = sorted(gpu_indices)
    status["sample_count"] = len(samples) if samples else (
        max(1, status["rows"] // max(1, len(gpu_indices))) if gpu_indices else 0
    )
    status["covers_expected_gpus"] = len(gpu_indices) >= expected_gpus
    return status


def _ensure_artifact_link(source: Path | None, target: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": str(source) if source else None,
        "target": str(target) if target else None,
        "created": False,
        "mode": None,
        "exists": False,
    }
    if source is None or target is None:
        return payload
    if target.exists() or target.is_symlink():
        payload["exists"] = target.exists()
        payload["mode"] = "existing"
        payload["sha256"] = _sha256(target)
        return payload
    if not source.exists() or not source.is_file():
        payload["mode"] = "missing_source"
        return payload
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source)
        payload["mode"] = "symlink"
    except OSError:
        shutil.copy2(source, target)
        payload["mode"] = "copy"
    payload["created"] = True
    payload["exists"] = target.exists()
    payload["sha256"] = _sha256(target)
    return payload


def _paper_metrics_paths(config: dict[str, Any]) -> tuple[list[str], list[str], str]:
    metrics_cfg = dict(config.get("paper_metrics", config))
    prediction_value = metrics_cfg.get("prediction_paths", metrics_cfg.get("predictions_path"))
    target_value = metrics_cfg.get("target_paths", metrics_cfg.get("target_path"))
    output = metrics_cfg.get("output_path") or metrics_cfg.get("metrics_path")
    if prediction_value is None or target_value is None or output is None:
        raise ValueError("paper metrics config requires prediction(s), target(s), and output_path")
    prediction_paths = prediction_value if isinstance(prediction_value, list) else [prediction_value]
    target_paths = target_value if isinstance(target_value, list) else [target_value]
    return [str(p) for p in prediction_paths], [str(p) for p in target_paths], str(output)


def _write_run_summary(
    *,
    root: Path,
    run_summary_path: Path,
    paper_config_path: str,
    split_config_path: str,
    paper_config: dict[str, Any],
    split_config: dict[str, Any],
    recovery_summary_path: Path | None,
    source_run_summary_path: Path | None,
    source_checkpoint_path: Path | None,
    source_stats_path: Path | None,
    checkpoint_link: dict[str, Any],
    stats_link: dict[str, Any],
    expected_gpus: int,
    nproc_per_node: int,
    exit_code: int,
    recovery_command: str | None,
    claim_boundary: str,
) -> dict[str, Any]:
    paths = dict(paper_config.get("paths", {}))
    checkpoint_path = _rooted(root, paths.get("checkpoint"))
    metadata_path = _rooted(root, paths.get("checkpoint_metadata"))
    train_summary_path = _rooted(root, paths.get("train_summary"))
    split_summary_path = _rooted(root, paths.get("split_stats_summary"))
    gpu_monitor_path = _rooted(root, paths.get("gpu_monitor"))
    paper_metrics_cfg = dict(paper_config.get("paper_metrics", {}))
    paper_metrics_path = _rooted(root, paths.get("paper_metrics") or paper_metrics_cfg.get("output_path"))

    metadata = _load_json(metadata_path)
    train_summary = _load_json(train_summary_path)
    split_summary = _load_json(split_summary_path)
    paper_metrics = _load_json(paper_metrics_path)
    recovery_summary = _load_json(recovery_summary_path)
    source_run_summary = _load_json(source_run_summary_path)
    predictions_path = _rooted(root, paper_metrics_cfg.get("predictions_path"))
    pseudolabels_path = None
    if metadata and metadata.get("pseudo_label_path"):
        pseudolabels_path = _rooted(root, str(metadata.get("pseudo_label_path")))

    payload: dict[str, Any] = {
        "schema": "g005_idm_checkpoint_recovery_finalization_run.v1",
        "status": "pass" if exit_code == 0 else "fail",
        "exit_code": exit_code,
        "paper_target_config": paper_config_path,
        "split_stats_config": split_config_path,
        "run_summary_path": str(run_summary_path),
        "expected_gpus": expected_gpus,
        "nproc_per_node": nproc_per_node,
        "git_head": _git_output(["rev-parse", "HEAD"], root=root),
        "git_status_short": _git_output(["status", "--short"], root=root),
        "recovery_command": recovery_command,
        "source_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else None,
        "source_checkpoint_sha256": _sha256(source_checkpoint_path),
        "source_stats_path": str(source_stats_path) if source_stats_path else None,
        "source_stats_sha256": _sha256(source_stats_path),
        "source_run_summary_path": str(source_run_summary_path) if source_run_summary_path else None,
        "source_run_summary_status": source_run_summary.get("status") if source_run_summary else None,
        "recovery_summary_path": str(recovery_summary_path) if recovery_summary_path else None,
        "recovery_summary_status": recovery_summary.get("status") if recovery_summary else None,
        "checkpoint_link": checkpoint_link,
        "stats_link": stats_link,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "metadata_exists": metadata is not None,
        "train_summary_path": str(train_summary_path) if train_summary_path else None,
        "train_summary_exists": train_summary is not None,
        "split_stats_summary_path": str(split_summary_path) if split_summary_path else None,
        "split_stats_status": split_summary.get("status") if split_summary else None,
        "split_stats_outputs": split_summary.get("outputs", []) if split_summary else [],
        "paper_metrics_path": str(paper_metrics_path) if paper_metrics_path else None,
        "paper_metrics_status": paper_metrics.get("status") if paper_metrics else None,
        "paper_metrics_rows": (paper_metrics or {}).get("alignment", {}).get("rows_seen") if paper_metrics else None,
        "gpu_monitor_log": str(gpu_monitor_path) if gpu_monitor_path else None,
        "gpu_monitor_sha256": _sha256(gpu_monitor_path),
        "gpu_monitor_status": _gpu_monitor_status(gpu_monitor_path, expected_gpus),
        "predictions_path": str(predictions_path) if predictions_path else None,
        "predictions_exists": bool(predictions_path and predictions_path.exists()),
        "predictions_bytes": predictions_path.stat().st_size if predictions_path and predictions_path.exists() else 0,
        "pseudolabels_path": str(pseudolabels_path) if pseudolabels_path else None,
        "pseudolabels_exists": bool(pseudolabels_path and pseudolabels_path.exists()),
        "pseudolabels_bytes": pseudolabels_path.stat().st_size if pseudolabels_path and pseudolabels_path.exists() else 0,
        "train_records": (metadata or {}).get("train_records"),
        "target_records": (metadata or {}).get("target_records") or (recovery_summary or {}).get("target_records"),
        "prediction_resume": (train_summary or {}).get("prediction_resume") if train_summary else None,
        "split_statistics_output_dir": split_config.get("output_dir"),
        "claim_boundary": claim_boundary,
    }
    run_summary_path.parent.mkdir(parents=True, exist_ok=True)
    run_summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize a full-corpus G005 streaming IDM checkpoint-recovery candidate."
    )
    parser.add_argument("--paper-target-config", required=True)
    parser.add_argument("--split-stats-config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-summary")
    parser.add_argument("--recovery-summary")
    parser.add_argument("--source-run-summary")
    parser.add_argument("--checkpoint-source")
    parser.add_argument("--stats-source")
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument("--expected-gpus", type=int)
    parser.add_argument("--recovery-command")
    parser.add_argument("--skip-split-stats", action="store_true")
    parser.add_argument("--skip-paper-metrics", action="store_true")
    parser.add_argument("--allow-audit-fail", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paper_config = load_config(args.paper_target_config)
    split_config = load_config(args.split_stats_config)
    paths = dict(paper_config.get("paths", {}))
    expected_gpus = args.expected_gpus or int(paper_config.get("expected_gpus", 4))
    nproc_per_node = args.nproc_per_node or expected_gpus
    run_summary_path = _rooted(root, args.run_summary or paths.get("run_summary"))
    if run_summary_path is None:
        raise ValueError("run summary path must be provided or configured under paths.run_summary")

    checkpoint_link = _ensure_artifact_link(
        _rooted(root, args.checkpoint_source),
        _rooted(root, paths.get("checkpoint")),
    )
    stats_link = _ensure_artifact_link(
        _rooted(root, args.stats_source),
        _rooted(root, split_config.get("train_stats_path")),
    )

    if not args.skip_split_stats:
        split_payload = write_split_statistical_comparisons(split_config, root=root)
        print(
            "split statistical comparisons: "
            f"status={split_payload['status']} outputs={len(split_payload.get('outputs', []))}"
        )

    if not args.skip_paper_metrics:
        metrics_cfg = dict(paper_config.get("paper_metrics", paper_config))
        prediction_paths, target_paths, output = _paper_metrics_paths(paper_config)
        metrics_payload = write_paper_idm_metrics(
            prediction_paths=[
                str(_rooted(root, path)) if not Path(path).is_absolute() else path for path in prediction_paths
            ],
            target_paths=[
                str(_rooted(root, path)) if not Path(path).is_absolute() else path for path in target_paths
            ],
            output_path=_rooted(root, output) or Path(output),
            split_tags=[str(tag) for tag in metrics_cfg.get("split_tags", ["temporal", "heldout_recording", "heldout_game"])],
            model_name=str(metrics_cfg.get("model_name", paper_config.get("model_name", "model"))),
            max_rows=metrics_cfg.get("max_rows"),
            progress_output_path=_rooted(root, metrics_cfg.get("progress_output_path")),
            progress_rows=int(metrics_cfg.get("progress_rows", 1_000_000)),
            empty_bins_as_correct=bool(metrics_cfg.get("empty_bins_as_correct", False)),
        )
        print(
            "g005 paper metrics: "
            f"status={metrics_payload['status']} rows={metrics_payload['alignment']['rows_seen']}"
        )

    claim_boundary = (
        "Checkpoint recovery finalization validates a full-corpus G005 IDM candidate from an already-trained "
        "4xH200 checkpoint. It is evidence for the recovered candidate, not a fresh training run and not FDM-1 parity."
    )
    _write_run_summary(
        root=root,
        run_summary_path=run_summary_path,
        paper_config_path=args.paper_target_config,
        split_config_path=args.split_stats_config,
        paper_config=paper_config,
        split_config=split_config,
        recovery_summary_path=_rooted(root, args.recovery_summary),
        source_run_summary_path=_rooted(root, args.source_run_summary),
        source_checkpoint_path=_rooted(root, args.checkpoint_source),
        source_stats_path=_rooted(root, args.stats_source),
        checkpoint_link=checkpoint_link,
        stats_link=stats_link,
        expected_gpus=expected_gpus,
        nproc_per_node=nproc_per_node,
        exit_code=0,
        recovery_command=args.recovery_command,
        claim_boundary=claim_boundary,
    )
    audit = write_g005_idm_paper_target_audit(paper_config, root=root)
    run_payload = _load_json(run_summary_path) or {}
    run_payload.update(
        {
            "g005_audit_path": str(_rooted(root, paper_config.get("output_path"))),
            "g005_audit_status": audit.get("status"),
            "g005_audit_error_count": audit.get("error_count"),
            "aggregate_target_results": audit.get("aggregate_target_results", []),
            "strict_target_results": audit.get("strict_target_results", []),
        }
    )
    run_summary_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "g005 recovery finalization: "
        f"audit_status={audit['status']} errors={audit['error_count']} run_summary={run_summary_path}"
    )
    return 0 if audit["status"] == "pass" or args.allow_audit_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
