#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _gpu_monitor_status(path: Path, expected_gpus: int) -> dict[str, Any]:
    status: dict[str, Any] = {
        "rows": 0,
        "sample_count": 0,
        "unique_gpu_indices": [],
        "expected_gpus": expected_gpus,
        "covers_expected_gpus": False,
    }
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return status
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header: list[str] | None = None
        index_col = None
        sample_col = None
        samples: set[str] = set()
        gpu_indices: set[str] = set()
        for row in reader:
            row = [cell.strip() for cell in row]
            if not row:
                continue
            lowered = [cell.lower() for cell in row]
            if "index" in lowered:
                header = lowered
                index_col = lowered.index("index")
                sample_col = lowered.index("sample_unix") if "sample_unix" in lowered else None
                continue
            if header is None:
                # Fallback for nvidia-smi output without a header.
                index_col = 1 if len(row) > 1 else 0
            if index_col is not None and index_col < len(row):
                gpu_indices.add(row[index_col])
            if sample_col is not None and sample_col < len(row):
                samples.add(row[sample_col])
            status["rows"] += 1
    status["unique_gpu_indices"] = sorted(gpu_indices)
    if samples:
        status["sample_count"] = len(samples)
    elif gpu_indices:
        status["sample_count"] = max(1, status["rows"] // max(1, len(gpu_indices)))
    status["covers_expected_gpus"] = len(gpu_indices) >= expected_gpus
    return status


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    integrated = _load(Path(args.integrated_run_evidence))
    idm_summary = _load(Path(args.idm_summary))
    metadata = _load(Path(args.checkpoint_metadata))
    monitor_meta = _load(Path(args.attached_monitor_metadata))
    gpu_monitor = Path(args.gpu_monitor)
    metrics_path = Path(args.metrics)
    findings: list[dict[str, Any]] = []
    if integrated is None:
        findings.append({"severity": "error", "code": "missing_integrated_run_evidence", "path": args.integrated_run_evidence})
    if idm_summary is None:
        findings.append({"severity": "error", "code": "missing_idm_summary", "path": args.idm_summary})
    if metadata is None:
        findings.append({"severity": "error", "code": "missing_checkpoint_metadata", "path": args.checkpoint_metadata})
    if not gpu_monitor.exists() or gpu_monitor.stat().st_size == 0:
        findings.append({"severity": "error", "code": "missing_gpu_monitor", "path": args.gpu_monitor})
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        findings.append({"severity": "error", "code": "missing_metrics", "path": args.metrics})
    if monitor_meta is None:
        findings.append({"severity": "error", "code": "missing_attached_monitor_metadata", "path": args.attached_monitor_metadata})
    elif int(monitor_meta.get("samples") or 0) <= 0:
        findings.append({"severity": "error", "code": "attached_monitor_has_no_samples", "path": args.attached_monitor_metadata, "samples": monitor_meta.get("samples")})
    nproc = int((integrated or {}).get("idm_nproc_per_node") or (metadata or {}).get("distributed", {}).get("world_size") or args.nproc_per_node)
    expected_gpus = int(args.expected_gpus)
    gpu_monitor_status = _gpu_monitor_status(gpu_monitor, expected_gpus)
    if gpu_monitor.exists() and not gpu_monitor_status["covers_expected_gpus"]:
        findings.append(
            {
                "severity": "error",
                "code": "gpu_monitor_does_not_cover_expected_gpus",
                "path": args.gpu_monitor,
                "expected_gpus": expected_gpus,
                "unique_gpu_indices": gpu_monitor_status["unique_gpu_indices"],
                "rows": gpu_monitor_status["rows"],
            }
        )
    if nproc != expected_gpus:
        findings.append({"severity": "error", "code": "nproc_expected_gpu_mismatch", "nproc_per_node": nproc, "expected_gpus": expected_gpus})
    distributed = (metadata or {}).get("distributed", {}) if isinstance((metadata or {}).get("distributed", {}), dict) else {}
    if distributed.get("world_size") is not None and int(distributed.get("world_size")) != nproc:
        findings.append({"severity": "error", "code": "metadata_world_size_mismatch", "metadata_world_size": distributed.get("world_size"), "nproc_per_node": nproc})
    exit_code = 0 if not findings else 2
    payload = {
        "schema": "g003_idm_4xh200_train_run.v1",
        "source": "attached_summary_from_integrated_parallel_run",
        "integrated_run_evidence": args.integrated_run_evidence,
        "integrated_run_evidence_sha256": _sha256(Path(args.integrated_run_evidence)),
        "attached_monitor_metadata": args.attached_monitor_metadata,
        "attached_monitor_samples": (monitor_meta or {}).get("samples"),
        "gpu_monitor_log": args.gpu_monitor,
        "gpu_monitor_sha256": _sha256(gpu_monitor),
        "gpu_monitor_status": gpu_monitor_status,
        "nproc_per_node": nproc,
        "expected_gpus": expected_gpus,
        "exit_code": exit_code,
        "git_head": _git_head(),
        "summary_path": args.idm_summary,
        "summary_exists": idm_summary is not None,
        "metadata_path": args.checkpoint_metadata,
        "metadata_exists": metadata is not None,
        "metrics_path": args.metrics,
        "metrics_exists": metrics_path.exists(),
        "train_records_count": (metadata or {}).get("train_records"),
        "target_records_count": (metadata or {}).get("target_records"),
        "checkpoint_path": (metadata or {}).get("checkpoint_path"),
        "label_quality_report_path": (metadata or {}).get("label_quality_report_path"),
        "statistical_comparison_path": (metadata or {}).get("statistical_comparison_path"),
        "convergence_report_path": (metadata or {}).get("convergence_report_path"),
        "convergence_plateau_met": (metadata or {}).get("convergence_plateau_met"),
        "distributed": distributed,
        "findings": findings,
        "claim_boundary": "4xH200 run evidence synthesized from a successful integrated G003 parallel extraction+training run plus attached GPU monitor; not a completion claim without G003 audit pass.",
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the required G003 4xH200 train-run summary after an integrated parallel run finishes.")
    parser.add_argument("--integrated-run-evidence", default="artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json")
    parser.add_argument("--idm-summary", default="artifacts/idm/idm_streaming_d2e_full_compact_summary.json")
    parser.add_argument("--checkpoint-metadata", default="outputs/idm_streaming_d2e_full_compact/checkpoint_metadata.json")
    parser.add_argument("--metrics", default="outputs/idm_streaming_d2e_full_compact/metrics.json")
    parser.add_argument("--gpu-monitor", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv")
    parser.add_argument("--attached-monitor-metadata", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json")
    parser.add_argument("--output", default="artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_summary(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"g003 attached train summary: exit_code={payload['exit_code']} findings={len(payload['findings'])} output={out}")
    return 0 if payload["exit_code"] == 0 or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
