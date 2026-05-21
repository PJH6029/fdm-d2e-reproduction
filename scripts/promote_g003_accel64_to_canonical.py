#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_g003_attached_train_run_summary import build_summary as build_g003_train_summary
from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.g003_completion import write_g003_full_idm_completion_audit


DEFAULT_SOURCE_CONFIG = "configs/eval/g003_full_idm_completion_accel64.yaml"
DEFAULT_CANONICAL_CONFIG = "configs/eval/g003_full_idm_completion.yaml"
DEFAULT_OUTPUT = "artifacts/idm/g003_accel64_promotion_manifest.json"


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _pid_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_status(root: Path, pid_file: str) -> dict[str, Any]:
    path = _path(root, pid_file)
    pid = _read_pid(path)
    return {"pid_file": pid_file, "pid": pid, "running": _pid_running(pid)}


def _same_target(src: Path, dst: Path) -> bool:
    if not dst.exists() and not dst.is_symlink():
        return False
    try:
        return dst.resolve() == src.resolve()
    except FileNotFoundError:
        return False


def _relative_symlink_target(src: Path, dst: Path) -> str:
    return os.path.relpath(src.resolve(), start=dst.parent.resolve())


def _add_mapping(mappings: list[dict[str, str]], *, key: str, source: str | Path, dest: str | Path, kind: str = "file") -> None:
    source_s = str(source)
    dest_s = str(dest)
    if source_s == dest_s:
        return
    if any(row["dest"] == dest_s for row in mappings):
        return
    mappings.append({"key": key, "source": source_s, "dest": dest_s, "kind": kind})


def build_promotion_mappings(
    *,
    root: Path,
    source_config: dict[str, Any],
    canonical_config: dict[str, Any],
    source_shard_root: str = "outputs/data/d2e_full_corpus_shards_accel64",
    canonical_shard_root: str = "outputs/data/d2e_full_corpus_shards",
    source_log_dir: str = "artifacts/sources/g003_accel64",
    canonical_log_dir: str = "artifacts/sources",
    source_integrated_run_evidence: str = "artifacts/idm/g003_d2e_full_idm_run_accel64.json",
    canonical_integrated_run_evidence: str = "artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json",
    source_attached_monitor_metadata: str = "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64_attached.json",
    canonical_attached_monitor_metadata: str = "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json",
) -> list[dict[str, str]]:
    source_paths = dict(source_config.get("paths", {}))
    canonical_paths = dict(canonical_config.get("paths", {}))
    mappings: list[dict[str, str]] = []

    # Preserve progress-monitor continuity without copying the full shard tree.
    _add_mapping(mappings, key="shard_root", source=source_shard_root, dest=canonical_shard_root, kind="directory")

    # Promote all immediate merged JSONLs and IDM artifacts, including files the
    # final gate checks but the G003 completion config does not enumerate.
    for key in ["train_records", "target_records"]:
        if key in source_paths and key in canonical_paths:
            src_dir = Path(str(source_paths[key])).parent
            dst_dir = Path(str(canonical_paths[key])).parent
            for src in sorted((_path(root, src_dir)).glob("*")):
                if src.is_file():
                    _add_mapping(mappings, key=f"data_output:{src.name}", source=src.relative_to(root), dest=dst_dir / src.name)
            break
    for key in ["checkpoint", "metrics", "predictions", "pseudolabels"]:
        if key in source_paths and key in canonical_paths:
            src_dir = Path(str(source_paths[key])).parent
            dst_dir = Path(str(canonical_paths[key])).parent
            for src in sorted((_path(root, src_dir)).glob("*")):
                if src.is_file():
                    _add_mapping(mappings, key=f"idm_output:{src.name}", source=src.relative_to(root), dest=dst_dir / src.name)
            break

    # Promote singleton artifacts outside the merged-data/model-output dirs.
    skip_keys = {"data_universe", "train_records", "target_records", "run_summary"}
    for key, source_rel in sorted(source_paths.items()):
        if key in skip_keys or key not in canonical_paths:
            continue
        _add_mapping(mappings, key=f"completion_path:{key}", source=source_rel, dest=canonical_paths[key])

    _add_mapping(
        mappings,
        key="integrated_run_evidence",
        source=source_integrated_run_evidence,
        dest=canonical_integrated_run_evidence,
    )
    _add_mapping(
        mappings,
        key="attached_monitor_metadata",
        source=source_attached_monitor_metadata,
        dest=canonical_attached_monitor_metadata,
    )

    for src in sorted(_path(root, source_log_dir).glob("d2e_full_corpus_shard_*.log")):
        _add_mapping(mappings, key=f"shard_log:{src.name}", source=src.relative_to(root), dest=Path(canonical_log_dir) / src.name)

    return mappings


def _apply_symlink_mapping(root: Path, mapping: dict[str, str], *, backup_root: Path, dry_run: bool) -> dict[str, Any]:
    src = _path(root, mapping["source"])
    dst = _path(root, mapping["dest"])
    action: dict[str, Any] = {**mapping, "source_exists": src.exists(), "dest_exists_before": dst.exists() or dst.is_symlink()}
    if not src.exists():
        return {**action, "status": "missing_source"}
    if _same_target(src, dst):
        return {**action, "status": "already_promoted"}
    backup_path: Path | None = None
    if dst.exists() or dst.is_symlink():
        backup_path = backup_root / mapping["dest"]
        action["backup_path"] = str(backup_path.relative_to(root) if backup_path.is_relative_to(root) else backup_path)
    if dry_run:
        return {**action, "status": "would_link"}

    dst.parent.mkdir(parents=True, exist_ok=True)
    if backup_path is not None:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if backup_path.exists() or backup_path.is_symlink():
            raise FileExistsError(f"backup path already exists: {backup_path}")
        shutil.move(str(dst), str(backup_path))
    dst.symlink_to(_relative_symlink_target(src, dst), target_is_directory=src.is_dir())
    return {**action, "status": "linked"}


def _build_canonical_train_summary(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    payload = build_g003_train_summary(
        Namespace(
            integrated_run_evidence=str(_path(root, args.canonical_integrated_run_evidence)),
            idm_summary=str(_path(root, args.canonical_idm_summary)),
            checkpoint_metadata=str(_path(root, args.canonical_checkpoint_metadata)),
            metrics=str(_path(root, args.canonical_metrics)),
            gpu_monitor=str(_path(root, args.canonical_gpu_monitor)),
            attached_monitor_metadata=str(_path(root, args.canonical_attached_monitor_metadata)),
            nproc_per_node=int(args.nproc_per_node),
            expected_gpus=int(args.expected_gpus),
        )
    )
    write_json(_path(root, args.canonical_train_run_summary), payload)
    return payload


def promote(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    source_config = load_config(_path(root, args.source_config))
    canonical_config = load_config(_path(root, args.canonical_config))
    source_audit = _load_json(_path(root, args.source_audit))
    source_pid = _pid_status(root, args.source_pid_file)
    primary_pid = _pid_status(root, args.primary_pid_file)
    findings: list[dict[str, Any]] = []

    if not args.skip_source_audit_check and (source_audit or {}).get("status") != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "source_accel64_audit_not_pass",
                "path": args.source_audit,
                "actual": (source_audit or {}).get("status"),
            }
        )
    if source_pid["running"] and not args.allow_active_source:
        findings.append({"severity": "error", "code": "source_accel64_parent_still_running", **source_pid})
    if primary_pid["running"] and not args.allow_active_primary:
        findings.append({"severity": "error", "code": "primary_parent_still_running", **primary_pid})

    mappings = build_promotion_mappings(
        root=root,
        source_config=source_config,
        canonical_config=canonical_config,
        source_shard_root=args.source_shard_root,
        canonical_shard_root=args.canonical_shard_root,
        source_log_dir=args.source_log_dir,
        canonical_log_dir=args.canonical_log_dir,
        source_integrated_run_evidence=args.source_integrated_run_evidence,
        canonical_integrated_run_evidence=args.canonical_integrated_run_evidence,
        source_attached_monitor_metadata=args.source_attached_monitor_metadata,
        canonical_attached_monitor_metadata=args.canonical_attached_monitor_metadata,
    )
    backup_root = _path(root, args.backup_root) / time.strftime("%Y%m%dT%H%M%S")
    actions: list[dict[str, Any]] = []
    if not findings:
        for mapping in mappings:
            action = _apply_symlink_mapping(root, mapping, backup_root=backup_root, dry_run=args.dry_run)
            actions.append(action)
            if action["status"] == "missing_source":
                findings.append({"severity": "error", "code": "missing_promotion_source", **action})

    train_summary: dict[str, Any] | None = None
    canonical_audit: dict[str, Any] | None = None
    if not findings and not args.dry_run and not args.skip_finalize:
        train_summary = _build_canonical_train_summary(args, root)
        if train_summary.get("exit_code") != 0:
            findings.append({"severity": "error", "code": "canonical_train_summary_failed", "exit_code": train_summary.get("exit_code")})
        canonical_audit = write_g003_full_idm_completion_audit(canonical_config, root=root, output_path=args.canonical_audit)
        if canonical_audit.get("status") != "pass":
            findings.append(
                {
                    "severity": "error",
                    "code": "canonical_g003_audit_not_pass",
                    "status": canonical_audit.get("status"),
                    "error_count": canonical_audit.get("error_count"),
                }
            )

    status = "dry_run" if args.dry_run and not findings else "pass" if not findings else "fail"
    payload = {
        "schema": "g003_accel64_promotion_manifest.v1",
        "status": status,
        "root": str(root),
        "source_config": args.source_config,
        "canonical_config": args.canonical_config,
        "source_audit": args.source_audit,
        "source_audit_status": (source_audit or {}).get("status"),
        "source_pid": source_pid,
        "primary_pid": primary_pid,
        "method": "relative_symlink",
        "dry_run": bool(args.dry_run),
        "skip_finalize": bool(args.skip_finalize),
        "backup_root": str(backup_root.relative_to(root) if backup_root.is_relative_to(root) else backup_root),
        "actions": actions,
        "canonical_train_summary_exit_code": train_summary.get("exit_code") if train_summary else None,
        "canonical_audit_status": canonical_audit.get("status") if canonical_audit else None,
        "canonical_audit_error_count": canonical_audit.get("error_count") if canonical_audit else None,
        "findings": findings,
        "claim_boundary": "Promotes an already-passing isolated accel64 G003 run into canonical paths via symlinks; it does not checkpoint OMX state and does not weaken the canonical G003 completion audit.",
    }
    write_json(_path(root, args.output), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a passing isolated G003 accel64 run into canonical G003 artifact paths.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--source-config", default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--canonical-config", default=DEFAULT_CANONICAL_CONFIG)
    parser.add_argument("--source-audit", default="artifacts/idm/g003_full_idm_completion_accel64_audit.json")
    parser.add_argument("--canonical-audit", default="artifacts/idm/g003_full_idm_completion_audit.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--skip-source-audit-check", action="store_true")
    parser.add_argument("--allow-active-source", action="store_true")
    parser.add_argument("--allow-active-primary", action="store_true")
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--source-pid-file", default="outputs/cluster/g003_full_compact_accel64.pid")
    parser.add_argument("--primary-pid-file", default="outputs/cluster/g003_full_compact_parallel.pid")
    parser.add_argument("--source-shard-root", default="outputs/data/d2e_full_corpus_shards_accel64")
    parser.add_argument("--canonical-shard-root", default="outputs/data/d2e_full_corpus_shards")
    parser.add_argument("--source-log-dir", default="artifacts/sources/g003_accel64")
    parser.add_argument("--canonical-log-dir", default="artifacts/sources")
    parser.add_argument("--backup-root", default="artifacts/idm/g003_accel64_promotion_backups")
    parser.add_argument("--source-integrated-run-evidence", default="artifacts/idm/g003_d2e_full_idm_run_accel64.json")
    parser.add_argument("--canonical-integrated-run-evidence", default="artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json")
    parser.add_argument("--source-attached-monitor-metadata", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64_attached.json")
    parser.add_argument("--canonical-attached-monitor-metadata", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json")
    parser.add_argument("--canonical-idm-summary", default="artifacts/idm/idm_streaming_d2e_full_compact_summary.json")
    parser.add_argument("--canonical-checkpoint-metadata", default="outputs/idm_streaming_d2e_full_compact/checkpoint_metadata.json")
    parser.add_argument("--canonical-metrics", default="outputs/idm_streaming_d2e_full_compact/metrics.json")
    parser.add_argument("--canonical-gpu-monitor", default="artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv")
    parser.add_argument("--canonical-train-run-summary", default="artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--expected-gpus", type=int, default=4)
    args = parser.parse_args()
    payload = promote(args)
    print(
        "g003 accel64 promotion: "
        f"status={payload['status']} actions={len(payload['actions'])} "
        f"findings={len(payload['findings'])} output={args.output}"
    )
    return 0 if payload["status"] in {"pass", "dry_run"} or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
