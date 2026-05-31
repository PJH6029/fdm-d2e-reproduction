#!/usr/bin/env python3
"""Collect pod-side status for the G003 FDM-1 action-slot pipeline.

This helper is intentionally evidence-oriented: it does not claim completion from
process exit alone.  Completion remains tied to the G003 completion audit and the
small-artifact evidence bundle.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import sha256_file, write_json
from fdm_d2e.process_liveness import pid_running

DEFAULT_PID = "outputs/cluster/fdm1_g003_action_dataset_pipeline.pid"
DEFAULT_LOG = "artifacts/logs/fdm1_g003_action_dataset_pipeline.log"
DEFAULT_COMPLETION_CONFIG = "configs/eval/fdm1_g003_action_dataset_completion.yaml"
DEFAULT_OUTPUT = "artifacts/cluster/fdm1_g003_action_dataset_pod_monitor.json"
DEFAULT_BUNDLE_MANIFEST = "artifacts/sources/fdm1_g003_evidence_bundle_manifest.json"
FATAL_PATTERNS = ("Traceback", "RuntimeError", "CalledProcessError", "No space left", "Killed", "CUDA out of memory")
LARGE_SUFFIXES = {".jsonl", ".mcap", ".mp4", ".pt", ".bin"}


def _path(root: str | Path, rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        return int(text.split()[0])
    except ValueError:
        return None


def tail_lines(path: Path, lines: int) -> list[str]:
    if not path.exists() or not path.is_file() or lines <= 0:
        return []
    # Logs are small enough for the status tail; avoid clever reverse readers for reliability.
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def file_record(path: Path, *, root: str | Path = ".", include_sha: bool = True) -> dict[str, Any]:
    rel = str(path if path.is_absolute() else path)
    try:
        display = str(path.relative_to(root))
    except Exception:
        display = rel
    entry: dict[str, Any] = {"path": display, "exists": path.exists(), "bytes": 0, "sha256": None}
    if path.exists() and path.is_file():
        entry["bytes"] = path.stat().st_size
        if include_sha and path.suffix not in LARGE_SUFFIXES:
            entry["sha256"] = sha256_file(path)
    return entry


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid_json", "error": str(exc)}
    return data if isinstance(data, dict) else {"status": "invalid_json", "error": "top-level JSON is not an object"}


def run_command(cmd: list[str], *, root: str | Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout.splitlines()[-40:], "stderr_tail": proc.stderr.splitlines()[-40:]}


def artifact_matrix(completion_config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    omit = set(map(str, completion_config.get("omit_sha256_artifact_keys", [])))
    matrix: dict[str, Any] = {}
    for key, rel in sorted(dict(completion_config.get("paths", {})).items()):
        path = _path(root, str(rel))
        matrix[str(key)] = file_record(path, root=root, include_sha=str(key) not in omit)
    return matrix


def determine_status(*, pid_is_running: bool, audit: dict[str, Any] | None, bundle: dict[str, Any] | None, fatal_matches: list[str]) -> str:
    if audit and audit.get("status") == "pass" and bundle and bundle.get("status") == "pass":
        return "pass"
    if audit and audit.get("status") == "pass":
        return "audit_pass_bundle_missing"
    if pid_is_running:
        return "running"
    if fatal_matches:
        return "failed_or_interrupted"
    return "incomplete"


def collect_status(
    *,
    root: str | Path = ".",
    pid_file: str | Path = DEFAULT_PID,
    log_path: str | Path = DEFAULT_LOG,
    completion_config_path: str | Path = DEFAULT_COMPLETION_CONFIG,
    bundle_manifest_path: str | Path = DEFAULT_BUNDLE_MANIFEST,
    tail: int = 80,
) -> dict[str, Any]:
    root_path = Path(root)
    config = load_config(_path(root_path, completion_config_path))
    pid_path = _path(root_path, pid_file)
    log_file = _path(root_path, log_path)
    pid = read_pid(pid_path)
    running = pid_running(pid) if pid is not None else False
    log_tail = tail_lines(log_file, tail)
    fatal_matches = [line for line in log_tail if any(pattern in line for pattern in FATAL_PATTERNS)]
    audit_path = _path(root_path, str(config.get("output_path", "artifacts/sources/fdm1_g003_action_dataset_completion_audit.json")))
    audit = load_json_if_exists(audit_path)
    bundle = load_json_if_exists(_path(root_path, bundle_manifest_path))
    status = determine_status(pid_is_running=running, audit=audit, bundle=bundle, fatal_matches=fatal_matches)
    return {
        "schema": "fdm1_g003_pod_monitor.v1",
        "status": status,
        "pid": pid,
        "pid_running": running,
        "pid_file": file_record(pid_path, root=root_path),
        "log": file_record(log_file, root=root_path, include_sha=False),
        "log_tail": log_tail,
        "fatal_log_matches": fatal_matches,
        "completion_audit": {
            "path": str(audit_path.relative_to(root_path) if not audit_path.is_absolute() else audit_path),
            "exists": audit_path.exists(),
            "status": audit.get("status") if audit else None,
            "error_count": len([f for f in audit.get("findings", []) if f.get("severity") == "error"]) if audit and isinstance(audit.get("findings"), list) else None,
        },
        "evidence_bundle": {
            "path": str(Path(bundle_manifest_path)),
            "exists": _path(root_path, bundle_manifest_path).exists(),
            "status": bundle.get("status") if bundle else None,
        },
        "artifacts": artifact_matrix(config, root=root_path),
        "claim_boundary": "Pod monitor status is operational evidence only; G003 completion requires audit pass, evidence bundle pass, and OMX checkpoint.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect G003 action-token pod launch/materialization status.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--pid-file", default=DEFAULT_PID)
    parser.add_argument("--log-path", default=DEFAULT_LOG)
    parser.add_argument("--completion-config", default=DEFAULT_COMPLETION_CONFIG)
    parser.add_argument("--bundle-manifest", default=DEFAULT_BUNDLE_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--tail", type=int, default=80)
    parser.add_argument("--refresh-audit", action="store_true", help="Run the G003 completion validator with --allow-fail before collecting status.")
    parser.add_argument("--build-bundle-if-pass", action="store_true", help="Build the evidence bundle when the refreshed/current audit is pass.")
    parser.add_argument("--require-pass", action="store_true", help="Exit nonzero unless monitor status is pass.")
    args = parser.parse_args(argv)

    commands: list[dict[str, Any]] = []
    if args.refresh_audit:
        commands.append(
            run_command(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/validate_fdm1_g003_action_dataset_completion.py",
                    "--config",
                    str(args.completion_config),
                    "--allow-fail",
                ],
                root=args.root,
            )
        )
    status = collect_status(
        root=args.root,
        pid_file=args.pid_file,
        log_path=args.log_path,
        completion_config_path=args.completion_config,
        bundle_manifest_path=args.bundle_manifest,
        tail=args.tail,
    )
    if args.build_bundle_if_pass and status["completion_audit"]["status"] == "pass":
        commands.append(
            run_command(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/build_fdm1_g003_evidence_bundle.py",
                    "--completion-config",
                    str(args.completion_config),
                ],
                root=args.root,
            )
        )
        status = collect_status(
            root=args.root,
            pid_file=args.pid_file,
            log_path=args.log_path,
            completion_config_path=args.completion_config,
            bundle_manifest_path=args.bundle_manifest,
            tail=args.tail,
        )
    if commands:
        status["commands"] = commands
    write_json(_path(args.root, args.output), status)
    print(
        "G003 pod monitor: "
        f"status={status['status']} pid={status['pid']} running={status['pid_running']} "
        f"audit={status['completion_audit']['status']} bundle={status['evidence_bundle']['status']} output={args.output}"
    )
    return 0 if status["status"] == "pass" or not args.require_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
