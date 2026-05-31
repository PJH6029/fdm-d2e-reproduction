#!/usr/bin/env python3
"""Fail-closed launch preflight for G003 FDM-1 action-slot materialization."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import sha256_file, write_json
from fdm_d2e.process_liveness import pid_running

DEFAULT_OUTPUT = "artifacts/cluster/fdm1_g003_action_dataset_preflight.json"
DEFAULT_EXTRACT_CONFIG = "configs/data/fdm1_d2e_480p_full_corpus_extract.yaml"
DEFAULT_FINALIZATION_CONFIG = "configs/data/fdm1_g003_action_dataset_finalization.yaml"
DEFAULT_COMPLETION_CONFIG = "configs/eval/fdm1_g003_action_dataset_completion.yaml"
DEFAULT_PID = "outputs/cluster/fdm1_g003_action_dataset_pipeline.pid"
DEFAULT_BRANCH = "research/fdm1-d2e-ultragoal"


def _path(root: str | Path, rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def _safe_exists(path: Path) -> tuple[bool, str | None]:
    try:
        return path.exists(), None
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8", errors="replace").strip().split()[0])
    except Exception:
        return None


def _git(args: list[str], *, root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _artifact(path: Path, *, root: Path, require_json: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    try:
        display = str(path.relative_to(root))
    except Exception:
        display = str(path)
    entry: dict[str, Any] = {"path": display, "exists": path.exists(), "bytes": 0, "sha256": None, "json_ok": None}
    if not path.exists():
        findings.append({"severity": "error", "code": "missing_required_file", "path": display})
        return entry, findings
    if path.is_file():
        entry["bytes"] = path.stat().st_size
        if entry["bytes"] < 32:
            findings.append({"severity": "warning", "code": "suspiciously_small_file", "path": display, "bytes": entry["bytes"]})
        if path.suffix.lower() not in {".jsonl", ".mcap", ".mp4", ".pt", ".bin"}:
            entry["sha256"] = sha256_file(path)
        if require_json:
            try:
                json.loads(path.read_text(encoding="utf-8"))
                entry["json_ok"] = True
            except Exception as exc:
                entry["json_ok"] = False
                findings.append({"severity": "error", "code": "invalid_json", "path": display, "error": str(exc)})
    return entry, findings


def _check_equal(findings: list[dict[str, Any]], actual: Any, expected: Any, *, path: str) -> None:
    if actual != expected:
        findings.append({"severity": "error", "code": "unexpected_config_value", "path": path, "expected": expected, "actual": actual})


def _strictly_increasing(values: list[Any]) -> bool:
    try:
        nums = [float(value) for value in values]
    except Exception:
        return False
    return len(nums) >= 2 and all(a < b for a, b in zip(nums, nums[1:]))


def _check_tokenization_invariants(findings: list[dict[str, Any]], config: dict[str, Any], *, path: str) -> None:
    required_special = {
        "MASK_ACTION",
        "NO_ACTION",
        "PAD_ACTION",
        "BOS_ACTION",
        "EOS_ACTION",
        "EVENT_OVERFLOW",
    }
    _check_equal(findings, config.get("canonical_roadmap"), "ROADMAP.md", path=f"{path}:canonical_roadmap")
    _check_equal(findings, config.get("bin_ms"), 50, path=f"{path}:bin_ms")
    _check_equal(findings, config.get("video_fps"), 20, path=f"{path}:video_fps")
    _check_equal(findings, config.get("k_event_slots_default"), 8, path=f"{path}:k_event_slots_default")
    special = set(map(str, config.get("special_tokens", [])))
    missing_special = sorted(required_special - special)
    if missing_special:
        findings.append(
            {
                "severity": "error",
                "code": "tokenization_missing_special_tokens",
                "path": f"{path}:special_tokens",
                "missing": missing_special,
            }
        )
    mouse = config.get("mouse_move", {}) if isinstance(config.get("mouse_move"), dict) else {}
    _check_equal(findings, mouse.get("default"), "compound", path=f"{path}:mouse_move.default")
    _check_equal(findings, mouse.get("axis_bins"), 49, path=f"{path}:mouse_move.axis_bins")
    boundaries = list(mouse.get("positive_boundaries_default", []))
    if len(boundaries) != 24 or not _strictly_increasing(boundaries):
        findings.append(
            {
                "severity": "error",
                "code": "tokenization_invalid_mouse_boundaries",
                "path": f"{path}:mouse_move.positive_boundaries_default",
                "expected_count": 24,
                "actual_count": len(boundaries),
            }
        )
    click = config.get("click_position_aux", {}) if isinstance(config.get("click_position_aux"), dict) else {}
    _check_equal(findings, click.get("default_horizon_seconds"), 1.0, path=f"{path}:click_position_aux.default_horizon_seconds")
    _check_equal(findings, click.get("default_grid"), [32, 18], path=f"{path}:click_position_aux.default_grid")


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    findings: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    required_files = {
        "roadmap": "ROADMAP.md",
        "extract_config": args.extract_config,
        "finalization_config": args.finalization_config,
        "completion_config": args.completion_config,
        "g002_validation": "artifacts/sources/fdm1_d2e_g002_validation.json",
        "game_metadata": "artifacts/sources/fdm1_d2e_game_metadata.json",
        "recording_level_split_manifest": "artifacts/sources/fdm1_d2e_recording_level_split_manifest.json",
        "heldout_game_split_manifest": "artifacts/sources/fdm1_d2e_heldout_game_split_manifest.json",
        "pseudo_label_split_manifest": "artifacts/sources/fdm1_d2e_pseudo_label_split_manifest.json",
        "scale_split_manifest": "artifacts/sources/fdm1_d2e_scale_split_manifest.json",
        "base_tokenization_config": "configs/tokenization/fdm1_action_slots.json",
    }
    for key, rel in required_files.items():
        require_json = key != "roadmap"
        entry, file_findings = _artifact(_path(root, rel), root=root, require_json=require_json)
        artifacts[key] = entry
        findings.extend(file_findings)

    extract = load_config(_path(root, args.extract_config)) if _path(root, args.extract_config).exists() else {}
    finalization = load_config(_path(root, args.finalization_config)) if _path(root, args.finalization_config).exists() else {}
    completion = load_config(_path(root, args.completion_config)) if _path(root, args.completion_config).exists() else {}
    tokenization_path = "configs/tokenization/fdm1_action_slots.json"
    tokenization = load_config(_path(root, tokenization_path)) if _path(root, tokenization_path).exists() else {}

    _check_equal(findings, extract.get("canonical_roadmap"), "ROADMAP.md", path=f"{args.extract_config}:canonical_roadmap")
    _check_equal(findings, extract.get("split_mode"), "fdm1-g002", path=f"{args.extract_config}:split_mode")
    _check_equal(findings, extract.get("source_ids"), ["d2e_480p"], path=f"{args.extract_config}:source_ids")
    _check_equal(findings, extract.get("resolution_tiers"), ["480p"], path=f"{args.extract_config}:resolution_tiers")
    _check_equal(findings, extract.get("bin_ms"), 50, path=f"{args.extract_config}:bin_ms")
    _check_equal(findings, extract.get("frame_fps"), 20, path=f"{args.extract_config}:frame_fps")
    _check_equal(findings, finalization.get("k_event_slots"), 8, path=f"{args.finalization_config}:k_event_slots")
    _check_equal(findings, completion.get("expected_recording_variants"), 459, path=f"{args.completion_config}:expected_recording_variants")
    _check_equal(findings, completion.get("expected_split_mode"), "fdm1-g002", path=f"{args.completion_config}:expected_split_mode")
    _check_equal(findings, completion.get("required_source_ids"), ["d2e_480p"], path=f"{args.completion_config}:required_source_ids")
    _check_equal(findings, completion.get("required_resolution_tiers"), ["480p"], path=f"{args.completion_config}:required_resolution_tiers")
    if tokenization:
        _check_tokenization_invariants(findings, tokenization, path=tokenization_path)

    g002_path = _path(root, "artifacts/sources/fdm1_d2e_g002_validation.json")
    if g002_path.exists():
        g002 = json.loads(g002_path.read_text(encoding="utf-8"))
        _check_equal(findings, g002.get("status"), "pass", path="artifacts/sources/fdm1_d2e_g002_validation.json:status")

    current_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root=root)
    git_head = _git(["rev-parse", "HEAD"], root=root)
    dirty = _git(["status", "--porcelain"], root=root)
    if args.expected_branch and current_branch != args.expected_branch:
        findings.append({"severity": "error", "code": "unexpected_git_branch", "expected": args.expected_branch, "actual": current_branch})
    if args.require_clean and dirty:
        findings.append({"severity": "error", "code": "dirty_git_worktree", "status_porcelain": dirty.splitlines()[:40]})

    pid_path = _path(root, args.pid_file)
    active_pid = _read_pid(pid_path)
    active = pid_running(active_pid) if active_pid is not None else False
    if active and not args.allow_active_pid:
        findings.append({"severity": "error", "code": "active_g003_pid", "pid": active_pid, "pid_file": str(pid_path)})

    cache_dir = _path(root, str(extract.get("cache_dir", ""))) if extract.get("cache_dir") else None
    cache_exists, cache_error = _safe_exists(cache_dir) if cache_dir else (None, None)
    if cache_error:
        findings.append({"severity": "warning", "code": "cache_dir_stat_error", "path": str(cache_dir), "error": cache_error})
    if args.require_cache_dir and cache_dir and not cache_exists:
        findings.append({"severity": "error", "code": "missing_cache_dir", "path": str(cache_dir), "error": cache_error})

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        findings.append({"severity": "error", "code": "missing_ffmpeg", "path": "PATH"})

    disk_target = _path(root, str(extract.get("output_dir", "outputs/data/fdm1_d2e_480p_window_records"))).parent
    disk_target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(disk_target)
    free_gb = usage.free / (1024**3)
    if free_gb < args.min_free_gb:
        findings.append({"severity": "error", "code": "insufficient_disk_free_gb", "path": str(disk_target), "min_free_gb": args.min_free_gb, "actual_free_gb": round(free_gb, 3)})

    in_pod = bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    if args.require_pod and not in_pod:
        findings.append({"severity": "error", "code": "not_inside_kubernetes_pod"})

    errors = [f for f in findings if f.get("severity") == "error"]
    return {
        "schema": "fdm1_g003_action_dataset_preflight.v1",
        "status": "ready" if not errors else "blocked",
        "canonical_roadmap": "ROADMAP.md",
        "git": {"branch": current_branch, "head": git_head, "dirty": bool(dirty)},
        "runtime": {
            "inside_kubernetes_pod": in_pod,
            "require_pod": args.require_pod,
            "cache_dir": str(cache_dir) if cache_dir else None,
            "cache_dir_exists": cache_exists,
            "cache_dir_error": cache_error,
            "ffmpeg_path": ffmpeg_path,
        },
        "disk": {"path": str(disk_target), "free_gb": round(free_gb, 3), "min_free_gb": args.min_free_gb},
        "pid": {"path": str(pid_path), "pid": active_pid, "running": active},
        "artifacts": artifacts,
        "findings": findings,
        "claim_boundary": "Preflight readiness only; G003 completion still requires full materialization, completion audit pass, evidence bundle pass, and OMX checkpoint.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan/verify G003 pod launch readiness before full-corpus materialization.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--extract-config", default=DEFAULT_EXTRACT_CONFIG)
    parser.add_argument("--finalization-config", default=DEFAULT_FINALIZATION_CONFIG)
    parser.add_argument("--completion-config", default=DEFAULT_COMPLETION_CONFIG)
    parser.add_argument("--pid-file", default=DEFAULT_PID)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-branch", default=DEFAULT_BRANCH)
    parser.add_argument("--min-free-gb", type=float, default=0.0)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--require-pod", action="store_true")
    parser.add_argument("--require-cache-dir", action="store_true")
    parser.add_argument("--allow-active-pid", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args(argv)
    payload = build_preflight(args)
    write_json(_path(args.root, args.output), payload)
    print(f"G003 preflight: status={payload['status']} errors={len([f for f in payload['findings'] if f.get('severity') == 'error'])} output={args.output}")
    return 0 if payload["status"] == "ready" or args.allow_blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
