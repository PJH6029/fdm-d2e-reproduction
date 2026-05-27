from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from fdm_d2e.eval.gidm_adapter import convert_gidm_mcap_predictions
from fdm_d2e.eval.gidm_runner import run_gidm_manifest_inference
from fdm_d2e.eval.gidm_targets import TARGET_SPLIT_TAGS, extract_gidm_target_records
from fdm_d2e.eval.paper_idm_metrics import write_paper_idm_metrics
from fdm_d2e.io_utils import read_json, write_json


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _maybe_relative(root: Path, value: str | Path) -> str:
    path = _path(root, value)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _list(value: Any, default: Sequence[str] = ()) -> list[str]:
    if value is None:
        return [str(item) for item in default]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _load_status(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": str(path), "exists": False}
    try:
        payload = read_json(path)
    except Exception as exc:
        return {"path": str(path), "exists": True, "json_error": str(exc)}
    if not isinstance(payload, dict):
        return {"path": str(path), "exists": True, "status": None}
    return {
        "path": str(path),
        "exists": True,
        "status": payload.get("status"),
        "error_count": payload.get("error_count"),
        "rows": payload.get("rows_written") or payload.get("target_rows") or payload.get("recording_count"),
        "alignment_rows": (payload.get("alignment") or {}).get("rows_seen") if isinstance(payload.get("alignment"), dict) else None,
    }


def _log_wandb_artifacts(
    *,
    root: Path,
    config: dict[str, Any],
    summary_path: Path,
    stage: str,
    enabled: bool,
) -> dict[str, Any] | None:
    wandb_cfg = dict(config.get("wandb", {}))
    if not enabled or not bool(wandb_cfg.get("enabled", False)):
        return None
    output = _path(root, wandb_cfg.get("output", f"artifacts/eval/g006_gidm_exact_split_{stage}_wandb_status.json"))
    json_paths = [
        summary_path,
        _path(root, config.get("inference_summary", "artifacts/eval/g006_gidm_exact_split_manifest_runner_summary.json")),
        _path(root, config.get("target_summary", "artifacts/eval/g006_gidm_exact_split_target_extraction_summary.json")),
        _path(root, config.get("conversion_summary", "artifacts/eval/g006_gidm_exact_split_conversion_summary.json")),
        _path(root, dict(config.get("paper_metrics", {})).get("output_path", "artifacts/eval/g006_gidm_exact_split_paper_metrics.json")),
    ]
    cmd = [
        sys.executable,
        "scripts/log_wandb_artifacts.py",
        "--env-file",
        str(_path(root, wandb_cfg.get("env_file", ".env"))),
        "--run-name",
        str(wandb_cfg.get("run_name", f"g006-gidm-exact-split-{stage}")),
        "--group",
        str(wandb_cfg.get("group", "g006-gidm-exact-split")),
        "--job-type",
        str(wandb_cfg.get("job_type", "eval-artifacts")),
        "--tags",
        str(wandb_cfg.get("tags", f"g006,gidm,d2e,{stage}")),
        "--artifact-name",
        str(wandb_cfg.get("artifact_name", f"g006-gidm-exact-split-{stage}")),
        "--output",
        str(output),
    ]
    for path in json_paths:
        if path.exists():
            cmd.extend(["--json", str(path)])
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "exit_code": int(proc.returncode),
        "output": str(output),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_gidm_exact_split_pipeline(
    config: dict[str, Any],
    *,
    root: str | Path = ".",
    stage: str = "all",
    allow_partial: bool = False,
    dry_run: bool = False,
    max_recordings: int | None = None,
    recording_keys: Sequence[str] | None = None,
    cuda_devices: Sequence[str] | None = None,
    workers: int | None = None,
    log_wandb: bool = True,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    started = time.time()
    stage = str(stage)
    if stage not in {"all", "inference", "finalize"}:
        raise ValueError("stage must be one of: all, inference, finalize")
    manifest_path = _path(root_path, config["manifest_path"])
    active_manifest_path = manifest_path
    summary_path = _path(root_path, config.get("summary_out", "artifacts/eval/g006_gidm_exact_split_pipeline_summary.json"))
    statuses: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []

    selected_cuda_devices = [str(item) for item in (cuda_devices if cuda_devices is not None else _list(config.get("cuda_devices"), ["0", "1", "2", "3"]))]
    selected_workers = int(workers if workers is not None else config.get("workers", len(selected_cuda_devices)))
    selected_recording_keys = [str(key) for key in (recording_keys or config.get("recording_keys", []) or [])]
    selected_max_recordings = int(max_recordings) if max_recordings is not None else config.get("max_recordings")

    try:
        if stage in {"all", "inference"}:
            inference_summary = run_gidm_manifest_inference(
                manifest_path=manifest_path,
                d2e_repo=_path(root_path, config.get("d2e_repo", "outputs/external/D2E")),
                output_summary=_path(root_path, config.get("inference_summary", "artifacts/eval/g006_gidm_exact_split_manifest_runner_summary.json")),
                cuda_devices=selected_cuda_devices,
                workers=selected_workers,
                model=str(config.get("model", "open-world-agents/Generalist-IDM-1B")),
                max_recordings=selected_max_recordings,
                recording_keys=selected_recording_keys,
                max_context_length=int(config.get("max_context_length", 2048)),
                max_duration=config.get("max_duration"),
                chunk_seconds=config.get("chunk_seconds"),
                chunk_context_seconds=float(config.get("chunk_context_seconds", 1.0)),
                chunk_manifest_output=_path(root_path, config["chunk_manifest_output"]) if config.get("chunk_manifest_output") else None,
                bin_ms=int(config.get("bin_ms", 50)),
                max_chunks=config.get("max_chunks"),
                uv_cache_dir=_path(root_path, config.get("uv_cache_dir", "outputs/external/uv-cache-desktop-minimal")),
                hf_home=_path(root_path, config.get("hf_home", "outputs/external/hf-home")),
                log_dir=_path(root_path, config.get("inference_log_dir", "artifacts/eval/gidm_exact_split_inference_logs")),
                resume=bool(config.get("resume", True)),
                dry_run=dry_run,
            )
            statuses["inference"] = {
                "status": "pass" if inference_summary.get("dry_run") or int(inference_summary.get("failed_recordings", 0) or 0) == 0 else "fail",
                "planned_recordings": inference_summary.get("planned_recordings"),
                "completed_recordings": inference_summary.get("completed_recordings"),
                "failed_recordings": inference_summary.get("failed_recordings"),
            }
            if statuses["inference"]["status"] != "pass":
                findings.append({"severity": "error", "code": "inference_failed", **statuses["inference"]})
            if config.get("chunk_manifest_output"):
                chunk_manifest_path = _path(root_path, config["chunk_manifest_output"])
                if chunk_manifest_path.exists():
                    active_manifest_path = chunk_manifest_path

        if stage in {"all", "finalize"} and not dry_run:
            if stage == "finalize" and config.get("chunk_manifest_output"):
                chunk_manifest_path = _path(root_path, config["chunk_manifest_output"])
                if chunk_manifest_path.exists():
                    active_manifest_path = chunk_manifest_path
            target_summary = extract_gidm_target_records(
                manifest_path=active_manifest_path,
                by_recording_roots=[_maybe_relative(root_path, item) for item in _list(config.get("by_recording_roots"))],
                output_path=_path(root_path, config.get("target_records", "outputs/gidm_exact_split/full_targets.jsonl")),
                summary_out=_path(root_path, config.get("target_summary", "artifacts/eval/g006_gidm_exact_split_target_extraction_summary.json")),
                recording_keys=selected_recording_keys,
                split_tags=_list(dict(config.get("paper_metrics", {})).get("split_tags"), TARGET_SPLIT_TAGS),
                only_existing_predictions=bool(allow_partial),
            )
            statuses["target_extraction"] = {
                "status": target_summary.get("status"),
                "rows_written": target_summary.get("rows_written"),
                "recording_count": target_summary.get("recording_count"),
                "error_count": target_summary.get("error_count"),
            }
            if target_summary.get("status") != "pass":
                findings.append({"severity": "error", "code": "target_extraction_failed", **statuses["target_extraction"]})

            conversion_summary = convert_gidm_mcap_predictions(
                manifest_path=active_manifest_path,
                target_record_paths=[_path(root_path, config.get("target_records", "outputs/gidm_exact_split/full_targets.jsonl"))],
                output_path=_path(root_path, config.get("predictions", "outputs/gidm_exact_split/full_predictions.jsonl")),
                summary_out=_path(root_path, config.get("conversion_summary", "artifacts/eval/g006_gidm_exact_split_conversion_summary.json")),
                bin_ms=int(config.get("bin_ms", 50)),
                timestamp_shift_ns=int(config.get("timestamp_shift_ns", 0) or 0),
                auto_timestamp_shift_from_screen=bool(config.get("auto_timestamp_shift_from_screen", True)),
                allow_missing=bool(allow_partial),
            )
            statuses["conversion"] = {
                "status": "pass" if not conversion_summary.get("missing_prediction_count") or allow_partial else "fail",
                "rows_written": conversion_summary.get("rows_written"),
                "recording_count": conversion_summary.get("recording_count"),
                "missing_prediction_count": conversion_summary.get("missing_prediction_count"),
            }
            if statuses["conversion"]["status"] != "pass":
                findings.append({"severity": "error", "code": "conversion_missing_predictions", **statuses["conversion"]})

            metrics_cfg = dict(config.get("paper_metrics", {}))
            paper_metrics = write_paper_idm_metrics(
                prediction_paths=[_path(root_path, path) for path in _list(metrics_cfg.get("prediction_paths"), [config.get("predictions", "outputs/gidm_exact_split/full_predictions.jsonl")])],
                target_paths=[_path(root_path, path) for path in _list(metrics_cfg.get("target_paths"), [config.get("target_records", "outputs/gidm_exact_split/full_targets.jsonl")])],
                output_path=_path(root_path, metrics_cfg.get("output_path", "artifacts/eval/g006_gidm_exact_split_paper_metrics.json")),
                split_tags=_list(metrics_cfg.get("split_tags"), TARGET_SPLIT_TAGS),
                model_name=str(metrics_cfg.get("model_name", config.get("model_name", "released_generalist_idm_1b_exact_split"))),
                max_rows=metrics_cfg.get("max_rows"),
                progress_output_path=_path(root_path, metrics_cfg["progress_output_path"]) if metrics_cfg.get("progress_output_path") else None,
                progress_rows=int(metrics_cfg.get("progress_rows", 1_000_000)),
                empty_bins_as_correct=bool(metrics_cfg.get("empty_bins_as_correct", False)),
            )
            statuses["paper_metrics"] = {
                "status": paper_metrics.get("status"),
                "rows_seen": (paper_metrics.get("alignment") or {}).get("rows_seen"),
                "error_count": paper_metrics.get("error_count"),
            }
            if paper_metrics.get("status") != "pass":
                findings.append({"severity": "error", "code": "paper_metrics_failed", **statuses["paper_metrics"]})
    except Exception as exc:
        findings.append({"severity": "error", "code": "pipeline_exception", "message": str(exc), "type": type(exc).__name__})

    artifact_status = {
        "manifest": _metadata(manifest_path),
        "inference_summary": _load_status(_path(root_path, config.get("inference_summary", "artifacts/eval/g006_gidm_exact_split_manifest_runner_summary.json"))),
        "target_summary": _load_status(_path(root_path, config.get("target_summary", "artifacts/eval/g006_gidm_exact_split_target_extraction_summary.json"))),
        "conversion_summary": _load_status(_path(root_path, config.get("conversion_summary", "artifacts/eval/g006_gidm_exact_split_conversion_summary.json"))),
        "paper_metrics": _load_status(_path(root_path, dict(config.get("paper_metrics", {})).get("output_path", "artifacts/eval/g006_gidm_exact_split_paper_metrics.json"))),
    }
    errors = [item for item in findings if item.get("severity") == "error"]
    payload = {
        "schema": "g006_gidm_exact_split_pipeline_summary.v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "goal_id": str(config.get("goal_id", "G006-g-idm-exact-split")),
        "stage": stage,
        "allow_partial": bool(allow_partial),
        "dry_run": bool(dry_run),
        "model": str(config.get("model", "open-world-agents/Generalist-IDM-1B")),
        "manifest_path": str(manifest_path),
        "active_manifest_path": str(active_manifest_path),
        "chunk_manifest_output": str(_path(root_path, config["chunk_manifest_output"])) if config.get("chunk_manifest_output") else None,
        "cuda_devices": selected_cuda_devices,
        "workers": selected_workers,
        "max_recordings": selected_max_recordings,
        "max_chunks": config.get("max_chunks"),
        "recording_keys": selected_recording_keys,
        "statuses": statuses,
        "artifacts": artifact_status,
        "wall_clock_seconds": time.time() - started,
        "findings": findings,
        "claim_boundary": str(config.get("claim_boundary", "Released G-IDM exact-split pipeline evidence; not our-IDM paper-target success.")),
    }
    write_json(summary_path, payload)
    wandb_status = _log_wandb_artifacts(root=root_path, config=config, summary_path=summary_path, stage=stage, enabled=log_wandb and not dry_run)
    if wandb_status is not None:
        payload["wandb_status"] = wandb_status
        write_json(summary_path, payload)
    return payload
