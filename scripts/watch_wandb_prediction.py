#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _load_env_file(path: Path) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    loaded: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))
        loaded.append(key)
    return loaded


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_size(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists() and path.is_file(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _prediction_status(parts_dir: Path, predictions_path: Path, pseudolabels_path: Path) -> dict[str, Any]:
    part_predictions = sorted(parts_dir.glob("part_*/predictions.jsonl")) if parts_dir.exists() else []
    part_pseudolabels = sorted(parts_dir.glob("part_*/pseudolabels.jsonl")) if parts_dir.exists() else []
    part_prediction_bytes = sum(path.stat().st_size for path in part_predictions if path.exists())
    part_pseudolabel_bytes = sum(path.stat().st_size for path in part_pseudolabels if path.exists())
    return {
        "parts_dir": str(parts_dir),
        "part_prediction_count": len(part_predictions),
        "part_prediction_bytes": part_prediction_bytes,
        "part_pseudolabel_count": len(part_pseudolabels),
        "part_pseudolabel_bytes": part_pseudolabel_bytes,
        "predictions": _jsonl_size(predictions_path),
        "pseudolabels": _jsonl_size(pseudolabels_path),
    }


def _process_running(pattern: str) -> bool | None:
    if not pattern:
        return None
    try:
        result = subprocess.run(["pgrep", "-af", pattern], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return None
    return result.returncode == 0


def _nested_get(payload: dict[str, Any] | None, keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _prediction_target_records(payload: dict[str, Any] | None) -> int | None:
    value = _nested_get(payload, ["target_records"])
    if value is None:
        value = _nested_get(payload, ["records"])
    if value is None:
        value = _nested_get(payload, ["metadata", "target_records"])
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Log full-target prediction recovery progress to Weights & Biases.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--prediction-parts-dir", required=True)
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--pseudolabels-path", required=True)
    parser.add_argument("--prediction-summary", required=True)
    parser.add_argument("--recovery-summary", required=True)
    parser.add_argument("--paper-metrics", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid-file")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", default="g005-idm-paper-target")
    parser.add_argument("--job-type", default="prediction-sidecar")
    parser.add_argument("--tags", default="g005,idm,d2e,prediction,sidecar")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--finish-rows", type=int, default=0)
    parser.add_argument("--process-pattern", default="recover_idm_video_outputs.py")
    args = parser.parse_args()

    if args.pid_file:
        pid_path = Path(args.pid_file)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    env_keys = _load_env_file(Path(args.env_file))
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        raise SystemExit("WANDB_PROJECT is required in environment or .env")
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise SystemExit("wandb is not installed; run with `uv run --with wandb ...`") from exc

    wandb_dir = Path(os.environ.setdefault("WANDB_DIR", "outputs/wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=args.run_name,
        group=args.group,
        job_type=args.job_type,
        tags=[tag.strip() for tag in args.tags.split(",") if tag.strip()],
        dir=str(wandb_dir),
        resume=os.environ.get("WANDB_RESUME", "allow"),
        config={
            "schema": "wandb_prediction_sidecar_config.v1",
            "prediction_parts_dir": args.prediction_parts_dir,
            "predictions_path": args.predictions_path,
            "pseudolabels_path": args.pseudolabels_path,
            "prediction_summary": args.prediction_summary,
            "recovery_summary": args.recovery_summary,
            "paper_metrics": args.paper_metrics,
            "audit": args.audit,
            "finish_rows": args.finish_rows,
            "env_keys_loaded": sorted(set(env_keys)),
            "claim_boundary": "Sidecar logs prediction/evaluation artifacts only; it does not run inference or mutate outputs.",
        },
    )
    loop_index = 0
    final_status = "running"
    try:
        while True:
            loop_index += 1
            prediction = _prediction_status(
                Path(args.prediction_parts_dir),
                Path(args.predictions_path),
                Path(args.pseudolabels_path),
            )
            process_running = _process_running(args.process_pattern)
            prediction_summary = _read_json(Path(args.prediction_summary))
            recovery_summary = _read_json(Path(args.recovery_summary))
            paper_metrics = _read_json(Path(args.paper_metrics))
            audit = _read_json(Path(args.audit))
            target_records = _prediction_target_records(prediction_summary)
            recovery_status = recovery_summary.get("status") if recovery_summary else None
            audit_status = audit.get("status") if audit else None
            run.log(
                {
                    "prediction/part_prediction_count": prediction["part_prediction_count"],
                    "prediction/part_prediction_bytes": prediction["part_prediction_bytes"],
                    "prediction/part_pseudolabel_count": prediction["part_pseudolabel_count"],
                    "prediction/part_pseudolabel_bytes": prediction["part_pseudolabel_bytes"],
                    "prediction/canonical_prediction_bytes": prediction["predictions"]["bytes"],
                    "prediction/canonical_pseudolabel_bytes": prediction["pseudolabels"]["bytes"],
                    "prediction/target_records": target_records,
                    "prediction/process_running": process_running,
                    "prediction/summary_exists": Path(args.prediction_summary).exists(),
                    "prediction/recovery_summary_exists": Path(args.recovery_summary).exists(),
                    "eval/paper_metrics_status": paper_metrics.get("status") if paper_metrics else None,
                    "eval/audit_status": audit_status,
                    "sidecar/loop": loop_index,
                    "sidecar/time": time.time(),
                }
            )
            done_by_rows = bool(args.finish_rows and isinstance(target_records, int) and target_records >= args.finish_rows)
            done_by_recovery = recovery_status in {"pass", "fail"}
            if done_by_rows or done_by_recovery:
                final_status = "complete" if recovery_status != "fail" else "failed"
            payload = {
                "schema": "wandb_prediction_sidecar_status.v1",
                "status": final_status,
                "wandb_run_id": getattr(run, "id", None),
                "wandb_run_url": getattr(run, "url", None),
                "project": project,
                "entity_configured": bool(os.environ.get("WANDB_ENTITY")),
                "run_name": args.run_name,
                "loop_index": loop_index,
                "prediction": prediction,
                "process_running": process_running,
                "prediction_summary_target_records": target_records,
                "recovery_summary_status": recovery_status,
                "paper_metrics_status": paper_metrics.get("status") if paper_metrics else None,
                "audit_status": audit_status,
                "updated_at_epoch": time.time(),
                "claim_boundary": "W&B prediction sidecar status contains no secrets and is logging-only evidence.",
            }
            _write_json(Path(args.output), payload)
            if final_status in {"complete", "failed"}:
                break
            time.sleep(max(1.0, args.poll_seconds))
    finally:
        run.finish()
        if args.pid_file:
            pid_path = Path(args.pid_file)
            if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pid_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
