#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float | int], *, max_items: int) -> None:
    if len(out) >= max_items:
        return
    if isinstance(value, bool):
        out[prefix] = int(value)
        return
    if isinstance(value, (int, float)):
        out[prefix] = value
        return
    if isinstance(value, dict):
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            if len(out) >= max_items:
                break
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            _flatten_numeric(child_prefix, child, out, max_items=max_items)


def _artifact_file_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists() and path.is_file(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Log evaluation JSON artifacts to Weights & Biases without exposing secrets.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", default="g005-idm-paper-target")
    parser.add_argument("--job-type", default="eval-artifacts")
    parser.add_argument("--tags", default="g005,idm,d2e,eval")
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--artifact-type", default="evaluation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", action="append", default=[], help="JSON artifact path to attach and flatten numeric values from.")
    parser.add_argument("--file", action="append", default=[], help="Non-JSON file path to attach without metric flattening.")
    parser.add_argument("--max-flattened-metrics", type=int, default=256)
    args = parser.parse_args()

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

    json_paths = [Path(path) for path in args.json]
    file_paths = [Path(path) for path in args.file]
    payloads = {str(path): _read_json(path) for path in json_paths}
    metrics: dict[str, float | int] = {}
    for path, payload in payloads.items():
        if payload is not None:
            stem = Path(path).stem
            _flatten_numeric(stem, payload, metrics, max_items=max(0, int(args.max_flattened_metrics)))

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
            "schema": "wandb_eval_artifact_logger_config.v1",
            "json_paths": [str(path) for path in json_paths],
            "file_paths": [str(path) for path in file_paths],
            "env_keys_loaded": sorted(set(env_keys)),
            "claim_boundary": "Logging-only W&B artifact run; this script does not train, evaluate, or mutate research outputs.",
        },
    )
    status = "complete"
    artifact = wandb.Artifact(args.artifact_name, type=args.artifact_type)
    attached: list[dict[str, Any]] = []
    try:
        if metrics:
            run.log(metrics)
        for path in [*json_paths, *file_paths]:
            meta = _artifact_file_metadata(path)
            attached.append(meta)
            if meta["exists"]:
                artifact.add_file(str(path))
            else:
                status = "partial"
        if attached:
            run.log_artifact(artifact)
    finally:
        run.finish()

    output_payload = {
        "schema": "wandb_eval_artifact_logger_status.v1",
        "status": status,
        "wandb_run_id": getattr(run, "id", None),
        "wandb_run_url": getattr(run, "url", None),
        "project": project,
        "entity_configured": bool(os.environ.get("WANDB_ENTITY")),
        "run_name": args.run_name,
        "artifact_name": args.artifact_name,
        "attached_files": attached,
        "flattened_metric_count": len(metrics),
        "updated_at_epoch": time.time(),
        "claim_boundary": "Status contains no W&B secrets; artifact contents are limited to the user-provided paths.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "wandb_run_url": getattr(run, "url", None)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
