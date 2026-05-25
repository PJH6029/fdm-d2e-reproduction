from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fdm_d2e.io_utils import ensure_dir, read_json, sha256_file, write_json

REMOVED_INLINE_DEPS = ("owa-cli @", "owa-env-gst @")


@dataclass(frozen=True)
class GidmRunPlan:
    index: int
    universe_row_id: str
    video_path: str
    prediction_mcap_path: str
    cuda_device: str
    log_path: str


def prepare_desktop_minimal_inference_script(d2e_repo: str | Path) -> Path:
    """Create the smallest upstream D2E inference script needed for batch runs.

    The public D2E script includes CLI/GStreamer desktop environment packages
    that are not imported by inference.py but can fail to build in lean cluster
    images. We keep owa-env-desktop because owa-data imports VK constants.
    """

    repo = Path(d2e_repo)
    source = repo / "inference.py"
    if not source.exists():
        raise FileNotFoundError(f"missing upstream D2E inference.py: {source}")
    target = repo / "inference_desktop_minimal.py"
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if any(fragment in line for fragment in REMOVED_INLINE_DEPS):
            continue
        lines.append(line)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def build_gidm_run_plan(
    *,
    manifest_path: str | Path,
    cuda_devices: Sequence[str],
    log_dir: str | Path,
    max_recordings: int | None = None,
    recording_keys: Sequence[str] | None = None,
    resume: bool = True,
) -> list[GidmRunPlan]:
    if not cuda_devices:
        raise ValueError("at least one CUDA device is required")
    manifest = read_json(manifest_path)
    wanted = set(str(key) for key in (recording_keys or []))
    rows = []
    for row in manifest.get("recordings", []):
        if not isinstance(row, dict):
            continue
        key = str(row.get("universe_row_id") or "")
        if wanted and key not in wanted:
            continue
        pred = Path(str(row.get("prediction_mcap_path") or ""))
        if resume and pred.exists() and pred.stat().st_size > 0:
            continue
        rows.append(row)
        if max_recordings is not None and len(rows) >= int(max_recordings):
            break

    log_root = ensure_dir(log_dir)
    plans = []
    for index, row in enumerate(rows):
        key = str(row["universe_row_id"])
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key).strip("_")
        plans.append(
            GidmRunPlan(
                index=index,
                universe_row_id=key,
                video_path=str(row["video_path"]),
                prediction_mcap_path=str(row["prediction_mcap_path"]),
                cuda_device=str(cuda_devices[index % len(cuda_devices)]),
                log_path=str(log_root / f"{index:05d}_{safe}.log"),
            )
        )
    return plans


def _run_one(
    plan: GidmRunPlan,
    *,
    script_path: Path,
    d2e_repo: Path,
    model: str,
    max_context_length: int,
    max_duration: float | None,
    uv_cache_dir: str | Path,
    hf_home: str | Path,
) -> dict[str, Any]:
    manifest_output_path = Path(plan.prediction_mcap_path)
    output_path = manifest_output_path if manifest_output_path.is_absolute() else (Path.cwd() / manifest_output_path).resolve()
    ensure_dir(output_path.parent)
    ensure_dir(Path(plan.log_path).parent)
    temp_output_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}.{plan.index}")
    if temp_output_path.exists():
        temp_output_path.unlink()
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": plan.cuda_device,
            "UV_CACHE_DIR": str(Path(uv_cache_dir).resolve()),
            "HF_HOME": str(Path(hf_home).resolve()),
            "TRANSFORMERS_CACHE": str((Path(hf_home) / "transformers").resolve()),
        }
    )
    cmd = [
        "uv",
        "run",
        str(script_path.name),
        plan.video_path,
        str(temp_output_path),
        "--model",
        model,
        "--device",
        "cuda",
        "--max-context-length",
        str(int(max_context_length)),
    ]
    if max_duration is not None:
        cmd.extend(["--max-duration", str(float(max_duration))])
    started = time.time()
    with Path(plan.log_path).open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(cmd, cwd=d2e_repo, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    ended = time.time()
    if proc.returncode == 0 and temp_output_path.exists() and temp_output_path.stat().st_size > 0:
        shutil.move(str(temp_output_path), str(output_path))
    elif temp_output_path.exists() and proc.returncode != 0:
        temp_output_path.unlink()
    success = int(proc.returncode) == 0 and output_path.exists() and output_path.stat().st_size > 0
    return {
        "universe_row_id": plan.universe_row_id,
        "video_path": plan.video_path,
        "prediction_mcap_path": plan.prediction_mcap_path,
        "resolved_prediction_mcap_path": str(output_path),
        "temp_prediction_mcap_path": str(temp_output_path),
        "cuda_device": plan.cuda_device,
        "log_path": plan.log_path,
        "exit_code": int(proc.returncode),
        "success": bool(success),
        "elapsed_seconds": ended - started,
        "output_exists": output_path.exists(),
        "output_size": output_path.stat().st_size if output_path.exists() else 0,
        "output_sha256": sha256_file(output_path) if output_path.exists() and output_path.is_file() else None,
    }


def run_gidm_manifest_inference(
    *,
    manifest_path: str | Path,
    d2e_repo: str | Path,
    output_summary: str | Path,
    cuda_devices: Sequence[str],
    workers: int,
    model: str = "open-world-agents/Generalist-IDM-1B",
    max_recordings: int | None = None,
    recording_keys: Sequence[str] | None = None,
    max_context_length: int = 2048,
    max_duration: float | None = None,
    uv_cache_dir: str | Path = "outputs/external/uv-cache-desktop-minimal",
    hf_home: str | Path = "outputs/external/hf-home",
    log_dir: str | Path = "artifacts/eval/gidm_manifest_inference_logs",
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo = Path(d2e_repo)
    script = prepare_desktop_minimal_inference_script(repo)
    plans = build_gidm_run_plan(
        manifest_path=manifest_path,
        cuda_devices=cuda_devices,
        log_dir=log_dir,
        max_recordings=max_recordings,
        recording_keys=recording_keys,
        resume=resume,
    )
    if dry_run:
        rows = [plan.__dict__ for plan in plans]
    else:
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = [
                executor.submit(
                    _run_one,
                    plan,
                    script_path=script,
                    d2e_repo=repo,
                    model=model,
                    max_context_length=max_context_length,
                    max_duration=max_duration,
                    uv_cache_dir=uv_cache_dir,
                    hf_home=hf_home,
                )
                for plan in plans
            ]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda row: str(row.get("universe_row_id", "")))
    payload = {
        "schema": "gidm_manifest_inference_run.v1",
        "manifest_path": str(manifest_path),
        "d2e_repo": str(repo),
        "inference_script": str(script),
        "model": model,
        "workers": int(workers),
        "cuda_devices": [str(device) for device in cuda_devices],
        "max_recordings": max_recordings,
        "recording_keys": [str(key) for key in (recording_keys or [])],
        "max_context_length": int(max_context_length),
        "max_duration": max_duration,
        "resume": bool(resume),
        "dry_run": bool(dry_run),
        "planned_recordings": len(plans),
        "completed_recordings": sum(1 for row in rows if bool(row.get("success", int(row.get("exit_code", 0)) == 0 and row.get("output_exists")))) if not dry_run else 0,
        "failed_recordings": sum(1 for row in rows if not bool(row.get("success", int(row.get("exit_code", 0)) == 0 and row.get("output_exists")))) if not dry_run else 0,
        "rows": rows,
        "claim_boundary": "Released G-IDM inference execution evidence only; metric claims require conversion plus paper-metric artifacts.",
    }
    write_json(output_summary, payload)
    return payload
