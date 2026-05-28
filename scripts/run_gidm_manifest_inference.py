#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.gidm_runner import run_gidm_manifest_inference


def main() -> int:
    parser = argparse.ArgumentParser(description="Run released Generalist-IDM inference over a manifest shard.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--d2e-repo", default="outputs/external/D2E")
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--cuda-devices", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="open-world-agents/Generalist-IDM-1B")
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--recording-key", action="append", default=[])
    parser.add_argument("--max-context-length", type=int, default=2048)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--chunk-seconds", type=float, help="Run each manifest recording as timestamp-aligned video chunks.")
    parser.add_argument("--chunk-context-seconds", type=float, default=1.0)
    parser.add_argument("--chunk-manifest-output")
    parser.add_argument(
        "--chunk-timestamp-mode",
        default="ground_truth_aligned",
        choices=["ground_truth_aligned", "video_relative", "ground_truth_plus_base"],
        help=(
            "Timestamp offset mode for chunked released-GIDM pilots. Keep the default for completion evidence; "
            "non-default modes are diagnostics for timing-alignment hypotheses."
        ),
    )
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--max-chunks", type=int, help="Limit planned chunks for bounded pilots.")
    parser.add_argument("--uv-cache-dir", default="outputs/external/uv-cache-desktop-minimal")
    parser.add_argument("--hf-home", default="outputs/external/hf-home")
    parser.add_argument("--log-dir", default="artifacts/eval/gidm_manifest_inference_logs")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = run_gidm_manifest_inference(
        manifest_path=args.manifest,
        d2e_repo=args.d2e_repo,
        output_summary=args.output_summary,
        cuda_devices=[item.strip() for item in args.cuda_devices.split(",") if item.strip()],
        workers=args.workers,
        model=args.model,
        max_recordings=args.max_recordings,
        recording_keys=args.recording_key,
        max_context_length=args.max_context_length,
        max_duration=args.max_duration,
        chunk_seconds=args.chunk_seconds,
        chunk_context_seconds=args.chunk_context_seconds,
        chunk_manifest_output=args.chunk_manifest_output,
        chunk_timestamp_mode=args.chunk_timestamp_mode,
        bin_ms=args.bin_ms,
        max_chunks=args.max_chunks,
        uv_cache_dir=args.uv_cache_dir,
        hf_home=args.hf_home,
        log_dir=args.log_dir,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["dry_run"] or payload["failed_recordings"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
