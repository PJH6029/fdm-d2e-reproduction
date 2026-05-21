#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.cluster.g003_monitor import build_g003_resume_plan, write_g003_resume_plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan safe G003 shard resume commands without executing them.")
    parser.add_argument("--progress-report", help="Existing monitor JSON to plan from; omit to build a fresh local report.")
    parser.add_argument("--allow-active-parent", action="store_true", help="Mark plan runnable even when original parent PID is active.")
    parser.add_argument("--shard-root", default="outputs/data/d2e_full_corpus_shards")
    parser.add_argument("--log-dir", default="artifacts/sources")
    parser.add_argument("--data-universe", default="artifacts/sources/d2e_full_data_universe_manifest.json")
    parser.add_argument("--output-dir", default="outputs/data/d2e_full_corpus")
    parser.add_argument("--idm-output-dir", default="outputs/idm_streaming_d2e_full_compact")
    parser.add_argument("--pid-file", default="outputs/cluster/g003_full_compact_parallel.pid")
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--stale-seconds", type=float, default=3600.0)
    parser.add_argument("--cache-dir", default="/root/work/data/d2e/cache")
    parser.add_argument("--uv-bin", default="uv")
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--frame-fps", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--video-mode", default="download", choices=["download", "remote"])
    parser.add_argument("--output", default="artifacts/idm/g003_resume_plan.json")
    args = parser.parse_args()

    progress = None
    if args.progress_report:
        progress = json.loads(Path(args.progress_report).read_text(encoding="utf-8"))
    plan = write_g003_resume_plan(
        args.output,
        progress_report=progress,
        allow_active_parent=args.allow_active_parent,
        shard_root=args.shard_root,
        log_dir=args.log_dir,
        data_universe=args.data_universe,
        output_dir=args.output_dir,
        idm_output_dir=args.idm_output_dir,
        pid_file=args.pid_file,
        num_shards=args.num_shards,
        stale_seconds=args.stale_seconds,
        cache_dir=args.cache_dir,
        uv_bin=args.uv_bin,
        bin_ms=args.bin_ms,
        frame_fps=args.frame_fps,
        image_size=args.image_size,
        video_mode=args.video_mode,
    )
    print(
        "g003 resume plan: "
        f"status={plan['status']} runnable={plan['runnable']} "
        f"incomplete={plan['incomplete_shards']} stale={plan['stale_shards']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
