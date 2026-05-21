#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.cluster.g003_monitor import write_g003_live_health_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a non-mutating G003 live process-topology and shard-health report."
    )
    parser.add_argument("--shard-root", default="outputs/data/d2e_full_corpus_shards")
    parser.add_argument("--log-dir", default="artifacts/sources")
    parser.add_argument("--data-universe", default="artifacts/sources/d2e_full_data_universe_manifest.json")
    parser.add_argument("--output-dir", default="outputs/data/d2e_full_corpus")
    parser.add_argument("--idm-output-dir", default="outputs/idm_streaming_d2e_full_compact")
    parser.add_argument("--pid-file", default="outputs/cluster/g003_full_compact_parallel.pid")
    parser.add_argument("--watcher-pid-file", default="outputs/cluster/g003_postrun_watcher.pid")
    parser.add_argument("--gpu-monitor-pid-file", default="outputs/cluster/g003_attached_gpu_monitor.pid")
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--stale-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--min-active-extractors",
        type=int,
        default=None,
        help="Override expected live extractor count; defaults to all incomplete shards during extraction.",
    )
    parser.add_argument("--output", default="artifacts/idm/g003_live_health_report.json")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit non-zero when the generated report status is blocked_live_health.",
    )
    args = parser.parse_args()
    report = write_g003_live_health_report(
        args.output,
        shard_root=args.shard_root,
        log_dir=args.log_dir,
        data_universe=args.data_universe,
        output_dir=args.output_dir,
        idm_output_dir=args.idm_output_dir,
        pid_file=args.pid_file,
        watcher_pid_file=args.watcher_pid_file,
        gpu_monitor_pid_file=args.gpu_monitor_pid_file,
        num_shards=args.num_shards,
        stale_seconds=args.stale_seconds,
        min_active_extractors=args.min_active_extractors,
    )
    print(
        "g003 live health: "
        f"status={report['status']} stage={report['stage']} "
        f"decoded={report['progress']['decoded_recording_variants']}/{report['progress']['expected_recording_variants']} "
        f"active_extractors={len(report['active_extractor_shards'])}/{report['expected_active_extractors']} "
        f"warnings={len(report['warnings'])} errors={len(report['errors'])} output={args.output}"
    )
    if args.fail_on_blocked and report["status"] == "blocked_live_health":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
