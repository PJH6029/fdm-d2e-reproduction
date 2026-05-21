#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.cluster.g003_monitor import write_g003_progress_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize G003 full-corpus shard extraction/training progress without mutating the run.")
    parser.add_argument("--shard-root", default="outputs/data/d2e_full_corpus_shards")
    parser.add_argument("--log-dir", default="artifacts/sources")
    parser.add_argument("--data-universe", default="artifacts/sources/d2e_full_data_universe_manifest.json")
    parser.add_argument("--output-dir", default="outputs/data/d2e_full_corpus")
    parser.add_argument("--idm-output-dir", default="outputs/idm_streaming_d2e_full_compact")
    parser.add_argument("--pid-file", default="outputs/cluster/g003_full_compact_parallel.pid")
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--stale-seconds", type=float, default=3600.0)
    parser.add_argument("--output", default="artifacts/idm/g003_full_compact_parallel_progress.json")
    args = parser.parse_args()
    report = write_g003_progress_report(
        args.output,
        shard_root=args.shard_root,
        log_dir=args.log_dir,
        data_universe=args.data_universe,
        output_dir=args.output_dir,
        idm_output_dir=args.idm_output_dir,
        pid_file=args.pid_file,
        num_shards=args.num_shards,
        stale_seconds=args.stale_seconds,
    )
    print(
        "g003 progress: "
        f"status={report['status']} decoded={report['decoded_recording_variants']}/{report['expected_recording_variants']} "
        f"complete_shards={report['complete_shards']}/{report['num_shards']} "
        f"stale={report['stale_shards']} no_progress={report['no_progress_shards']} "
        f"long_running={report['long_running_shards']} "
        f"recommendation={report['recommendation']['code']} "
        f"pid_running={report['pid_running']} output={args.output}"
    )
    return 0 if report["status"] != "not_started_or_unknown" else 2


if __name__ == "__main__":
    raise SystemExit(main())
