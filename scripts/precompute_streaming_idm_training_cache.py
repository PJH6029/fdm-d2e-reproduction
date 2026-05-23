#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import read_json, write_json
from fdm_d2e.training.streaming_idm import (
    MOUSE_AXIS_CLASSES,
    _build_training_cache_manifests,
    _record_paths_from_config,
    _training_cache_manifest_byte_count,
    _training_cache_manifest_row_count,
    scan_streaming_idm_stats,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute streaming IDM tensor cache outside DDP.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stats-path")
    parser.add_argument("--stats-seed-path")
    parser.add_argument("--force-rescan-stats", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--output", default="artifacts/idm/streaming_idm_training_cache_precompute_summary.json")
    parser.add_argument("--progress-output")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.workers is not None:
        config["training_cache_num_workers"] = int(args.workers)
    config["force_rebuild_training_cache"] = bool(args.force_rebuild)
    if not config.get("training_cache_dir"):
        raise ValueError("config must define training_cache_dir")

    out_dir = Path(config.get("output_dir", "outputs/idm_streaming_full"))
    stats_path = Path(args.stats_path) if args.stats_path else out_dir / "streaming_stats.json"
    if args.force_rescan_stats and stats_path.exists():
        stats_path.unlink()
    if not stats_path.exists() and args.stats_seed_path:
        seed = Path(args.stats_seed_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_bytes(seed.read_bytes())
    record_paths = _record_paths_from_config(
        config,
        primary_key="train_records",
        paths_key="train_record_paths",
        glob_key="train_records_glob",
    )
    if not stats_path.exists():
        started_stats = time.time()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats = scan_streaming_idm_stats(
            record_paths if len(record_paths) > 1 else record_paths[0],
            feature_mode=str(config.get("feature_mode", "summary_compact_grid8_shift_surface_time")),
            categorical_min_count=int(config.get("categorical_min_count", 1)),
            num_workers=int(config.get("precompute_num_workers", config.get("stats_num_workers", 1))),
        )
        stats["stats_precompute_wall_clock_seconds"] = time.time() - started_stats
        write_json(stats_path, stats)
    else:
        stats = read_json(stats_path)
    category_vocab = [str(token) for token in stats.get("category_vocab", [])]
    mouse_axis_classes = [str(value) for value in config.get("mouse_axis_classes", MOUSE_AXIS_CLASSES)]
    started = time.time()
    progress_out = Path(args.progress_output) if args.progress_output else None
    if progress_out:
        write_json(
            progress_out,
            {
                "schema": "streaming_idm_training_cache_precompute_progress.v1",
                "status": "running",
                "config": str(config_path),
                "stats_path": str(stats_path),
                "cache_dir": str(config["training_cache_dir"]),
                "record_paths": len(record_paths),
                "workers": int(config.get("training_cache_num_workers", 1)),
                "started_at_unix": started,
            },
        )
    manifests = _build_training_cache_manifests(
        record_paths,
        stats=stats,
        config=config,
        category_vocab=category_vocab,
        mouse_axis_classes=mouse_axis_classes,
    )
    payload = {
        "schema": "streaming_idm_training_cache_precompute_summary.v1",
        "status": "pass",
        "config": str(config_path),
        "stats_path": str(stats_path),
        "stats_precomputed": bool(stats.get("stats_precompute_wall_clock_seconds") is not None),
        "cache_dir": str(config["training_cache_dir"]),
        "record_paths": len(record_paths),
        "workers": int(config.get("training_cache_num_workers", 1)),
        "force_rebuild": bool(args.force_rebuild),
        "manifest_count": len(manifests),
        "rows": sum(_training_cache_manifest_row_count(row) for row in manifests),
        "bytes": sum(_training_cache_manifest_byte_count(row) for row in manifests),
        "manifest_paths": [str(row.get("manifest_path")) for row in manifests],
        "wall_clock_seconds": time.time() - started,
        "claim_boundary": "Standalone cache precompute prepares streaming IDM DDP input; it is not model-quality evidence.",
    }
    write_json(args.output, payload)
    if progress_out:
        write_json(progress_out, {**payload, "schema": "streaming_idm_training_cache_precompute_progress.v1"})
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
