#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.data.d2e_real import build_window_records, decode_mcap_events, download_recording_ref, extract_video_frame_features
from fdm_d2e.data.fdm1_g003_splits import annotate_window_records_with_fdm1_splits, load_g002_split_index
from fdm_d2e.data.full_corpus import (
    annotate_window_records,
    d2e_ref_from_universe_row,
    included_universe_rows,
    split_output_paths,
    universe_row_id,
)
from fdm_d2e.io_utils import read_json, read_jsonl, sha256_file, stable_hash_json, write_json, write_jsonl


def _selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = read_json(args.data_universe)
    rows = included_universe_rows(
        manifest,
        source_ids=args.source_id,
        resolution_tiers=args.resolution_tier,
    )
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >=1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    if args.max_recordings is not None:
        rows = rows[: int(args.max_recordings)]
    return rows


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _reset_outputs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")


def _materialize_splits(paths: dict[str, Path], records: list[dict[str, Any]]) -> dict[str, int]:
    buckets = {
        "all": records,
        "train_core": [row for row in records if row.get("split") == "train_core"],
        "target_temporal": [row for row in records if "temporal" in row.get("eval_split_tags", [])],
        "target_heldout_recording": [row for row in records if "heldout_recording" in row.get("eval_split_tags", [])],
        "target_heldout_game": [row for row in records if "heldout_game" in row.get("eval_split_tags", [])],
        "target_all_eval": [row for row in records if row.get("eval_split_tags")],
    }
    for name, rows in buckets.items():
        _append_jsonl(paths[name], rows)
    return {name: len(rows) for name, rows in buckets.items()}


def _recording_dir(output_dir: Path, row: dict[str, Any]) -> Path:
    return output_dir / "by_recording" / str(row["source_id"]) / str(row["game"]) / str(row["recording_id"])


def _emit_stage(row: dict[str, Any], stage: str, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "stage": stage,
                "timestamp_unix": time.time(),
                "universe_row_id": universe_row_id(row),
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _load_or_extract_recording(args: argparse.Namespace, split_contract: Any, row: dict[str, Any]) -> dict[str, Any]:
    rec_dir = _recording_dir(Path(args.output_dir), row)
    rec_dir.mkdir(parents=True, exist_ok=True)
    records_path = rec_dir / "all_records.jsonl"
    summary_path = rec_dir / "decode_summary.json"
    if records_path.exists() and summary_path.exists() and not args.force:
        _emit_stage(row, "recording_resume_cached")
        records = read_jsonl(records_path)
        summary = read_json(summary_path)
        return {"records": records, "summary": {**summary, "resumed": True}}

    ref = d2e_ref_from_universe_row(row)
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    # D2E-480p and D2E-Original share game/recording filenames but differ in
    # video payloads.  Namespace the cache by source_id to prevent one tier from
    # silently reusing the other's media file.
    cache_dir = Path(args.cache_dir) / str(row["source_id"])
    stage_start = time.time()
    _emit_stage(row, "download_mcap_start")
    downloaded = download_recording_ref(ref, cache_dir, token=token, kinds=("mcap",))
    _emit_stage(row, "download_mcap_done", elapsed_seconds=round(time.time() - stage_start, 3))
    video_source = ref.video_url
    if args.video_mode == "download":
        stage_start = time.time()
        _emit_stage(row, "download_video_start")
        video_source = download_recording_ref(ref, cache_dir, token=token, kinds=("video",))["video"]
        _emit_stage(row, "download_video_done", elapsed_seconds=round(time.time() - stage_start, 3))

    event_limit = int(args.event_limit) if args.event_limit is not None else None
    stage_start = time.time()
    _emit_stage(row, "decode_mcap_start")
    decoded_events = decode_mcap_events(downloaded["mcap"], limit=event_limit)
    _emit_stage(row, "decode_mcap_done", elapsed_seconds=round(time.time() - stage_start, 3), decoded_events=len(decoded_events))
    max_bins = int(args.max_bins_per_recording) if args.max_bins_per_recording is not None else None
    max_frames = max_bins
    try:
        stage_start = time.time()
        _emit_stage(row, "extract_frames_start", video_mode=args.video_mode)
        frame_features = extract_video_frame_features(
            video_source,
            rec_dir,
            max_frames=max_frames,
            fps=int(args.frame_fps),
            image_size=int(args.image_size),
            compact_features=True,
            keep_frames=bool(args.keep_frames),
        )
        _emit_stage(row, "extract_frames_done", elapsed_seconds=round(time.time() - stage_start, 3), frame_features=len(frame_features))
    except subprocess.CalledProcessError:
        if args.video_mode == "download":
            raise
        video_source = download_recording_ref(ref, cache_dir, token=token, kinds=("video",))["video"]
        stage_start = time.time()
        _emit_stage(row, "extract_frames_start", video_mode="download_fallback")
        frame_features = extract_video_frame_features(
            video_source,
            rec_dir,
            max_frames=max_frames,
            fps=int(args.frame_fps),
            image_size=int(args.image_size),
            compact_features=True,
            keep_frames=bool(args.keep_frames),
        )
        _emit_stage(row, "extract_frames_done", elapsed_seconds=round(time.time() - stage_start, 3), frame_features=len(frame_features))
    stage_start = time.time()
    _emit_stage(row, "build_records_start", frame_features=len(frame_features), decoded_events=len(decoded_events))
    raw_records = build_window_records(
        ref,
        decoded_events,
        split="full_corpus",
        bin_ms=int(args.bin_ms),
        max_bins=max_bins,
        frame_features=frame_features,
    )
    _emit_stage(row, "build_records_done", elapsed_seconds=round(time.time() - stage_start, 3), records=len(raw_records))
    if args.split_mode == "fdm1-g002":
        records = annotate_window_records_with_fdm1_splits(raw_records, universe_row=row, split_index=split_contract)
    else:
        records = annotate_window_records(raw_records, universe_row=row, split_contract=split_contract)
    _emit_stage(row, "annotate_records_done", records=len(records))
    write_jsonl(records_path, records)
    summary = {
        "schema": "d2e_full_recording_decode_summary.v1",
        "universe_row_id": universe_row_id(row),
        "source_id": row["source_id"],
        "resolution_tier": row.get("resolution_tier"),
        "game": row["game"],
        "recording_id": row["recording_id"],
        "cross_resolution_key": row["cross_resolution_key"],
        "repo_id": row["repo_id"],
        "revision": row.get("resolved_revision") or row.get("requested_revision"),
        "mcap_path": downloaded["mcap"],
        "mcap_sha256": sha256_file(downloaded["mcap"]),
        "video_mode": args.video_mode,
        "video_source": video_source,
        "num_decoded_events": len(decoded_events),
        "num_frame_features": len(frame_features),
        "num_window_records": len(records),
        "split_counts": {
            "train_core": sum(1 for item in records if item.get("split") == "train_core"),
            "target_temporal": sum(1 for item in records if "temporal" in item.get("eval_split_tags", [])),
            "target_heldout_recording": sum(1 for item in records if "heldout_recording" in item.get("eval_split_tags", [])),
            "target_heldout_game": sum(1 for item in records if "heldout_game" in item.get("eval_split_tags", [])),
        },
        "record_fingerprint": stable_hash_json(
            {
                "universe_row_id": universe_row_id(row),
                "mcap_sha256": sha256_file(downloaded["mcap"]),
                "num_window_records": len(records),
                "token_rows": [item.get("ground_truth_tokens", []) for item in records[: min(len(records), 1024)]],
            }
        ),
    }
    write_json(summary_path, summary)
    return {"records": records, "summary": summary}


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", help="JSON-compatible YAML defaults for this extraction run.")
    known, _ = pre.parse_known_args()
    defaults: dict[str, Any] = {}
    if known.config:
        cfg = load_config(known.config)
        defaults = {
            "data_universe": cfg.get("data_universe"),
            "split_contract": cfg.get("split_contract"),
            "split_mode": cfg.get("split_mode"),
            "recording_level_split_manifest": cfg.get("recording_level_split_manifest"),
            "heldout_game_split_manifest": cfg.get("heldout_game_split_manifest"),
            "pseudo_label_split_manifest": cfg.get("pseudo_label_split_manifest"),
            "scale_split_manifest": cfg.get("scale_split_manifest"),
            "output_dir": cfg.get("output_dir"),
            "summary_out": cfg.get("summary_out"),
            "cache_dir": cfg.get("cache_dir"),
            "source_id": cfg.get("source_ids"),
            "resolution_tier": cfg.get("resolution_tiers"),
            "bin_ms": cfg.get("bin_ms"),
            "frame_fps": cfg.get("frame_fps"),
            "image_size": cfg.get("image_size"),
            "video_mode": cfg.get("video_mode"),
            "keep_frames": bool(cfg.get("keep_frames", False)),
        }
        defaults = {key: value for key, value in defaults.items() if value is not None}

    parser = argparse.ArgumentParser(description="Decode/materialize full D2E compact-feature train/eval JSONL splits.", parents=[pre])
    parser.set_defaults(**defaults)
    parser.add_argument("--data-universe", default=defaults.get("data_universe", "artifacts/sources/d2e_full_data_universe_manifest.json"))
    parser.add_argument("--split-contract", default=defaults.get("split_contract", "artifacts/sources/d2e_full_split_contract.json"))
    parser.add_argument("--split-mode", choices=["legacy", "fdm1-g002"], default=defaults.get("split_mode", "legacy"))
    parser.add_argument("--recording-level-split-manifest", default=defaults.get("recording_level_split_manifest", "artifacts/sources/fdm1_d2e_recording_level_split_manifest.json"))
    parser.add_argument("--heldout-game-split-manifest", default=defaults.get("heldout_game_split_manifest", "artifacts/sources/fdm1_d2e_heldout_game_split_manifest.json"))
    parser.add_argument("--pseudo-label-split-manifest", default=defaults.get("pseudo_label_split_manifest", "artifacts/sources/fdm1_d2e_pseudo_label_split_manifest.json"))
    parser.add_argument("--scale-split-manifest", default=defaults.get("scale_split_manifest", "artifacts/sources/fdm1_d2e_scale_split_manifest.json"))
    parser.add_argument("--output-dir", default=defaults.get("output_dir", "outputs/data/d2e_full_corpus"))
    parser.add_argument("--summary-out", default=defaults.get("summary_out", "artifacts/sources/d2e_full_corpus_decode_summary.json"))
    parser.add_argument("--cache-dir", default=defaults.get("cache_dir", "/root/work/data/d2e/cache"))
    parser.add_argument("--source-id", action="append", help="Restrict to one or more source_id values, e.g. d2e_480p. Default: all included D2E sources.")
    parser.add_argument("--resolution-tier", action="append", help="Restrict to one or more resolution tiers. Default: all included tiers.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--max-bins-per-recording", type=int)
    parser.add_argument("--event-limit", type=int)
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--frame-fps", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--video-mode", choices=["remote", "download"], default="download")
    parser.add_argument("--keep-frames", action="store_true", help="Keep transient PPM frames. Default stores compact JSON features only.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hf-token")
    args = parser.parse_args()

    if args.split_mode == "fdm1-g002":
        split_contract = load_g002_split_index(
            recording_level_split_path=args.recording_level_split_manifest,
            heldout_game_split_path=args.heldout_game_split_manifest,
            pseudo_label_split_path=args.pseudo_label_split_manifest,
            scale_split_path=args.scale_split_manifest,
        )
    else:
        split_contract = read_json(args.split_contract)
    rows = _selected_rows(args)
    output_dir = Path(args.output_dir)
    paths = split_output_paths(output_dir)
    _reset_outputs(paths)
    summaries = []
    aggregate_counts = {name: 0 for name in paths}
    failures = []
    for idx, row in enumerate(rows, 1):
        try:
            result = _load_or_extract_recording(args, split_contract, row)
        except Exception as exc:
            failures.append({"universe_row_id": universe_row_id(row), "error": repr(exc)})
            if not args.force:
                raise
            continue
        counts = _materialize_splits(paths, result["records"])
        for name, count in counts.items():
            aggregate_counts[name] += count
        summaries.append(result["summary"])
        print(
            json.dumps(
                {
                    "decoded": idx,
                    "total_selected": len(rows),
                    "universe_row_id": universe_row_id(row),
                    "records": len(result["records"]),
                    "train_core": counts["train_core"],
                    "eval": counts["target_all_eval"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "schema": "d2e_full_corpus_decode_summary.v1",
        "output_dir": str(output_dir),
        "data_universe": args.data_universe,
        "split_contract": args.split_contract if args.split_mode == "legacy" else None,
        "split_mode": args.split_mode,
        "fdm1_split_manifests": {
            "recording_level_split": args.recording_level_split_manifest,
            "heldout_game_split": args.heldout_game_split_manifest,
            "pseudo_label_split": args.pseudo_label_split_manifest,
            "scale_split": args.scale_split_manifest,
        } if args.split_mode == "fdm1-g002" else None,
        "selected_recording_variants": len(rows),
        "shard": {"index": args.shard_index, "num_shards": args.num_shards},
        "source_ids": sorted({str(row.get("source_id")) for row in rows}),
        "resolution_tiers": sorted({str(row.get("resolution_tier")) for row in rows}),
        "counts": aggregate_counts,
        "paths": {name: str(path) for name, path in paths.items()},
        "recordings": summaries,
        "failures": failures,
        "dataset_fingerprint": stable_hash_json(
            {
                "data_universe": read_json(args.data_universe).get("dataset_fingerprint"),
                "split_mode": args.split_mode,
                "split_contract": split_contract.get("dataset_fingerprint") if isinstance(split_contract, dict) else getattr(split_contract, "fingerprints", {}),
                "rows": [universe_row_id(row) for row in rows],
                "counts": aggregate_counts,
                "shard": {"index": args.shard_index, "num_shards": args.num_shards},
            }
        ),
    }
    write_json(args.summary_out, summary)
    print(
        "decoded D2E full-corpus shard: "
        f"variants={len(rows)} records={aggregate_counts['all']} train_core={aggregate_counts['train_core']} "
        f"eval={aggregate_counts['target_all_eval']} failures={len(failures)}"
    )


if __name__ == "__main__":
    main()
