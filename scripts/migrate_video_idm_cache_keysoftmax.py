#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import ensure_dir, read_json, write_json
from fdm_d2e.training.streaming_idm import _record_paths_from_config, iter_jsonl
from fdm_d2e.training.torch_idm import require_torch
from fdm_d2e.training.video_idm import (
    _cache_source_identity,
    _is_keyboard_token,
    _is_mouse_button_token,
    _keyboard_label,
    _video_cache_manifest_path,
    load_video_idm_cache_manifests,
    scan_video_idm_stats,
)


def _record_batches(path: str | Path, sizes: Iterable[int]) -> Iterable[list[dict[str, Any]]]:
    rows = iter_jsonl(path)
    for size in sizes:
        batch = []
        for _ in range(int(size)):
            batch.append(next(rows))
        yield batch


def _category_matrix_from_rows(torch, rows: list[dict[str, Any]], vocab: list[str]):
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    cat_y = torch.zeros((len(rows), len(vocab)), dtype=torch.float32)
    for row_idx, row in enumerate(rows):
        for token in set(str(token) for token in row.get("ground_truth_tokens", [])):
            idx = vocab_index.get(token)
            if idx is not None:
                cat_y[row_idx, idx] = 1.0
    return cat_y


def _keyboard_targets(torch, rows: list[dict[str, Any]], classes: list[tuple[str, ...]]):
    class_index = {tokens: idx for idx, tokens in enumerate(classes)}
    return torch.tensor([class_index.get(_keyboard_label(row), 0) for row in rows], dtype=torch.long)


def _copy_chunk_payload(
    torch,
    *,
    source_chunk: dict[str, Any],
    target_chunk_path: Path,
    rows: list[dict[str, Any]],
    target_category_vocab: list[str],
    keyboard_classes: list[tuple[str, ...]],
) -> dict[str, Any]:
    rows_count = int(source_chunk["rows"])
    if rows_count != len(rows):
        raise ValueError(f"source chunk row mismatch for {source_chunk['path']}: manifest={rows_count} rows={len(rows)}")
    migrated = {
        "schema": "video_idm_cache_chunk.v1",
        "rows": rows_count,
        "payload_source_path": str(source_chunk["path"]),
        "cat_y": _category_matrix_from_rows(torch, rows, target_category_vocab),
        "keyboard_y": _keyboard_targets(torch, rows, keyboard_classes),
    }
    target_chunk_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_chunk_path.with_suffix(target_chunk_path.suffix + ".tmp")
    torch.save(migrated, tmp)
    tmp.replace(target_chunk_path)
    return {"path": str(target_chunk_path), "rows": int(migrated["rows"]), "bytes": int(target_chunk_path.stat().st_size)}


def _migrate_manifest(
    torch,
    *,
    source_manifest: dict[str, Any],
    target_manifest_path: Path,
    target_identity: dict[str, Any],
    record_path: Path,
    target_category_vocab: list[str],
    keyboard_classes: list[tuple[str, ...]],
    force: bool,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if target_manifest_path.exists() and not force:
        existing = read_json(target_manifest_path)
        chunks = existing.get("chunks", [])
        if existing.get("identity") == target_identity and chunks and all(Path(row["path"]).exists() for row in chunks):
            return existing
    chunk_dir = ensure_dir(target_manifest_path.with_suffix(""))
    for old in chunk_dir.glob("chunk_*.pt"):
        old.unlink()
    chunks = []
    source_chunks = list(source_manifest.get("chunks", []))
    row_batches = _record_batches(record_path, [int(chunk["rows"]) for chunk in source_chunks])
    started = time.time()
    rows_done = 0
    bytes_done = 0
    for idx, (source_chunk, rows) in enumerate(zip(source_chunks, row_batches)):
        chunk = _copy_chunk_payload(
            torch,
            source_chunk=source_chunk,
            target_chunk_path=chunk_dir / f"chunk_{idx:06d}.pt",
            rows=rows,
            target_category_vocab=target_category_vocab,
            keyboard_classes=keyboard_classes,
        )
        chunks.append(chunk)
        rows_done += int(chunk["rows"])
        bytes_done += int(chunk.get("bytes", 0))
        if progress is not None:
            progress(
                {
                    "event": "chunk_migrated",
                    "manifest_path": str(target_manifest_path),
                    "record_path": str(record_path),
                    "chunks_done": idx + 1,
                    "chunks_total": len(source_chunks),
                    "rows_done": rows_done,
                    "bytes_done": bytes_done,
                    "wall_clock_seconds": time.time() - started,
                }
            )
    manifest = {
        "schema": "video_idm_cache_manifest.v1",
        "identity": target_identity,
        "split_name": str(source_manifest.get("split_name", "")),
        "source_path": str(record_path),
        "manifest_path": str(target_manifest_path),
        "chunk_size": int(source_manifest.get("chunk_size", 0)),
        "rows": int(sum(int(chunk["rows"]) for chunk in chunks)),
        "bytes": int(sum(int(chunk.get("bytes", 0)) for chunk in chunks)),
        "chunks": chunks,
        "provider_summary": {
            "migration_source_manifest": str(source_manifest.get("manifest_path", "")),
            "wall_clock_seconds": time.time() - started,
        },
    }
    tmp = target_manifest_path.with_suffix(target_manifest_path.suffix + ".tmp")
    write_json(tmp, manifest)
    tmp.replace(target_manifest_path)
    return manifest


def migrate_cache(
    source_config: dict[str, Any],
    target_config: dict[str, Any],
    *,
    force: bool = False,
    progress_output: str | Path | None = None,
    splits: Sequence[str] = ("train", "target"),
) -> dict[str, Any]:
    torch = require_torch()
    started = time.time()
    progress_path = Path(progress_output) if progress_output else None
    requested_splits = {str(split).strip() for split in splits if str(split).strip()}
    if not requested_splits or requested_splits - {"train", "target"}:
        raise ValueError("splits must contain train and/or target")

    def _progress(payload: dict[str, Any]) -> None:
        if progress_path is None:
            return
        snapshot = {
            "schema": "video_idm_keysoftmax_cache_migration_progress.v1",
            "status": "running",
            "started_at_wall_time": started,
            "wall_clock_seconds": time.time() - started,
            **payload,
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(progress_path, snapshot)

    target_train_paths = _record_paths_from_config(
        target_config,
        primary_key="train_records",
        paths_key="train_record_paths",
        glob_key="train_records_glob",
    )
    target_target_paths = _record_paths_from_config(
        target_config,
        primary_key="target_records",
        paths_key="target_record_paths",
        glob_key="target_records_glob",
    )
    source_stats = read_json(source_config["stats_path"])
    target_stats_path = Path(target_config.get("stats_path", Path(target_config["output_dir"]) / "video_idm_stats.json"))
    if target_stats_path.exists() and not force:
        target_stats = read_json(target_stats_path)
    else:
        target_stats = scan_video_idm_stats(target_train_paths, config=target_config)
        write_json(target_stats_path, target_stats)
    if str(target_config.get("keyboard_head_mode")) != "softmax":
        raise ValueError("target config must set keyboard_head_mode=softmax")
    if any(_is_keyboard_token(token) or _is_mouse_button_token(token) for token in target_stats.get("category_vocab", [])):
        raise ValueError("target category_vocab must exclude keyboard/button tokens for exact-set heads")
    source_train_manifests = (
        load_video_idm_cache_manifests(
            target_train_paths,
            stats=source_stats,
            config=source_config,
            split_name="train",
        )
        if "train" in requested_splits
        else []
    )
    source_target_manifests = (
        load_video_idm_cache_manifests(
            target_target_paths,
            stats=source_stats,
            config=source_config,
            split_name="target",
        )
        if "target" in requested_splits
        else []
    )
    keyboard_classes = [tuple(str(token) for token in row) for row in target_stats.get("keyboard_classes", [])]
    target_category_vocab = [str(token) for token in target_stats.get("category_vocab", [])]
    migrated: dict[str, list[dict[str, Any]]] = {"train": [], "target": []}
    split_jobs = []
    if "train" in requested_splits:
        split_jobs.append(("train", target_train_paths, source_train_manifests))
    if "target" in requested_splits:
        split_jobs.append(("target", target_target_paths, source_target_manifests))
    for split_name, paths, source_manifests in split_jobs:
        for record_path, source_manifest in zip(paths, source_manifests):
            target_manifest_path = _video_cache_manifest_path(
                target_config["video_cache_dir"],
                record_path,
                stats=target_stats,
                config=target_config,
                split_name=split_name,
            )
            identity = _cache_source_identity(record_path, stats=target_stats, config=target_config, split_name=split_name)
            migrated[split_name].append(
                _migrate_manifest(
                    torch,
                    source_manifest=source_manifest,
                    target_manifest_path=target_manifest_path,
                    target_identity=identity,
                    record_path=record_path,
                    target_category_vocab=target_category_vocab,
                    keyboard_classes=keyboard_classes,
                    force=force,
                    progress=lambda payload, split_name=split_name: _progress({"split_name": split_name, **payload}),
                )
            )
    summary = {
        "schema": "video_idm_keysoftmax_cache_migration_summary.v1",
        "status": "pass",
        "splits": sorted(requested_splits),
        "source_stats_path": str(source_config["stats_path"]),
        "target_stats_path": str(target_stats_path),
        "target_keyboard_classes": len(keyboard_classes),
        "target_category_vocab": len(target_category_vocab),
        "train_cache": {
            "manifest_paths": [str(row["manifest_path"]) for row in migrated["train"]],
            "rows": sum(int(row.get("rows", 0)) for row in migrated["train"]),
            "bytes": sum(int(row.get("bytes", 0)) for row in migrated["train"]),
        },
        "target_cache": {
            "manifest_paths": [str(row["manifest_path"]) for row in migrated["target"]],
            "rows": sum(int(row.get("rows", 0)) for row in migrated["target"]),
            "bytes": sum(int(row.get("bytes", 0)) for row in migrated["target"]),
        },
    }
    if progress_path is not None:
        write_json(
            progress_path,
            {
                "schema": "video_idm_keysoftmax_cache_migration_progress.v1",
                "status": "pass",
                "started_at_wall_time": started,
                "wall_clock_seconds": time.time() - started,
                "summary_path": "",
                "train_rows": summary["train_cache"]["rows"],
                "target_rows": summary["target_cache"]["rows"],
            },
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build key-softmax video IDM cache by reusing existing raw112 frame chunks.")
    parser.add_argument("--source-config", default="configs/model/idm_video_pair_d2e_full_raw112_paper_target.yaml")
    parser.add_argument("--target-config", default="configs/model/idm_video_pair_d2e_full_raw112_keysoftmax_paper_target.yaml")
    parser.add_argument("--output", default="artifacts/idm/g005_video_pair_raw112_keysoftmax_cache_migration_summary.json")
    parser.add_argument("--progress-output", default="artifacts/idm/g005_video_pair_raw112_keysoftmax_cache_migration_progress.json")
    parser.add_argument("--splits", default="train,target", help="comma-separated cache splits to migrate: train,target")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = migrate_cache(
        load_config(args.source_config),
        load_config(args.target_config),
        force=args.force,
        progress_output=args.progress_output,
        splits=[item for item in args.splits.split(",") if item],
    )
    if args.progress_output:
        progress_path = Path(args.progress_output)
        if progress_path.exists():
            progress = read_json(progress_path)
            progress["summary_path"] = str(args.output)
            write_json(progress_path, progress)
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
