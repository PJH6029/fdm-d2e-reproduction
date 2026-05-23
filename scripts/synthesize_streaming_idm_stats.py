#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import read_json, stable_hash_json, write_json
from fdm_d2e.training.neural_idm import record_features
from fdm_d2e.training.streaming_idm import _record_paths_from_config


def _action_history_dim(history_vocab: list[str], history_len: int) -> int:
    if history_len <= 0:
        return 0
    return (2 * history_len) + (len(history_vocab) * history_len) + 3


def _first_jsonl_row(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError(f"JSONL row must be an object in {path}")
                    return payload
    raise ValueError("no non-empty train records found")


def synthesize_streaming_idm_stats(config: dict[str, Any], *, source_stats_path: Path) -> dict[str, Any]:
    source_stats = read_json(source_stats_path)
    if not isinstance(source_stats, dict):
        raise ValueError(f"source stats must be a JSON object: {source_stats_path}")
    record_paths = _record_paths_from_config(
        config,
        primary_key="train_records",
        paths_key="train_record_paths",
        glob_key="train_records_glob",
    )
    feature_mode = str(config.get("feature_mode", source_stats.get("feature_mode", "summary")))
    base_input_dim = len(record_features(_first_jsonl_row(record_paths), feature_mode=feature_mode))
    history_len = int(config.get("action_history_len", 0) or 0)
    if history_len < 0:
        raise ValueError("action_history_len must be non-negative")
    history_vocab = [str(token) for token in source_stats.get("category_vocab", [])]
    history_dim = _action_history_dim(history_vocab, history_len)
    input_dim = base_input_dim + history_dim
    identity = {
        "strategy": "unit_normalizer_from_existing_corpus_stats",
        "source_stats_path": str(source_stats_path),
        "source_dataset_fingerprint": source_stats.get("dataset_fingerprint"),
        "source_num_examples": source_stats.get("num_examples"),
        "source_category_vocab": source_stats.get("category_vocab", []),
        "train_records": [str(path) for path in record_paths],
        "feature_mode": feature_mode,
        "base_input_dim": base_input_dim,
        "action_history_len": history_len,
        "action_history_dim": history_dim,
        "input_dim": input_dim,
    }
    stats = {
        "schema": "streaming_idm_stats.v1",
        "train_records": [str(path) for path in record_paths],
        "num_examples": int(source_stats["num_examples"]),
        "feature_mode": feature_mode,
        "input_dim": input_dim,
        "mean": [0.0 for _ in range(input_dim)],
        "std": [1.0 for _ in range(input_dim)],
        "category_vocab": [str(token) for token in source_stats.get("category_vocab", [])],
        "category_counts": dict(source_stats.get("category_counts", {})),
        "global_majority_tokens": list(source_stats.get("global_majority_tokens", ["NOOP"])),
        "last_tokens_by_recording": dict(source_stats.get("last_tokens_by_recording", {})),
        "last_tokens_by_game": dict(source_stats.get("last_tokens_by_game", {})),
        "source_ids": list(source_stats.get("source_ids", [])),
        "resolution_tiers": list(source_stats.get("resolution_tiers", [])),
        "split_names": list(source_stats.get("split_names", [])),
        "eval_split_tags": list(source_stats.get("eval_split_tags", [])),
        "dataset_fingerprint": stable_hash_json(identity),
        "action_history_len": history_len,
        "action_history_vocab": history_vocab if history_len > 0 else [],
        "action_history_dim": history_dim,
        "action_history_feedback": "teacher_forced_train" if history_len > 0 else "none",
        "stats_synthesis": {
            **identity,
            "status": "synthetic_unit_normalizer",
            "created_at_unix": time.time(),
            "claim_boundary": (
                "This stats file reuses full-corpus count/vocabulary provenance from an existing exact stats scan "
                "and intentionally uses zero-mean/unit-std features to avoid an additional full JSONL pass. "
                "Tensor-cache materialization and model training still consume all train rows."
            ),
        },
    }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize streaming IDM stats for a new feature mode without a full stats pass.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-stats", required=True)
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output = Path(args.output) if args.output else Path(config.get("output_dir", "outputs/idm_streaming_full")) / "streaming_stats.json"
    if output.exists() and not args.force:
        raise FileExistsError(f"stats output already exists: {output}; use --force to overwrite")
    stats = synthesize_streaming_idm_stats(config, source_stats_path=Path(args.source_stats))
    write_json(output, stats)
    print(
        "synthesized streaming IDM stats: "
        f"path={output} examples={stats['num_examples']} input_dim={stats['input_dim']} "
        f"fingerprint={stats['dataset_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
