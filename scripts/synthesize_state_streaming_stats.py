#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json

try:  # pragma: no cover - exercised on cluster when present.
    import orjson  # type: ignore
except Exception:  # pragma: no cover - fallback is covered.
    orjson = None


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield _loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc


def _expand(pattern: str | list[str]) -> list[Path]:
    patterns = pattern if isinstance(pattern, list) else [pattern]
    out: list[Path] = []
    for item in patterns:
        matches = sorted(glob.glob(str(item)))
        out.extend(Path(path) for path in matches) if matches else out.append(Path(item))
    return out


def _is_category_token(token: str) -> bool:
    return token.startswith("KEY_") or (
        token.startswith("MOUSE_") and not token.startswith("MOUSE_DX_") and not token.startswith("MOUSE_DY_")
    )


def _label_key(tokens: Iterable[str]) -> str:
    return json.dumps(sorted(set(str(token) for token in tokens)), ensure_ascii=False, separators=(",", ":"))


def _tokens(row: dict[str, Any]) -> list[str]:
    value = row.get("ground_truth_tokens")
    return [str(token) for token in value] if isinstance(value, list) else ["NOOP"]


def _latest_update(target: dict[str, tuple[int, list[str]]], key: str, timestamp_ns: Any, tokens: list[str]) -> None:
    try:
        timestamp = int(timestamp_ns)
    except (TypeError, ValueError):
        timestamp = -1
    previous = target.get(key)
    if previous is None or timestamp >= previous[0]:
        target[key] = (timestamp, tokens)


def _scan_partition(path: str) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    keyboard_class_counts: Counter[str] = Counter()
    button_class_counts: Counter[str] = Counter()
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    last_tokens_by_recording: dict[str, tuple[int, list[str]]] = {}
    last_tokens_by_game: dict[str, tuple[int, list[str]]] = {}
    source_ids: set[str] = set()
    resolution_tiers: set[str] = set()
    split_names: set[str] = set()
    eval_split_tags: set[str] = set()
    fingerprint = hashlib.sha256()
    rows = 0
    for row in _iter_jsonl(Path(path)):
        tokens = _tokens(row)
        rows += 1
        category_counts.update(token for token in tokens if _is_category_token(token))
        keyboard_class_counts[_label_key(token for token in tokens if token.startswith("KEY_"))] += 1
        button_class_counts[
            _label_key(token for token in tokens if token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")))
        ] += 1
        sequence_counts[tuple(tokens)] += 1
        timestamp_ns = row.get("timestamp_ns")
        _latest_update(last_tokens_by_recording, str(row.get("recording_id", "")), timestamp_ns, tokens)
        _latest_update(last_tokens_by_game, str(row.get("game", "unknown")), timestamp_ns, tokens)
        if row.get("source_id") is not None:
            source_ids.add(str(row["source_id"]))
        if row.get("resolution_tier") is not None:
            resolution_tiers.add(str(row["resolution_tier"]))
        if row.get("split") is not None:
            split_names.add(str(row["split"]))
        for tag in row.get("eval_split_tags", []) or []:
            eval_split_tags.add(str(tag))
        fingerprint.update(json.dumps({"sequence_id": row.get("sequence_id"), "tokens": tokens}, sort_keys=True).encode("utf-8"))
        fingerprint.update(b"\n")
    return {
        "path": path,
        "rows": rows,
        "category_counts": dict(category_counts),
        "keyboard_class_counts": dict(keyboard_class_counts),
        "button_class_counts": dict(button_class_counts),
        "sequence_counts": {"\u241f".join(key): value for key, value in sequence_counts.items()},
        "last_tokens_by_recording": last_tokens_by_recording,
        "last_tokens_by_game": last_tokens_by_game,
        "source_ids": sorted(source_ids),
        "resolution_tiers": sorted(resolution_tiers),
        "split_names": sorted(split_names),
        "eval_split_tags": sorted(eval_split_tags),
        "fingerprint": fingerprint.hexdigest(),
    }


def _merge_counter(target: Counter[str], values: dict[str, int]) -> None:
    target.update({str(key): int(value) for key, value in values.items()})


def _feature_prefix_from_seed(seed: dict[str, Any]) -> tuple[list[float], list[float], int]:
    mean = [float(value) for value in seed.get("mean", [])]
    std = [float(value) for value in seed.get("std", [])]
    if len(mean) != len(std) or not mean:
        raise ValueError("seed stats must contain non-empty mean/std arrays of equal length")
    action_history_dim = int(seed.get("action_history_dim", 0) or 0)
    base_dim = len(mean) - action_history_dim
    if base_dim <= 0:
        raise ValueError("seed action_history_dim leaves no base feature dimensions")
    return mean[:base_dim], std[:base_dim], base_dim


def synthesize_stats(config_path: Path, *, seed_stats_path: Path, output_path: Path, summary_path: Path, workers: int) -> dict[str, Any]:
    started = time.time()
    config = load_config(config_path)
    seed_stats = json.loads(seed_stats_path.read_text(encoding="utf-8"))
    mean, std, input_dim = _feature_prefix_from_seed(seed_stats)
    train_paths = _expand(config.get("train_records_glob") or config["train_records"])
    category_counts: Counter[str] = Counter()
    keyboard_class_counts: Counter[str] = Counter()
    button_class_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    last_tokens_by_recording: dict[str, tuple[int, list[str]]] = {}
    last_tokens_by_game: dict[str, tuple[int, list[str]]] = {}
    source_ids: set[str] = set()
    resolution_tiers: set[str] = set()
    split_names: set[str] = set()
    eval_split_tags: set[str] = set()
    parts = []
    with ProcessPoolExecutor(max_workers=min(max(1, int(workers)), len(train_paths))) as pool:
        futures = [pool.submit(_scan_partition, str(path)) for path in train_paths]
        for future in as_completed(futures):
            part = future.result()
            parts.append(part)
            _merge_counter(category_counts, part["category_counts"])
            _merge_counter(keyboard_class_counts, part["keyboard_class_counts"])
            _merge_counter(button_class_counts, part["button_class_counts"])
            _merge_counter(sequence_counts, part["sequence_counts"])
            for key, value in part["last_tokens_by_recording"].items():
                _latest_update(last_tokens_by_recording, key, value[0], value[1])
            for key, value in part["last_tokens_by_game"].items():
                _latest_update(last_tokens_by_game, key, value[0], value[1])
            source_ids.update(part["source_ids"])
            resolution_tiers.update(part["resolution_tiers"])
            split_names.update(part["split_names"])
            eval_split_tags.update(part["eval_split_tags"])
    rows = sum(int(part["rows"]) for part in parts)
    if rows <= 0:
        raise ValueError("no state training rows found")
    categorical_min_count = int(config.get("categorical_min_count", 1))
    category_vocab = sorted(token for token, count in category_counts.items() if count >= categorical_min_count)
    majority_key, _majority_count = sequence_counts.most_common(1)[0] if sequence_counts else ("NOOP", 0)
    dataset_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "feature_seed": seed_stats.get("dataset_fingerprint"),
                "feature_seed_path": str(seed_stats_path),
                "feature_prefix_dim": input_dim,
                "state_token_parts": sorted(
                    ({"path": part["path"], "rows": part["rows"], "fingerprint": part["fingerprint"]} for part in parts),
                    key=lambda row: str(row["path"]),
                ),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    stats = {
        "schema": "streaming_idm_stats.v1",
        "train_records": [str(path) for path in train_paths],
        "num_examples": rows,
        "feature_mode": str(config.get("feature_mode", seed_stats.get("feature_mode"))),
        "input_dim": input_dim,
        "mean": mean,
        "std": std,
        "category_vocab": category_vocab,
        "category_counts": dict(sorted(category_counts.items())),
        "keyboard_class_counts": dict(sorted(keyboard_class_counts.items())),
        "button_class_counts": dict(sorted(button_class_counts.items())),
        "global_majority_tokens": majority_key.split("\u241f") if majority_key else ["NOOP"],
        "last_tokens_by_recording": {key: list(tokens) for key, (_ts, tokens) in sorted(last_tokens_by_recording.items())},
        "last_tokens_by_game": {key: list(tokens) for key, (_ts, tokens) in sorted(last_tokens_by_game.items())},
        "source_ids": sorted(source_ids),
        "resolution_tiers": sorted(resolution_tiers),
        "split_names": sorted(split_names),
        "eval_split_tags": sorted(eval_split_tags),
        "dataset_fingerprint": dataset_fingerprint,
        "action_history_len": 0,
        "action_history_vocab": [],
        "action_history_dim": 0,
        "action_history_feedback": "none",
        "synthesized_from_feature_seed": str(seed_stats_path),
    }
    write_json(output_path, stats)
    summary = {
        "schema": "state_streaming_stats_synthesis_summary.v1",
        "status": "pass",
        "config": str(config_path),
        "seed_stats_path": str(seed_stats_path),
        "output_path": str(output_path),
        "rows": rows,
        "input_dim": input_dim,
        "category_vocab_size": len(category_vocab),
        "keyboard_class_count": len(keyboard_class_counts),
        "button_class_count": len(button_class_counts),
        "workers": int(workers),
        "wall_clock_seconds": time.time() - started,
        "claim_boundary": "Fast state-label stats synthesis reuses visual feature moments from a same-feature full-corpus seed and rescans only state-token labels.",
    }
    write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize streaming IDM stats for a state-token corpus from same-feature visual seed stats.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    payload = synthesize_stats(
        Path(args.config),
        seed_stats_path=Path(args.seed_stats),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        workers=max(1, int(args.workers)),
    )
    print(json.dumps({"status": payload["status"], "rows": payload["rows"], "input_dim": payload["input_dim"]}, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
