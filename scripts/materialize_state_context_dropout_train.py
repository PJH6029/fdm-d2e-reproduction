#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            yield row


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_unit_interval(value: str, *, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _mask_state_context(row: dict[str, Any], *, seed: int, dropout_rate: float) -> tuple[dict[str, Any], bool]:
    key = str(row.get("sequence_id") or row.get("timestamp_ns") or "")
    applied = _stable_unit_interval(key, seed=seed) < float(dropout_rate)
    out = dict(row)
    if applied:
        out["prior_action_tokens"] = ["NOOP"]
        out["prior_key_hold_bins"] = {}
        out["prior_button_hold_bins"] = {}
        out["prior_since_key_transition_bins"] = None
        out["prior_since_button_transition_bins"] = None
        out["previous_event_tokens"] = ["NOOP"]
        out["prior_action_source"] = "deterministic_train_context_dropout"
        out["state_context_schema"] = "d2e_event_state_context_dropout_train.v1"
    out["state_context_dropout"] = {
        "schema": "state_context_dropout_marker.v1",
        "applied": bool(applied),
        "dropout_rate": float(dropout_rate),
        "seed": int(seed),
        "source": "deterministic_sequence_hash",
    }
    return out, applied


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    return sorted(dict.fromkeys(paths))


def materialize_state_context_dropout_train(
    *,
    input_paths: list[str],
    output_root: str | Path,
    dropout_rate: float,
    seed: int,
    summary_out: str | Path,
    progress_out: str | Path | None = None,
    max_rows_per_file: int | None = None,
) -> dict[str, Any]:
    if not 0.0 <= float(dropout_rate) <= 1.0:
        raise ValueError("dropout_rate must be in [0, 1]")
    paths = _expand_inputs(input_paths)
    if not paths:
        raise FileNotFoundError(f"no input paths matched: {input_paths}")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total_rows = 0
    dropped_rows = 0
    outputs: list[dict[str, Any]] = []
    for input_idx, path in enumerate(paths):
        shard_name = path.parent.name if path.parent.name.startswith("shard_") else f"shard_{input_idx:02d}"
        out_path = root / shard_name / "train_core.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        dropped = 0
        fingerprint = hashlib.sha256()
        with out_path.open("w", encoding="utf-8", buffering=1024 * 1024) as out_handle:
            for row in _iter_jsonl(path):
                if max_rows_per_file is not None and rows >= int(max_rows_per_file):
                    break
                transformed, applied = _mask_state_context(row, seed=seed, dropout_rate=dropout_rate)
                line = json.dumps(transformed, ensure_ascii=False, sort_keys=True)
                out_handle.write(line + "\n")
                fingerprint.update(line.encode("utf-8")); fingerprint.update(b"\n")
                rows += 1
                dropped += int(applied)
        total_rows += rows
        dropped_rows += dropped
        item = {
            "input_path": str(path),
            "output_path": str(out_path),
            "rows": rows,
            "dropped_rows": dropped,
            "dropout_fraction": dropped / rows if rows else None,
            "sha256": fingerprint.hexdigest(),
        }
        outputs.append(item)
        if progress_out:
            _write_json(progress_out, {
                "schema": "state_context_dropout_materialization_progress.v1",
                "status": "running",
                "processed_files": len(outputs),
                "total_files": len(paths),
                "rows": total_rows,
                "dropped_rows": dropped_rows,
                "latest": item,
            })
    payload = {
        "schema": "state_context_dropout_train_materialization.v1",
        "status": "pass",
        "input_paths": [str(path) for path in paths],
        "output_root": str(root),
        "dropout_rate": float(dropout_rate),
        "seed": int(seed),
        "file_count": len(outputs),
        "rows": total_rows,
        "dropped_rows": dropped_rows,
        "dropout_fraction": dropped_rows / total_rows if total_rows else None,
        "outputs": outputs,
        "wall_clock_seconds": time.time() - started,
        "claim_boundary": "Training-data augmentation for closed-loop robustness only; no model-quality claim.",
    }
    _write_json(summary_out, payload)
    if progress_out:
        _write_json(progress_out, {**payload, "schema": "state_context_dropout_materialization_progress.v1"})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic train-core shards with prior state context dropout.")
    parser.add_argument("--input", action="append", required=True, help="Input train_core.jsonl path or glob. Repeatable.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dropout-rate", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--progress-out")
    parser.add_argument("--max-rows-per-file", type=int)
    args = parser.parse_args()
    payload = materialize_state_context_dropout_train(
        input_paths=args.input,
        output_root=args.output_root,
        dropout_rate=args.dropout_rate,
        seed=args.seed,
        summary_out=args.summary_out,
        progress_out=args.progress_out,
        max_rows_per_file=args.max_rows_per_file,
    )
    print(json.dumps({"status": payload["status"], "rows": payload["rows"], "dropped_rows": payload["dropped_rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
