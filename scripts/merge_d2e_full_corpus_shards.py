#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.data.full_corpus import split_output_paths
from fdm_d2e.io_utils import stable_hash_json, write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_shard_index(path: Path) -> int | None:
    prefix = "shard_"
    if not path.name.startswith(prefix):
        return None
    suffix = path.name[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _shard_sort_key(path: Path) -> tuple[int, int, str]:
    index = _parse_shard_index(path)
    if index is None:
        return (1, 10**9, path.name)
    return (0, index, path.name)


def _concat_jsonl(inputs: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as out:
        for path in inputs:
            if not path.exists():
                continue
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    out.write(line)
                    count += 1
    return count


def _union_list(summaries: list[dict[str, Any]], key: str) -> list[str]:
    values: set[str] = set()
    for summary in summaries:
        values.update(str(item) for item in summary.get(key, []) if item is not None)
    return sorted(values)


def _first_non_empty(summaries: list[dict[str, Any]], key: str) -> Any:
    for summary in summaries:
        value = summary.get(key)
        if value:
            return value
    return None


def merge_shards(
    *,
    shard_root: str | Path,
    output_dir: str | Path,
    summary_out: str | Path,
    expected_shards: int | None = None,
) -> dict[str, Any]:
    shard_root_path = Path(shard_root)
    shard_dirs = sorted(
        (path for path in shard_root_path.glob("shard_*") if path.is_dir()),
        key=_shard_sort_key,
    )
    if not shard_dirs:
        raise ValueError(f"no shard directories found under {shard_root_path}")
    shard_pairs: list[tuple[Path, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    shard_indices = [_parse_shard_index(path) for path in shard_dirs]
    invalid_shards = [
        path.name
        for path, index in zip(shard_dirs, shard_indices, strict=True)
        if index is None
    ]
    if invalid_shards:
        failures.append(
            {
                "shard": "merge",
                "error": "invalid_shard_names",
                "invalid_shards": invalid_shards,
            }
        )
    if expected_shards is not None:
        expected_count = int(expected_shards)
        expected_indices = list(range(expected_count))
        actual_indices = [index for index in shard_indices if index is not None]
        duplicates = sorted(
            index for index, count in Counter(actual_indices).items() if count > 1
        )
        missing = [idx for idx in expected_indices if idx not in actual_indices]
        extra = sorted(idx for idx in actual_indices if idx not in expected_indices)
        if missing or extra or duplicates or len(shard_dirs) != expected_count or invalid_shards:
            failures.append(
                {
                    "shard": "merge",
                    "error": "shard_coverage_mismatch",
                    "expected_shards": expected_count,
                    "actual_shards": len(shard_dirs),
                    "missing_indices": missing,
                    "extra_indices": extra,
                    "duplicate_indices": duplicates,
                    "invalid_shards": invalid_shards,
                }
            )
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "decode_summary.json"
        if not summary_path.exists():
            failures.append({"shard": shard_dir.name, "error": "missing decode_summary.json"})
            continue
        summary = _read_json(summary_path)
        shard_pairs.append((shard_dir, summary))
        failures.extend(summary.get("failures", []))

    shard_summaries = [summary for _, summary in shard_pairs]
    output_paths = split_output_paths(output_dir)
    counts = {}
    for name, output_path in output_paths.items():
        inputs = [split_output_paths(shard_dir)[name] for shard_dir in shard_dirs]
        counts[name] = _concat_jsonl(inputs, output_path)

    selected = sum(int(summary.get("selected_recording_variants", 0) or 0) for summary in shard_summaries)
    recordings = [recording for summary in shard_summaries for recording in summary.get("recordings", [])]
    split_modes = sorted({str(summary.get("split_mode")) for summary in shard_summaries if summary.get("split_mode")})
    split_mode = split_modes[0] if len(split_modes) == 1 else None
    if len(split_modes) > 1:
        failures.append({"shard": "merge", "error": "mixed_split_modes", "split_modes": split_modes})
    data_universe = _first_non_empty(shard_summaries, "data_universe")
    split_contract = _first_non_empty(shard_summaries, "split_contract")
    fdm1_split_manifests = _first_non_empty(shard_summaries, "fdm1_split_manifests")
    summary = {
        "schema": "d2e_full_corpus_decode_summary.v1",
        "output_dir": str(output_dir),
        "data_universe": data_universe,
        "split_contract": split_contract,
        "split_mode": split_mode,
        "fdm1_split_manifests": fdm1_split_manifests,
        "shard_root": str(shard_root_path),
        "num_shards": len(shard_dirs),
        "expected_shards": expected_shards,
        "shard_indices": [
            {"name": path.name, "index": index}
            for path, index in zip(shard_dirs, shard_indices, strict=True)
        ],
        "selected_recording_variants": selected,
        "source_ids": _union_list(shard_summaries, "source_ids"),
        "resolution_tiers": _union_list(shard_summaries, "resolution_tiers"),
        "counts": counts,
        "paths": {name: str(path) for name, path in output_paths.items()},
        "recordings": recordings,
        "failures": failures,
        "shards": [
            {
                "path": str(shard_dir),
                "summary_path": str(shard_dir / "decode_summary.json"),
                "counts": summary.get("counts", {}),
                "selected_recording_variants": summary.get("selected_recording_variants"),
                "split_mode": summary.get("split_mode"),
                "source_ids": summary.get("source_ids", []),
                "resolution_tiers": summary.get("resolution_tiers", []),
            }
            for shard_dir, summary in shard_pairs
        ],
        "dataset_fingerprint": stable_hash_json(
            {
                "data_universe": data_universe,
                "split_mode": split_mode,
                "fdm1_split_manifests": fdm1_split_manifests,
                "shard_summaries": [summary.get("dataset_fingerprint") for summary in shard_summaries],
                "counts": counts,
                "recordings": [recording.get("universe_row_id") for recording in recordings],
            }
        ),
        "claim_boundary": "Merged decode summary only; downstream G003 completion still requires action-slot finalization and completion audit pass.",
    }
    write_json(summary_out, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge parallel D2E full-corpus extraction shards into train/eval JSONL files.")
    parser.add_argument("--shard-root", default="outputs/data/d2e_full_corpus_shards")
    parser.add_argument("--output-dir", default="outputs/data/d2e_full_corpus")
    parser.add_argument("--summary-out", default="artifacts/sources/d2e_full_corpus_decode_summary.json")
    parser.add_argument("--expected-shards", type=int)
    args = parser.parse_args()
    try:
        summary = merge_shards(
            shard_root=args.shard_root,
            output_dir=args.output_dir,
            summary_out=args.summary_out,
            expected_shards=args.expected_shards,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "merged D2E full-corpus shards: "
        f"shards={summary['num_shards']} variants={summary['selected_recording_variants']} records={summary['counts'].get('all', 0)} "
        f"train_core={summary['counts'].get('train_core', 0)} eval={summary['counts'].get('target_all_eval', 0)} failures={len(summary['failures'])}"
    )


if __name__ == "__main__":
    main()
