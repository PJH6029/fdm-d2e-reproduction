#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.data.full_corpus import split_output_paths
from fdm_d2e.io_utils import stable_hash_json, write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge parallel D2E full-corpus extraction shards into train/eval JSONL files.")
    parser.add_argument("--shard-root", default="outputs/data/d2e_full_corpus_shards")
    parser.add_argument("--output-dir", default="outputs/data/d2e_full_corpus")
    parser.add_argument("--summary-out", default="artifacts/sources/d2e_full_corpus_decode_summary.json")
    args = parser.parse_args()

    shard_root = Path(args.shard_root)
    shard_dirs = sorted(path for path in shard_root.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        raise SystemExit(f"no shard directories found under {shard_root}")
    shard_summaries = []
    failures = []
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "decode_summary.json"
        if not summary_path.exists():
            failures.append({"shard": shard_dir.name, "error": "missing decode_summary.json"})
            continue
        summary = _read_json(summary_path)
        shard_summaries.append(summary)
        failures.extend(summary.get("failures", []))

    output_paths = split_output_paths(args.output_dir)
    counts = {}
    for name, output_path in output_paths.items():
        inputs = [split_output_paths(shard_dir)[name] for shard_dir in shard_dirs]
        counts[name] = _concat_jsonl(inputs, output_path)

    selected = sum(int(summary.get("selected_recording_variants", 0)) for summary in shard_summaries)
    recordings = [recording for summary in shard_summaries for recording in summary.get("recordings", [])]
    summary = {
        "schema": "d2e_full_corpus_decode_summary.v1",
        "output_dir": str(args.output_dir),
        "shard_root": str(shard_root),
        "num_shards": len(shard_dirs),
        "selected_recording_variants": selected,
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
            }
            for shard_dir, summary in zip(shard_dirs, shard_summaries, strict=False)
        ],
        "dataset_fingerprint": stable_hash_json(
            {
                "shard_summaries": [summary.get("dataset_fingerprint") for summary in shard_summaries],
                "counts": counts,
                "recordings": [recording.get("universe_row_id") for recording in recordings],
            }
        ),
    }
    write_json(args.summary_out, summary)
    print(
        "merged D2E full-corpus shards: "
        f"shards={len(shard_dirs)} variants={selected} records={counts.get('all', 0)} "
        f"train_core={counts.get('train_core', 0)} eval={counts.get('target_all_eval', 0)} failures={len(failures)}"
    )


if __name__ == "__main__":
    main()
