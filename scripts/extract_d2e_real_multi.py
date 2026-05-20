#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.data.d2e_real import prepare_decoded_sample
from fdm_d2e.io_utils import stable_hash_json, write_json, write_jsonl
from fdm_d2e.schema import validate_named


def _sample_config(base: dict, sample: dict) -> dict:
    cfg = dict(base)
    cfg["max_recordings"] = 1
    cfg["games"] = [sample["game"]]
    cfg["recording_ids"] = [sample["recording_id"]]
    cfg["sample_root"] = f"{base.get('dataset_name', 'real_multi')}_per_recording"
    for key, value in sample.get("overrides", {}).items():
        cfg[key] = value
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode multiple real D2E recordings and combine train/heldout contracts.")
    parser.add_argument("--config", default="configs/data/d2e_real_multi_apex.yaml")
    parser.add_argument("--summary-copy", default="artifacts/sources/d2e_multi_decode_summary.json")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs")) / "data" / str(config.get("dataset_name", "real_multi"))
    samples = list(config.get("samples", []))
    if len(samples) < 2:
        raise SystemExit("multi decode requires at least two samples")

    all_records = []
    train = []
    heldout = []
    summaries = []
    sequences = []
    for sample in samples:
        result = prepare_decoded_sample(_sample_config(config, sample))
        summary = result["summary"]
        summaries.append(summary)
        records = result["records"]
        all_records.extend(records)
        train.extend(row for row in records if row.get("split") == "train")
        heldout.extend(row for row in records if row.get("split") == "heldout")
        sequences.extend(result["sequence_pack"]["sequences"])

    dataset_fingerprint = stable_hash_json(
        {
            "config": {k: v for k, v in config.items() if k != "hf_token"},
            "recordings": [{"pair_id": row["pair_id"], "mcap_sha256": row["mcap_sha256"], "window_start_ns": row["window_start_ns"]} for row in summaries],
            "num_records": len(all_records),
        }
    )
    sequence_pack = {
        "schema": "sequence_pack.v2",
        "dataset_fingerprint": dataset_fingerprint,
        "timebase": {"timestamp_unit": "nanoseconds", "bin_ms": int(config.get("bin_ms", 50))},
        "sequences": sequences,
    }
    validate_named(sequence_pack, "sequence_pack_v2.schema.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "all_records.jsonl", all_records)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "heldout.jsonl", heldout)
    write_json(output_dir / "sequence_pack.v2.json", sequence_pack)
    summary = {
        "schema": "d2e_multi_decode_summary.v1",
        "dataset_name": config.get("dataset_name", "real_multi"),
        "output_dir": str(output_dir),
        "num_recordings": len(summaries),
        "num_records": len(all_records),
        "splits": {"train": len(train), "heldout": len(heldout)},
        "heldout_recording_clusters": sorted({row.get("recording_id") for row in heldout}),
        "games": sorted({row.get("game") for row in all_records}),
        "recordings": summaries,
        "dataset_fingerprint": dataset_fingerprint,
    }
    write_json(output_dir / "decode_summary.json", summary)
    write_json(args.summary_copy, summary)
    print(
        "decoded multi real D2E: "
        f"recordings={summary['num_recordings']} records={summary['num_records']} "
        f"train={summary['splits']['train']} heldout={summary['splits']['heldout']} "
        f"clusters={len(summary['heldout_recording_clusters'])} fingerprint={dataset_fingerprint[:12]}"
    )


if __name__ == "__main__":
    main()
