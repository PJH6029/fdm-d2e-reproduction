from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fdm_d2e.data.full_corpus import split_output_paths
from fdm_d2e.io_utils import write_json, write_jsonl
from scripts.merge_d2e_full_corpus_shards import merge_shards


def _write_shard(root: Path, index: int, *, split_mode: str = "fdm1-g002") -> None:
    shard = root / f"shard_{index}"
    paths = split_output_paths(shard)
    row = {
        "schema": "d2e_window_record.v1",
        "sequence_id": f"d2e_480p:Toy/rec{index}#000000",
        "recording_id": f"d2e_480p:Toy/rec{index}",
        "universe_row_id": f"d2e_480p:Toy/rec{index}",
        "split": "train_core" if index == 0 else "eval",
        "eval_split_tags": [] if index == 0 else ["recording_test"],
    }
    for name, path in paths.items():
        rows = [row] if name in {"all", "train_core" if index == 0 else "target_all_eval"} else []
        write_jsonl(path, rows)
    write_json(
        shard / "decode_summary.json",
        {
            "schema": "d2e_full_corpus_decode_summary.v1",
            "data_universe": "artifacts/sources/d2e_full_data_universe_manifest.json",
            "split_contract": None,
            "split_mode": split_mode,
            "fdm1_split_manifests": {"recording_level_split": "recording.json"},
            "selected_recording_variants": 1,
            "source_ids": ["d2e_480p"],
            "resolution_tiers": ["480p"],
            "counts": {name: len(open(path).read().splitlines()) if path.exists() else 0 for name, path in paths.items()},
            "recordings": [{"universe_row_id": f"d2e_480p:Toy/rec{index}"}],
            "failures": [],
            "dataset_fingerprint": f"fp-{index}",
        },
    )


def test_merge_shards_preserves_fdm1_metadata_for_completion_audit(tmp_path: Path):
    shard_root = tmp_path / "shards"
    _write_shard(shard_root, 0)
    _write_shard(shard_root, 1)
    summary = merge_shards(shard_root=shard_root, output_dir=tmp_path / "merged", summary_out=tmp_path / "summary.json")
    assert summary["split_mode"] == "fdm1-g002"
    assert summary["source_ids"] == ["d2e_480p"]
    assert summary["resolution_tiers"] == ["480p"]
    assert summary["selected_recording_variants"] == 2
    assert summary["counts"]["all"] == 2
    assert summary["counts"]["train_core"] == 1
    assert summary["counts"]["target_all_eval"] == 1
    assert summary["fdm1_split_manifests"] == {"recording_level_split": "recording.json"}


def test_merge_shards_records_mixed_split_mode_failure(tmp_path: Path):
    shard_root = tmp_path / "shards"
    _write_shard(shard_root, 0, split_mode="fdm1-g002")
    _write_shard(shard_root, 1, split_mode="legacy")
    summary = merge_shards(shard_root=shard_root, output_dir=tmp_path / "merged", summary_out=tmp_path / "summary.json")
    assert summary["split_mode"] is None
    assert any(item.get("error") == "mixed_split_modes" for item in summary["failures"])


def test_merge_shards_cli_writes_summary(tmp_path: Path):
    shard_root = tmp_path / "shards"
    _write_shard(shard_root, 0)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/merge_d2e_full_corpus_shards.py",
            "--shard-root",
            str(shard_root),
            "--output-dir",
            str(tmp_path / "merged"),
            "--summary-out",
            str(tmp_path / "summary.json"),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "merged D2E full-corpus shards" in completed.stdout
    assert json.loads((tmp_path / "summary.json").read_text())["split_mode"] == "fdm1-g002"
