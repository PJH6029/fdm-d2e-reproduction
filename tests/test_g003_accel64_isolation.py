from __future__ import annotations

import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def test_g003_accel64_configs_use_isolated_paths():
    model = _load("configs/model/idm_streaming_d2e_full_compact_accel64.yaml")
    split = _load("configs/eval/g003_split_statistics_accel64.yaml")
    completion = _load("configs/eval/g003_full_idm_completion_accel64.yaml")

    assert model["train_records"] == "outputs/data/d2e_full_corpus_accel64/train_core.jsonl"
    assert model["target_records"] == "outputs/data/d2e_full_corpus_accel64/target_all_eval.jsonl"
    assert model["output_dir"] == "outputs/idm_streaming_d2e_full_compact_accel64"
    assert model["summary_out"] == "artifacts/idm/idm_streaming_d2e_full_compact_accel64_summary.json"

    assert split["predictions_path"] == "outputs/idm_streaming_d2e_full_compact_accel64/predictions.jsonl"
    assert split["ground_truth_path"] == model["target_records"]
    assert split["train_records_path"] == model["train_records"]
    assert split["summary_out"] == "artifacts/eval/g003_split_statistical_comparisons_accel64_summary.json"

    paths = completion["paths"]
    assert completion["expected_shards"] == 64
    assert completion["allowed_shard_counts"] == [64]
    assert paths["decode_summary"] == "artifacts/sources/d2e_full_corpus_decode_summary_accel64.json"
    assert paths["train_records"] == model["train_records"]
    assert paths["target_records"] == model["target_records"]
    assert paths["checkpoint"].startswith("outputs/idm_streaming_d2e_full_compact_accel64/")
    assert paths["run_summary"] == "artifacts/idm/g003_d2e_full_idm_4xh200_train_run_accel64.json"
    assert paths["gpu_monitor"] == "artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64.csv"
    assert paths["split_stats_summary"] == split["summary_out"]


def test_g003_accel64_launch_scripts_preserve_canonical_outputs():
    run_text = Path("scripts/run_g003_d2e_full_idm_accel64_isolated.sh").read_text()
    launch_text = Path("scripts/launch_g003_accel64_isolated.sh").read_text()

    assert "NUM_SHARDS=\"${NUM_SHARDS:-64}\"" in run_text
    assert "SHARD_ROOT=\"${SHARD_ROOT:-outputs/data/d2e_full_corpus_shards_accel64}\"" in run_text
    assert "LOG_DIR=\"${LOG_DIR:-artifacts/sources/g003_accel64}\"" in run_text
    assert "CACHE_DIR=\"${CACHE_DIR:-/root/work/data/d2e/cache_accel64}\"" in run_text
    assert "configs/model/idm_streaming_d2e_full_compact_accel64.yaml" in run_text

    assert "outputs/cluster/g003_full_compact_accel64.pid" in launch_text
    assert "g003_full_idm_completion_accel64.yaml" in launch_text
    assert "g003_full_idm_completion_accel64_audit.json" in launch_text
    assert "outputs/data/d2e_full_corpus_accel64" in launch_text
    assert "outputs/idm_streaming_d2e_full_compact_accel64" in launch_text
