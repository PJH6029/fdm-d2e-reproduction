from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdm_d2e.training.streaming_idm import (
    _distributed_runtime,
    _training_cache_identity,
    _training_cache_assignment_plan,
    _training_cache_rank_assignment,
    predict_streaming_idm_checkpoint,
    recover_streaming_idm_outputs_from_checkpoint,
    scan_streaming_idm_stats,
    train_streaming_idm,
)
from fdm_d2e.training.torch_idm import torch_available


def _record(idx: int, split: str) -> dict:
    token = "MOUSE_DX_P1" if idx % 2 else "MOUSE_DX_N1"
    return {
        "schema": "d2e_window_record.v1",
        "sequence_id": f"d2e_480p:Apex/rec#%06d" % idx,
        "recording_id": "d2e_480p:Apex/rec",
        "cross_resolution_key": "Apex/rec",
        "game": "Apex",
        "source_id": "d2e_480p",
        "resolution_tier": "480p",
        "split": split,
        "eval_split_tags": ["temporal"] if split == "eval" else [],
        "timestamp_ns": idx,
        "bin_index": idx,
        "frame": {
            "path": "",
            "index": idx,
            "features": [0.1 * idx, 0.2, 0.3, 0.4, 0.5],
            "grid8": [0.01 * (idx + 1)] * (8 * 8 * 3),
            "luma16": [0.02 * (idx + 1)] * (16 * 16),
        },
        "next_frame_features": [0.1 * (idx + 1), 0.2, 0.3, 0.4, 0.5],
        "frame_delta_features": [0.1, 0.0, 0.0, 0.0, 0.0],
        "next_frame_grid8": [0.01 * (idx + 2)] * (8 * 8 * 3),
        "next_frame_luma16": [0.02 * (idx + 2)] * (16 * 16),
        "events": [],
        "ground_truth_tokens": [token, "MOUSE_DY_Z0", "KEY_PRESS_87"] if idx % 3 == 0 else [token, "MOUSE_DY_Z0"],
        "source": "test",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_training_cache_assignment_balances_manifest_rows():
    manifests = [
        {"rows": 1_121_684, "chunks": [{"rows": 1_121_684}]},
        {"rows": 1_633_458, "chunks": [{"rows": 1_633_458}]},
        {"rows": 1_698_060, "chunks": [{"rows": 1_698_060}]},
        {"rows": 1_117_045, "chunks": [{"rows": 1_117_045}]},
        {"rows": 1_201_218, "chunks": [{"rows": 1_201_218}]},
        {"rows": 857_970, "chunks": [{"rows": 857_970}]},
        {"rows": 1_597_891, "chunks": [{"rows": 1_597_891}]},
        {"rows": 1_144_425, "chunks": [{"rows": 1_144_425}]},
        {"rows": 1_104_112, "chunks": [{"rows": 1_104_112}]},
        {"rows": 1_351_897, "chunks": [{"rows": 1_351_897}]},
        {"rows": 1_078_306, "chunks": [{"rows": 1_078_306}]},
        {"rows": 1_204_409, "chunks": [{"rows": 1_204_409}]},
        {"rows": 988_859, "chunks": [{"rows": 988_859}]},
        {"rows": 1_170_339, "chunks": [{"rows": 1_170_339}]},
        {"rows": 810_509, "chunks": [{"rows": 810_509}]},
        {"rows": 1_130_824, "chunks": [{"rows": 1_130_824}]},
    ]

    modulo_loads = [
        sum(int(manifests[idx]["rows"]) for idx in range(len(manifests)) if idx % 4 == rank)
        for rank in range(4)
    ]
    plan = _training_cache_assignment_plan(manifests, world_size=4, mode="greedy_rows")
    greedy_loads = [row["rows"] for row in plan["ranks"]]
    assigned = [
        idx
        for rank in range(4)
        for idx in _training_cache_rank_assignment(manifests, rank=rank, world_size=4, mode="greedy_rows")
    ]

    assert sorted(assigned) == list(range(len(manifests)))
    assert plan["mode"] == "greedy_rows"
    assert max(greedy_loads) - min(greedy_loads) < max(modulo_loads) - min(modulo_loads)
    assert plan["row_load_max"] == max(greedy_loads)
    assert plan["row_load_min"] == min(greedy_loads)


def test_training_cache_identity_includes_mouse_target_mode(tmp_path: Path):
    records = tmp_path / "train.jsonl"
    _write_jsonl(records, [_record(0, "train_core")])
    stats = {
        "feature_mode": "summary_compact_grid8_shift_surface_time",
        "input_dim": 620,
        "dataset_fingerprint": "abc123",
    }
    common = {
        "mouse_head_mode": "axis_softmax",
    }

    mean_identity = _training_cache_identity(
        records,
        stats=stats,
        config={**common, "mouse_target_mode": "mean"},
        category_vocab=["KEY_PRESS_87"],
        mouse_axis_classes=["N1", "Z0", "P1"],
    )
    sum_identity = _training_cache_identity(
        records,
        stats=stats,
        config={**common, "mouse_target_mode": "sum"},
        category_vocab=["KEY_PRESS_87"],
        mouse_axis_classes=["N1", "Z0", "P1"],
    )

    assert mean_identity["mouse_target_mode"] == "mean"
    assert sum_identity["mouse_target_mode"] == "sum"
    assert mean_identity != sum_identity


def test_streaming_idm_action_history_changes_stats_and_cache_identity(tmp_path: Path):
    records = tmp_path / "train.jsonl"
    rows = [_record(idx, "train_core") for idx in range(4)]
    _write_jsonl(records, rows)

    base_stats = scan_streaming_idm_stats(
        records,
        feature_mode="summary_compact_grid8_shift_surface_time",
        categorical_min_count=1,
    )
    history_stats = scan_streaming_idm_stats(
        records,
        feature_mode="summary_compact_grid8_shift_surface_time",
        categorical_min_count=1,
        action_history_len=2,
    )

    assert history_stats["action_history_len"] == 2
    assert history_stats["action_history_dim"] == 2 * 2 + len(history_stats["action_history_vocab"]) * 2 + 3
    assert history_stats["input_dim"] == base_stats["input_dim"] + history_stats["action_history_dim"]
    assert history_stats["dataset_fingerprint"] != base_stats["dataset_fingerprint"]

    common = {
        "mouse_head_mode": "axis_softmax",
        "mouse_target_mode": "mean",
        "action_history_len": 2,
    }
    base_identity = _training_cache_identity(
        records,
        stats=base_stats,
        config={**common, "action_history_len": 0},
        category_vocab=base_stats["category_vocab"],
        mouse_axis_classes=["N1", "Z0", "P1"],
    )
    history_identity = _training_cache_identity(
        records,
        stats=history_stats,
        config=common,
        category_vocab=history_stats["category_vocab"],
        mouse_axis_classes=["N1", "Z0", "P1"],
    )

    assert history_identity["action_history_len"] == 2
    assert history_identity["input_dim"] == history_stats["input_dim"]
    assert history_identity != base_identity


def test_streaming_idm_action_history_prediction_does_not_peek_target_labels(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    target_mutated_path = tmp_path / "target_mutated.jsonl"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    target_rows = [_record(idx + 8, "eval") for idx in range(4)]
    mutated_rows = []
    for row in target_rows:
        mutated = dict(row)
        mutated["ground_truth_tokens"] = ["KEY_PRESS_999", "MOUSE_LEFT_DOWN", "MOUSE_DX_P5", "MOUSE_DY_N5"]
        mutated_rows.append(mutated)
    _write_jsonl(target_path, target_rows)
    _write_jsonl(target_mutated_path, mutated_rows)

    idm_out = tmp_path / "idm_history"
    summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm_history",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(idm_out),
            "summary_out": str(tmp_path / "summary.json"),
            "config_path": "test_action_history_config",
            "source_namespace": "unit_d2e_stream",
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "eval_interval_epochs": 1,
            "batch_size": 4,
            "training_cache_dir": str(tmp_path / "idm_history_cache"),
            "training_cache_chunk_size": 3,
            "categorical_min_count": 1,
            "mouse_head_mode": "axis_softmax",
            "action_history_len": 2,
            "seed": 41,
            "force_cpu": True,
        }
    )
    assert summary["metadata"]["action_history_len"] == 2
    assert summary["metadata"]["action_history_feedback"] == "autoregressive_predicted"

    original_prediction = predict_streaming_idm_checkpoint(
        {
            "checkpoint_path": str(idm_out / "checkpoint.pt"),
            "checkpoint_metadata_path": str(idm_out / "checkpoint_metadata.json"),
            "records_path": str(target_path),
            "output_dir": str(tmp_path / "predict_original"),
            "force_cpu": True,
            "validate_pseudolabels": False,
        }
    )
    mutated_prediction = predict_streaming_idm_checkpoint(
        {
            "checkpoint_path": str(idm_out / "checkpoint.pt"),
            "checkpoint_metadata_path": str(idm_out / "checkpoint_metadata.json"),
            "records_path": str(target_mutated_path),
            "output_dir": str(tmp_path / "predict_mutated"),
            "force_cpu": True,
            "validate_pseudolabels": False,
        }
    )

    original_tokens = [
        json.loads(line)["predicted_tokens"]
        for line in Path(original_prediction["predictions_path"]).read_text().splitlines()
    ]
    mutated_tokens = [
        json.loads(line)["predicted_tokens"]
        for line in Path(mutated_prediction["predictions_path"]).read_text().splitlines()
    ]
    assert original_tokens == mutated_tokens

    target_part_a = tmp_path / "target_part_a.jsonl"
    target_part_b = tmp_path / "target_part_b.jsonl"
    _write_jsonl(target_part_a, target_rows[:2])
    _write_jsonl(target_part_b, target_rows[2:])
    with pytest.raises(ValueError, match="parallel prediction with action_history_len>0 requires"):
        predict_streaming_idm_checkpoint(
            {
                "checkpoint_path": str(idm_out / "checkpoint.pt"),
                "records_path": str(target_part_a),
                "record_paths": [str(target_part_a), str(target_part_b)],
                "output_dir": str(tmp_path / "predict_parallel_blocked"),
                "prediction_workers": 2,
                "force_cpu": True,
                "validate_pseudolabels": False,
            }
        )
    safe_part_a = tmp_path / "target_part_safe_a.jsonl"
    safe_part_b = tmp_path / "target_part_safe_b.jsonl"
    safe_rows_a = [{**row, "recording_id": "d2e_480p:Apex/rec_a", "sequence_id": row["sequence_id"].replace("rec#", "rec_a#")} for row in target_rows[:2]]
    safe_rows_b = [{**row, "recording_id": "d2e_480p:Apex/rec_b", "sequence_id": row["sequence_id"].replace("rec#", "rec_b#")} for row in target_rows[2:]]
    _write_jsonl(safe_part_a, safe_rows_a)
    _write_jsonl(safe_part_b, safe_rows_b)
    parallel_prediction = predict_streaming_idm_checkpoint(
        {
            "checkpoint_path": str(idm_out / "checkpoint.pt"),
            "records_path": str(safe_part_a),
            "record_paths": [str(safe_part_a), str(safe_part_b)],
            "output_dir": str(tmp_path / "predict_parallel_allowed"),
            "prediction_workers": 2,
            "action_history_parallel_by_path": True,
            "force_cpu": True,
            "validate_pseudolabels": False,
        }
    )
    assert parallel_prediction["records"] == 4
    assert "_action_history_seed_state" not in parallel_prediction["prediction_config"]
    assert parallel_prediction["prediction_config"]["action_history_seed_state_summary"]["source"] == "parent_train_scan"


def test_streaming_idm_trains_tiny_compact_feature_checkpoint(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    _write_jsonl(target_path, [_record(idx + 8, "eval") for idx in range(4)])

    summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "idm"),
            "summary_out": str(tmp_path / "summary.json"),
            "config_path": "test_inline_config",
            "source_namespace": "unit_d2e_stream",
            "endpoints": "configs/eval/primary_endpoints.yaml",
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "eval_interval_epochs": 1,
            "batch_size": 4,
            "training_cache_dir": str(tmp_path / "idm_train_cache"),
            "training_cache_chunk_size": 3,
            "training_cache_shard_by_path": False,
            "training_progress_interval_batches": 1,
            "categorical_min_count": 1,
            "category_threshold_mode": "group_fbeta_calibrated",
            "category_calibration_beta": 0.5,
            "category_calibration_grid": [0.1, 0.5, 0.9],
            "category_calibration_max_examples": 8,
            "mouse_head_mode": "axis_softmax",
            "mouse_output_gain_mode": "train_abs_ratio",
            "mouse_gain_calibration_max_examples": 8,
            "seed": 7,
            "force_cpu": True,
        }
    )

    assert summary["metadata"]["train_records"] == 8
    assert summary["metadata"]["target_records"] == 4
    assert summary["metadata"]["config_fingerprint"]
    assert summary["metadata"]["config_path"] == "test_inline_config"
    assert summary["metadata"]["source_namespace"] == "unit_d2e_stream"
    assert summary["metadata"]["source_ids"] == ["d2e_480p"]
    assert summary["metadata"]["resolution_tiers"] == ["480p"]
    assert summary["metadata"]["target_source_ids"] == ["d2e_480p"]
    assert summary["metadata"]["target_resolution_tiers"] == ["480p"]
    assert summary["metadata"]["split_names"] == ["train_core"]
    assert summary["metadata"]["target_eval_split_tags"] == ["temporal"]
    assert summary["metadata"]["training_cache"]["enabled"] is True
    assert summary["metadata"]["training_cache"]["rows"] == 8
    assert summary["metadata"]["training_cache"]["chunk_size"] == 3
    assert summary["metadata"]["training_cache"]["shard_by_path"] is False
    assert summary["metadata"]["training_cache"]["progress_interval_batches"] == 1
    progress = json.loads((tmp_path / "idm" / "rank_progress" / "train_rank0.json").read_text())
    assert progress["training_cache_shard_by_path"] is False
    assert progress["batches"] >= 1
    assert summary["metadata"]["calibration"]["mode"] == "group_fbeta_calibrated"
    assert summary["metadata"]["calibration"]["status"] == "computed"
    assert "KEY_PRESS_87" in summary["metadata"]["calibration"]["category_thresholds"]
    assert summary["metadata"]["calibration"]["mouse_output_gain_info"]["mode"] == "train_abs_ratio"
    assert summary["metadata"]["calibration"]["mouse_output_gain"] > 0
    assert all(Path(path).exists() for path in summary["metadata"]["training_cache"]["manifest_paths"])
    assert Path(summary["metadata"]["resolved_config_path"]).exists()
    assert Path(summary["metadata"]["train_records_path"]).exists()
    assert Path(summary["metadata"]["target_records_path"]).exists()
    assert Path(summary["metadata"]["checkpoint_path"]).exists()
    assert Path(summary["metadata"]["pseudo_label_path"]).exists()
    assert Path(summary["metadata"]["label_quality_report_path"]).exists()
    assert Path(summary["metadata"]["convergence_report_path"]).exists()
    assert summary["convergence_report"]["num_validation_checkpoints"] == 1
    assert summary["convergence_report"]["history"][0]["validation_score"]["mode"] == "composite_primary"
    assert summary["label_quality_report"]["baseline_metrics"]["noop"]["num_examples"] == 4
    assert "game:Apex" in summary["label_quality_report"]["groups_by_model"]["tiny_streaming_idm"]
    assert Path(summary["metadata"]["statistical_comparison_path"]).exists()
    assert summary["statistical_comparison"]["schema"] == "stat_comparison.v1"


def test_streaming_idm_predicts_train_core_pseudolabels_without_retraining(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    fdm_train_path = tmp_path / "fdm_train_core.jsonl"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    _write_jsonl(target_path, [_record(idx + 8, "eval") for idx in range(4)])
    _write_jsonl(fdm_train_path, [_record(idx + 20, "train_core") for idx in range(5)])
    idm_out = tmp_path / "idm"

    train_summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm_predict_only",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(idm_out),
            "summary_out": str(tmp_path / "summary.json"),
            "config_path": "test_inline_config",
            "source_namespace": "unit_d2e_stream",
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "eval_interval_epochs": 1,
            "batch_size": 4,
            "categorical_min_count": 1,
            "mouse_head_mode": "axis_softmax",
            "seed": 11,
            "force_cpu": True,
        }
    )

    pred_summary = predict_streaming_idm_checkpoint(
        {
            "checkpoint_path": train_summary["metadata"]["checkpoint_path"],
            "checkpoint_metadata_path": str(idm_out / "checkpoint_metadata.json"),
            "records_path": str(fdm_train_path),
            "output_dir": str(tmp_path / "fdm_train_core_pseudolabels"),
            "summary_out": str(tmp_path / "fdm_train_core_summary.json"),
            "force_cpu": True,
            "eval_batch_size": 2,
        }
    )

    assert pred_summary["schema"] == "streaming_idm_predict_summary.v1"
    assert pred_summary["records"] == 5
    assert pred_summary["source_checkpoint_artifact"]["exists"] is True
    assert pred_summary["source_checkpoint_metadata"]["exists"] is True
    assert Path(pred_summary["pseudo_label_path"]).exists()
    assert Path(pred_summary["predictions_path"]).exists()
    assert Path(tmp_path / "fdm_train_core_summary.json").exists()


def test_streaming_idm_predicts_over_multiple_record_files(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    shard_a = tmp_path / "fdm_train_core_a.jsonl"
    shard_b = tmp_path / "fdm_train_core_b.jsonl"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    _write_jsonl(target_path, [_record(idx + 8, "eval") for idx in range(4)])
    _write_jsonl(shard_a, [_record(idx + 20, "train_core") for idx in range(3)])
    _write_jsonl(shard_b, [_record(idx + 23, "train_core") for idx in range(2)])

    train_summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm_predict_shards",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "idm_sharded_predict"),
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "batch_size": 4,
            "categorical_min_count": 1,
            "mouse_head_mode": "axis_softmax",
            "seed": 13,
            "force_cpu": True,
        }
    )

    pred_summary = predict_streaming_idm_checkpoint(
        {
            "checkpoint_path": train_summary["metadata"]["checkpoint_path"],
            "records_path": str(shard_a),
            "record_paths": [str(shard_a), str(shard_b)],
            "output_dir": str(tmp_path / "fdm_train_core_pseudolabels_sharded"),
            "force_cpu": True,
            "eval_batch_size": 2,
        }
    )

    assert pred_summary["records"] == 5
    assert pred_summary["record_paths"] == [str(shard_a), str(shard_b)]
    assert len(Path(pred_summary["pseudo_label_path"]).read_text().strip().splitlines()) == 5


def test_streaming_idm_predicts_over_multiple_record_files_with_parallel_workers(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    shard_a = tmp_path / "fdm_train_core_a.jsonl"
    shard_b = tmp_path / "fdm_train_core_b.jsonl"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    _write_jsonl(target_path, [_record(idx + 8, "eval") for idx in range(4)])
    _write_jsonl(shard_a, [_record(idx + 20, "train_core") for idx in range(3)])
    _write_jsonl(shard_b, [_record(idx + 23, "train_core") for idx in range(2)])

    train_summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm_predict_parallel_shards",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "idm_parallel_sharded_predict"),
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "batch_size": 4,
            "categorical_min_count": 1,
            "mouse_head_mode": "axis_softmax",
            "seed": 17,
            "force_cpu": True,
        }
    )

    pred_summary = predict_streaming_idm_checkpoint(
        {
            "checkpoint_path": train_summary["metadata"]["checkpoint_path"],
            "records_path": str(shard_a),
            "record_paths": [str(shard_a), str(shard_b)],
            "output_dir": str(tmp_path / "fdm_train_core_pseudolabels_parallel_sharded"),
            "force_cpu": True,
            "eval_batch_size": 2,
            "prediction_workers": 2,
            "validate_pseudolabels": False,
        }
    )

    assert pred_summary["records"] == 5
    assert pred_summary["record_paths"] == [str(shard_a), str(shard_b)]
    assert pred_summary["prediction_resume"]["write_mode"] == "parallel_parts"
    assert pred_summary["prediction_resume"]["workers"] == 2
    assert pred_summary["prediction_resume"]["pseudolabel_validation"] is False
    assert len(pred_summary["prediction_resume"]["parts"]) == 2
    assert len(Path(pred_summary["pseudo_label_path"]).read_text().strip().splitlines()) == 5
    assert len(Path(pred_summary["predictions_path"]).read_text().strip().splitlines()) == 5


def test_streaming_idm_prediction_can_resume_partial_outputs(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    predict_path = tmp_path / "predict.jsonl"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    _write_jsonl(target_path, [_record(idx + 8, "eval") for idx in range(4)])
    _write_jsonl(predict_path, [_record(idx + 20, "eval") for idx in range(5)])

    train_summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm_resume_predict",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "idm_resume_predict"),
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "batch_size": 4,
            "categorical_min_count": 1,
            "mouse_head_mode": "axis_softmax",
            "seed": 19,
            "force_cpu": True,
        }
    )
    output_dir = tmp_path / "resumed_predictions"
    first_summary = predict_streaming_idm_checkpoint(
        {
            "checkpoint_path": train_summary["metadata"]["checkpoint_path"],
            "records_path": str(predict_path),
            "output_dir": str(output_dir),
            "force_cpu": True,
            "eval_batch_size": 2,
            "resume_predictions": True,
        }
    )
    pseudo_path = Path(first_summary["pseudo_label_path"])
    predictions_path = Path(first_summary["predictions_path"])
    pseudo_lines = pseudo_path.read_text().splitlines()
    prediction_lines = predictions_path.read_text().splitlines()
    pseudo_path.write_text("\n".join(pseudo_lines[:2]) + "\n")
    predictions_path.write_text("\n".join(prediction_lines[:2]) + "\n")

    resumed_summary = predict_streaming_idm_checkpoint(
        {
            "checkpoint_path": train_summary["metadata"]["checkpoint_path"],
            "records_path": str(predict_path),
            "output_dir": str(output_dir),
            "force_cpu": True,
            "eval_batch_size": 2,
            "resume_predictions": True,
        }
    )

    assert resumed_summary["records"] == 5
    assert resumed_summary["prediction_resume"]["existing_rows"] == 2
    assert Path(resumed_summary["pseudo_label_path"]).read_text().splitlines()[:2] == pseudo_lines[:2]
    assert Path(resumed_summary["predictions_path"]).read_text().splitlines()[:2] == prediction_lines[:2]
    assert len(Path(resumed_summary["pseudo_label_path"]).read_text().splitlines()) == 5


def test_streaming_idm_recovers_outputs_from_checkpoint_without_retraining(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    output_dir = tmp_path / "idm_recover"
    summary_path = tmp_path / "recovered_summary.json"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    _write_jsonl(target_path, [_record(idx + 8, "eval") for idx in range(5)])

    train_summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm_recover",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(output_dir),
            "summary_out": str(summary_path),
            "config_path": "test_recover_config",
            "source_namespace": "unit_d2e_stream",
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "eval_interval_epochs": 1,
            "batch_size": 4,
            "categorical_min_count": 1,
            "mouse_head_mode": "axis_softmax",
            "seed": 23,
            "force_cpu": True,
        }
    )
    pseudo_path = Path(train_summary["metadata"]["pseudo_label_path"])
    predictions_path = Path(train_summary["predictions_path"])
    pseudo_lines = pseudo_path.read_text().splitlines()
    prediction_lines = predictions_path.read_text().splitlines()
    pseudo_path.write_text("\n".join(pseudo_lines[:2]) + "\n")
    predictions_path.write_text("\n".join(prediction_lines[:2]) + "\n")
    for rel in [
        "metrics.json",
        "label_quality_report.json",
        "statistical_comparison.json",
        "checkpoint_metadata.json",
    ]:
        path = output_dir / rel
        if path.exists():
            path.unlink()
    summary_path.unlink()

    recovery = recover_streaming_idm_outputs_from_checkpoint(
        {
            "checkpoint_path": train_summary["metadata"]["checkpoint_path"],
            "output_dir": str(output_dir),
            "summary_out": str(summary_path),
            "resume_predictions": True,
            "force_cpu": True,
        }
    )

    recovered_summary = json.loads(summary_path.read_text())
    metadata = json.loads((output_dir / "checkpoint_metadata.json").read_text())
    assert recovery["status"] == "pass"
    assert recovery["target_records"] == 5
    assert recovery["prediction_resume"]["existing_rows"] == 2
    assert metadata["target_records"] == 5
    assert metadata["recovery"]["source_checkpoint_path"] == train_summary["metadata"]["checkpoint_path"]
    assert recovered_summary["schema"] == "streaming_idm_train_summary.v1"
    assert len(pseudo_path.read_text().splitlines()) == 5


def test_streaming_idm_recovers_outputs_with_parallel_prediction_workers(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_a = tmp_path / "target_a.jsonl"
    target_b = tmp_path / "target_b.jsonl"
    output_dir = tmp_path / "idm_parallel_recover"
    summary_path = tmp_path / "parallel_recovered_summary.json"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    _write_jsonl(target_a, [_record(idx + 8, "eval") for idx in range(3)])
    _write_jsonl(target_b, [_record(idx + 11, "eval") for idx in range(2)])

    train_summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm_parallel_recover",
            "train_records": str(train_path),
            "target_records": str(target_a),
            "target_record_paths": [str(target_a), str(target_b)],
            "output_dir": str(output_dir),
            "summary_out": str(summary_path),
            "config_path": "test_parallel_recover_config",
            "source_namespace": "unit_d2e_stream",
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "eval_interval_epochs": 1,
            "batch_size": 4,
            "categorical_min_count": 1,
            "mouse_head_mode": "axis_softmax",
            "seed": 29,
            "force_cpu": True,
        }
    )
    pseudo_path = Path(train_summary["metadata"]["pseudo_label_path"])
    for rel in [
        "pseudolabels.jsonl",
        "predictions.jsonl",
        "metrics.json",
        "label_quality_report.json",
        "statistical_comparison.json",
        "checkpoint_metadata.json",
    ]:
        path = output_dir / rel
        if path.exists():
            path.unlink()
    summary_path.unlink()

    recovery = recover_streaming_idm_outputs_from_checkpoint(
        {
            "checkpoint_path": train_summary["metadata"]["checkpoint_path"],
            "output_dir": str(output_dir),
            "summary_out": str(summary_path),
            "prediction_workers": 2,
            "eval_batch_size": 3,
            "validate_pseudolabels": False,
            "force_cpu": True,
        }
    )

    recovered_summary = json.loads(summary_path.read_text())
    metadata = json.loads((output_dir / "checkpoint_metadata.json").read_text())
    assert recovery["status"] == "pass"
    assert recovery["target_records"] == 5
    assert recovery["prediction_resume"]["write_mode"] == "parallel_parts"
    assert recovery["prediction_resume"]["workers"] == 2
    assert recovery["prediction_resume"]["pseudolabel_validation"] is False
    assert metadata["target_records"] == 5
    assert recovered_summary["prediction_resume"]["workers"] == 2
    assert recovered_summary["prediction_resume"]["pseudolabel_validation"] is False
    assert len(pseudo_path.read_text().splitlines()) == 5
    assert len((output_dir / "predictions.jsonl").read_text().splitlines()) == 5


def test_streaming_idm_recovers_action_history_outputs_with_parallel_prediction_workers(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_a = tmp_path / "target_history_a.jsonl"
    target_b = tmp_path / "target_history_b.jsonl"
    output_dir = tmp_path / "idm_history_parallel_recover"
    summary_path = tmp_path / "history_parallel_recovered_summary.json"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    rows_a = [
        {**_record(idx + 8, "eval"), "recording_id": "d2e_480p:Apex/rec_a", "sequence_id": f"d2e_480p:Apex/rec_a#{idx:06d}"}
        for idx in range(3)
    ]
    rows_b = [
        {**_record(idx + 11, "eval"), "recording_id": "d2e_480p:Apex/rec_b", "sequence_id": f"d2e_480p:Apex/rec_b#{idx:06d}"}
        for idx in range(2)
    ]
    _write_jsonl(target_a, rows_a)
    _write_jsonl(target_b, rows_b)

    train_summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm_history_parallel_recover",
            "train_records": str(train_path),
            "target_records": str(target_a),
            "target_record_paths": [str(target_a), str(target_b)],
            "output_dir": str(output_dir),
            "summary_out": str(summary_path),
            "config_path": "test_history_parallel_recover_config",
            "source_namespace": "unit_d2e_stream",
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "eval_interval_epochs": 1,
            "batch_size": 4,
            "categorical_min_count": 1,
            "mouse_head_mode": "axis_softmax",
            "action_history_len": 2,
            "action_history_parallel_by_path": True,
            "prediction_workers": 2,
            "validate_pseudolabels": False,
            "seed": 37,
            "force_cpu": True,
        }
    )
    for rel in [
        "pseudolabels.jsonl",
        "predictions.jsonl",
        "metrics.json",
        "label_quality_report.json",
        "statistical_comparison.json",
        "checkpoint_metadata.json",
    ]:
        path = output_dir / rel
        if path.exists():
            path.unlink()
    summary_path.unlink()

    recovery = recover_streaming_idm_outputs_from_checkpoint(
        {
            "checkpoint_path": train_summary["metadata"]["checkpoint_path"],
            "output_dir": str(output_dir),
            "summary_out": str(summary_path),
            "prediction_workers": 2,
            "action_history_parallel_by_path": True,
            "action_history_seed_state_mode": "empty",
            "force_cpu": True,
            "validate_pseudolabels": False,
        }
    )

    recovered_summary = json.loads(summary_path.read_text())
    metadata = json.loads((output_dir / "checkpoint_metadata.json").read_text())
    assert recovery["status"] == "pass"
    assert recovery["target_records"] == 5
    assert recovery["prediction_resume"]["write_mode"] == "parallel_parts"
    assert recovery["prediction_resume"]["workers"] == 2
    assert recovery["prediction_resume"]["action_history_seed_state_summary"]["source"] == "empty"
    assert recovered_summary["prediction_resume"]["workers"] == 2
    assert metadata["action_history_len"] == 2
    assert metadata["action_history_feedback"] == "autoregressive_predicted"
    assert len((output_dir / "predictions.jsonl").read_text().splitlines()) == 5


def test_streaming_idm_train_summary_uses_parallel_final_prediction(tmp_path: Path):
    if not torch_available():
        pytest.skip("torch extra is not installed")
    train_path = tmp_path / "train.jsonl"
    target_a = tmp_path / "target_a.jsonl"
    target_b = tmp_path / "target_b.jsonl"
    output_dir = tmp_path / "idm_parallel_train_predict"
    _write_jsonl(train_path, [_record(idx, "train_core") for idx in range(8)])
    _write_jsonl(target_a, [_record(idx + 8, "eval") for idx in range(3)])
    _write_jsonl(target_b, [_record(idx + 11, "eval") for idx in range(2)])

    summary = train_streaming_idm(
        {
            "model_name": "tiny_streaming_idm_parallel_train_predict",
            "train_records": str(train_path),
            "target_records": str(target_a),
            "target_record_paths": [str(target_a), str(target_b)],
            "output_dir": str(output_dir),
            "config_path": "test_parallel_train_predict_config",
            "source_namespace": "unit_d2e_stream",
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "hidden_dim": 8,
            "depth": 1,
            "epochs": 1,
            "eval_interval_epochs": 1,
            "batch_size": 4,
            "eval_batch_size": 3,
            "categorical_min_count": 1,
            "mouse_head_mode": "axis_softmax",
            "prediction_workers": 2,
            "validate_pseudolabels": False,
            "seed": 31,
            "force_cpu": True,
        }
    )

    assert summary["metadata"]["target_records"] == 5
    assert summary["prediction_resume"]["write_mode"] == "parallel_parts"
    assert summary["prediction_resume"]["workers"] == 2
    assert summary["prediction_resume"]["pseudolabel_validation"] is False
    assert len(Path(summary["metadata"]["pseudo_label_path"]).read_text().splitlines()) == 5
    assert len(Path(summary["predictions_path"]).read_text().splitlines()) == 5


def test_distributed_runtime_passes_configured_timeout(monkeypatch):
    calls = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def set_device(local_rank):
            calls["set_device"] = local_rank

    class FakeDistributed:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def init_process_group(**kwargs):
            calls["init_kwargs"] = kwargs

    class FakeTorch:
        cuda = FakeCuda()
        distributed = FakeDistributed()

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")

    dist = _distributed_runtime(FakeTorch(), {"distributed_timeout_seconds": 21600})

    assert calls["set_device"] == 2
    assert calls["init_kwargs"]["backend"] == "nccl"
    assert calls["init_kwargs"]["timeout"].total_seconds() == 21600
    assert dist["timeout_seconds"] == 21600


def test_distributed_runtime_preserves_timeout_when_already_initialized(monkeypatch):
    calls = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def set_device(local_rank):
            calls["set_device"] = local_rank

    class FakeDistributed:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def init_process_group(**_kwargs):
            raise AssertionError("process group should not be initialized twice")

    class FakeTorch:
        cuda = FakeCuda()
        distributed = FakeDistributed()

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")

    dist = _distributed_runtime(FakeTorch(), {"distributed_timeout_seconds": 86400})

    assert calls["set_device"] == 1
    assert dist["enabled"] is True
    assert dist["timeout_seconds"] == 86400


def test_streaming_idm_stats_can_scan_multiple_record_files(tmp_path: Path):
    shard_a = tmp_path / "shard_a.jsonl"
    shard_b = tmp_path / "shard_b.jsonl"
    _write_jsonl(shard_a, [_record(idx, "train_core") for idx in range(3)])
    _write_jsonl(shard_b, [_record(idx + 3, "train_core") for idx in range(3)])

    stats = scan_streaming_idm_stats(
        [shard_a, shard_b],
        feature_mode="summary_compact_grid8_shift_surface_time",
        categorical_min_count=1,
        num_workers=1,
    )

    assert stats["num_examples"] == 6
    assert stats["input_dim"] > 0
    assert stats["source_ids"] == ["d2e_480p"]
    assert "KEY_PRESS_87" in stats["category_vocab"]
    assert stats["last_tokens_by_recording"]["d2e_480p:Apex/rec"]
