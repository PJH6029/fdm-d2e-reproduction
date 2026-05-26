from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.paper_idm_metrics import build_paper_idm_metrics
from fdm_d2e.io_utils import write_json, write_jsonl
from fdm_d2e.reporting.g005_idm_paper_target import validate_g005_idm_paper_target


def test_paper_idm_metrics_match_d2e_like_token_counts(tmp_path: Path):
    preds = tmp_path / "preds.jsonl"
    targets = tmp_path / "targets.jsonl"
    write_jsonl(
        preds,
        [
            {"sequence_id": "r#0", "predicted_tokens": ["KEY_PRESS_87", "MOUSE_LEFT_DOWN", "MOUSE_DX_P1", "MOUSE_DY_Z0"]},
            {"sequence_id": "r#1", "predicted_tokens": ["KEY_PRESS_65", "MOUSE_DX_P2", "MOUSE_DY_P1"]},
            {"sequence_id": "r#2", "predicted_tokens": ["MOUSE_DX_N1", "MOUSE_DY_P1"]},
        ],
    )
    write_jsonl(
        targets,
        [
            {
                "sequence_id": "r#0",
                "eval_split_tags": ["temporal"],
                "ground_truth_tokens": ["KEY_PRESS_87", "MOUSE_LEFT_DOWN", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
            },
            {
                "sequence_id": "r#1",
                "eval_split_tags": ["heldout_recording"],
                "ground_truth_tokens": ["MOUSE_DX_P2", "MOUSE_DY_P1"],
            },
            {
                "sequence_id": "r#2",
                "eval_split_tags": ["heldout_game"],
                "ground_truth_tokens": ["MOUSE_DX_N1", "MOUSE_DY_P1"],
            },
        ],
    )
    payload = build_paper_idm_metrics(
        prediction_paths=[preds],
        target_paths=[targets],
        split_tags=["temporal", "heldout_recording", "heldout_game"],
    )
    all_metrics = payload["groups"]["all"]
    assert payload["status"] == "pass"
    assert all_metrics["paper_compatible"]["keyboard"]["key_accuracy"] == 0.5
    assert all_metrics["paper_compatible"]["mouse_button"]["button_accuracy"] == 1.0
    assert all_metrics["paper_compatible"]["mouse_move"]["pearson_x"] == 1.0
    assert all_metrics["strict_local"]["mouse_button"]["f1"] == 1.0
    assert payload["groups"]["eval_split:heldout_game"]["rows"] == 1


def test_g005_idm_paper_target_audit_passes_with_full_evidence(tmp_path: Path):
    paper_metrics = tmp_path / "artifacts/idm/paper_metrics.json"
    contract = tmp_path / "artifacts/eval/contract.json"
    metadata = tmp_path / "outputs/model/checkpoint_metadata.json"
    summary = tmp_path / "artifacts/idm/summary.json"
    run = tmp_path / "artifacts/idm/run.json"
    split = tmp_path / "artifacts/eval/split_summary.json"
    checkpoint = tmp_path / "outputs/model/checkpoint.pt"
    gpu = tmp_path / "artifacts/idm/gpu.csv"

    write_json(
        contract,
        {
            "status": "pass",
            "target_sequence": {
                "phase_1": {
                    "primary_targets": {
                        "pearson_x": 0.8,
                        "pearson_y": 0.7,
                        "keyboard_accuracy": 0.7,
                        "mouse_button_accuracy": 0.9,
                        "scale_ratio_x_max": 1.3,
                        "scale_ratio_y_max": 1.4,
                    }
                }
            },
            "official_metric_protocol": {"empty_bins_as_correct": False},
        },
    )
    write_json(
        paper_metrics,
        {
            "status": "pass",
            "groups": {
                "all": {
                    "paper_compatible": {
                        "empty_bins_as_correct": False,
                        "mouse_move": {"pearson_x": 0.81, "pearson_y": 0.71, "scale_ratio_x": 1.2, "scale_ratio_y": 1.3},
                        "keyboard": {"key_accuracy": 0.75},
                        "mouse_button": {"button_accuracy": 0.95},
                    },
                    "strict_local": {
                        "mouse_button": {
                            "f1": 0.04,
                            "no_button_false_positive_rate": 0.01,
                        }
                    },
                },
                "eval_split:temporal": {"paper_compatible": {"mouse_move": {}, "keyboard": {}, "mouse_button": {}}},
                "eval_split:heldout_recording": {"paper_compatible": {"mouse_move": {}, "keyboard": {}, "mouse_button": {}}},
                "eval_split:heldout_game": {"paper_compatible": {"mouse_move": {}, "keyboard": {}, "mouse_button": {}}},
            },
        },
    )
    write_json(metadata, {"train_records": 10, "target_records": 9, "distributed": {"world_size": 4}})
    write_json(summary, {"schema": "streaming_idm_train_summary.v1"})
    write_json(
        run,
        {
            "exit_code": 0,
            "nproc_per_node": 4,
            "gpu_monitor_status": {"covers_expected_gpus": True, "unique_gpu_indices": ["0", "1", "2", "3"], "rows": 4},
        },
    )
    write_json(
        split,
        {
            "status": "pass",
            "outputs": [
                {"split": "temporal", "status": "pass"},
                {"split": "heldout_recording", "status": "pass"},
                {"split": "heldout_game", "status": "pass"},
            ],
        },
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    gpu.parent.mkdir(parents=True, exist_ok=True)
    gpu.write_text("sample_unix,parent_pid,timestamp,index\n1,2,now,0\n1,2,now,1\n1,2,now,2\n1,2,now,3\n")

    config = {
        "expected_gpus": 4,
        "min_gpu_monitor_rows": 4,
        "min_train_records": 10,
        "min_target_records": 9,
        "paths": {
            "gidm_baseline_contract": "artifacts/eval/contract.json",
            "paper_metrics": "artifacts/idm/paper_metrics.json",
            "checkpoint": "outputs/model/checkpoint.pt",
            "checkpoint_metadata": "outputs/model/checkpoint_metadata.json",
            "train_summary": "artifacts/idm/summary.json",
            "run_summary": "artifacts/idm/run.json",
            "split_stats_summary": "artifacts/eval/split_summary.json",
            "gpu_monitor": "artifacts/idm/gpu.csv",
        },
        "strict_local_targets": [
            {"name": "mouse_button_f1", "path": ["strict_local", "mouse_button", "f1"], "direction": "higher", "baseline": 0.01, "min_delta": 0.02},
            {"name": "no_button_false_positive_rate", "path": ["strict_local", "mouse_button", "no_button_false_positive_rate"], "direction": "lower", "target": 0.10},
        ],
    }
    payload = validate_g005_idm_paper_target(config, root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["aggregate_target_results"]


def test_g005_idm_paper_target_audit_rejects_non_official_state_metric_protocol(tmp_path: Path):
    write_json(
        tmp_path / "contract.json",
        {
            "status": "pass",
            "official_metric_protocol": {"empty_bins_as_correct": False},
            "target_sequence": {"phase_1": {"primary_targets": {"keyboard_accuracy": 0.7}}},
        },
    )
    write_json(
        tmp_path / "paper.json",
        {
            "status": "pass",
            "groups": {
                "all": {
                    "paper_compatible": {
                        "empty_bins_as_correct": True,
                        "keyboard": {"key_accuracy": 0.9},
                    }
                }
            },
        },
    )
    payload = validate_g005_idm_paper_target(
        {
            "expected_gpus": 4,
            "paths": {
                "gidm_baseline_contract": "contract.json",
                "paper_metrics": "paper.json",
            },
            "paper_metrics": {
                "target_path": "outputs/data/d2e_state_corpus_shards_accel64/shard_*/target_all_eval.jsonl",
            },
        },
        root=tmp_path,
    )
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "paper_metric_protocol_empty_bins_mismatch" in codes
    assert "paper_metric_target_uses_held_state_corpus" in codes


def test_g005_idm_paper_target_audit_fails_missing_paper_target(tmp_path: Path):
    write_json(
        tmp_path / "contract.json",
        {
            "status": "pass",
            "target_sequence": {"phase_1": {"primary_targets": {"keyboard_accuracy": 0.9}}},
        },
    )
    write_json(
        tmp_path / "paper.json",
        {
            "status": "pass",
            "groups": {"all": {"paper_compatible": {"keyboard": {"key_accuracy": 0.1}}}},
        },
    )
    payload = validate_g005_idm_paper_target(
        {
            "expected_gpus": 4,
            "paths": {
                "gidm_baseline_contract": "contract.json",
                "paper_metrics": "paper.json",
            },
        },
        root=tmp_path,
    )
    assert payload["status"] == "fail"
    assert any(item["code"] == "paper_target_not_met" for item in payload["findings"])
