from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import read_json
from fdm_d2e.training.masked_diffusion_idm_trainer import (
    _button_class_conditional_prior_offsets,
    _button_probabilities_from_output,
    _calibrate_button_event_budget,
    _calibrate_button_event_budget_multiplier,
    _calibrate_button_event_threshold,
    _predict_factorized_tokens,
    _predict_factorized_tokens_batch,
    torch_available,
    train_masked_diffusion_idm,
    video_feature_vector,
)


def _row(idx: int, *, split: str) -> dict:
    return {
        "sequence_id": f"unit#{idx:03d}",
        "split": split,
        "eval_split_tags": ["temporal"] if split != "train_core" else [],
        "frame": {"features": [idx / 10, 1.0], "width": 854, "height": 480},
        "next_frame_features": [idx / 10 + 0.1, 0.5],
        "frame_delta_features": [0.1, -0.1],
        "ground_truth_tokens": ["KEY_PRESS_W", "MOUSE_DX_P1", "MOUSE_DY_Z0"] if idx % 2 == 0 else [],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_video_feature_vector_uses_configured_paths_and_padding():
    row = _row(1, split="train_core")
    features = video_feature_vector(row, feature_paths=["frame.features", "next_frame_features"], dim=6)
    assert features[:4] == [0.1, 1.0, 0.2, 0.5]
    assert features[4:] == [0.0, 0.0]


def test_video_feature_vector_flattens_luma_window_tokens():
    row = _row(2, split="train_core")
    row["compact_luma_window"] = [[0.1, 0.2], [0.3, 0.4]]
    row["compact_luma_window_mask"] = [1, 0]
    row["frame"]["stats"] = {"b": 0.6, "a": 0.5}
    features = video_feature_vector(
        row,
        feature_paths=["compact_luma_window", "compact_luma_window_mask", "frame.stats"],
        dim=10,
    )
    assert features == [0.1, 0.2, 0.3, 0.4, 1.0, 0.0, 0.5, 0.6, 0.0, 0.0]


def test_button_event_calibration_uses_dynamic_probability_thresholds():
    rows = [
        {"button_event_prob": 0.491, "button_event_label": 1},
        {"button_event_prob": 0.489, "button_event_label": 1},
        {"button_event_prob": 0.487, "button_event_label": 0},
        {"button_event_prob": 0.480, "button_event_label": 0},
        {"button_event_prob": 0.470, "button_event_label": 0},
        {"button_event_prob": 0.460, "button_event_label": 0},
    ]
    coarse = _calibrate_button_event_threshold(
        rows,
        candidates=[0.45, 0.5],
        max_false_positive_rate=0.25,
        beta=2.0,
        dynamic_thresholds=False,
    )
    dynamic = _calibrate_button_event_threshold(
        rows,
        candidates=[0.45, 0.5],
        max_false_positive_rate=0.25,
        beta=2.0,
        dynamic_thresholds=True,
        dynamic_max_candidates=8,
    )
    assert coarse["selected_row"]["recall"] == 0.0
    assert dynamic["selected_row"]["recall"] == 1.0
    assert dynamic["selected_row"]["false_positive_rate"] <= 0.25
    assert dynamic["candidate_count"] > len([0.45, 0.5])


def test_button_event_calibration_can_jointly_gate_token_confidence():
    rows = [
        {"button_event_prob": 0.60, "button_probs": [0.91, 0.10], "button_event_label": 1},
        {"button_event_prob": 0.59, "button_probs": [0.86, 0.20], "button_event_label": 1},
        {"button_event_prob": 0.61, "button_probs": [0.40, 0.30], "button_event_label": 0},
        {"button_event_prob": 0.58, "button_probs": [0.35, 0.20], "button_event_label": 0},
        {"button_event_prob": 0.57, "button_probs": [0.25, 0.10], "button_event_label": 0},
        {"button_event_prob": 0.56, "button_probs": [0.15, 0.12], "button_event_label": 0},
    ]
    event_only = _calibrate_button_event_threshold(
        rows,
        candidates=[0.50],
        max_false_positive_rate=0.10,
        beta=2.0,
    )
    joint = _calibrate_button_event_threshold(
        rows,
        candidates=[0.50],
        max_false_positive_rate=0.10,
        beta=2.0,
        min_token_candidates=[0.0, 0.50, 0.80],
        calibrate_min_token_probability=True,
    )
    assert event_only["selected_row"]["false_positive_rate"] == 1.0
    assert joint["selected_min_token_probability"] == 0.80
    assert joint["selected_row"]["recall"] == 1.0
    assert joint["selected_row"]["false_positive_rate"] == 0.0


def test_button_event_budget_uses_train_event_rate_without_target_labels():
    probability_rows = [
        {"button_event_prob": 0.90, "button_probs": [0.90]},
        {"button_event_prob": 0.80, "button_probs": [0.80]},
        {"button_event_prob": 0.70, "button_probs": [0.70]},
        {"button_event_prob": 0.60, "button_probs": [0.60]},
        {"button_event_prob": 0.50, "button_probs": [0.50]},
    ]
    rate_rows = [
        {"ground_truth_tokens": ["MOUSE_LEFT_DOWN"]},
        {"ground_truth_tokens": []},
        {"ground_truth_tokens": []},
        {"ground_truth_tokens": []},
        {"ground_truth_tokens": []},
    ]
    budget = _calibrate_button_event_budget(
        probability_rows,
        rate_rows=rate_rows,
        button_vocab=["MOUSE_LEFT_DOWN"],
        config={
            "button_event_threshold": 0.0,
            "button_event_min_token_probability": 0.0,
            "button_event_budget_rate_multiplier": 1.0,
        },
    )
    assert budget["rate_source_positive_rate"] == 0.2
    assert budget["max_forced_events"] == 1
    assert budget["score_threshold"] == 0.81
    assert budget["selected_preview"][0]["index"] == 0


def test_button_event_budget_can_rank_all_scores_when_absolute_gate_does_not_transfer():
    probability_rows = [
        {"button_event_prob": 0.40, "button_probs": [0.90]},
        {"button_event_prob": 0.30, "button_probs": [0.80]},
        {"button_event_prob": 0.20, "button_probs": [0.70]},
    ]
    rate_rows = [
        {"ground_truth_tokens": ["MOUSE_LEFT_DOWN"]},
        {"ground_truth_tokens": []},
        {"ground_truth_tokens": []},
    ]
    strict = _calibrate_button_event_budget(
        probability_rows,
        rate_rows=rate_rows,
        button_vocab=["MOUSE_LEFT_DOWN"],
        config={
            "button_event_threshold": 0.99,
            "button_event_min_token_probability": 0.99,
            "button_event_budget_rate_multiplier": 1.0,
        },
    )
    relaxed = _calibrate_button_event_budget(
        probability_rows,
        rate_rows=rate_rows,
        button_vocab=["MOUSE_LEFT_DOWN"],
        config={
            "button_event_threshold": 0.99,
            "button_event_min_token_probability": 0.99,
            "button_event_budget_rate_multiplier": 1.0,
            "button_event_budget_rank_all_scores": True,
        },
    )
    assert strict["threshold_candidate_count"] == 0
    assert strict["score_threshold"] == 2.0
    assert relaxed["threshold_candidate_count"] == 3
    assert abs(relaxed["score_threshold"] - 0.36) < 1e-9


def test_button_event_budget_multiplier_selects_calibration_recall_under_fpr_cap():
    probability_rows = [
        {"button_event_prob": 0.90, "button_probs": [0.90], "button_event_label": 1},
        {"button_event_prob": 0.80, "button_probs": [0.80], "button_event_label": 1},
        {"button_event_prob": 0.70, "button_probs": [0.70], "button_event_label": 0},
        {"button_event_prob": 0.60, "button_probs": [0.60], "button_event_label": 0},
        {"button_event_prob": 0.50, "button_probs": [0.50], "button_event_label": 0},
    ]
    rate_rows = [
        {"ground_truth_tokens": ["MOUSE_LEFT_DOWN"]},
        {"ground_truth_tokens": []},
        {"ground_truth_tokens": []},
        {"ground_truth_tokens": []},
        {"ground_truth_tokens": []},
    ]
    payload = _calibrate_button_event_budget_multiplier(
        probability_rows,
        rate_rows=rate_rows,
        button_vocab=["MOUSE_LEFT_DOWN"],
        config={
            "button_event_threshold": 0.0,
            "button_event_min_token_probability": 0.0,
            "button_event_budget_rate_multiplier_candidates": [1.0, 2.0, 3.0],
            "button_event_budget_cap_rate": 1.0,
            "button_event_budget_applies_to_all_buttons": True,
            "button_threshold": 0.0,
            "button_event_budget_calibration_max_no_button_fpr": 0.34,
        },
    )
    assert payload["status"] == "pass"
    assert payload["selected_multiplier"] == 2.0
    assert payload["selected_row"]["metrics"]["recall"] == 1.0
    assert payload["selected_row"]["metrics"]["false_positive_rate"] <= 0.34


def test_button_class_prior_offsets_boost_rare_transition_tokens_without_target_labels():
    rows = []
    for idx in range(9):
        row = _row(idx, split="train_core")
        row["ground_truth_tokens"] = ["MOUSE_LEFT_DOWN"]
        rows.append(row)
    rare = _row(99, split="train_core")
    rare["ground_truth_tokens"] = ["MOUSE_RIGHT_DOWN"]
    rows.append(rare)
    offsets = _button_class_conditional_prior_offsets(
        rows,
        button_vocab=["MOUSE_LEFT_DOWN", "MOUSE_RIGHT_DOWN"],
        config={"button_class_conditional_prior_correction": True, "button_class_conditional_prior_alpha": 1.0},
    )
    assert len(offsets) == 2
    assert offsets[1] > offsets[0]


def test_button_class_prior_correction_keeps_event_probability_and_redistributes_tokens():
    if not torch_available():
        return
    import torch

    logits = torch.tensor([[0.0, 4.0, 3.0]], dtype=torch.float32)
    raw_probs, raw_event = _button_probabilities_from_output(
        {"button_class": logits, "button": None, "button_event": None},
        torch,
        config={"button_probability_source": "button_class", "button_event_probability_source": "button_class"},
    )
    corrected_probs, corrected_event = _button_probabilities_from_output(
        {"button_class": logits, "button": None, "button_event": None},
        torch,
        config={
            "button_probability_source": "button_class",
            "button_event_probability_source": "button_class",
            "button_class_conditional_prior_correction": True,
            "button_class_conditional_logit_offsets": [-0.8, 0.8],
        },
    )
    assert corrected_event == raw_event
    assert sum(corrected_probs) == pytest.approx(raw_event)
    assert raw_probs[0] > raw_probs[1]
    assert corrected_probs[1] > corrected_probs[0]


def test_batched_factorized_prediction_matches_single_row_path():
    if not torch_available():
        return
    import torch

    class DummyFactorizedModel:
        def eval(self):
            return self

        def __call__(self, features):
            batch = features.shape[0]
            mouse_x = torch.zeros((batch, 49), dtype=torch.float32)
            mouse_y = torch.zeros((batch, 49), dtype=torch.float32)
            mouse_x[:, 24] = 5.0
            mouse_y[:, 24] = 5.0
            key = torch.stack([features[:, 0] * 4.0, 1.0 - features[:, 0]], dim=1)
            button_class = torch.stack(
                [
                    torch.full((batch,), -1.0),
                    features[:, 0] * 3.0,
                    (1.0 - features[:, 0]) * 3.0,
                ],
                dim=1,
            )
            return {
                "mouse_x": mouse_x,
                "mouse_y": mouse_y,
                "key": key,
                "button": None,
                "button_class": button_class,
                "button_event": None,
            }

    rows = [
        {"sequence_id": "a", "frame": {"features": [0.9], "width": 854, "height": 480}, "ground_truth_tokens": []},
        {"sequence_id": "b", "frame": {"features": [0.1], "width": 854, "height": 480}, "ground_truth_tokens": []},
    ]
    config = {
        "video_feature_paths": ["frame.features"],
        "video_feature_dim": 1,
        "key_threshold": 0.65,
        "button_threshold": 0.2,
        "button_probability_source": "button_class",
        "button_event_probability_source": "button_class",
        "max_predicted_keys": 2,
        "max_predicted_buttons": 1,
    }
    key_vocab = ["KEY_PRESS_W", "KEY_PRESS_A"]
    button_vocab = ["MOUSE_LEFT_DOWN", "MOUSE_RIGHT_DOWN"]
    device = torch.device("cpu")
    model = DummyFactorizedModel()
    single = [
        _predict_factorized_tokens(model, torch, row, config=config, key_vocab=key_vocab, button_vocab=button_vocab, device=device)
        for row in rows
    ]
    batched = _predict_factorized_tokens_batch(model, torch, rows, config=config, key_vocab=key_vocab, button_vocab=button_vocab, device=device)
    assert batched == single
    assert batched[0] == ["KEY_PRESS_W", "MOUSE_LEFT_DOWN"]
    assert batched[1] == ["KEY_PRESS_A", "MOUSE_RIGHT_DOWN"]


def test_train_masked_diffusion_idm_tiny_smoke(tmp_path: Path):
    if not torch_available():
        return
    train_path = tmp_path / "train.jsonl"
    target_path = tmp_path / "target.jsonl"
    _write_jsonl(train_path, [_row(i, split="train_core") for i in range(6)])
    _write_jsonl(target_path, [_row(i, split="eval") for i in range(6, 9)])
    summary = train_masked_diffusion_idm(
        {
            "model_name": "unit_masked_diffusion_idm",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "out"),
            "summary_out": str(tmp_path / "summary.json"),
            "max_train_rows": 6,
            "max_target_rows": 3,
            "max_action_tokens_per_bin": 4,
            "video_feature_paths": ["frame.features", "next_frame_features", "frame_delta_features"],
            "video_feature_dim": 6,
            "mask_probability": 0.75,
            "random_token_probability": 0.0,
            "diffusion_steps": 4,
            "hidden_dim": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "force_cpu": True,
            "seed": 11,
            "noop_loss_weight": 0.2,
            "keyboard_loss_weight": 2.0,
        }
    )
    assert summary["status"] == "pass"
    assert summary["loss_weights"]["noop_loss_weight"] == 0.2
    assert summary["loss_weights"]["keyboard_loss_weight"] == 2.0
    assert summary["train_rows"] == 6
    assert summary["target_rows"] == 3
    assert Path(summary["checkpoint_path"]).exists()
    assert Path(summary["predictions_path"]).exists()
    assert Path(summary["metrics_path"]).exists()
    assert len(Path(summary["predictions_path"]).read_text(encoding="utf-8").strip().splitlines()) == 3
    metrics = read_json(summary["metrics_path"])
    assert metrics["status"] == "pass"
    assert metrics["alignment"]["rows_seen"] == 3
    assert "masked-diffusion IDM" in summary["recipe_alignment"]


def test_train_factorized_masked_diffusion_idm_tiny_smoke(tmp_path: Path):
    if not torch_available():
        return
    train_path = tmp_path / "train_factorized.jsonl"
    target_path = tmp_path / "target_factorized.jsonl"
    rows = [_row(i, split="train_core") for i in range(8)]
    rows[1]["ground_truth_tokens"] = ["KEY_RELEASE_W", "MOUSE_LEFT_DOWN", "MOUSE_DX_N1", "MOUSE_DY_P1"]
    rows[3]["ground_truth_tokens"] = ["KEY_PRESS_A", "MOUSE_LEFT_UP", "MOUSE_DX_P1", "MOUSE_DY_N1"]
    _write_jsonl(train_path, rows)
    _write_jsonl(target_path, [_row(i, split="eval") for i in range(8, 11)])
    summary = train_masked_diffusion_idm(
        {
            "model_name": "unit_factorized_masked_diffusion_idm",
            "factorized_action_tokens": True,
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "out_factorized"),
            "summary_out": str(tmp_path / "summary_factorized.json"),
            "max_train_rows": 8,
            "max_target_rows": 3,
            "video_feature_paths": ["frame.features", "next_frame_features", "frame_delta_features"],
            "video_feature_dim": 6,
            "hidden_dim": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "force_cpu": True,
            "key_threshold": 0.99,
            "button_threshold": 0.99,
            "button_event_auxiliary": True,
            "button_event_loss_weight": 1.0,
            "button_event_threshold": 0.5,
            "button_event_force_topk": 1,
            "calibration_dynamic_thresholds": True,
            "calibration_dynamic_threshold_max_candidates": 8,
            "calibrate_thresholds": True,
            "factorized_calibration_fraction": 0.25,
            "factorized_calibration_max_rows": 2,
            "threshold_candidates": [0.25, 0.5, 0.75],
            "calibrate_per_token_thresholds": True,
        }
    )
    assert summary["schema"] == "factorized_masked_diffusion_idm_train_summary.v1"
    assert summary["status"] == "pass"
    assert summary["key_vocab_size"] >= 2
    assert summary["button_vocab_size"] >= 2
    assert summary["factorization"]["mouse_axis_bins"] == 49
    assert summary["threshold_calibration"]["status"] == "pass"
    assert summary["threshold_calibration"]["selected"]["key_threshold"] in {0.25, 0.5, 0.75}
    assert summary["threshold_calibration"]["per_token"]["status"] == "pass"
    assert summary["factorization"]["key_token_threshold_count"] == summary["key_vocab_size"]
    assert Path(summary["checkpoint_path"]).exists()
    metrics = read_json(summary["metrics_path"])
    assert metrics["status"] == "pass"
    assert metrics["alignment"]["rows_seen"] == 3
    assert "typed masked action-token planes" in summary["recipe_alignment"]


def test_train_factorized_masked_diffusion_idm_luma_cnn_tiny_smoke(tmp_path: Path):
    if not torch_available():
        return
    train_path = tmp_path / "train_luma_factorized.jsonl"
    target_path = tmp_path / "target_luma_factorized.jsonl"
    train_rows = []
    for i in range(8):
        row = _row(i, split="train_core")
        row["compact_luma_window"] = [[float(i + j + k) / 10.0 for k in range(4)] for j in range(2)]
        row["compact_luma_window_mask"] = [1.0, 1.0]
        if i in {1, 3}:
            row["ground_truth_tokens"] = ["MOUSE_LEFT_DOWN", "KEY_PRESS_A", "MOUSE_DX_P1", "MOUSE_DY_Z0"]
        train_rows.append(row)
    target_rows = []
    for i in range(8, 11):
        row = _row(i, split="eval")
        row["compact_luma_window"] = [[float(i + j + k) / 10.0 for k in range(4)] for j in range(2)]
        row["compact_luma_window_mask"] = [1.0, 1.0]
        target_rows.append(row)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(target_path, target_rows)
    summary = train_masked_diffusion_idm(
        {
            "model_name": "unit_factorized_masked_diffusion_idm_luma_cnn",
            "factorized_action_tokens": True,
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "out_luma_factorized"),
            "summary_out": str(tmp_path / "summary_luma_factorized.json"),
            "max_train_rows": 8,
            "max_target_rows": 3,
            "video_feature_paths": ["compact_luma_window", "compact_luma_window_mask", "frame.features"],
            "video_feature_dim": 12,
            "video_encoder_arch": "compact_luma_window_cnn",
            "luma_window_frames": 2,
            "luma_window_size": 2,
            "luma_encoder_channels": 4,
            "luma_encoder_pool_hw": 1,
            "luma_aux_hidden_dim": 4,
            "hidden_dim": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "video_encoder_pretrain_epochs": 1,
            "video_encoder_pretrain_lr": 0.001,
            "video_encoder_pretrain_mask_probability": 0.5,
            "video_reconstruction_aux_weight": 0.05,
            "force_cpu": True,
            "key_threshold": 0.99,
            "button_threshold": 0.99,
            "button_event_auxiliary": True,
            "button_event_loss_weight": 1.0,
            "button_transition_softmax": True,
            "button_probability_source": "button_class",
            "button_event_probability_source": "button_class",
            "button_class_loss_weight": 1.0,
            "button_class_no_button_weight": 0.2,
            "button_event_threshold": 0.5,
            "button_event_force_topk": 1,
            "button_event_budgeted_unmasking": True,
            "button_event_budget_rate_multiplier": 1.0,
            "button_event_budget_rate_multiplier_candidates": [1.0, 2.0],
            "button_event_budget_applies_to_all_buttons": True,
            "button_event_budget_rank_all_scores": True,
            "calibrate_thresholds": True,
            "factorized_calibration_fraction": 0.25,
            "factorized_calibration_max_rows": 2,
            "threshold_candidates": [0.25, 0.5, 0.75],
            "calibrate_per_token_thresholds": True,
        }
    )
    assert summary["status"] == "pass"
    assert summary["threshold_calibration"]["status"] == "pass"
    assert summary["threshold_calibration"]["per_token"]["button_event_threshold"]["status"] == "pass"
    assert summary["button_event_budget"]["status"] == "pass"
    assert summary["button_event_budget"]["multiplier_calibration"]["status"] == "pass"
    assert summary["factorization"]["button_event_auxiliary"] is True
    assert summary["factorization"]["button_transition_softmax"] is True
    assert summary["factorization"]["button_probability_source"] == "button_class"
    assert summary["factorization"]["button_event_probability_source"] == "button_class"
    assert any("button_class" in row for row in summary["history"])
    assert any("video_reconstruction" in row for row in summary["history"])
    assert summary["video_encoder_pretrain_history"]
    assert summary["video_encoder_pretrain_history"][0]["video_reconstruction_loss"] >= 0.0
    assert summary["factorization"]["video_encoder_pretrain_objective"] == "masked_luma_reconstruction"
    assert summary["factorization"]["video_reconstruction_aux_weight"] == 0.05
    assert "button_event_min_token_probability" in summary["factorization"]
    assert "button_event_budget_score_threshold" in summary["factorization"]
    assert summary["factorization"]["button_event_budget_applies_to_all_buttons"] is True
    assert summary["factorization"]["button_event_budget_rank_all_scores"] is True
    assert Path(summary["checkpoint_path"]).exists()
    assert read_json(summary["metrics_path"])["alignment"]["rows_seen"] == 3
