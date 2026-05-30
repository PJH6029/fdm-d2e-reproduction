from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import read_json
from fdm_d2e.eval.state_prediction_events import event_tokens_from_state_prediction
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
from fdm_d2e.training.masked_diffusion_idm import canonical_fdm1_action_tokens
from fdm_d2e.training.temporal_masked_diffusion_idm_trainer import (
    _adapt_temporal_family_budget_to_unlabeled_distribution,
    _apply_candidate_score_reranker_to_probability_rows,
    _build_vocab_for_config,
    _calibrate_temporal_family_non_noop_budget,
    _calibrate_temporal_non_noop_budget,
    _candidate_family_diagnostics,
    _candidate_token_prior_weights,
    _family_token_presence_rank_loss,
    _fit_candidate_score_reranker,
    _mouse_axis_class_vocab,
    _maybe_tensorize_features,
    _precompute_features,
    _precompute_features_with_distributed_cache,
    _temporal_calibration_split_indices,
    _temporal_button_class_targets,
    _temporal_candidate_rows_from_probabilities,
    _temporal_center_candidates,
    _temporal_family_count_targets,
    _temporal_family_class_targets,
    _temporal_family_token_presence_targets,
    _target_slots,
    _target_slots_for_config,
    _token_presence_targets,
    _tokens_from_family_budget_candidates,
    _tokens_from_non_noop_candidates,
    train_temporal_masked_diffusion_idm,
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


def _write_pgm(path: Path, values: list[int], *, width: int = 2, height: int = 2) -> None:
    payload = bytes(max(0, min(255, int(value))) for value in values)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + payload)


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


def test_temporal_raw_video_feature_source_reads_frame_offsets(tmp_path: Path):
    _write_pgm(tmp_path / "frame_000000.ppm", [0, 64, 128, 255])
    _write_pgm(tmp_path / "frame_000001.ppm", [255, 128, 64, 0])
    row = {
        "sequence_id": "raw-video#0",
        "frame": {"path": str(tmp_path / "frame_000000.ppm"), "index": 0, "width": 2, "height": 2},
        "ground_truth_tokens": [],
    }
    features = _precompute_features(
        [row],
        config={
            "video_feature_source": "raw_frames",
            "raw_video_image_size": 2,
            "raw_video_frame_offsets": [0, 1],
            "raw_video_missing_frame_policy": "error",
            "video_feature_dim": 8,
        },
    )
    assert len(features) == 1
    assert features[0] == [0.0, 64 / 255.0, 128 / 255.0, 1.0, 1.0, 128 / 255.0, 64 / 255.0, 0.0]


def test_tensorize_features_stacks_raw_video_tensor_storage():
    if not torch_available():
        return
    import torch

    features = [torch.tensor([0.0, 1.0], dtype=torch.float32), torch.tensor([0.5, 0.25], dtype=torch.float32)]
    tensor = _maybe_tensorize_features(
        torch,
        features,
        config={"precompute_features_as_tensor": True, "precompute_feature_tensor_dtype": "float16"},
        split_name="fit",
    )
    assert tuple(tensor.shape) == (2, 2)
    assert tensor.dtype == torch.float16
    assert tensor.tolist() == [[0.0, 1.0], [0.5, 0.25]]


def test_temporal_target_slots_support_held_state_action_tokens():
    row = {
        "prior_action_tokens": ["KEY_DOWN_87", "MOUSE_LEFT_DOWN"],
        "ground_truth_tokens": ["KEY_RELEASE_87", "KEY_PRESS_65", "MOUSE_DX_P2", "MOUSE_DY_Z0"],
    }
    slots = _target_slots_for_config(
        row,
        max_slots=8,
        config={
            "action_target_mode": "held_state_tokens",
            "action_mouse_tokenization": "d2e_metric_aggregate_decomposed_bins",
        },
        preserve_pad_slots=True,
    )

    assert "KEY_DOWN_87" not in slots
    assert "KEY_DOWN_65" in slots
    assert "MOUSE_LEFT_DOWN" in slots
    assert "MOUSE_DX_P2" in slots


def test_state_prediction_eventification_from_row_prior_produces_transitions():
    converted = event_tokens_from_state_prediction(
        ["MOUSE_DX_P1", "MOUSE_DY_Z0", "KEY_DOWN_65", "MOUSE_LEFT_DOWN"],
        prior_tokens=["KEY_DOWN_87", "MOUSE_LEFT_DOWN"],
    )

    assert "KEY_PRESS_65" in converted
    assert "KEY_RELEASE_87" in converted
    assert "MOUSE_LEFT_DOWN" not in converted
    assert "MOUSE_DX_P1" in converted


def test_distributed_raw_video_feature_cache_loads_ordered_chunks(tmp_path: Path):
    if not torch_available():
        return
    import torch

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for idx in range(4):
        _write_pgm(frame_dir / f"frame_{idx:06d}.ppm", [idx, idx + 1, idx + 2, idx + 3])
    rows = [
        {
            "sequence_id": f"raw-video-cache#{idx}",
            "frame": {"path": str(frame_dir / f"frame_{idx:06d}.ppm"), "index": idx, "width": 2, "height": 2},
        }
        for idx in range(4)
    ]
    config = {
        "video_feature_source": "raw_frames",
        "raw_video_image_size": 2,
        "raw_video_frame_offsets": [0],
        "raw_video_missing_frame_policy": "zero",
        "video_feature_dim": 4,
        "distributed_feature_cache_dir": str(tmp_path / "feature_cache"),
        "raw_video_feature_tensor_dtype": "float32",
    }
    # The helper is normally synchronized by torch.distributed.  Pre-create the
    # other-rank chunk so this unit test can exercise rank0 ordering without
    # launching a process group.
    split_dir = next((tmp_path / "feature_cache").glob("fit-*"), None)
    assert split_dir is None
    # First call rank0 once without distributed caching to compute the expected
    # row values, then create the rank1 chunk matching the helper contract.
    expected = _precompute_features(rows, config=config)
    split_name = "fit"
    from fdm_d2e.training.temporal_masked_diffusion_idm_trainer import _feature_cache_fingerprint

    fingerprint = _feature_cache_fingerprint(rows, split_name=split_name, config=config)
    split_dir = tmp_path / "feature_cache" / f"{split_name}-{fingerprint}"
    split_dir.mkdir(parents=True)
    torch.save(
        {
            "schema": "temporal_raw_video_feature_cache_chunk.v1",
            "split_name": split_name,
            "fingerprint": fingerprint,
            "rank": 1,
            "world_size": 2,
            "start": 2,
            "end": 4,
            "feature_dim": 4,
            "dtype": "float32",
            "features": torch.tensor(expected[2:4], dtype=torch.float32),
        },
        split_dir / "chunk_rank001_of_002.pt",
    )
    features = _precompute_features_with_distributed_cache(
        rows,
        config=config,
        split_name=split_name,
        torch=torch,
        distributed=True,
        rank=0,
        world_size=2,
        dist=None,
    )
    assert tuple(features.shape) == (4, 4)
    assert features.tolist() == [pytest.approx(row) for row in expected]
    assert (split_dir / "summary.json").exists()
    cached_single_rank = _precompute_features_with_distributed_cache(
        rows,
        config=config,
        split_name=split_name,
        torch=torch,
        distributed=False,
        rank=0,
        world_size=1,
        dist=None,
    )
    assert tuple(cached_single_rank.shape) == (4, 4)
    assert cached_single_rank.tolist() == [pytest.approx(row) for row in expected]
    cached_prefix = _precompute_features_with_distributed_cache(
        rows[:2],
        config={**config, "allow_prefix_feature_cache_reuse": True, "prefix_feature_cache_reuse_splits": ["fit"]},
        split_name=split_name,
        torch=torch,
        distributed=False,
        rank=0,
        world_size=1,
        dist=None,
    )
    assert tuple(cached_prefix.shape) == (2, 4)
    assert cached_prefix.tolist() == [pytest.approx(row) for row in expected[:2]]


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


def test_temporal_non_noop_budget_selects_threshold_from_train_heldout_rows():
    rows = [
        {
            "row": {"ground_truth_tokens": ["KEY_PRESS_A"], "frame": {"width": 854, "height": 480}},
            "ground_truth_tokens": ["KEY_PRESS_A"],
            "candidates": [{"score": 0.91, "token": "KEY_PRESS_A", "slot": 0, "token_index": 2}],
        },
        {
            "row": {"ground_truth_tokens": ["MOUSE_LEFT_DOWN"], "frame": {"width": 854, "height": 480}},
            "ground_truth_tokens": ["MOUSE_LEFT_DOWN"],
            "candidates": [{"score": 0.88, "token": "MOUSE_LEFT_DOWN", "slot": 1, "token_index": 3}],
        },
        {
            "row": {"ground_truth_tokens": [], "frame": {"width": 854, "height": 480}},
            "ground_truth_tokens": [],
            "candidates": [{"score": 0.10, "token": "MOUSE_LEFT_DOWN", "slot": 0, "token_index": 3}],
        },
    ]
    payload = _calibrate_temporal_non_noop_budget(
        rows,
        config={
            "non_noop_budget_max_tokens_per_row": 2,
            "non_noop_budget_max_no_button_fpr": 0.0,
            "non_noop_budget_max_threshold_candidates": 32,
        },
    )
    assert payload["status"] == "pass"
    assert payload["selected_threshold"] <= 0.88
    assert payload["selected_row"]["predicted_non_noop_tokens"] >= 2
    assert payload["selected_row"]["no_button_false_positive_rate"] == 0.0


def test_temporal_calibration_split_can_stratify_sparse_action_families():
    rows: list[dict] = []
    for idx in range(40):
        row = _row(idx, split="train_core")
        if idx in {3, 23}:
            row["ground_truth_tokens"] = ["MOUSE_LEFT_DOWN"]
        elif idx in {7, 27}:
            row["ground_truth_tokens"] = ["KEY_PRESS_A"]
        elif idx in {11, 31}:
            row["ground_truth_tokens"] = ["MOUSE_DX_P1", "MOUSE_DY_N1"]
        else:
            row["ground_truth_tokens"] = ["NOOP"]
        rows.append(row)

    fit_indices, calibration_indices = _temporal_calibration_split_indices(
        rows,
        {
            "calibrate_non_noop_budget": True,
            "temporal_calibration_fraction": 0.25,
            "temporal_calibration_max_rows": 8,
            "temporal_calibration_strategy": "stratified_action",
        },
    )

    calibration_rows = [rows[index] for index in calibration_indices]
    assert len(calibration_indices) == 8
    assert set(fit_indices).isdisjoint(calibration_indices)
    assert any("KEY_PRESS_A" in row["ground_truth_tokens"] for row in calibration_rows)
    assert any("MOUSE_LEFT_DOWN" in row["ground_truth_tokens"] for row in calibration_rows)
    assert any(any(token.startswith("MOUSE_DX") for token in row["ground_truth_tokens"]) for row in calibration_rows)
    assert any(row["ground_truth_tokens"] == ["NOOP"] for row in calibration_rows)


def test_temporal_family_non_noop_budget_calibrates_separate_action_families():
    rows = [
        {
            "row": {"ground_truth_tokens": ["KEY_PRESS_A"], "frame": {"width": 854, "height": 480}},
            "ground_truth_tokens": ["KEY_PRESS_A"],
            "candidates": [
                {"score": 0.92, "token": "KEY_PRESS_A", "slot": 0, "token_index": 2},
                {"score": 0.40, "token": "MOUSE_LEFT_DOWN", "slot": 1, "token_index": 3},
                {"score": 0.95, "token": "FDM1_MOUSE_DX_P01", "slot": 2, "token_index": 4},
            ],
        },
        {
            "row": {"ground_truth_tokens": ["MOUSE_LEFT_DOWN"], "frame": {"width": 854, "height": 480}},
            "ground_truth_tokens": ["MOUSE_LEFT_DOWN"],
            "candidates": [
                {"score": 0.30, "token": "KEY_PRESS_A", "slot": 0, "token_index": 2},
                {"score": 0.89, "token": "MOUSE_LEFT_DOWN", "slot": 1, "token_index": 3},
                {"score": 0.93, "token": "FDM1_MOUSE_DX_P01", "slot": 2, "token_index": 4},
            ],
        },
        {
            "row": {"ground_truth_tokens": [], "frame": {"width": 854, "height": 480}},
            "ground_truth_tokens": [],
            "candidates": [
                {"score": 0.10, "token": "KEY_PRESS_A", "slot": 0, "token_index": 2},
                {"score": 0.15, "token": "MOUSE_LEFT_DOWN", "slot": 1, "token_index": 3},
                {"score": 0.20, "token": "FDM1_MOUSE_DX_P01", "slot": 2, "token_index": 4},
            ],
        },
    ]
    payload = _calibrate_temporal_family_non_noop_budget(
        rows,
        config={
            "family_non_noop_budget_families": ["keyboard", "mouse_button", "mouse_move"],
            "family_non_noop_budget_keyboard_max_tokens_per_row": 1,
            "family_non_noop_budget_mouse_button_max_tokens_per_row": 1,
            "family_non_noop_budget_mouse_move_max_tokens_per_row": 1,
            "family_non_noop_budget_max_no_button_fpr": 0.0,
            "family_non_noop_budget_max_threshold_candidates": 16,
        },
    )
    assert payload["status"] == "pass"
    assert payload["families"]["keyboard"]["selected_threshold"] <= 0.92
    assert payload["families"]["mouse_button"]["selected_threshold"] <= 0.89
    assert payload["families"]["mouse_button"]["selected_row"]["no_button_false_positive_rate"] == 0.0
    tokens = _tokens_from_family_budget_candidates(
        rows[1]["candidates"],
        family_budgets=payload,
    )
    assert "MOUSE_LEFT_DOWN" in tokens
    assert "KEY_PRESS_A" not in tokens


def test_candidate_decode_can_preserve_mouse_trajectory_duplicates():
    candidates = [
        {"score": 0.95, "token": "MOUSE_DX_P5", "slot": 0, "token_index": 2},
        {"score": 0.94, "token": "MOUSE_DX_P5", "slot": 1, "token_index": 2},
        {"score": 0.93, "token": "MOUSE_DY_P4", "slot": 2, "token_index": 3},
        {"score": 0.92, "token": "MOUSE_DY_P4", "slot": 3, "token_index": 3},
    ]

    collapsed = _tokens_from_non_noop_candidates(candidates, threshold=0.5, max_tokens=8)
    trajectory = _tokens_from_non_noop_candidates(
        candidates,
        threshold=0.5,
        max_tokens=8,
        config={"candidate_duplicate_families": ["mouse_move"]},
    )
    assert collapsed == ["MOUSE_DX_P5", "MOUSE_DY_P4"]
    assert trajectory == ["MOUSE_DX_P5", "MOUSE_DX_P5", "MOUSE_DY_P4", "MOUSE_DY_P4"]

    family_tokens = _tokens_from_family_budget_candidates(
        candidates,
        family_budgets={"families": {"mouse_move": {"status": "pass", "selected_threshold": 0.5, "max_tokens_per_row": 8}}},
        config={"candidate_duplicate_families": ["mouse_move"]},
    )
    assert family_tokens == trajectory


def test_unlabeled_family_budget_adaptation_raises_shifted_button_threshold_without_labels():
    family_budget = {
        "schema": "temporal_family_non_noop_budget_calibration.v1",
        "status": "pass",
        "rows": 100,
        "families": {
            "mouse_button": {
                "status": "pass",
                "selected_threshold": 0.10,
                "max_tokens_per_row": 1,
                "calibration_predicted_tokens_per_row": 0.20,
            }
        },
    }
    target_rows = [
        {
            "row": {"ground_truth_tokens": ["MOUSE_LEFT_DOWN"]},
            "ground_truth_tokens": ["MOUSE_LEFT_DOWN"],
            "candidates": [{"score": score, "token": "MOUSE_LEFT_DOWN", "slot": 0, "token_index": 3}],
        }
        for score in [0.90, 0.80, 0.70, 0.60, 0.50]
    ]
    adapted = _adapt_temporal_family_budget_to_unlabeled_distribution(
        family_budget,
        target_rows,
        config={"adaptive_family_budget_families": ["mouse_button"]},
    )
    budget = adapted["families"]["mouse_button"]
    assert budget["selected_threshold"] == pytest.approx(0.90)
    assert budget["unlabeled_pre_adaptation_threshold"] == pytest.approx(0.10)
    assert adapted["unlabeled_distribution_adaptation"]["families"]["mouse_button"]["target_token_budget"] == 1
    # The helper is allowed to receive diagnostic rows that include labels, but
    # the selected threshold depends only on candidate scores and train-heldout
    # emission budget.
    target_rows[0]["ground_truth_tokens"] = []
    adapted_without_labels = _adapt_temporal_family_budget_to_unlabeled_distribution(
        family_budget,
        target_rows,
        config={"adaptive_family_budget_families": ["mouse_button"]},
    )
    assert adapted_without_labels["families"]["mouse_button"]["selected_threshold"] == pytest.approx(
        budget["selected_threshold"]
    )


def test_temporal_retrieval_prior_biases_action_token_candidates():
    if not torch_available():
        return
    import torch

    probabilities = torch.zeros((1, 2, 5), dtype=torch.float32)
    probabilities[:, :, 2] = 0.01
    probabilities[:, :, 3] = 0.01
    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "KEY_PRESS_A", "MOUSE_LEFT_DOWN", "FDM1_MOUSE_DX_P01"]
    candidates = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={"retrieval_action_prior_blend": 0.9, "non_noop_budget_candidates_per_row": 8},
        retrieval_priors=[{"MOUSE_LEFT_DOWN": 0.95}],
    )
    assert candidates[0][0]["token"] == "MOUSE_LEFT_DOWN"
    assert candidates[0][0]["retrieval_score"] == 0.95


def test_temporal_token_presence_biases_action_token_identity():
    if not torch_available():
        return
    import torch

    probabilities = torch.zeros((1, 2, 5), dtype=torch.float32)
    probabilities[:, :, 2] = 0.01
    probabilities[:, :, 3] = 0.02
    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "KEY_PRESS_A", "MOUSE_LEFT_DOWN", "FDM1_MOUSE_DX_P01"]
    candidates = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={"token_presence_candidate_score_blend": 0.9, "non_noop_budget_candidates_per_row": 8},
        event_probabilities={"token_presence": torch.tensor([[0.98, 0.0, 0.99, 0.05, 0.04]])},
    )
    assert candidates[0][0]["token"] == "KEY_PRESS_A"


def test_temporal_candidates_can_preserve_low_scoring_family_candidates():
    if not torch_available():
        return
    import torch

    vocab = [
        "<FDM1_ACTION_PAD>",
        "<FDM1_ACTION_MASK>",
        "KEY_PRESS_A",
        "MOUSE_LEFT_DOWN",
        "FDM1_MOUSE_DX_P01",
        "FDM1_MOUSE_DY_P01",
    ]
    probabilities = torch.zeros((1, 2, len(vocab)), dtype=torch.float32)
    probabilities[:, :, 4] = 0.90
    probabilities[:, :, 5] = 0.80
    probabilities[:, :, 2] = 0.02
    probabilities[:, :, 3] = 0.01

    global_only = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={"non_noop_budget_candidates_per_row": 2},
    )
    family_preserved = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={
            "non_noop_budget_candidates_per_row": 2,
            "non_noop_budget_min_candidates_per_family": 1,
            "non_noop_budget_candidate_families": ["keyboard", "mouse_button", "mouse_move"],
        },
    )

    assert {candidate["family"] for candidate in global_only[0]} == {"mouse_move"}
    assert "KEY_PRESS_A" in {candidate["token"] for candidate in family_preserved[0]}
    assert "MOUSE_LEFT_DOWN" in {candidate["token"] for candidate in family_preserved[0]}


def test_candidate_token_prior_weights_boost_rare_train_tokens_without_target_labels():
    rows = []
    for idx in range(9):
        row = _row(idx, split="train_core")
        row["ground_truth_tokens"] = ["MOUSE_LEFT_DOWN"]
        rows.append(row)
    rare = _row(99, split="train_core")
    rare["ground_truth_tokens"] = ["MOUSE_RIGHT_DOWN"]
    rows.append(rare)

    weights, summary = _candidate_token_prior_weights(
        rows,
        vocab=["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "MOUSE_LEFT_DOWN", "MOUSE_RIGHT_DOWN"],
        max_slots=4,
        preserve_pad_slots=True,
        config={
            "candidate_token_prior_correction": True,
            "candidate_token_prior_families": ["mouse_button"],
            "candidate_token_prior_strength": 1.0,
            "candidate_token_prior_smoothing": 1.0,
            "candidate_token_prior_max_weight": 4.0,
        },
    )

    assert summary["status"] == "pass"
    assert summary["families"]["mouse_button"]["total_count"] == 10
    assert weights["MOUSE_RIGHT_DOWN"] > weights["MOUSE_LEFT_DOWN"]
    assert summary["claim_boundary"].startswith("Train-fit action-token prior")


def test_candidate_token_prior_does_not_boost_tokens_unseen_in_fit_rows():
    row = _row(1, split="train_core")
    row["ground_truth_tokens"] = ["MOUSE_LEFT_DOWN"]

    weights, summary = _candidate_token_prior_weights(
        [row],
        vocab=[
            "<FDM1_ACTION_PAD>",
            "<FDM1_ACTION_MASK>",
            "NOOP",
            "MOUSE_LEFT_DOWN",
            "MOUSE_MIDDLE_DOWN",
        ],
        max_slots=4,
        preserve_pad_slots=True,
        config={
            "candidate_token_prior_correction": True,
            "candidate_token_prior_families": ["mouse_button"],
            "candidate_token_prior_strength": 1.0,
            "candidate_token_prior_smoothing": 1.0,
            "candidate_token_prior_unseen_weight": 0.25,
            "candidate_token_prior_min_weight": 0.25,
            "candidate_token_prior_max_weight": 4.0,
        },
    )

    assert summary["families"]["mouse_button"]["unseen_tokens"] == 1
    assert summary["unseen_weight"] == 0.25
    assert weights["MOUSE_MIDDLE_DOWN"] == 0.25
    assert weights["MOUSE_LEFT_DOWN"] >= 1.0


def test_temporal_candidate_token_prior_adjusts_recipe_candidate_ranking():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "MOUSE_LEFT_DOWN", "MOUSE_RIGHT_DOWN"]
    probabilities = torch.zeros((1, 1, len(vocab)), dtype=torch.float32)
    probabilities[:, :, 3] = 0.50
    probabilities[:, :, 4] = 0.30

    candidates = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={"non_noop_budget_candidates_per_row": 4},
        token_prior_weights={"MOUSE_LEFT_DOWN": 0.5, "MOUSE_RIGHT_DOWN": 2.0},
    )

    assert candidates[0][0]["token"] == "MOUSE_RIGHT_DOWN"
    assert candidates[0][0]["prior_weight"] == 2.0
    assert candidates[0][1]["token"] == "MOUSE_LEFT_DOWN"


def test_temporal_candidate_rows_can_merge_neighbor_source_offsets():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "KEY_PRESS_A", "MOUSE_LEFT_DOWN"]
    probabilities = torch.zeros((1, 3, 1, len(vocab)), dtype=torch.float32)
    probabilities[:, 0, :, 3] = 0.90  # offset -1
    probabilities[:, 1, :, 4] = 0.80  # center offset 0
    probabilities[:, 2, :, 3] = 0.10  # offset +1

    candidates = _temporal_candidate_rows_from_probabilities(
        probabilities,
        offsets=[-1, 0, 1],
        vocab=vocab,
        config={
            "temporal_candidate_source_offsets": [-1, 0],
            "candidate_source_offset_weights": {"-1": 0.5},
            "non_noop_budget_candidates_per_row": 8,
        },
    )

    key_from_neighbor = next(
        candidate
        for candidate in candidates[0]
        if candidate["token"] == "KEY_PRESS_A" and candidate["source_offset"] == -1
    )
    button_from_center = next(
        candidate
        for candidate in candidates[0]
        if candidate["token"] == "MOUSE_LEFT_DOWN" and candidate["source_offset"] == 0
    )
    assert key_from_neighbor["pre_source_offset_score"] == pytest.approx(0.90)
    assert key_from_neighbor["score"] == pytest.approx(0.45)
    assert button_from_center["score"] == pytest.approx(0.80)


def test_candidate_score_reranker_uses_train_heldout_labels_to_rerank_candidates():
    if not torch_available():
        return
    import torch

    rows = []
    for idx in range(16):
        rows.append(
            {
                "row": {"ground_truth_tokens": ["KEY_PRESS_A"] if idx % 2 == 0 else ["KEY_PRESS_D"]},
                "ground_truth_fdm1_tokens": ["KEY_PRESS_A"] if idx % 2 == 0 else ["KEY_PRESS_D"],
                "candidates": [
                    {
                        "score": 0.90,
                        "token_probability": 0.90,
                        "key_presence_score": 0.10,
                        "token": "KEY_PRESS_D" if idx % 2 == 0 else "KEY_PRESS_A",
                        "family": "keyboard",
                        "slot": 0,
                        "token_index": 4,
                    },
                    {
                        "score": 0.10,
                        "token_probability": 0.10,
                        "key_presence_score": 0.95,
                        "token": "KEY_PRESS_A" if idx % 2 == 0 else "KEY_PRESS_D",
                        "family": "keyboard",
                        "slot": 1,
                        "token_index": 3,
                    },
                ],
            }
        )

    reranker = _fit_candidate_score_reranker(
        rows,
        torch=torch,
        config={
            "candidate_score_reranker_enabled": True,
            "candidate_score_reranker_families": ["keyboard"],
            "candidate_score_reranker_features": ["score", "key_presence_score"],
            "candidate_score_reranker_epochs": 80,
            "candidate_score_reranker_lr": 0.1,
            "candidate_score_reranker_blend": 1.0,
        },
    )
    assert reranker["status"] == "pass"
    assert reranker["families"]["keyboard"]["train_pairwise_auc"] >= 0.99

    applied = _apply_candidate_score_reranker_to_probability_rows(
        [rows[0]],
        config={"candidate_score_reranker": reranker},
    )
    assert applied[0]["candidates"][0]["token"] == "KEY_PRESS_A"
    assert applied[0]["candidates"][0]["reranker_score"] > applied[0]["candidates"][1]["reranker_score"]
    assert applied[0]["candidates"][0]["pre_reranker_score"] == pytest.approx(0.10)


def test_temporal_center_candidates_applies_candidate_score_reranker_before_cutoff():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "KEY_PRESS_A", "KEY_PRESS_D"]
    probabilities = torch.zeros((1, 1, len(vocab)), dtype=torch.float32)
    probabilities[:, :, 3] = 0.10
    probabilities[:, :, 4] = 0.90
    reranker = {
        "status": "pass",
        "feature_names": ["score"],
        "score_blend": 1.0,
        "families": {
            "keyboard": {
                "status": "pass",
                "mean": [0.5],
                "scale": [1.0],
                "weights": [-10.0],
                "bias": 0.0,
            }
        },
    }

    candidates = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={"non_noop_budget_candidates_per_row": 1, "candidate_score_reranker": reranker},
    )

    assert candidates[0][0]["token"] == "KEY_PRESS_A"
    assert candidates[0][0]["pre_reranker_score"] == pytest.approx(0.10)


def test_temporal_button_class_targets_map_click_tokens_to_row_classes():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "MOUSE_LEFT_DOWN", "MOUSE_RIGHT_UP"]
    ids = torch.tensor([[[0, 2, 2], [4, 0, 2], [3, 4, 0]]], dtype=torch.long)
    targets = _temporal_button_class_targets(torch, ids, vocab, ["MOUSE_LEFT_DOWN", "MOUSE_RIGHT_UP"])

    assert targets.tolist() == [[0, 2, 2]]


def test_temporal_button_class_biases_mouse_button_candidate_identity():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "MOUSE_LEFT_DOWN", "MOUSE_RIGHT_UP"]
    probabilities = torch.zeros((1, 1, len(vocab)), dtype=torch.float32)
    probabilities[:, :, 3] = 0.40
    probabilities[:, :, 4] = 0.20
    button_class = torch.tensor([[0.05, 0.05, 0.90]], dtype=torch.float32)

    candidates = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={"non_noop_budget_candidates_per_row": 4, "button_class_candidate_score_blend": 0.9},
        event_probabilities={"button_class": button_class},
    )

    assert candidates[0][0]["token"] == "MOUSE_RIGHT_UP"
    assert candidates[0][0]["button_class_score"] == pytest.approx(0.90)


def test_temporal_direct_auxiliary_candidates_can_override_slot_identity_noise():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "KEY_PRESS_A", "KEY_PRESS_D", "MOUSE_LEFT_DOWN"]
    probabilities = torch.zeros((1, 2, len(vocab)), dtype=torch.float32)
    probabilities[:, :, 3] = 0.02
    probabilities[:, :, 4] = 0.02
    probabilities[:, :, 5] = 0.02

    candidates = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={
            "non_noop_budget_candidates_per_row": 8,
            "direct_auxiliary_candidate_families": ["keyboard", "mouse_button"],
            "direct_auxiliary_button_class_blend": 1.0,
        },
        event_probabilities={
            "key_token_presence": torch.tensor([[0.05, 0.91]], dtype=torch.float32),
            "button_class": torch.tensor([[0.05, 0.94]], dtype=torch.float32),
        },
    )

    assert candidates[0][0]["token"] == "MOUSE_LEFT_DOWN"
    assert candidates[0][0]["direct_auxiliary_candidate"] == "button_presence_class"
    assert any(
        row["token"] == "KEY_PRESS_D"
        and row.get("direct_auxiliary_candidate") == "key_token_presence"
        and row["score"] == pytest.approx(0.91)
        for row in candidates[0]
    )


def test_temporal_direct_auxiliary_candidates_can_add_mouse_move_presence_tokens():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "FDM1_MOUSE_DX_N01", "FDM1_MOUSE_DY_P02"]
    probabilities = torch.zeros((1, 1, len(vocab)), dtype=torch.float32)
    probabilities[:, :, 3] = 0.01
    probabilities[:, :, 4] = 0.01
    token_presence = torch.zeros((1, len(vocab)), dtype=torch.float32)
    token_presence[:, 4] = 0.88

    candidates = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={
            "non_noop_budget_candidates_per_row": 4,
            "direct_auxiliary_candidate_families": ["mouse_move"],
        },
        event_probabilities={"token_presence": token_presence},
    )

    assert candidates[0][0]["token"] == "FDM1_MOUSE_DY_P02"
    assert candidates[0][0]["direct_auxiliary_candidate"] == "token_presence_mouse_move"
    assert candidates[0][0]["score"] == pytest.approx(0.88)


def test_temporal_direct_auxiliary_candidates_prefer_dedicated_mouse_move_head():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "FDM1_MOUSE_DX_N01", "FDM1_MOUSE_DY_P02"]
    probabilities = torch.zeros((1, 1, len(vocab)), dtype=torch.float32)
    # Dedicated mouse-move head is family-scoped: columns map only to mouse-move
    # vocab order, not global action vocab indices.
    mouse_move_presence = torch.tensor([[0.13, 0.91]], dtype=torch.float32)

    candidates = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={
            "non_noop_budget_candidates_per_row": 4,
            "direct_auxiliary_candidate_families": ["mouse_move"],
            "mouse_move_token_presence_candidate_score_blend": 1.0,
        },
        event_probabilities={"mouse_move_token_presence": mouse_move_presence},
    )

    assert candidates[0][0]["token"] == "FDM1_MOUSE_DY_P02"
    assert candidates[0][0]["direct_auxiliary_candidate"] == "mouse_move_token_presence"
    assert candidates[0][0]["mouse_move_presence_score"] == pytest.approx(0.91)


def test_temporal_video_presence_heads_can_rank_direct_action_candidates():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "KEY_PRESS_A", "KEY_PRESS_D", "MOUSE_DX_P1"]
    probabilities = torch.zeros((1, 1, len(vocab)), dtype=torch.float32)
    probabilities[:, :, 3:] = 0.01

    candidates = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={
            "non_noop_budget_candidates_per_row": 6,
            "direct_auxiliary_candidate_families": ["keyboard", "mouse_move"],
            "video_key_token_presence_candidate_score_blend": 0.9,
            "video_mouse_move_token_presence_candidate_score_blend": 0.9,
        },
        event_probabilities={
            "video_key_token_presence": torch.tensor([[0.05, 0.97]], dtype=torch.float32),
            "video_mouse_move_token_presence": torch.tensor([[0.88]], dtype=torch.float32),
        },
    )

    assert candidates[0][0]["token"] == "KEY_PRESS_D"
    assert candidates[0][0]["direct_auxiliary_candidate"] == "video_key_token_presence"
    assert candidates[0][0]["video_key_presence_score"] == pytest.approx(0.97)
    assert any(
        row["token"] == "MOUSE_DX_P1"
        and row.get("direct_auxiliary_candidate") == "video_mouse_move_token_presence"
        and row["video_mouse_move_presence_score"] == pytest.approx(0.88)
        for row in candidates[0]
    )


def test_family_token_presence_rank_loss_rewards_positive_over_hard_negative():
    if not torch_available():
        return
    import torch

    targets = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    offset_mask = torch.tensor([True])
    bad_logits = torch.tensor([[[0.0, 2.0, 1.0]]], dtype=torch.float32)
    good_logits = torch.tensor([[[3.0, 1.0, 0.0]]], dtype=torch.float32)

    bad_loss = _family_token_presence_rank_loss(
        torch,
        bad_logits,
        targets,
        offset_mask,
        margin=1.0,
        top_negatives=1,
    )
    good_loss = _family_token_presence_rank_loss(
        torch,
        good_logits,
        targets,
        offset_mask,
        margin=1.0,
        top_negatives=1,
    )

    assert float(good_loss) < float(bad_loss)


def test_temporal_family_token_presence_targets_are_family_scoped_multihot():
    if not torch_available():
        return
    import torch

    vocab = [
        "<FDM1_ACTION_PAD>",
        "<FDM1_ACTION_MASK>",
        "NOOP",
        "KEY_PRESS_W",
        "KEY_RELEASE_W",
        "MOUSE_LEFT_DOWN",
    ]
    ids = torch.tensor([[[3, 4, 2], [5, 0, 2], [3, 5, 0]]], dtype=torch.long)

    key_targets = _temporal_family_token_presence_targets(torch, ids, vocab, ["KEY_PRESS_W", "KEY_RELEASE_W"])
    button_targets = _temporal_family_token_presence_targets(torch, ids, vocab, ["MOUSE_LEFT_DOWN"])

    assert key_targets.tolist() == [[[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]]
    assert button_targets.tolist() == [[[0.0], [1.0], [1.0]]]


def test_temporal_family_count_targets_cap_sparse_action_counts():
    if not torch_available():
        return
    import torch

    vocab = [
        "<FDM1_ACTION_PAD>",
        "<FDM1_ACTION_MASK>",
        "NOOP",
        "KEY_PRESS_W",
        "KEY_RELEASE_W",
        "MOUSE_LEFT_DOWN",
        "MOUSE_DX_P1",
        "MOUSE_DY_N1",
    ]
    ids = torch.tensor([[[3, 4, 5, 6, 7], [0, 2, 3, 6, 6]]], dtype=torch.long)
    key_counts = _temporal_family_count_targets(torch, ids, vocab, "keyboard", max_count=1)
    button_counts = _temporal_family_count_targets(torch, ids, vocab, "mouse_button", max_count=2)
    move_counts = _temporal_family_count_targets(torch, ids, vocab, "mouse_move", max_count=2)

    assert key_counts.tolist() == [[1, 1]]
    assert button_counts.tolist() == [[1, 0]]
    assert move_counts.tolist() == [[2, 2]]


def test_temporal_family_count_head_gates_candidate_scores_and_budget():
    if not torch_available():
        return
    import torch

    vocab = [
        "<FDM1_ACTION_PAD>",
        "<FDM1_ACTION_MASK>",
        "NOOP",
        "KEY_PRESS_W",
        "MOUSE_LEFT_DOWN",
    ]
    probabilities = torch.tensor([[[0.01, 0.01, 0.04, 0.55, 0.39]]], dtype=torch.float32)
    candidates = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={
            "non_noop_budget_candidates_per_row": 4,
            "keyboard_count_candidate_score_blend": 1.0,
            "keyboard_count_candidate_gate_power": 1.0,
            "keyboard_count_candidate_gate_floor": 0.0,
        },
        event_probabilities={"keyboard_count": torch.tensor([[0.90, 0.08, 0.02]], dtype=torch.float32)},
    )
    key_candidate = next(item for item in candidates[0] if item["token"] == "KEY_PRESS_W")
    assert key_candidate["family_count_nonzero_score"] == pytest.approx(0.10)
    assert key_candidate["family_count_predicted"] == 0
    assert key_candidate["score"] == pytest.approx(0.01)

    emitted = _tokens_from_family_budget_candidates(
        candidates[0],
        family_budgets={"families": {"keyboard": {"status": "pass", "selected_threshold": 0.0, "max_tokens_per_row": 2}}},
        config={"family_count_candidate_budget": True},
    )
    assert emitted == []


def test_temporal_family_token_presence_biases_key_and_button_candidate_identity():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "KEY_PRESS_A", "KEY_PRESS_D", "MOUSE_LEFT_DOWN"]
    probabilities = torch.zeros((1, 1, len(vocab)), dtype=torch.float32)
    probabilities[:, :, 3] = 0.20
    probabilities[:, :, 4] = 0.45
    probabilities[:, :, 5] = 0.10

    candidates = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={
            "non_noop_budget_candidates_per_row": 4,
            "key_token_presence_candidate_score_blend": 0.9,
            "button_token_presence_candidate_score_blend": 0.9,
        },
        event_probabilities={
            "key_token_presence": torch.tensor([[0.95, 0.05]], dtype=torch.float32),
            "button_token_presence": torch.tensor([[0.80]], dtype=torch.float32),
        },
    )

    assert candidates[0][0]["token"] == "KEY_PRESS_A"
    assert candidates[0][0]["key_presence_score"] == pytest.approx(0.95)
    assert any(row["token"] == "MOUSE_LEFT_DOWN" and row["button_presence_score"] == pytest.approx(0.80) for row in candidates[0])


def test_candidate_family_diagnostics_reports_exact_coverage_and_ranks():
    rows = [
        {
            "ground_truth_tokens": ["MOUSE_LEFT_DOWN"],
            "candidates": [
                {"score": 0.90, "token": "FDM1_MOUSE_DX_P01", "slot": 0, "token_index": 4, "family": "mouse_move"},
                {"score": 0.20, "token": "MOUSE_LEFT_DOWN", "slot": 1, "token_index": 3, "family": "mouse_button"},
            ],
        },
        {
            "ground_truth_tokens": ["MOUSE_RIGHT_DOWN"],
            "candidates": [
                {"score": 0.40, "token": "MOUSE_LEFT_DOWN", "slot": 0, "token_index": 3, "family": "mouse_button"},
            ],
        },
    ]

    diagnostics = _candidate_family_diagnostics(rows, config={"candidate_diagnostic_families": ["mouse_button"]})
    button = diagnostics["families"]["mouse_button"]

    assert diagnostics["status"] == "pass"
    assert button["positive_rows"] == 2
    assert button["positive_rows_with_family_candidate"] == 2
    assert button["positive_rows_with_exact_candidate"] == 1
    assert button["exact_candidate_rank"]["p50"] == 2.0


def test_candidate_family_diagnostics_compares_recipe_mouse_tokens():
    rows = [
        {
            "ground_truth_tokens": ["MOUSE_DX_P1"],
            "ground_truth_fdm1_tokens": ["FDM1_MOUSE_DX_P03"],
            "candidates": [
                {"score": 0.70, "token": "FDM1_MOUSE_DX_P03", "slot": 0, "token_index": 4, "family": "mouse_move"}
            ],
        }
    ]

    diagnostics = _candidate_family_diagnostics(rows, config={"candidate_diagnostic_families": ["mouse_move"]})
    move = diagnostics["families"]["mouse_move"]

    assert move["positive_rows"] == 1
    assert move["positive_rows_with_exact_candidate"] == 1
    assert move["positive_rows_with_exact_candidate_rate"] == 1.0


def test_d2e_metric_bin_mouse_tokenization_preserves_metric_tokens():
    row = {
        "ground_truth_tokens": ["KEY_PRESS_A", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
        "frame": {"width": 854, "height": 480},
    }

    recipe_tokens = canonical_fdm1_action_tokens(row)
    metric_tokens = canonical_fdm1_action_tokens(row, mouse_token_mode="d2e_metric_bins")

    assert "FDM1_MOUSE_DX_P03" in recipe_tokens
    assert "MOUSE_DX_P1" in metric_tokens
    assert "MOUSE_DY_Z0" in metric_tokens
    assert not any(token.startswith("FDM1_MOUSE_") for token in metric_tokens)


def test_temporal_config_vocab_uses_d2e_metric_mouse_bins():
    rows = [
        {"ground_truth_tokens": ["KEY_PRESS_A", "MOUSE_DX_P1", "MOUSE_DY_Z0"], "frame": {"width": 854, "height": 480}},
        {"ground_truth_tokens": ["MOUSE_DX_N1", "MOUSE_DY_P1"], "frame": {"width": 854, "height": 480}},
    ]
    config = {"action_mouse_tokenization": "d2e_metric_bins"}

    slots = _target_slots_for_config(rows[0], max_slots=4, config=config)
    vocab = _build_vocab_for_config(rows, max_slots=4, config=config)

    assert slots[:3] == ["KEY_PRESS_A", "MOUSE_DX_P1", "MOUSE_DY_Z0"]
    assert "MOUSE_DX_P1" in vocab
    assert "MOUSE_DY_P1" in vocab
    assert not any(token.startswith("FDM1_MOUSE_") for token in vocab)


def test_temporal_family_class_targets_and_mouse_axis_vocab():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "KEY_PRESS_A", "MOUSE_DX_N1", "MOUSE_DY_P1"]
    target_ids = torch.tensor([[[3, 4, 5, 0], [2, 0, 0, 0]]], dtype=torch.long)

    assert _mouse_axis_class_vocab(vocab, "x") == ["MOUSE_DX_N1"]
    assert _mouse_axis_class_vocab(vocab, "y") == ["MOUSE_DY_P1"]
    key_targets = _temporal_family_class_targets(torch, target_ids, vocab, ["KEY_PRESS_A"])
    dx_targets = _temporal_family_class_targets(torch, target_ids, vocab, ["MOUSE_DX_N1"])

    assert key_targets.tolist() == [[1, 0]]
    assert dx_targets.tolist() == [[1, 0]]


def test_mouse_move_family_budget_can_constrain_one_token_per_axis():
    candidates = [
        {"score": 0.90, "token": "MOUSE_DY_Z0"},
        {"score": 0.80, "token": "MOUSE_DY_P1"},
        {"score": 0.70, "token": "MOUSE_DX_N1"},
        {"score": 0.60, "token": "MOUSE_DX_P1"},
    ]
    budgets = {"families": {"mouse_move": {"status": "pass", "selected_threshold": 0.0, "max_tokens_per_row": 2}}}

    unconstrained = _tokens_from_family_budget_candidates(candidates, family_budgets=budgets, config={})
    constrained = _tokens_from_family_budget_candidates(
        candidates,
        family_budgets=budgets,
        config={"mouse_move_axis_constrained_budget": True},
    )

    assert unconstrained == ["MOUSE_DY_Z0", "MOUSE_DY_P1"]
    assert constrained == ["MOUSE_DY_Z0", "MOUSE_DX_N1"]


def test_temporal_target_slots_can_preserve_padding_for_sparse_action_sequences():
    row = {"ground_truth_tokens": ["KEY_PRESS_A"], "frame": {"width": 854, "height": 480}}

    legacy_slots = _target_slots(row, max_slots=4)
    pad_aware_slots = _target_slots(row, max_slots=4, preserve_pad_slots=True)

    assert legacy_slots == ["KEY_PRESS_A", "NOOP", "NOOP", "NOOP"]
    assert pad_aware_slots == ["KEY_PRESS_A", "<FDM1_ACTION_PAD>", "<FDM1_ACTION_PAD>", "<FDM1_ACTION_PAD>"]


def test_temporal_target_slots_support_aggregate_decomposed_mouse_config():
    row = {
        "frame": {"width": 854, "height": 480},
        "events": [
            {"type": "mouse_move", "dx": 66, "dy": 18},
            {"type": "mouse_move", "dx": 67, "dy": 17},
            {"type": "keyboard", "event_type": "press", "key": "w"},
        ],
    }

    slots = _target_slots_for_config(
        row,
        max_slots=20,
        config={
            "action_mouse_tokenization": "d2e_metric_aggregate_decomposed_bins",
            "action_mouse_max_tokens_per_axis": 8,
        },
        preserve_pad_slots=True,
    )

    assert "KEY_PRESS_W" in slots
    assert slots.count("MOUSE_DX_P5") > 2
    assert any(token.startswith("MOUSE_DY_P") for token in slots)
    assert slots[-1] == "<FDM1_ACTION_PAD>"


def test_temporal_token_presence_targets_exclude_pad_and_noop_by_default():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "KEY_PRESS_A", "MOUSE_LEFT_DOWN"]
    ids = torch.tensor([[[3, 0, 2], [4, 0, 0]]], dtype=torch.long)
    targets = _token_presence_targets(torch, ids, vocab)

    assert targets.shape == (1, 2, len(vocab))
    assert targets[0, 0, 3].item() == 1.0
    assert targets[0, 1, 4].item() == 1.0
    assert targets[0, 0, 0].item() == 0.0
    assert targets[0, 0, 2].item() == 0.0


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
            "key_span_diffusion": True,
            "key_span_offsets": [-1, 0, 1],
            "key_span_loss_weight": 0.5,
            "key_probability_source": "key_span",
            "key_span_probability_aggregation": "max",
            "button_span_diffusion": True,
            "button_span_offsets": [-1, 0, 1],
            "button_span_loss_weight": 0.5,
            "button_probability_source": "button_span_class",
            "button_event_probability_source": "button_span_class",
            "button_span_probability_aggregation": "max",
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
    assert summary["factorization"]["key_probability_source"] == "key_span"
    assert summary["factorization"]["key_span_diffusion"] is True
    assert summary["factorization"]["key_span_offsets"] == [-1, 0, 1]
    assert summary["factorization"]["key_span_loss_weight"] == 0.5
    assert summary["factorization"]["key_span_probability_aggregation"] == "max"
    assert summary["factorization"]["button_probability_source"] == "button_span_class"
    assert summary["factorization"]["button_event_probability_source"] == "button_span_class"
    assert summary["factorization"]["button_span_probability_aggregation"] == "max"
    assert summary["factorization"]["button_span_diffusion"] is True
    assert summary["factorization"]["button_span_offsets"] == [-1, 0, 1]
    assert summary["factorization"]["button_span_loss_weight"] == 0.5
    assert any("button_class" in row for row in summary["history"])
    assert any("key_span" in row for row in summary["history"])
    assert any("button_span_class" in row for row in summary["history"])
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


def test_train_temporal_masked_diffusion_idm_tiny_smoke(tmp_path: Path):
    if not torch_available():
        return
    train_path = tmp_path / "train_temporal.jsonl"
    target_path = tmp_path / "target_temporal.jsonl"
    train_rows = []
    for i in range(10):
        row = _row(i, split="train_core")
        row["compact_luma_window"] = [[float(i + frame + pix) / 20.0 for pix in range(4)] for frame in range(2)]
        row["compact_luma_window_mask"] = [1.0, 1.0]
        if i in {2, 4, 6}:
            row["ground_truth_tokens"] = ["KEY_PRESS_A", "MOUSE_LEFT_DOWN", "MOUSE_DX_P1", "MOUSE_DY_Z0"]
        train_rows.append(row)
    target_rows = []
    for i in range(10, 14):
        row = _row(i, split="eval")
        row["compact_luma_window"] = [[float(i + frame + pix) / 20.0 for pix in range(4)] for frame in range(2)]
        row["compact_luma_window_mask"] = [1.0, 1.0]
        target_rows.append(row)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(target_path, target_rows)
    summary = train_temporal_masked_diffusion_idm(
        {
            "model_name": "unit_temporal_masked_diffusion_idm",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "out_temporal"),
            "summary_out": str(tmp_path / "summary_temporal.json"),
            "max_train_rows": 10,
            "max_target_rows": 4,
            "max_action_tokens_per_bin": 4,
            "temporal_offsets": [-1, 0, 1],
            "temporal_loss_offsets": [-1, 0, 1],
            "video_feature_paths": ["compact_luma_window", "compact_luma_window_mask", "frame.features"],
            "video_feature_dim": 12,
            "video_encoder_arch": "compact_luma_window_cnn",
            "luma_window_frames": 2,
            "luma_window_size": 2,
            "luma_encoder_channels": 4,
            "luma_encoder_pool_hw": 1,
            "luma_aux_hidden_dim": 4,
            "hidden_dim": 24,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "batch_size": 2,
            "prediction_batch_size": 2,
            "precompute_features_as_tensor": True,
            "precompute_feature_tensor_dtype": "float32",
            "epochs": 1,
            "lr": 0.001,
            "mask_probability": 0.75,
            "random_token_probability": 0.0,
            "diffusion_steps": 4,
            "token_loss_type": "focal",
            "token_focal_gamma": 1.5,
            "temporal_event_auxiliary": True,
            "event_aux_weight": 0.2,
            "key_event_pos_weight": 2.0,
            "button_event_pos_weight": 3.0,
            "event_auxiliary_candidate_score_blend": 0.25,
            "temporal_button_class_auxiliary": True,
            "button_class_aux_weight": 0.3,
            "button_class_no_button_weight": 0.1,
            "button_class_button_weight": 4.0,
            "button_class_candidate_score_blend": 0.4,
            "temporal_key_class_auxiliary": True,
            "key_class_aux_weight": 0.2,
            "key_class_key_weight": 3.0,
            "key_class_no_key_weight": 0.1,
            "key_class_candidate_score_blend": 0.25,
            "temporal_mouse_axis_class_auxiliary": True,
            "mouse_axis_class_aux_weight": 0.2,
            "mouse_axis_class_axis_weight": 2.0,
            "mouse_axis_class_no_axis_weight": 0.1,
            "mouse_axis_class_candidate_score_blend": 0.3,
            "mouse_move_axis_constrained_budget": True,
            "temporal_key_token_presence_auxiliary": True,
            "key_token_presence_aux_weight": 0.2,
            "key_token_presence_rank_weight": 0.1,
            "key_token_presence_pos_weight": 3.0,
            "key_token_presence_negative_weight": 0.1,
            "key_token_presence_candidate_score_blend": 0.35,
            "temporal_button_token_presence_auxiliary": True,
            "button_token_presence_aux_weight": 0.25,
            "button_token_presence_rank_weight": 0.1,
            "button_token_presence_pos_weight": 5.0,
            "button_token_presence_negative_weight": 0.1,
            "button_token_presence_candidate_score_blend": 0.45,
            "temporal_mouse_move_token_presence_auxiliary": True,
            "mouse_move_token_presence_aux_weight": 0.15,
            "mouse_move_token_presence_rank_weight": 0.1,
            "mouse_move_token_presence_pos_weight": 2.0,
            "mouse_move_token_presence_negative_weight": 0.1,
            "mouse_move_token_presence_candidate_score_blend": 0.5,
            "temporal_family_count_auxiliary": True,
            "family_count_aux_weight": 0.12,
            "keyboard_count_max": 2,
            "mouse_button_count_max": 1,
            "mouse_move_count_max": 2,
            "family_count_candidate_score_blend": 0.2,
            "family_count_candidate_gate_power": 0.5,
            "family_count_candidate_budget": True,
            "temporal_video_key_token_presence_auxiliary": True,
            "video_key_token_presence_aux_weight": 0.1,
            "video_key_token_presence_rank_weight": 0.05,
            "video_key_token_presence_candidate_score_blend": 0.3,
            "temporal_video_button_token_presence_auxiliary": True,
            "video_button_token_presence_aux_weight": 0.1,
            "video_button_token_presence_rank_weight": 0.05,
            "video_button_token_presence_candidate_score_blend": 0.3,
            "temporal_video_mouse_move_token_presence_auxiliary": True,
            "video_mouse_move_token_presence_aux_weight": 0.1,
            "video_mouse_move_token_presence_rank_weight": 0.05,
            "video_mouse_move_token_presence_candidate_score_blend": 0.3,
            "retrieval_action_prior_enabled": True,
            "retrieval_action_prior_top_k": 2,
            "retrieval_action_prior_temperature": 0.1,
            "retrieval_action_prior_blend": 0.2,
            "candidate_token_prior_correction": True,
            "candidate_token_prior_families": ["keyboard", "mouse_button"],
            "candidate_token_prior_strength": 0.5,
            "candidate_token_prior_smoothing": 1.0,
            "calibrate_non_noop_budget": True,
            "temporal_calibration_fraction": 0.2,
            "temporal_calibration_max_rows": 2,
            "non_noop_budget_max_tokens_per_row": 3,
            "non_noop_budget_max_threshold_candidates": 16,
            "video_encoder_pretrain_epochs": 1,
            "video_encoder_pretrain_max_batches": 1,
            "video_encoder_pretrain_lr": 0.001,
            "video_encoder_pretrain_mask_probability": 0.5,
            "video_reconstruction_aux_weight": 0.05,
            "force_cpu": True,
            "noop_loss_weight": 0.1,
            "keyboard_loss_weight": 2.0,
            "mouse_button_loss_weight": 3.0,
        }
    )
    assert summary["schema"] == "temporal_masked_diffusion_idm_train_summary.v1"
    assert summary["status"] == "pass"
    assert summary["temporal_offsets"] == [-1, 0, 1]
    assert summary["temporal_window"] == 3
    assert summary["fit_rows"] == 8
    assert summary["calibration_rows"] == 2
    assert summary["vocab_size"] >= 4
    assert summary["video_encoder_pretrain_history"]
    assert summary["video_encoder_pretrain_history"][0]["truncated"] is True
    assert summary["video_encoder_pretrain_history"][0]["max_batches"] == 1
    assert summary["precompute_features_as_tensor"] is True
    assert summary["precompute_feature_tensor_dtype"] == "float32"
    assert summary["loss_weights"]["keyboard_loss_weight"] == 2.0
    assert summary["loss_weights"]["mouse_button_loss_weight"] == 3.0
    assert summary["loss_weights"]["token_loss_type"] == "focal"
    assert summary["loss_weights"]["temporal_event_auxiliary"] is True
    assert summary["loss_weights"]["key_event_aux_weight"] == 0.2
    assert summary["loss_weights"]["button_event_aux_weight"] == 0.2
    assert summary["loss_weights"]["event_auxiliary_candidate_score_blend"] == 0.25
    assert summary["loss_weights"]["temporal_button_class_auxiliary"] is True
    assert summary["loss_weights"]["button_class_aux_weight"] == 0.3
    assert summary["loss_weights"]["button_class_candidate_score_blend"] == 0.4
    assert summary["loss_weights"]["temporal_key_class_auxiliary"] is True
    assert summary["loss_weights"]["key_class_aux_weight"] == 0.2
    assert summary["loss_weights"]["key_class_candidate_score_blend"] == 0.25
    assert summary["loss_weights"]["temporal_mouse_axis_class_auxiliary"] is True
    assert summary["loss_weights"]["mouse_axis_class_aux_weight"] == 0.2
    assert summary["loss_weights"]["mouse_axis_class_candidate_score_blend"] == 0.3
    assert summary["loss_weights"]["mouse_move_axis_constrained_budget"] is True
    assert summary["loss_weights"]["temporal_key_token_presence_auxiliary"] is True
    assert summary["loss_weights"]["key_token_presence_aux_weight"] == 0.2
    assert summary["loss_weights"]["key_token_presence_rank_weight"] == 0.1
    assert summary["loss_weights"]["key_token_presence_candidate_score_blend"] == 0.35
    assert summary["loss_weights"]["temporal_button_token_presence_auxiliary"] is True
    assert summary["loss_weights"]["button_token_presence_aux_weight"] == 0.25
    assert summary["loss_weights"]["button_token_presence_rank_weight"] == 0.1
    assert summary["loss_weights"]["button_token_presence_candidate_score_blend"] == 0.45
    assert summary["loss_weights"]["temporal_mouse_move_token_presence_auxiliary"] is True
    assert summary["loss_weights"]["mouse_move_token_presence_aux_weight"] == 0.15
    assert summary["loss_weights"]["mouse_move_token_presence_rank_weight"] == 0.1
    assert summary["loss_weights"]["mouse_move_token_presence_candidate_score_blend"] == 0.5
    assert summary["loss_weights"]["temporal_family_count_auxiliary"] is True
    assert summary["loss_weights"]["temporal_keyboard_count_auxiliary"] is True
    assert summary["loss_weights"]["keyboard_count_aux_weight"] == 0.12
    assert summary["loss_weights"]["keyboard_count_max"] == 2
    assert summary["loss_weights"]["mouse_button_count_max"] == 1
    assert summary["loss_weights"]["mouse_move_count_max"] == 2
    assert summary["loss_weights"]["family_count_candidate_budget"] is True
    assert summary["loss_weights"]["temporal_video_key_token_presence_auxiliary"] is True
    assert summary["loss_weights"]["video_key_token_presence_aux_weight"] == 0.1
    assert summary["loss_weights"]["video_key_token_presence_rank_weight"] == 0.05
    assert summary["loss_weights"]["video_key_token_presence_candidate_score_blend"] == 0.3
    assert summary["loss_weights"]["temporal_video_button_token_presence_auxiliary"] is True
    assert summary["loss_weights"]["video_button_token_presence_aux_weight"] == 0.1
    assert summary["loss_weights"]["temporal_video_mouse_move_token_presence_auxiliary"] is True
    assert summary["loss_weights"]["video_mouse_move_token_presence_aux_weight"] == 0.1
    assert summary["loss_weights"]["retrieval_action_prior_blend"] == 0.2
    assert summary["loss_weights"]["candidate_token_prior_correction"] is True
    assert summary["candidate_token_prior"]["status"] == "pass"
    assert summary["retrieval_action_prior"]["status"] == "pass"
    assert summary["retrieval_action_prior"]["rows"] == 8
    assert any("key_event_loss" in row and "button_event_loss" in row for row in summary["history"])
    assert any("button_class_loss" in row for row in summary["history"])
    assert any("key_class_loss" in row and "mouse_axis_class_loss" in row for row in summary["history"])
    assert any("key_token_presence_loss" in row and "button_token_presence_loss" in row for row in summary["history"])
    assert any("mouse_move_token_presence_loss" in row and "mouse_move_token_presence_rank_loss" in row for row in summary["history"])
    assert any("keyboard_count_loss" in row and "mouse_button_count_loss" in row for row in summary["history"])
    assert any("video_key_token_presence_loss" in row and "video_button_token_presence_loss" in row for row in summary["history"])
    assert any("video_mouse_move_token_presence_loss" in row for row in summary["history"])
    assert summary["non_noop_budget"]["status"] in {"pass", "skipped"}
    assert "temporal action-token sequences" in summary["recipe_alignment"]
    assert Path(summary["checkpoint_path"]).exists()
    assert Path(summary["predictions_path"]).exists()
    assert read_json(summary["metrics_path"])["alignment"]["rows_seen"] == 4


def test_train_temporal_masked_diffusion_idm_held_state_eventifies_predictions(tmp_path: Path):
    if not torch_available():
        return
    train_path = tmp_path / "train_state_temporal.jsonl"
    target_path = tmp_path / "target_state_temporal.jsonl"
    train_rows = []
    for i in range(6):
        row = _row(i, split="train_core")
        row["compact_luma_window"] = [[float(i + pix) / 10.0 for pix in range(4)] for _ in range(2)]
        row["compact_luma_window_mask"] = [1.0, 1.0]
        row["prior_action_tokens"] = ["KEY_DOWN_87"] if i % 2 else []
        row["ground_truth_tokens"] = ["KEY_RELEASE_87", "KEY_PRESS_65", "MOUSE_DX_P1", "MOUSE_DY_Z0"] if i % 2 else [
            "KEY_PRESS_87",
            "MOUSE_DX_Z0",
            "MOUSE_DY_Z0",
        ]
        train_rows.append(row)
    target_rows = []
    for i in range(6, 8):
        row = _row(i, split="eval")
        row["recording_id"] = "unit-state"
        row["compact_luma_window"] = [[float(i + pix) / 10.0 for pix in range(4)] for _ in range(2)]
        row["compact_luma_window_mask"] = [1.0, 1.0]
        row["prior_action_tokens"] = ["KEY_DOWN_87"] if i == 6 else ["KEY_DOWN_65"]
        row["ground_truth_tokens"] = ["KEY_RELEASE_87", "KEY_PRESS_65", "MOUSE_DX_P1", "MOUSE_DY_Z0"]
        target_rows.append(row)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(target_path, target_rows)

    summary = train_temporal_masked_diffusion_idm(
        {
            "model_name": "unit_temporal_masked_diffusion_state_idm",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "out_state_temporal"),
            "summary_out": str(tmp_path / "summary_state_temporal.json"),
            "max_train_rows": 6,
            "max_target_rows": 2,
            "max_action_tokens_per_bin": 6,
            "action_target_mode": "held_state_tokens",
            "action_mouse_tokenization": "d2e_metric_aggregate_decomposed_bins",
            "eventify_state_predictions": True,
            "temporal_offsets": [0],
            "temporal_loss_offsets": [0],
            "video_feature_paths": ["compact_luma_window", "compact_luma_window_mask", "frame.features"],
            "video_feature_dim": 12,
            "video_encoder_arch": "compact_luma_window_cnn",
            "luma_window_frames": 2,
            "luma_window_size": 2,
            "luma_encoder_channels": 2,
            "luma_encoder_pool_hw": 1,
            "luma_aux_hidden_dim": 4,
            "hidden_dim": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "batch_size": 2,
            "prediction_batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "mask_probability": 1.0,
            "random_token_probability": 0.0,
            "diffusion_steps": 2,
            "force_cpu": True,
        }
    )

    assert summary["status"] == "pass"
    assert summary["action_target_mode"] == "held_state_tokens"
    assert summary["state_eventification"]["status"] == "pass"
    assert Path(summary["state_predictions_path"]).exists()
    assert Path(summary["predictions_path"]).exists()
    assert read_json(summary["metrics_path"])["alignment"]["rows_seen"] == 2


def test_train_temporal_masked_diffusion_idm_raw_video_cnn_tiny_smoke(tmp_path: Path):
    if not torch_available():
        return
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for idx in range(8):
        _write_pgm(
            frame_dir / f"frame_{idx:06d}.ppm",
            [1 + idx * 20, 240 - idx * 10, 40 + idx, 80 + idx],
        )
    train_path = tmp_path / "train_raw_temporal.jsonl"
    target_path = tmp_path / "target_raw_temporal.jsonl"

    def raw_row(idx: int, *, split: str) -> dict:
        row = _row(idx, split=split)
        row["frame"] = {
            "path": str(frame_dir / f"frame_{idx:06d}.ppm"),
            "index": idx,
            "width": 2,
            "height": 2,
        }
        row["ground_truth_tokens"] = ["KEY_PRESS_A", "MOUSE_LEFT_DOWN"] if idx % 3 == 0 else []
        return row

    _write_jsonl(train_path, [raw_row(i, split="train_core") for i in range(5)])
    _write_jsonl(target_path, [raw_row(i, split="eval") for i in range(5, 8)])
    summary = train_temporal_masked_diffusion_idm(
        {
            "model_name": "unit_temporal_masked_diffusion_raw_video_idm",
            "train_records": str(train_path),
            "target_records": str(target_path),
            "output_dir": str(tmp_path / "out_raw_temporal"),
            "summary_out": str(tmp_path / "summary_raw_temporal.json"),
            "max_train_rows": 5,
            "max_target_rows": 3,
            "max_action_tokens_per_bin": 4,
            "temporal_offsets": [0],
            "temporal_loss_offsets": [0],
            "video_feature_source": "raw_frames",
            "raw_video_image_size": 2,
            "raw_video_frame_offsets": [0, 1],
            "raw_video_feature_storage": "tensor",
            "precompute_features_as_tensor": True,
            "precompute_feature_tensor_dtype": "float16",
            "raw_video_missing_frame_policy": "zero",
            "video_feature_dim": 8,
            "video_encoder_arch": "raw_video_patch_cnn",
            "raw_video_encoder_channels": 2,
            "raw_video_encoder_token_frames": 2,
            "raw_video_encoder_token_hw": 1,
            "hidden_dim": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "batch_size": 2,
            "prediction_batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "mask_probability": 0.75,
            "full_action_mask_probability": 1.0,
            "random_token_probability": 0.0,
            "diffusion_steps": 2,
            "video_encoder_pretrain_epochs": 1,
            "video_encoder_pretrain_mask_probability": 0.5,
            "video_reconstruction_aux_weight": 0.05,
            "force_cpu": True,
            "noop_loss_weight": 0.1,
            "keyboard_loss_weight": 2.0,
            "mouse_button_loss_weight": 3.0,
        }
    )
    assert summary["schema"] == "temporal_masked_diffusion_idm_train_summary.v1"
    assert summary["status"] == "pass"
    assert summary["video_feature_source"] == "raw_frames"
    assert summary["video_encoder_arch"] == "raw_video_patch_cnn"
    assert summary["raw_video_frame_offsets"] == [0, 1]
    assert summary["raw_video_image_size"] == 2
    assert summary["video_tokens_per_offset"] == 2
    assert summary["loss_weights"]["full_action_mask_probability"] == 1.0
    assert summary["video_encoder_pretrain_history"]
    assert Path(summary["checkpoint_path"]).exists()
    assert read_json(summary["metrics_path"])["alignment"]["rows_seen"] == 3


def test_temporal_center_candidates_can_gate_buttons_by_no_button_class():
    if not torch_available():
        return
    import torch

    vocab = ["<FDM1_ACTION_PAD>", "<FDM1_ACTION_MASK>", "NOOP", "MOUSE_LEFT_DOWN"]
    probabilities = torch.tensor([[[0.01, 0.01, 0.18, 0.80]]], dtype=torch.float32)
    ungated = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={"non_noop_budget_candidates_per_row": 4},
        event_probabilities={"button_class": torch.tensor([[0.95, 0.05]], dtype=torch.float32)},
    )
    gated = _temporal_center_candidates(
        probabilities,
        vocab=vocab,
        config={
            "non_noop_budget_candidates_per_row": 4,
            "button_class_no_button_gate_power": 2.0,
            "button_class_no_button_gate_floor": 0.0,
        },
        event_probabilities={"button_class": torch.tensor([[0.95, 0.05]], dtype=torch.float32)},
    )
    ungated_button = next(item for item in ungated[0] if item["token"] == "MOUSE_LEFT_DOWN")
    gated_button = next(item for item in gated[0] if item["token"] == "MOUSE_LEFT_DOWN")
    assert ungated_button["score"] == pytest.approx(0.8)
    assert gated_button["button_class_no_button_gate_score"] == pytest.approx(0.05)
    assert gated_button["event_gate_multiplier"] == pytest.approx(0.05**2)
    assert gated_button["score"] < ungated_button["score"] * 0.01
