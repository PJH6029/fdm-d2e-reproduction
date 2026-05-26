import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.training.torch_idm import (
    _action_history_features,
    _axis_class_indices,
    _axis_class_to_delta,
    _axis_suffix_from_delta,
    _build_model,
    _button_target_indices,
    _calibrated_category_thresholds_from_scores,
    _calibrated_button_softmax_threshold_from_scores,
    _calibrated_group_fbeta_thresholds_from_scores,
    _calibrated_group_thresholds_from_scores,
    _prediction_from_output,
    _mouse_baseline_deltas,
    _seed_mouse_delta_state,
    _split_calibration_records,
    button_softmax_classes,
    require_torch,
    torch_available,
)
from fdm_d2e.tokenization.actions import token_to_delta_class


class TorchIDMContractTests(unittest.TestCase):
    def test_torch_availability_probe_is_boolean(self):
        self.assertIsInstance(torch_available(), bool)

    def test_luma_temporal_conv_model_accepts_feature_plus_history_tail(self):
        if not torch_available():
            self.skipTest("torch extra is not installed")
        torch = require_torch()
        model = _build_model(
            torch,
            input_dim=2332 + 2 + 13,
            output_dim=7,
            hidden_dim=8,
            depth=1,
            dropout=0.0,
            config={"model_arch": "luma_temporal_conv", "visual_conv_channels": 2, "visual_conv_pool_hw": 2},
            feature_mode="summary_luma16_stack5_time",
        )

        out = model(torch.zeros((3, 2332 + 2 + 13), dtype=torch.float32))

        self.assertEqual(tuple(out.shape), (3, 7))

    def test_luma_temporal_conv_model_accepts_compact_luma_pair_features(self):
        if not torch_available():
            self.skipTest("torch extra is not installed")
        torch = require_torch()
        model = _build_model(
            torch,
            input_dim=812,
            output_dim=7,
            hidden_dim=8,
            depth=1,
            dropout=0.0,
            config={"model_arch": "luma_temporal_conv", "visual_conv_channels": 2, "visual_conv_pool_hw": 2},
            feature_mode="summary_compact_luma16_pair_shift_time",
        )

        out = model(torch.zeros((3, 812), dtype=torch.float32))

        self.assertEqual(tuple(out.shape), (3, 7))

    def test_luma_temporal_conv_model_accepts_compact_luma_window_features(self):
        if not torch_available():
            self.skipTest("torch extra is not installed")
        torch = require_torch()
        model = _build_model(
            torch,
            input_dim=2337,
            output_dim=7,
            hidden_dim=8,
            depth=1,
            dropout=0.0,
            config={"model_arch": "luma_temporal_conv", "visual_conv_channels": 2, "visual_conv_pool_hw": 2},
            feature_mode="summary_compact_luma16_window5_time",
        )

        out = model(torch.zeros((3, 2337), dtype=torch.float32))

        self.assertEqual(tuple(out.shape), (3, 7))

    def test_luma_action_sequence_prior_parses_compact_luma_and_history(self):
        if not torch_available():
            self.skipTest("torch extra is not installed")
        torch = require_torch()
        history_len = 2
        history_vocab_dim = 5
        history_dim = (2 * history_len) + (history_vocab_dim * history_len) + 3
        model = _build_model(
            torch,
            input_dim=812 + history_dim,
            output_dim=7,
            hidden_dim=8,
            depth=1,
            dropout=0.0,
            config={
                "model_arch": "luma_action_sequence_prior",
                "action_history_len": history_len,
                "sequence_token_dim": 16,
                "sequence_transformer_heads": 2,
                "sequence_transformer_layers": 1,
                "sequence_transformer_ff_dim": 32,
            },
            feature_mode="summary_compact_luma16_pair_shift_time",
        )

        out = model(torch.zeros((3, 812 + history_dim), dtype=torch.float32))

        self.assertEqual(tuple(out.shape), (3, 7))

    def test_luma_action_sequence_prior_accepts_stack5_future_context(self):
        if not torch_available():
            self.skipTest("torch extra is not installed")
        torch = require_torch()
        history_len = 2
        history_vocab_dim = 5
        history_dim = (2 * history_len) + (history_vocab_dim * history_len) + 3
        model = _build_model(
            torch,
            input_dim=2332 + history_dim,
            output_dim=7,
            hidden_dim=8,
            depth=1,
            dropout=0.0,
            config={
                "model_arch": "luma_action_sequence_prior",
                "action_history_len": history_len,
                "visual_stack_frames": 5,
                "sequence_token_dim": 16,
                "sequence_transformer_heads": 2,
                "sequence_transformer_layers": 1,
                "sequence_transformer_ff_dim": 32,
            },
            feature_mode="summary_luma16_stack5_time",
        )

        out = model(torch.zeros((3, 2332 + history_dim), dtype=torch.float32))

        self.assertEqual(tuple(out.shape), (3, 7))

    def test_residual_mouse_baselines_are_causal_for_train_and_last_seen_for_target(self):
        train = [
            {"sequence_id": "r#0", "recording_id": "r", "game": "g", "timestamp_ns": 0, "ground_truth_tokens": ["MOUSE_DX_P1", "MOUSE_DY_Z0"]},
            {"sequence_id": "r#1", "recording_id": "r", "game": "g", "timestamp_ns": 1, "ground_truth_tokens": ["MOUSE_DX_P2", "MOUSE_DY_N1"]},
        ]
        target = [
            {"sequence_id": "r#2", "recording_id": "r", "game": "g", "timestamp_ns": 2, "ground_truth_tokens": ["MOUSE_DX_P3", "MOUSE_DY_Z0"]},
        ]

        self.assertEqual(_mouse_baseline_deltas(train, mode="causal_last_seen"), [(0.0, 0.0), (1.0, 0.0)])
        self.assertEqual(_mouse_baseline_deltas(target, mode="target_last_seen_train", train_records=train), [(2.0, -1.0)])

    def test_seed_mouse_delta_state_uses_latest_train_delta_per_recording_and_game(self):
        records = [
            {"sequence_id": "r#1", "recording_id": "r", "game": "g", "timestamp_ns": 1, "ground_truth_tokens": ["MOUSE_DX_P2", "MOUSE_DY_Z0"]},
            {"sequence_id": "r#0", "recording_id": "r", "game": "g", "timestamp_ns": 0, "ground_truth_tokens": ["MOUSE_DX_P1", "MOUSE_DY_N1"]},
            {"sequence_id": "s#0", "recording_id": "s", "game": "h", "timestamp_ns": 0, "ground_truth_tokens": ["MOUSE_DX_N1", "MOUSE_DY_P1"]},
        ]

        by_recording, by_game, fallback = _seed_mouse_delta_state(records)

        self.assertEqual(by_recording["r"], (2.0, 0.0))
        self.assertEqual(by_game["g"], (2.0, 0.0))
        self.assertEqual(by_recording["s"], (-1.0, 1.0))
        self.assertEqual(fallback, (-1.0, 1.0))

    def test_calibration_split_uses_training_tail_per_recording(self):
        records = [
            {"sequence_id": f"r#{idx}", "recording_id": "r", "timestamp_ns": idx, "ground_truth_tokens": []}
            for idx in range(5)
        ] + [
            {"sequence_id": f"s#{idx}", "recording_id": "s", "timestamp_ns": idx, "ground_truth_tokens": []}
            for idx in range(3)
        ]

        fit, calibration = _split_calibration_records(records, fraction=0.4)

        self.assertEqual([row["sequence_id"] for row in fit], ["r#0", "r#1", "r#2", "s#0", "s#1"])
        self.assertEqual([row["sequence_id"] for row in calibration], ["r#3", "r#4", "s#2"])

    def test_per_token_calibration_prefers_threshold_with_better_f_score(self):
        thresholds, diagnostics = _calibrated_category_thresholds_from_scores(
            score_rows=[[0.9], [0.8], [0.3], [0.1]],
            label_rows=[[1], [1], [0], [0]],
            vocab=["MOUSE_LEFT_DOWN"],
            default_threshold=0.5,
            grid=[0.2, 0.5, 0.85],
            beta=1.0,
        )

        self.assertEqual(thresholds["MOUSE_LEFT_DOWN"], 0.5)
        self.assertEqual(diagnostics["per_token"]["MOUSE_LEFT_DOWN"]["tp"], 2)
        self.assertEqual(diagnostics["per_token"]["MOUSE_LEFT_DOWN"]["fp"], 0)

    def test_group_exact_calibration_uses_separate_keyboard_and_button_thresholds(self):
        vocab = ["KEY_PRESS_87", "MOUSE_LEFT_DOWN"]
        thresholds, diagnostics = _calibrated_group_thresholds_from_scores(
            score_rows=[
                [0.8, 0.2],
                [0.4, 0.8],
                [0.7, 0.1],
            ],
            label_rows=[
                [1, 0],
                [0, 1],
                [1, 0],
            ],
            vocab=vocab,
            default_threshold=0.5,
            grid=[0.15, 0.5, 0.75],
        )

        self.assertEqual(thresholds["KEY_PRESS_87"], 0.5)
        self.assertEqual(thresholds["MOUSE_LEFT_DOWN"], 0.75)
        self.assertEqual(diagnostics["per_group"]["keyboard"]["accuracy"], 1.0)
        self.assertEqual(diagnostics["per_group"]["mouse_button"]["accuracy"], 1.0)

    def test_group_fbeta_calibration_penalizes_no_button_false_positives(self):
        vocab = ["MOUSE_LEFT_DOWN"]
        thresholds, diagnostics = _calibrated_group_fbeta_thresholds_from_scores(
            score_rows=[[0.9], [0.7], [0.6], [0.1]],
            label_rows=[[1], [0], [0], [0]],
            vocab=vocab,
            default_threshold=0.5,
            grid=[0.5, 0.8],
            beta=0.5,
        )

        self.assertEqual(thresholds["MOUSE_LEFT_DOWN"], 0.8)
        self.assertEqual(diagnostics["per_group"]["mouse_button"]["tp"], 1)
        self.assertEqual(diagnostics["per_group"]["mouse_button"]["fp"], 0)

    def test_button_softmax_classes_make_no_button_explicit(self):
        records = [
            {"sequence_id": "r#0", "ground_truth_tokens": ["MOUSE_LEFT_DOWN"]},
            {"sequence_id": "r#1", "ground_truth_tokens": []},
            {"sequence_id": "r#2", "ground_truth_tokens": ["MOUSE_LEFT_DOWN"]},
            {"sequence_id": "r#3", "ground_truth_tokens": ["MOUSE_RIGHT_UP"]},
        ]

        classes = button_softmax_classes(records, min_count=2)

        self.assertEqual(classes, [(), ("MOUSE_LEFT_DOWN",)])
        self.assertEqual(_button_target_indices(records, classes), [1, 0, 1, 0])

    def test_button_softmax_calibration_prefers_precision_over_spam(self):
        threshold, diagnostics = _calibrated_button_softmax_threshold_from_scores(
            score_rows=[
                [0.05, 0.95],
                [0.25, 0.75],
                [0.35, 0.65],
                [0.9, 0.1],
            ],
            label_indices=[1, 0, 0, 0],
            button_classes=[(), ("MOUSE_LEFT_DOWN",)],
            default_threshold=0.5,
            grid=[0.5, 0.8],
            beta=0.5,
        )

        self.assertEqual(threshold, 0.8)
        self.assertEqual(diagnostics["tp"], 1)
        self.assertEqual(diagnostics["fp"], 0)

    def test_mouse_axis_classes_match_delta_token_bins(self):
        records = [
            {"sequence_id": "r#0", "ground_truth_tokens": ["MOUSE_DX_P2", "MOUSE_DY_N1"]},
            {"sequence_id": "r#1", "ground_truth_tokens": ["MOUSE_DX_Z0", "MOUSE_DY_P3"]},
        ]

        dx_indices, dy_indices = _axis_class_indices(records, ["N3", "N2", "N1", "Z0", "P1", "P2", "P3"])

        self.assertEqual(_axis_suffix_from_delta(2.0, "MOUSE_DX_"), "P2")
        self.assertEqual(_axis_suffix_from_delta(-1.0, "MOUSE_DY_"), "N1")
        self.assertEqual(dx_indices, [5, 3])
        self.assertEqual(dy_indices, [2, 6])
        self.assertEqual(_axis_class_to_delta("P3"), 6.0)
        self.assertEqual(_axis_class_to_delta("N2"), -2.0)

    def test_prediction_softmax_button_head_emits_at_most_one_exact_set(self):
        _, _, tokens = _prediction_from_output(
            [0.0, 0.0, 3.0, 0.1, 2.5],
            base_dx=0.0,
            base_dy=0.0,
            residual_mouse=False,
            category_vocab=["KEY_PRESS_87"],
            category_thresholds={"KEY_PRESS_87": 0.5},
            category_threshold=0.5,
            button_head_mode="softmax",
            button_classes=[(), ("MOUSE_LEFT_DOWN",)],
            button_softmax_threshold=0.5,
        )

        self.assertIn("KEY_PRESS_87", tokens)
        self.assertIn("MOUSE_LEFT_DOWN", tokens)
        self.assertNotIn("MOUSE_LEFT_UP", tokens)

    def test_prediction_axis_softmax_overrides_regression_motion_tokens(self):
        _, _, tokens = _prediction_from_output(
            [0.0, 0.0, 0.1, 5.0, 0.2, 4.0, 0.3, 0.1],
            base_dx=0.0,
            base_dy=0.0,
            residual_mouse=False,
            category_vocab=[],
            category_thresholds={},
            category_threshold=0.5,
            button_head_mode="multilabel",
            mouse_head_mode="axis_softmax",
            mouse_axis_classes=["N1", "Z0", "P2"],
        )

        self.assertEqual(tokens[:2], ["MOUSE_DX_Z0", "MOUSE_DY_N1"])

    def test_prediction_axis_softmax_expected_decode_can_reduce_extreme_bins(self):
        dx, dy, tokens = _prediction_from_output(
            [0.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0],
            base_dx=0.0,
            base_dy=0.0,
            residual_mouse=False,
            category_vocab=[],
            category_thresholds={},
            category_threshold=0.5,
            button_head_mode="multilabel",
            mouse_head_mode="axis_softmax",
            mouse_axis_classes=["N1", "Z0", "P2"],
            mouse_axis_decode_mode="expected",
        )

        self.assertGreater(dx, 0.0)
        self.assertLess(dx, 1.0)
        self.assertLess(abs(dy), 0.01)
        self.assertEqual(tokens[0], "MOUSE_DX_P1")
        self.assertTrue(tokens[1].startswith("MOUSE_DY_"))

    def test_mouse_output_gain_rescales_decoded_motion_before_tokenization(self):
        dx, dy, tokens = _prediction_from_output(
            [2.0, -1.0],
            base_dx=0.0,
            base_dy=0.0,
            residual_mouse=False,
            category_vocab=[],
            category_thresholds={},
            category_threshold=0.5,
            mouse_output_gain=3.0,
        )

        self.assertEqual((dx, dy), (6.0, -3.0))
        self.assertEqual(tokens[:2], ["MOUSE_DX_P3", "MOUSE_DY_N2"])

    def test_prediction_can_emit_decomposed_mouse_tokens_for_paper_sums(self):
        dx, dy, tokens = _prediction_from_output(
            [30.0, -7.0],
            base_dx=0.0,
            base_dy=0.0,
            residual_mouse=False,
            category_vocab=[],
            category_thresholds={},
            category_threshold=0.5,
            mouse_emit_mode="decompose",
            mouse_max_tokens_per_axis=4,
        )

        self.assertEqual((dx, dy), (30.0, -7.0))
        self.assertGreater(len([token for token in tokens if token.startswith("MOUSE_DX_")]), 1)
        self.assertEqual(sum(float(token_to_delta_class(token) or 0.0) for token in tokens if token.startswith("MOUSE_DX_")), 30.0)
        self.assertEqual(sum(float(token_to_delta_class(token) or 0.0) for token in tokens if token.startswith("MOUSE_DY_")), -7.0)

    def test_action_history_features_are_causal_and_seedable(self):
        vocab = ["KEY_PRESS_87", "MOUSE_LEFT_DOWN", "MOUSE_LEFT_UP"]
        records = [
            {"sequence_id": "r#0", "recording_id": "r", "timestamp_ns": 0, "ground_truth_tokens": ["KEY_PRESS_87", "MOUSE_DX_P2", "MOUSE_DY_Z0", "MOUSE_LEFT_DOWN"]},
            {"sequence_id": "r#1", "recording_id": "r", "timestamp_ns": 1, "ground_truth_tokens": ["MOUSE_DX_N1", "MOUSE_DY_Z0", "MOUSE_LEFT_UP"]},
        ]

        features = _action_history_features(records, vocab, history_len=2)

        self.assertEqual(features[0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Slot 0 sees the previous row only: dx=+2 scaled by 8, key/button one-hots, and left button down.
        self.assertEqual(features[1][:4], [0.25, 0.0, 0.0, 0.0])
        self.assertEqual(features[1][4:7], [1.0, 1.0, 0.0])
        self.assertEqual(features[1][-3:], [1.0, 0.0, 0.0])

        target = [{"sequence_id": "r#2", "recording_id": "r", "timestamp_ns": 2, "ground_truth_tokens": []}]
        seeded = _action_history_features(target, vocab, history_len=2, seed_records=records)

        self.assertEqual(seeded[0][:4], [-0.125, 0.0, 0.25, 0.0])
        self.assertEqual(seeded[0][4:7], [0.0, 0.0, 1.0])
        self.assertEqual(seeded[0][7:10], [1.0, 1.0, 0.0])
        self.assertEqual(seeded[0][-3:], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
