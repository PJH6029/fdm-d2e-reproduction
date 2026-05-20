import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.training.torch_idm import (
    _calibrated_category_thresholds_from_scores,
    _mouse_baseline_deltas,
    _split_calibration_records,
    torch_available,
)


class TorchIDMContractTests(unittest.TestCase):
    def test_torch_availability_probe_is_boolean(self):
        self.assertIsInstance(torch_available(), bool)

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


if __name__ == "__main__":
    unittest.main()
