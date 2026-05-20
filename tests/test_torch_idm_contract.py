import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.training.torch_idm import _mouse_baseline_deltas, torch_available


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


if __name__ == "__main__":
    unittest.main()
