import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.training.calibrated_fdm import _scale_predictions


class CalibratedFDMContractTests(unittest.TestCase):
    def test_recording_scale_calibration_uses_train_only_motion_scale(self):
        train = [
            {"sequence_id": "r#0", "recording_id": "r", "game": "g", "ground_truth_tokens": ["MOUSE_DX_P3", "MOUSE_DY_Z0"]},
            {"sequence_id": "r#1", "recording_id": "r", "game": "g", "ground_truth_tokens": ["MOUSE_DX_P3", "MOUSE_DY_Z0"]},
        ]
        predictions = [
            {"sequence_id": "r#2", "recording_id": "r", "game": "g", "timestamp_ns": 2, "predicted_tokens": ["MOUSE_DX_P1", "MOUSE_DY_Z0", "KEY_PRESS_87"]},
        ]

        calibrated, diagnostics = _scale_predictions(predictions, train, min_gain=0.25, max_gain=4.0)

        self.assertEqual(calibrated[0]["predicted_tokens"], ["MOUSE_DX_P3", "MOUSE_DY_Z0", "KEY_PRESS_87"])
        self.assertEqual(diagnostics["recording_gains"][0]["source"], "recording")
        self.assertGreater(diagnostics["recording_gains"][0]["gain"], 1.0)


if __name__ == "__main__":
    unittest.main()
