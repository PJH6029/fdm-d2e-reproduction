import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import read_jsonl, write_json, write_jsonl
from fdm_d2e.training.calibrated_fdm import _scale_predictions, calibrate_fdm_predictions


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

    def test_calibration_records_can_be_separate_from_baseline_train_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_predictions = root / "source_predictions.jsonl"
            fdm_train = root / "fdm_train_pseudolabels.jsonl"
            calibration_train = root / "d2e_train_ground_truth.jsonl"
            target = root / "target.jsonl"
            labels = root / "labels.jsonl"
            endpoints = root / "endpoints.json"
            out = root / "out"
            write_jsonl(
                source_predictions,
                [
                    {
                        "schema": "fdm_prediction.v1",
                        "sequence_id": "r#2",
                        "recording_id": "r",
                        "game": "g",
                        "timestamp_ns": 2,
                        "predicted_tokens": ["MOUSE_DX_P1", "MOUSE_DY_Z0"],
                    }
                ],
            )
            calibration_predictions = root / "calibration_predictions.jsonl"
            write_jsonl(
                calibration_predictions,
                [
                    {
                        "schema": "fdm_prediction.v1",
                        "sequence_id": "r#1",
                        "recording_id": "r",
                        "game": "g",
                        "timestamp_ns": 1,
                        "predicted_tokens": ["MOUSE_DX_P1", "MOUSE_DY_Z0"],
                    }
                ],
            )
            write_jsonl(
                fdm_train,
                [
                    {
                        "sequence_id": "r#0",
                        "recording_id": "r",
                        "game": "g",
                        "timestamp_ns": 0,
                        "ground_truth_tokens": ["MOUSE_DX_P1", "MOUSE_DY_Z0"],
                    }
                ],
            )
            write_jsonl(
                calibration_train,
                [
                    {
                        "sequence_id": "r#0",
                        "recording_id": "r",
                        "game": "g",
                        "timestamp_ns": 0,
                        "ground_truth_tokens": ["MOUSE_DX_P3", "MOUSE_DY_Z0"],
                    }
                ],
            )
            write_jsonl(
                target,
                [
                    {
                        "sequence_id": "r#2",
                        "recording_id": "r",
                        "game": "g",
                        "timestamp_ns": 2,
                        "ground_truth_tokens": ["MOUSE_DX_P3", "MOUSE_DY_Z0"],
                    }
                ],
            )
            write_jsonl(labels, [{"sequence_id": "r#0", "predicted_tokens": ["MOUSE_DX_P1"]}])
            write_json(
                endpoints,
                {
                    "schema": "primary_endpoints.v1",
                    "reference_baseline": "noop",
                    "endpoints": [],
                },
            )

            summary = calibrate_fdm_predictions(
                {
                    "model_name": "calibrated_test",
                    "source_predictions_path": str(source_predictions),
                    "train_records_path": str(fdm_train),
                    "baseline_train_records_path": str(fdm_train),
                    "calibration_records_path": str(calibration_train),
                    "calibration_predictions_path": str(calibration_predictions),
                    "calibration_label_source": "d2e_train_ground_truth",
                    "target_records_path": str(target),
                    "labels_path": str(labels),
                    "endpoints": str(endpoints),
                    "output_dir": str(out),
                    "min_gain": 0.25,
                    "max_gain": 4.0,
                }
            )

            calibrated = read_jsonl(summary["predictions_path"])
            checkpoint = summary["checkpoint"]
            self.assertEqual(calibrated[0]["predicted_tokens"], ["MOUSE_DX_P3", "MOUSE_DY_Z0"])
            self.assertEqual(checkpoint["train_records_path"], str(fdm_train))
            self.assertEqual(checkpoint["baseline_train_records_path"], str(fdm_train))
            self.assertEqual(checkpoint["calibration_records_path"], str(calibration_train))
            self.assertEqual(checkpoint["calibration_predictions_path"], str(calibration_predictions))
            self.assertEqual(checkpoint["calibration_label_source"], "d2e_train_ground_truth")
            self.assertFalse(checkpoint["calibration_uses_target_ground_truth"])
            self.assertFalse(checkpoint["calibration_uses_target_prediction_distribution"])
            self.assertEqual(checkpoint["calibration"]["prediction_reference"], "calibration_predictions")


if __name__ == "__main__":
    unittest.main()
