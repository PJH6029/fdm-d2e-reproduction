import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import read_json, write_jsonl
from fdm_d2e.rollout.game_harness import candidate_catalog, prediction_controls, run_game_harness_eval


class GameHarnessContractTests(unittest.TestCase):
    def test_candidate_catalog_has_five_game_like_candidates(self):
        candidates = candidate_catalog()
        self.assertGreaterEqual(len(candidates), 5)
        self.assertTrue(all("game" in item["type"] for item in candidates))
        self.assertTrue(all(item["install"] == "repo_local_python_no_external_dependency" for item in candidates))

    def test_prediction_controls_decode_keyboard_mouse_and_clicks(self):
        frames = prediction_controls(
            [
                {"timestamp_ns": 0, "predicted_tokens": ["KEY_PRESS_68", "MOUSE_DX_N2", "MOUSE_DY_P1", "MOUSE_LEFT_DOWN"]},
                {"timestamp_ns": 1, "predicted_tokens": ["KEY_RELEASE_68"]},
            ]
        )
        self.assertEqual(frames[0].move_x, 1)
        self.assertEqual(frames[0].mouse_dx, -2.0)
        self.assertEqual(frames[0].mouse_dy, 1.0)
        self.assertTrue(frames[0].click)
        self.assertEqual(frames[1].move_x, 0)

    def test_game_harness_quality_gate_passes_on_controlled_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            output = root / "harness.json"
            rows = []
            for idx in range(20):
                rows.append({"timestamp_ns": idx, "predicted_tokens": ["KEY_PRESS_68", "KEY_PRESS_87", "MOUSE_DX_N3", "MOUSE_DY_P1"]})
            write_jsonl(predictions, rows)
            result = run_game_harness_eval(
                {
                    "predictions_path": str(predictions),
                    "output_path": str(output),
                    "action_limit": 20,
                    "thresholds": {"min_valid_action_rate": 0.98, "max_crashes": 0, "min_progress_score": 0.1},
                    "tasks": [
                        {"task": "grid", "environment": "grid_target_arena", "target_x": 16, "target_y": 2},
                        {"task": "aim", "environment": "aim_click_arena", "target_x": -30, "target_y": 10},
                        {"task": "runner", "environment": "dodge_runner_arena", "target_steps": 20},
                    ],
                }
            )
            self.assertEqual(result["quality_gate"]["status"], "pass")
            self.assertEqual(read_json(output)["quality_gate"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
