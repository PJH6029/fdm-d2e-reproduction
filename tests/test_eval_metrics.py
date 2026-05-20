import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fdm_d2e.eval.action_metrics import compute_metrics


class EvalMetricTests(unittest.TestCase):
    def test_metrics_compute_keyboard_and_mouse(self):
        gt = [{'sequence_id': 'a', 'ground_truth_tokens': ['KEY_PRESS_W', 'MOUSE_DX_P2', 'MOUSE_DY_Z0', 'MOUSE_LEFT_DOWN']}]
        pred = [{'sequence_id': 'a', 'timestamp_ns': 0, 'predicted_tokens': ['KEY_PRESS_W', 'MOUSE_DX_P2', 'MOUSE_DY_Z0', 'MOUSE_LEFT_DOWN']}]
        metrics = compute_metrics(pred, gt)
        self.assertEqual(metrics['keyboard']['accuracy'], 1.0)
        self.assertEqual(metrics['mouse_button']['accuracy'], 1.0)
        self.assertEqual(metrics['mouse_move']['status'], 'computed')


if __name__ == '__main__':
    unittest.main()
