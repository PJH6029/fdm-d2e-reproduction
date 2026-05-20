import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.training.neural_idm import TinyMouseIDM, record_features, target_mouse_delta, tokens_from_delta, train_idm_variant


def rec(idx, dx_token, dy_token):
    return {
        "sequence_id": f"r#{idx}",
        "recording_id": "r",
        "game": "g",
        "timestamp_ns": idx,
        "bin_index": idx,
        "frame": {"path": f"f{idx}.ppm", "index": idx, "features": [idx / 10, 0.0, 0.0, 0.0, 0.0]},
        "ground_truth_tokens": [dx_token, dy_token],
    }


class NeuralIDMTests(unittest.TestCase):
    def test_target_mouse_delta_reduces_tokens(self):
        self.assertEqual(target_mouse_delta(rec(0, "MOUSE_DX_P2", "MOUSE_DY_N1")), (2.0, -1.0))
        self.assertEqual(tokens_from_delta(2.2, -0.5), ["MOUSE_DX_P2", "MOUSE_DY_N1"])

    def test_tiny_mouse_idm_trains_and_writes_pseudolabels(self):
        train = [rec(i, "MOUSE_DX_P1" if i < 3 else "MOUSE_DX_N1", "MOUSE_DY_Z0") for i in range(6)]
        target = [rec(7, "MOUSE_DX_N1", "MOUSE_DY_Z0")]
        model = TinyMouseIDM(input_dim=16, hidden_dim=4, seed=1).fit(train, epochs=20, lr=0.01)
        dx, dy = model.predict_delta(target[0])
        self.assertIsInstance(dx, float)
        self.assertIsInstance(dy, float)
        with tempfile.TemporaryDirectory() as td:
            result = train_idm_variant(train, target, model_name="unit_idm", hidden_dim=4, epochs=20, lr=0.01, seed=1, confidence_threshold=0.0, output_dir=td)
            self.assertTrue((Path(td) / "unit_idm" / "pseudolabels.jsonl").exists())
            self.assertEqual(result["metadata"]["schema"], "idm_checkpoint_metadata.v1")

    def test_record_features_can_include_dependency_free_frame_pair_signal(self):
        with tempfile.TemporaryDirectory() as td:
            frame_dir = Path(td)
            first = frame_dir / "frame_000001.ppm"
            second = frame_dir / "frame_000002.ppm"

            def write_ppm(path: Path, offset: int) -> None:
                pixels = bytearray()
                for y in range(16):
                    for x in range(16):
                        value = (x * 7 + y * 3 + offset) % 256
                        pixels.extend([value, value // 2, 255 - value])
                path.write_bytes(b"P6\n16 16\n255\n" + bytes(pixels))

            write_ppm(first, 0)
            write_ppm(second, 11)
            row = rec(0, "MOUSE_DX_P1", "MOUSE_DY_Z0")
            row["frame"]["path"] = str(first)

            summary = record_features(row)
            rich = record_features(row, feature_mode="summary_grid4_shift")
            richer = record_features(row, feature_mode="summary_grid8_shift")
            timed = record_features(row, feature_mode="summary_grid8_shift_time")
            surfaced = record_features(row, feature_mode="summary_grid8_shift_surface_time")

            self.assertEqual(len(summary), 16)
            self.assertEqual(len(rich), 164)
            self.assertEqual(len(richer), 596)
            self.assertEqual(len(timed), 608)
            self.assertEqual(len(surfaced), 620)
            self.assertTrue(any(abs(value) > 0 for value in rich[16:]))
            self.assertTrue(any(abs(value) > 0 for value in richer[16:]))
            self.assertTrue(any(abs(value) > 0 for value in timed[596:]))
            self.assertTrue(any(abs(value) > 0 for value in surfaced[596:608]))
            self.assertTrue(any(abs(value) > 0 for value in surfaced[608:]))


if __name__ == "__main__":
    unittest.main()
