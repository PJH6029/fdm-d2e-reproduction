import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.baselines import build_baseline_predictions
from fdm_d2e.eval.statistics import cluster_bootstrap_delta, compare_systems, holm_bonferroni, values_by_cluster


GT = [
    {"sequence_id": "r1#0", "recording_id": "r1", "game": "g", "timestamp_ns": 0, "ground_truth_tokens": ["KEY_PRESS_W", "MOUSE_DX_P2", "MOUSE_DY_Z0"]},
    {"sequence_id": "r1#1", "recording_id": "r1", "game": "g", "timestamp_ns": 1, "ground_truth_tokens": ["KEY_PRESS_W", "MOUSE_DX_P2", "MOUSE_DY_Z0"]},
    {"sequence_id": "r2#0", "recording_id": "r2", "game": "g", "timestamp_ns": 0, "ground_truth_tokens": ["MOUSE_LEFT_DOWN"]},
    {"sequence_id": "r2#1", "recording_id": "r2", "game": "g", "timestamp_ns": 1, "ground_truth_tokens": ["MOUSE_LEFT_DOWN"]},
]


class EvalStatisticsTests(unittest.TestCase):
    def test_baselines_include_noop_and_majority(self):
        preds = build_baseline_predictions(GT[:2], GT[2:], baseline_names=["noop", "global_majority"])
        self.assertEqual(preds["noop"][0]["predicted_tokens"], ["NOOP"])
        self.assertEqual(preds["global_majority"][0]["predicted_tokens"], ["KEY_PRESS_W", "MOUSE_DX_P2", "MOUSE_DY_Z0"])

    def test_values_by_cluster_computes_endpoint(self):
        perfect = [{"sequence_id": row["sequence_id"], "predicted_tokens": row["ground_truth_tokens"], "timestamp_ns": row["timestamp_ns"]} for row in GT]
        endpoint = {"name": "keyboard_accuracy", "metric_path": ["keyboard", "accuracy"], "direction": "higher"}
        values = values_by_cluster(perfect, GT, endpoint)
        self.assertEqual(values["r1"], 1.0)
        self.assertNotIn("r2", values)

    def test_bootstrap_delta_positive_for_better_candidate(self):
        stats = cluster_bootstrap_delta({"r1": 1.0, "r2": 1.0}, {"r1": 0.0, "r2": 0.0}, direction="higher", n_resamples=100, seed=1)
        self.assertEqual(stats["status"], "computed")
        self.assertGreater(stats["delta"], 0)
        self.assertEqual(stats["ci"], [1.0, 1.0])

    def test_holm_bonferroni_adjusts_monotonically(self):
        adjusted = holm_bonferroni([{"p_value": 0.01}, {"p_value": 0.03}, {"p_value": 0.20}])
        self.assertLessEqual(adjusted[0]["p_adjusted_holm"], adjusted[1]["p_adjusted_holm"])
        self.assertTrue(adjusted[0]["reject_holm_0_05"])
        self.assertFalse(adjusted[2]["reject_holm_0_05"])

    def test_compare_systems_uses_endpoint_config(self):
        perfect = [{"sequence_id": row["sequence_id"], "predicted_tokens": row["ground_truth_tokens"], "timestamp_ns": row["timestamp_ns"]} for row in GT]
        noop = [{"sequence_id": row["sequence_id"], "predicted_tokens": ["NOOP"], "timestamp_ns": row["timestamp_ns"]} for row in GT]
        config = {
            "reference_baseline": "noop",
            "cluster_key": "recording_id",
            "bootstrap": {"n_resamples": 100, "confidence": 0.95, "seed": 7},
            "correction": "holm_bonferroni",
            "endpoints": [{"name": "keyboard_accuracy", "metric_path": ["keyboard", "accuracy"], "direction": "higher", "min_effect": 0.05}],
        }
        comparison = compare_systems({"noop": noop, "perfect": perfect}, GT, config)
        self.assertEqual(comparison["schema"], "stat_comparison.v1")
        self.assertEqual(comparison["comparisons"][0]["model"], "perfect")


if __name__ == "__main__":
    unittest.main()
