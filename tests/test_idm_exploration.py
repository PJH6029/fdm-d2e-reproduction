from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.idm_exploration import build_idm_exploration_summary, write_idm_exploration_summary


def _metrics(*, good: bool = True) -> dict:
    return {
        "schema": "metrics.v1",
        "num_examples": 100,
        "keyboard": {"accuracy": 0.25 if good else 0.01},
        "mouse_button": {
            "accuracy": 0.3 if good else 0.01,
            "f1": 0.24 if good else 0.01,
            "no_button_false_positive_rate": 0.03 if good else 0.3,
        },
        "mouse_move": {"pearson": 0.31 if good else 0.01, "scale_ratio": 1.2},
    }


def _fixture(root: Path) -> dict:
    write_json(root / "g002.json", {"status": "pass", "error_count": 0})
    write_json(root / "g003.json", {"status": "pass", "error_count": 0})
    write_json(root / "current_metrics.json", _metrics(good=False))
    write_json(
        root / "sweep.json",
        {
            "schema": "idm_torch_sweep.v1",
            "num_runs": 2,
            "top_variants": [
                {
                    "variant": "good",
                    "model_name": "candidate",
                    "config": {"feature_mode": "summary_grid8_shift_surface_time"},
                    "metrics": _metrics(good=True),
                }
            ],
        },
    )
    write_json(
        root / "portfolio.json",
        {
            "schema": "idm_prediction_portfolio_summary.v1",
            "model_name": "portfolio",
            "metrics": _metrics(good=True),
        },
    )
    (root / "streaming.py").write_text(
        "_calibrate_streaming_category_thresholds group_fbeta_calibrated category_thresholds "
        "_calibrate_streaming_mouse_output_gain train_abs_ratio mouse_output_gain_info",
        encoding="utf-8",
    )
    (root / "torch_idm.py").write_text("luma_temporal_conv _build_luma_temporal_conv_model", encoding="utf-8")
    write_json(
        root / "candidate_a.yaml",
        {
            "model_name": "candidate_a",
            "feature_mode": "summary_compact_grid8_shift_surface_time",
            "category_threshold_mode": "group_fbeta_calibrated",
            "category_calibration_beta": 0.5,
            "category_calibration_max_examples": 100,
            "mouse_output_gain_mode": "train_abs_ratio",
            "mouse_gain_calibration_max_examples": 100,
            "training_cache_shard_assignment": "greedy_rows",
            "prediction_workers": 2,
        },
    )
    return {
        "output_path": "summary.json",
        "prerequisites": {"g002": "g002.json", "g003": "g003.json"},
        "current_full_idm_metrics": "current_metrics.json",
        "sweep_evidence": [{"id": "sweep", "path": "sweep.json"}],
        "portfolio_evidence": [{"id": "portfolio", "path": "portfolio.json"}],
        "progress_criteria": {
            "min_small_medium_mouse_button_f1": 0.2,
            "max_small_medium_no_button_fpr": 0.1,
            "min_small_medium_mouse_pearson": 0.25,
        },
        "full_corpus_candidate_configs": ["candidate_a.yaml"],
        "implementation_hooks": {
            "streaming": {
                "path": "streaming.py",
                "patterns": ["_calibrate_streaming_category_thresholds", "train_abs_ratio"],
            },
            "luma": {"path": "torch_idm.py", "patterns": ["luma_temporal_conv"]},
        },
    }


def test_idm_exploration_summary_passes_complete_fixture(tmp_path):
    payload = build_idm_exploration_summary(_fixture(tmp_path), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert payload["ranked_candidates"][0]["metrics_snapshot"]["mouse_button_f1"] == 0.24
    assert payload["full_corpus_candidate_configs"][0]["config"]["category_threshold_mode"] == "group_fbeta_calibrated"


def test_idm_exploration_summary_fails_without_calibrated_candidate(tmp_path):
    config = _fixture(tmp_path)
    write_json(
        tmp_path / "candidate_a.yaml",
        {
            "model_name": "candidate_a",
            "category_threshold_mode": "global",
            "mouse_output_gain_mode": "fixed",
            "training_cache_shard_assignment": "round_robin",
        },
    )
    payload = build_idm_exploration_summary(config, root=tmp_path)
    assert payload["status"] == "fail"
    codes = {finding["code"] for finding in payload["findings"]}
    assert "candidate_missing_streaming_category_calibration" in codes
    assert "candidate_missing_mouse_gain_calibration" in codes


def test_idm_exploration_summary_writes_output(tmp_path):
    payload = write_idm_exploration_summary(_fixture(tmp_path), root=tmp_path)
    written = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert payload["schema"] == "idm_exploration_summary.v1"
    assert '"status": "pass"' in written
