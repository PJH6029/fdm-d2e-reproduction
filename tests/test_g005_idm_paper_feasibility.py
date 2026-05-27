from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.idm_paper_feasibility import build_idm_paper_target_feasibility
from fdm_d2e.io_utils import write_json, write_jsonl


def test_g005_feasibility_lookup_uses_train_context_without_target_leakage(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    contract = tmp_path / "contract.json"
    baseline = tmp_path / "baseline.json"
    feature = {
        "game": "Toy",
        "prior_action_tokens": ["KEY_DOWN_87", "MOUSE_LEFT_DOWN"],
        "prior_key_hold_bins": {"87": 2},
        "prior_button_hold_bins": {"LEFT": 1},
        "prior_since_key_transition_bins": 1,
        "prior_since_button_transition_bins": 1,
        "previous_event_tokens": ["MOUSE_DX_P1", "MOUSE_DY_Z0"],
    }
    write_jsonl(
        train,
        [
            {
                "sequence_id": "train#0",
                "recording_id": "train",
                **feature,
                "ground_truth_tokens": ["KEY_RELEASE_87", "MOUSE_LEFT_UP", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
            },
            {
                "sequence_id": "train#1",
                "recording_id": "train",
                **feature,
                "ground_truth_tokens": ["KEY_RELEASE_87", "MOUSE_LEFT_UP", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
            },
            {
                "sequence_id": "train#2",
                "recording_id": "train",
                "game": "Toy",
                "ground_truth_tokens": ["KEY_PRESS_65", "MOUSE_DX_Z0", "MOUSE_DY_Z0"],
            },
        ],
    )
    write_jsonl(
        target,
        [
            {
                "sequence_id": "target#0",
                "recording_id": "target",
                "eval_split_tags": ["temporal"],
                **feature,
                "ground_truth_tokens": ["KEY_RELEASE_87", "MOUSE_LEFT_UP", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
            }
        ],
    )
    write_json(
        contract,
        {
            "target_sequence": {
                "phase_1": {
                    "primary_targets": {
                        "pearson_x": 0.5,
                        "pearson_y": 0.5,
                        "keyboard_accuracy": 0.5,
                        "mouse_button_accuracy": 0.5,
                        "scale_ratio_x_max": 2.0,
                        "scale_ratio_y_max": 2.0,
                    }
                }
            }
        },
    )
    write_json(
        baseline,
        {
            "groups": {
                "all": {
                    "paper_compatible": {
                        "keyboard": {"key_accuracy": 0.1},
                        "mouse_button": {"button_accuracy": 0.1},
                    }
                }
            }
        },
    )

    payload = build_idm_paper_target_feasibility(
        train_paths=[train],
        target_paths=[target],
        split_tags=["temporal"],
        min_feature_count=2,
        paper_contract_path=contract,
        baseline_metrics_path=baseline,
    )

    by_name = {row["name"]: row for row in payload["predictor_summaries"]}
    lookup = by_name["lookup_game_prior_state_duration_mode"]
    assert payload["status"] == "pass"
    assert lookup["values"]["keyboard_accuracy"] == 1.0
    assert lookup["values"]["mouse_button_accuracy"] == 1.0
    assert payload["recommendation"]["scale_new_full_gpu_run"] is True


def test_g005_feasibility_recommends_no_scale_when_rules_do_not_beat_baseline(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    target = tmp_path / "target.jsonl"
    baseline = tmp_path / "baseline.json"
    write_jsonl(
        train,
        [
            {"sequence_id": "train#0", "game": "Toy", "ground_truth_tokens": ["KEY_PRESS_65"]},
            {"sequence_id": "train#1", "game": "Toy", "ground_truth_tokens": ["KEY_PRESS_65"]},
        ],
    )
    write_jsonl(
        target,
        [
            {
                "sequence_id": "target#0",
                "game": "Toy",
                "eval_split_tags": ["temporal"],
                "ground_truth_tokens": ["KEY_RELEASE_87", "MOUSE_LEFT_UP"],
            }
        ],
    )
    write_json(
        baseline,
        {
            "groups": {
                "all": {
                    "paper_compatible": {
                        "keyboard": {"key_accuracy": 0.5},
                        "mouse_button": {"button_accuracy": 0.5},
                    }
                }
            }
        },
    )

    payload = build_idm_paper_target_feasibility(
        train_paths=[train],
        target_paths=[target],
        split_tags=["temporal"],
        min_feature_count=1,
        baseline_metrics_path=baseline,
    )

    assert payload["status"] == "pass"
    assert payload["recommendation"]["scale_new_full_gpu_run"] is False
    assert payload["alignment"]["target_rows"] == 1

