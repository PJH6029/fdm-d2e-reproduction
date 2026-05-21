from __future__ import annotations

from fdm_d2e.data.full_corpus import annotate_window_records
from fdm_d2e.training.neural_idm import record_features


def _split_contract() -> dict:
    return {
        "manifests": {
            "temporal": {"split_policy": {"train_fraction": 0.5}},
            "heldout_recording": {"splits": {"heldout_recording": ["d2e_original:Apex_Legends/0805_01"]}},
            "heldout_game": {"splits": {"heldout_game": []}},
        }
    }


def _universe_row(source_id: str) -> dict:
    return {
        "source_id": source_id,
        "resolution_tier": "original" if source_id == "d2e_original" else "480p",
        "game": "Apex_Legends",
        "recording_id": "0805_01",
        "source_recording_key": "Apex_Legends/0805_01",
        "cross_resolution_key": "Apex_Legends/0805_01",
    }


def _records() -> list[dict]:
    return [
        {
            "schema": "d2e_window_record.v1",
            "sequence_id": f"Apex_Legends/0805_01#{idx:06d}",
            "recording_id": "0805_01",
            "game": "Apex_Legends",
            "split": "full_corpus",
            "timestamp_ns": idx,
            "bin_index": idx,
            "frame": {"path": "", "index": idx, "features": []},
            "events": [],
            "ground_truth_tokens": [],
            "source": "test",
        }
        for idx in range(4)
    ]


def test_full_corpus_annotation_decollides_resolution_variants_and_materializes_eval_tags():
    rows_480p = annotate_window_records(_records(), universe_row=_universe_row("d2e_480p"), split_contract=_split_contract())
    rows_original = annotate_window_records(_records(), universe_row=_universe_row("d2e_original"), split_contract=_split_contract())

    assert rows_480p[0]["sequence_id"].startswith("d2e_480p:")
    assert rows_original[0]["sequence_id"].startswith("d2e_original:")
    assert rows_480p[0]["sequence_id"] != rows_original[0]["sequence_id"]
    assert rows_480p[0]["split"] == "train_core"
    assert rows_480p[-1]["eval_split_tags"] == ["temporal"]
    assert all("heldout_recording" in row["eval_split_tags"] for row in rows_original)
    assert all(row["cross_resolution_key"] == "Apex_Legends/0805_01" for row in rows_480p + rows_original)


def test_compact_grid8_feature_mode_uses_json_features_without_frame_files():
    row = {
        "frame": {
            "features": [0.1, 0.2, 0.3, 0.4, 0.5],
            "grid8": [0.1] * (8 * 8 * 3),
            "luma16": [0.1] * (16 * 16),
        },
        "next_frame_features": [0.2, 0.3, 0.4, 0.5, 0.6],
        "frame_delta_features": [0.1] * 5,
        "next_frame_grid8": [0.2] * (8 * 8 * 3),
        "next_frame_luma16": [0.2] * (16 * 16),
        "bin_index": 3,
    }

    features = record_features(row, feature_mode="summary_compact_grid8_shift_surface_time")

    assert len(features) == 16 + (8 * 8 * 3 * 3) + 16 + 12
    assert features[16] == 0.1
    assert features[16 + (8 * 8 * 3)] == 0.2

    causal = record_features(
        {**row, "prior_action_tokens": ["MOUSE_DX_P1", "MOUSE_DY_N1", "KEY_PRESS_87"]},
        feature_mode="summary_causal_compact_grid8_time_prior_action",
    )
    assert len(causal) == 6 + (8 * 8 * 3) + (16 * 16) + 12 + 38
    assert causal[6] == 0.1
    assert causal[6 + (8 * 8 * 3)] == 0.1
    assert any(abs(value) > 0 for value in causal[-38:])
