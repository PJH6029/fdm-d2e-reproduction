from __future__ import annotations

from fdm_d2e.data.fdm1_g003_splits import annotate_window_records_with_fdm1_splits, build_g002_split_index, fdm1_split_metadata


def _index():
    recording = {
        "schema": "fdm1_recording_level_split_manifest.v1",
        "fingerprint": "recfp",
        "splits": [
            {"cross_resolution_key": "Game/train", "split": "train"},
            {"cross_resolution_key": "Game/val", "split": "val"},
            {"cross_resolution_key": "Game/test", "split": "test"},
        ],
    }
    heldout = {
        "schema": "fdm1_heldout_game_split_manifest.v1",
        "fingerprint": "heldfp",
        "splits": [
            {"cross_resolution_key": "Game/train", "split": "train_pool"},
            {"cross_resolution_key": "Game/val", "split": "train_pool"},
            {"cross_resolution_key": "Game/test", "split": "heldout_game_test"},
        ],
    }
    pseudo = {
        "schema": "fdm1_pseudo_label_simulation_split_manifest.v1",
        "fingerprint": "pseudofp",
        "splits": [
            {"cross_resolution_key": "Game/train", "pseudo_label_split": "D_PSEUDO_B"},
            {"cross_resolution_key": "Game/val", "pseudo_label_split": "D_FDM_GT_EVAL"},
        ],
    }
    scale = {
        "schema": "fdm1_data_scale_split_manifest.v1",
        "fingerprint": "scalefp",
        "scales": {
            "1pct": {"cross_resolution_keys": ["Game/train"]},
            "100pct": {"cross_resolution_keys": ["Game/train", "Game/val", "Game/test"]},
        },
    }
    return build_g002_split_index(recording_level_split=recording, heldout_game_split=heldout, pseudo_label_split=pseudo, scale_split=scale)


def test_fdm1_split_metadata_combines_recording_heldout_pseudo_and_scale_roles():
    index = _index()
    train = fdm1_split_metadata("Game/train", index)
    val = fdm1_split_metadata("Game/val", index)
    test = fdm1_split_metadata("Game/test", index)

    assert train["split"] == "train_core"
    assert train["fdm1_pseudo_label_split"] == "D_PSEUDO_B"
    assert train["fdm1_scale_memberships"] == ["1pct", "100pct"]
    assert val["split"] == "eval"
    assert val["eval_split_tags"] == ["recording_val", "pseudo_gt_eval"]
    assert test["eval_split_tags"] == ["recording_test", "heldout_game"]


def test_annotate_window_records_with_fdm1_splits_decollides_source_and_preserves_fingerprints():
    records = [
        {
            "schema": "d2e_window_record.v1",
            "sequence_id": "Game/train#000000",
            "recording_id": "train",
            "game": "Game",
            "split": "full_corpus",
            "timestamp_ns": 0,
            "bin_index": 0,
            "frame": {"path": "frame", "index": 0, "features": []},
            "events": [],
        }
    ]
    row = {"source_id": "d2e_480p", "resolution_tier": "480p", "cross_resolution_key": "Game/train", "source_recording_key": "Game/train"}
    annotated = annotate_window_records_with_fdm1_splits(records, universe_row=row, split_index=_index())

    assert annotated[0]["sequence_id"] == "d2e_480p:Game/train#000000"
    assert annotated[0]["recording_id"] == "d2e_480p:Game/train"
    assert annotated[0]["split"] == "train_core"
    assert annotated[0]["fdm1_split_fingerprints"]["recording_level_split"] == "recfp"
