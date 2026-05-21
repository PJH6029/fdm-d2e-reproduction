from __future__ import annotations

import pytest

from fdm_d2e.data.splits import build_generalization_split_contract, validate_generalization_splits


def _row(source_id: str, key: str, game: str) -> dict:
    return {
        "source_id": source_id,
        "source_recording_key": key,
        "cross_resolution_key": key,
        "recording_id": key.rsplit("/", 1)[1],
        "game": game,
        "status": "included",
    }


def _manifest() -> dict:
    rows = []
    for game in ["Apex", "Brotato", "Cyberpunk", "Dinkum", "Raft"]:
        for idx in range(3):
            key = f"{game}/rec_{idx:02d}"
            rows.append(_row("d2e_480p", key, game))
            rows.append(_row("d2e_original", key, game))
    return {"dataset_fingerprint": "fp", "recordings": rows}


def test_generalization_split_contract_keeps_resolution_variants_together():
    contract = build_generalization_split_contract(
        _manifest(),
        heldout_recording_fraction=0.2,
        heldout_game_fraction=0.2,
        seed="test",
    )
    assert contract["leakage_report"]["status"] == "pass"
    rec = contract["manifests"]["heldout_recording"]["splits"]
    row_to_split = {row_id: "train" for row_id in rec["train"]}
    row_to_split.update({row_id: "heldout" for row_id in rec["heldout_recording"]})
    by_group = {}
    for row_id, split in row_to_split.items():
        _, key = row_id.split(":", 1)
        by_group.setdefault(key, set()).add(split)
    assert all(len(splits) == 1 for splits in by_group.values())
    game = contract["manifests"]["heldout_game"]["splits"]
    heldout_games = set(game["heldout_games"])
    assert heldout_games
    assert not any(row.split(":", 1)[1].split("/", 1)[0] in heldout_games for row in game["train"])


def test_generalization_split_validation_rejects_cross_resolution_leakage():
    contract = build_generalization_split_contract(_manifest(), seed="test")
    rec = contract["manifests"]["heldout_recording"]
    leaked = rec["splits"]["heldout_recording"][0]
    rec["splits"]["train"].append(leaked)
    with pytest.raises(ValueError):
        validate_generalization_splits(
            _manifest()["recordings"],
            temporal_manifest=contract["manifests"]["temporal"],
            heldout_recording_manifest=rec,
            heldout_game_manifest=contract["manifests"]["heldout_game"],
        )
