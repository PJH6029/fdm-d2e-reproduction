from __future__ import annotations

from fdm_d2e.data.fdm1_g002 import (
    build_game_metadata,
    build_heldout_game_manifest,
    build_pseudo_label_split_manifest,
    build_recording_level_split_manifest,
    build_scale_split_manifest,
    parse_d2e_readme_game_hours,
    parse_d2e_readme_summary_hours,
    validate_g002_contract,
)


def _row(game: str, idx: int, source_id: str = "d2e_480p") -> dict:
    rec = f"rec_{idx:02d}"
    return {
        "source_id": source_id,
        "cross_resolution_key": f"{game}/{rec}",
        "source_recording_key": f"{game}/{rec}",
        "recording_id": rec,
        "game": game,
        "status": "included",
        "size_bytes": 10,
    }


def _universe() -> dict:
    # Use real taxonomy keys so the category-coverage heldout selector can satisfy all roles.
    games = {
        "Apex_Legends": 6,
        "Minecraft_1.21.8": 6,
        "Stardew_Valley": 6,
        "MapleStory_Worlds_Southperry": 6,
        "PEAK": 6,
    }
    rows = []
    for game, count in games.items():
        for idx in range(count):
            rows.append(_row(game, idx, "d2e_480p"))
            rows.append(_row(game, idx, "d2e_original"))
    return {
        "schema": "data_universe_manifest.v1",
        "dataset_fingerprint": "fp",
        "d2e_sources": [{"source_id": "d2e_480p", "resolved_revision": "rev", "license": "cc-by-nc-4.0"}],
        "recordings": rows,
        "coverage": {"games_count": 5, "recording_variants": len(rows), "status_counts": {"included": len(rows)}},
    }


def _readme() -> str:
    return """
# D2E-480p
This dataset has **268.7 hours**.
## Games
| Game | Hours |
| --- | ---: |
| Apex Legends | 25.6 |
| Minecraft | 8.6 |
| Stardew Valley | 14.6 |
| MapleStory Worlds | 14.1 |
| PEAK | 1.8 |
## Citation
"""


def test_parse_d2e_readme_hours_and_summary():
    assert parse_d2e_readme_summary_hours(_readme()) == 268.7
    hours = parse_d2e_readme_game_hours(_readme())
    assert hours["Apex Legends"] == 25.6
    assert hours["MapleStory Worlds"] == 14.1


def test_g002_split_bundle_is_leakage_safe_and_has_required_subsets():
    universe = _universe()
    metadata = build_game_metadata(universe, readme_text=_readme(), readme_revision="rev", readme_url="fixture")
    recording = build_recording_level_split_manifest(universe)
    heldout = build_heldout_game_manifest(universe, metadata)
    pseudo = build_pseudo_label_split_manifest(recording)
    scale = build_scale_split_manifest(recording, metadata)

    assert set(recording["counts"]) == {"train", "val", "test"}
    assert heldout["heldout_selection"]["missing_roles"] == {}
    assert set(pseudo["counts"]) == {"D_IDM_LABELED_A", "D_PSEUDO_B", "D_FDM_GT_EVAL"}
    assert set(scale["scales"]) == {"1pct", "5pct", "10pct", "25pct", "50pct", "100pct"}

    key_to_split = {}
    for row in recording["splits"]:
        previous = key_to_split.setdefault(row["cross_resolution_key"], row["split"])
        assert previous == row["split"]


def test_g002_validator_rejects_missing_pseudo_subset():
    universe = _universe()
    # Make the validator see 29 games to isolate the pseudo-subset check.
    universe["coverage"]["games_count"] = 29
    metadata = build_game_metadata(universe, readme_text=_readme(), readme_revision="rev", readme_url="fixture")
    # Duplicate rows to satisfy the game-metadata count expected by the validator in this unit scope.
    metadata["games"] = metadata["games"] * 6
    metadata["games"] = metadata["games"][:29]
    recording = build_recording_level_split_manifest(universe)
    heldout = build_heldout_game_manifest(universe, metadata)
    pseudo = build_pseudo_label_split_manifest(recording)
    pseudo["counts"].pop("D_PSEUDO_B", None)
    scale = build_scale_split_manifest(recording, metadata)
    payload = validate_g002_contract({"data_universe": universe, "game_metadata": metadata, "recording_level_split": recording, "heldout_game_split": heldout, "pseudo_label_split": pseudo, "scale_split": scale})
    assert payload["status"] == "fail"
    assert any(item["code"] == "pseudo_split_missing_subset" for item in payload["findings"])
