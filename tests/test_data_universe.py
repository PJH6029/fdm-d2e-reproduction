from __future__ import annotations

from fdm_d2e.data.data_universe import DataSourceConfig, build_data_universe_manifest


def _info(repo_id: str) -> dict:
    return {
        "id": repo_id,
        "sha": "abc123",
        "lastModified": "2026-05-21T00:00:00.000Z",
        "cardData": {"license": "cc-by-nc-4.0"},
        "private": False,
        "gated": False,
        "disabled": False,
        "usedStorage": 123,
    }


def _entry(path: str, size: int = 10) -> dict:
    return {
        "type": "file",
        "path": path,
        "size": size,
        "oid": f"git-{path}",
        "lfs": {"oid": f"sha256-{path}", "size": size, "pointerSize": 128},
        "xetHash": f"xet-{path}",
    }


def test_data_universe_pairs_resolution_variants_and_statuses_all_rows():
    sources = [
        DataSourceConfig("d2e_480p", "open-world-agents/D2E-480p", "main", "480p"),
        DataSourceConfig("d2e_original", "open-world-agents/D2E-Original", "main", "original_fhd_qhd"),
    ]
    trees = {
        "open-world-agents/D2E-480p": [
            _entry("Apex_Legends/0805_01.mcap"),
            _entry("Apex_Legends/0805_01.mkv", 100),
            _entry("Brotato/0901_01.mcap"),
        ],
        "open-world-agents/D2E-Original": [
            _entry("Apex_Legends/0805_01.mcap", 20),
            _entry("Apex_Legends/0805_01.mkv", 200),
        ],
    }
    infos = {source.repo_id: _info(source.repo_id) for source in sources}
    manifest = build_data_universe_manifest(
        sources=sources,
        repo_infos=infos,
        repo_trees=trees,
        auxiliary_candidates=[],
        generated_at_utc="2026-05-21T00:00:00Z",
    )
    assert manifest["schema"] == "data_universe_manifest.v1"
    assert manifest["coverage"]["recording_variants"] == 3
    assert manifest["coverage"]["unique_cross_resolution_recordings"] == 2
    assert manifest["coverage"]["status_counts"] == {"included": 2, "unsupported": 1}
    assert manifest["coverage"]["all_recording_variants_statused"] is True
    unsupported = [row for row in manifest["recordings"] if row["status"] == "unsupported"]
    assert unsupported[0]["audited_exclusion"]["reason"] == "missing_video_or_mcap_pair"


def test_data_universe_storage_budget_flags_large_source_total():
    source = DataSourceConfig("d2e_480p", "open-world-agents/D2E-480p", "main", "480p")
    tree = [_entry("Apex_Legends/0805_01.mcap", 1024**4), _entry("Apex_Legends/0805_01.mkv", 2 * 1024**4)]
    manifest = build_data_universe_manifest(
        sources=[source],
        repo_infos={source.repo_id: _info(source.repo_id)},
        repo_trees={source.repo_id: tree},
        auxiliary_candidates=[],
        budget_tib=2.0,
        generated_at_utc="2026-05-21T00:00:00Z",
    )
    assert manifest["storage_budget"]["total_source_bytes_within_budget"] is False
    assert manifest["storage_budget"]["requires_staged_cache_or_extra_storage"] is True
