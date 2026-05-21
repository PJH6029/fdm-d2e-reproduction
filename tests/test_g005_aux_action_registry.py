from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from build_g005_aux_action_registry import build_registry


def _args(root: Path, **overrides) -> Namespace:
    data = {"root": str(root), "aux_candidates": "artifacts/sources/aux.json", "source_id": None, "output": "artifacts/aux/action_registry.json"}
    data.update(overrides)
    return Namespace(**data)


def _write_candidates(root: Path) -> None:
    write_json(
        root / "artifacts/sources/aux.json",
        {
            "candidates": [
                {
                    "id": "minerl",
                    "selection_status": "selected_candidate",
                    "domain": "Minecraft human demonstrations",
                    "source_url": "https://zenodo.org/records/1",
                    "license_id": "mit",
                },
                {
                    "id": "atari",
                    "selection_status": "selected_candidate",
                    "domain": "Atari human demonstrations",
                    "source_url": "https://zenodo.org/records/2",
                    "license_id": "cc-by-4.0",
                },
                {
                    "id": "review",
                    "selection_status": "review_required_not_selected",
                    "domain": "Atari review candidate",
                },
            ]
        },
    )


def test_registry_builds_source_specific_action_heads_for_selected_aux_sources(tmp_path: Path):
    _write_candidates(tmp_path)
    payload = build_registry(_args(tmp_path))
    assert payload["status"] == "pass"
    assert payload["selected_aux_source_ids"] == ["atari", "minerl"]
    assert payload["source_specific_action_heads"] is True
    assert payload["no_cross_source_action_collapse"] is True
    heads = {row["id"]: row for row in payload["action_heads"]}
    assert heads["minerl"]["type"] == "minecraft_keyboard_mouse"
    assert heads["atari"]["type"] == "atari_discrete"
    assert heads["minerl"]["namespace"] == "minerl"
    assert heads["atari"]["d2e_endpoint_claims_allowed"] == []
    assert payload["d2e_endpoint_claim_boundary"]["no_aux_source_directly_claims_d2e_keyboard_mouse"] is True


def test_registry_rejects_unselected_source_id(tmp_path: Path):
    _write_candidates(tmp_path)
    try:
        build_registry(_args(tmp_path, source_id=["review"]))
    except SystemExit as exc:
        assert "not selected" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit")
