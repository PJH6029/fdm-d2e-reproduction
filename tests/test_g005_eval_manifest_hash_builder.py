from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_g005_eval_manifest_hashes import build_hash_manifest


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "output": "artifacts/aux/hashes.json",
        "d2e_only_temporal": "splits/temporal.json",
        "d2e_only_heldout_recording": "splits/heldout_recording.json",
        "d2e_only_heldout_game": "splits/heldout_game.json",
        "d2e_aux_temporal": None,
        "d2e_aux_heldout_recording": None,
        "d2e_aux_heldout_game": None,
        "allow_mismatch": False,
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_split_fixture(root: Path) -> None:
    _write(root / "splits/temporal.json", '{"split":"temporal"}\n')
    _write(root / "splits/heldout_recording.json", '{"split":"heldout_recording"}\n')
    _write(root / "splits/heldout_game.json", '{"split":"heldout_game"}\n')


def test_g005_eval_manifest_hash_builder_passes_with_byte_identical_defaults(tmp_path: Path):
    _write_split_fixture(tmp_path)
    payload = build_hash_manifest(_args(tmp_path))
    assert payload["status"] == "pass"
    assert payload["same_d2e_eval_manifests"] is True
    assert set(payload["splits"]) == {"temporal", "heldout_recording", "heldout_game"}
    assert payload["splits"]["temporal"]["same_hash"] is True
    assert payload["splits"]["temporal"]["d2e_only_manifest_sha256"] == payload["splits"]["temporal"]["d2e_aux_manifest_sha256"]
    assert "does not launch aux training" in payload["claim_boundary"]


def test_g005_eval_manifest_hash_builder_fails_missing_manifest(tmp_path: Path):
    _write(tmp_path / "splits/temporal.json", "temporal")
    payload = build_hash_manifest(_args(tmp_path))
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "missing_d2e_only_eval_manifest" in codes
    assert "missing_d2e_aux_eval_manifest" in codes


def test_g005_eval_manifest_hash_builder_rejects_aux_mismatch_by_default(tmp_path: Path):
    _write_split_fixture(tmp_path)
    _write(tmp_path / "aux/temporal.json", "different-temporal")
    _write(tmp_path / "aux/heldout_recording.json", '{"split":"heldout_recording"}\n')
    _write(tmp_path / "aux/heldout_game.json", '{"split":"heldout_game"}\n')
    payload = build_hash_manifest(
        _args(
            tmp_path,
            d2e_aux_temporal="aux/temporal.json",
            d2e_aux_heldout_recording="aux/heldout_recording.json",
            d2e_aux_heldout_game="aux/heldout_game.json",
        )
    )
    assert payload["status"] == "fail"
    assert any(item["code"] == "d2e_aux_eval_manifest_hash_mismatch" for item in payload["findings"])


def test_g005_eval_manifest_hash_builder_can_write_nonterminal_mismatch_warning(tmp_path: Path):
    _write_split_fixture(tmp_path)
    _write(tmp_path / "aux/temporal.json", "different-temporal")
    _write(tmp_path / "aux/heldout_recording.json", '{"split":"heldout_recording"}\n')
    _write(tmp_path / "aux/heldout_game.json", '{"split":"heldout_game"}\n')
    payload = build_hash_manifest(
        _args(
            tmp_path,
            d2e_aux_temporal="aux/temporal.json",
            d2e_aux_heldout_recording="aux/heldout_recording.json",
            d2e_aux_heldout_game="aux/heldout_game.json",
            allow_mismatch=True,
        )
    )
    assert payload["status"] == "pass"
    assert payload["same_d2e_eval_manifests"] is False
    assert any(item["severity"] == "warning" for item in payload["findings"])
