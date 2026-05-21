from __future__ import annotations

import sys
import zipfile
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from build_g005_aux_archive_inventory import build_inventory


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "aux_candidates": "artifacts/sources/aux.json",
        "namespace_root": "outputs/aux",
        "source_id": None,
        "max_members": 20,
        "hash_files": False,
        "output": "artifacts/aux/inventory.json",
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_candidates(root: Path) -> None:
    write_json(
        root / "artifacts/sources/aux.json",
        {
            "candidates": [
                {
                    "id": "zip_aux",
                    "selection_status": "selected_candidate",
                    "source_url": "https://zenodo.org/records/1",
                    "license_id": "cc0",
                },
                {
                    "id": "missing_aux",
                    "selection_status": "selected_candidate",
                    "source_url": "https://example.test/missing",
                    "license_id": "mit",
                },
            ]
        },
    )


def test_archive_inventory_detects_action_members_inside_zip(tmp_path: Path):
    _write_candidates(tmp_path)
    raw = tmp_path / "outputs/aux/zip_aux/raw"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "demo.zip", "w") as zf:
        zf.writestr("metadata/readme.txt", "hello")
        zf.writestr("episode_000/actions.jsonl", "{}\n")
        zf.writestr("episode_000/video.mp4", "not-real-video")

    payload = build_inventory(_args(tmp_path, source_id=["zip_aux"]))
    assert payload["status"] == "pass"
    source = payload["aux_sources"][0]
    assert source["raw_file_count"] == 1
    assert source["action_candidate_member_count"] == 1
    file_row = source["files"][0]
    assert file_row["archive_type"] == "zip"
    assert file_row["member_count"] == 3
    assert file_row["sha256"] is None
    assert file_row["action_candidate_members"][0]["path"] == "episode_000/actions.jsonl"


def test_archive_inventory_hashes_files_when_requested(tmp_path: Path):
    _write_candidates(tmp_path)
    raw = tmp_path / "outputs/aux/zip_aux/raw"
    raw.mkdir(parents=True)
    (raw / "actions.jsonl").write_text("{}\n", encoding="utf-8")
    payload = build_inventory(_args(tmp_path, source_id=["zip_aux"], hash_files=True))
    assert payload["status"] == "pass"
    assert payload["aux_sources"][0]["files"][0]["sha256"]


def test_archive_inventory_blocks_when_selected_namespace_is_missing(tmp_path: Path):
    _write_candidates(tmp_path)
    payload = build_inventory(_args(tmp_path))
    assert payload["status"] == "blocked"
    codes = {item["code"] for item in payload["findings"]}
    assert "aux_namespace_missing" in codes
