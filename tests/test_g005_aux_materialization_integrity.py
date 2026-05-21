from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from materialize_g005_aux_sources import build_or_execute
from validate_g005_aux_materialization_integrity import build_integrity


def _materializer_args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "aux_candidates": "artifacts/sources/aux.json",
        "namespace_root": "outputs/aux",
        "source_id": ["zenodo_aux"],
        "splits": ["train", "val", "test"],
        "output": "artifacts/aux/materialize_summary.json",
        "execute": True,
        "max_bytes": None,
        "allow_patterns": None,
        "ignore_patterns": None,
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _integrity_args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "aux_candidates": "artifacts/sources/aux.json",
        "namespace_root": "outputs/aux",
        "materialization_summary": "artifacts/aux/materialize_summary.json",
        "source_id": ["zenodo_aux"],
        "required_splits": ["train", "val", "test"],
        "output": "artifacts/aux/integrity.json",
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_zenodo_candidate(root: Path, metadata_url: str) -> None:
    write_json(
        root / "artifacts/sources/aux.json",
        {
            "candidates": [
                {
                    "id": "zenodo_aux",
                    "selection_status": "selected_candidate",
                    "source_url": "https://zenodo.org/records/1",
                    "metadata_api_url": metadata_url,
                    "license_id": "cc0",
                }
            ]
        },
    )


def _write_metadata(root: Path, *, payload: bytes = b"fixture-data", checksum: str | None = None) -> tuple[Path, Path]:
    remote = root / "remote/source.zip"
    remote.parent.mkdir(parents=True, exist_ok=True)
    remote.write_bytes(payload)
    digest = checksum or "md5:" + hashlib.md5(payload).hexdigest()  # noqa: S324 - mirrors Zenodo md5 checksum format
    metadata = root / "remote/metadata.json"
    metadata.write_text(
        json.dumps({"id": 1, "files": [{"key": "source.zip", "size": len(payload), "checksum": digest, "links": {"self": remote.as_uri()}}]}),
        encoding="utf-8",
    )
    return remote, metadata


def test_integrity_passes_for_materializer_output_with_size_checksum_and_splits(tmp_path: Path):
    _remote, metadata = _write_metadata(tmp_path)
    _write_zenodo_candidate(tmp_path, metadata.as_uri())
    summary = build_or_execute(_materializer_args(tmp_path))
    write_json(tmp_path / "artifacts/aux/materialize_summary.json", summary)

    payload = build_integrity(_integrity_args(tmp_path))

    assert payload["status"] == "pass"
    source = payload["aux_sources"][0]
    assert source["validated_file_count"] == 1
    assert all(row["referenced_files_exist"] for row in source["split_manifests"])


def test_integrity_blocks_when_raw_file_size_does_not_match_metadata(tmp_path: Path):
    _remote, metadata = _write_metadata(tmp_path, payload=b"expected")
    _write_zenodo_candidate(tmp_path, metadata.as_uri())
    namespace = tmp_path / "outputs/aux/zenodo_aux"
    raw = namespace / "raw/source.zip"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"bad")
    for split in ["train", "val", "test"]:
        write_json(namespace / split / "manifest.json", {"source_files": [{"path": str(raw)}]})
    write_json(tmp_path / "artifacts/aux/materialize_summary.json", {"status": "pass", "executions": [{"id": "zenodo_aux", "status": "pass"}]})

    payload = build_integrity(_integrity_args(tmp_path))

    assert payload["status"] == "blocked"
    codes = {item["code"] for item in payload["findings"]}
    assert "materialized_file_size_mismatch" in codes
    assert "materialized_file_checksum_mismatch" in codes


def test_integrity_blocks_missing_split_manifest_even_when_raw_file_exists(tmp_path: Path):
    _remote, metadata = _write_metadata(tmp_path)
    _write_zenodo_candidate(tmp_path, metadata.as_uri())
    namespace = tmp_path / "outputs/aux/zenodo_aux"
    raw = namespace / "raw/source.zip"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"fixture-data")
    write_json(tmp_path / "artifacts/aux/materialize_summary.json", {"status": "pass", "executions": [{"id": "zenodo_aux", "status": "pass"}]})

    payload = build_integrity(_integrity_args(tmp_path))

    assert payload["status"] == "blocked"
    assert any(item["code"] == "split_manifest_missing" for item in payload["findings"])
