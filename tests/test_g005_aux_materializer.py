from __future__ import annotations

import json
import hashlib
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from materialize_g005_aux_sources import build_or_execute


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "aux_candidates": "artifacts/sources/aux.json",
        "namespace_root": "outputs/aux",
        "source_id": None,
        "splits": ["train", "val", "test"],
        "output": "artifacts/aux/plan.json",
        "execute": False,
        "max_bytes": None,
        "allow_patterns": None,
        "ignore_patterns": None,
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_candidates(root: Path, metadata_url: str | None = None) -> None:
    write_json(
        root / "artifacts/sources/aux.json",
        {
            "candidates": [
                {
                    "id": "zenodo_aux",
                    "selection_status": "selected_candidate",
                    "source_url": "https://zenodo.org/records/1",
                    "metadata_api_url": metadata_url or "https://zenodo.org/api/records/1",
                    "license_id": "mit",
                    "domain": "Minecraft human demonstrations",
                    "size_bytes": 12,
                },
                {
                    "id": "hf_aux",
                    "selection_status": "selected_candidate",
                    "source_url": "https://huggingface.co/datasets/org/name",
                    "metadata_api_url": "https://huggingface.co/api/datasets/org/name",
                    "license_id": "cc0",
                    "domain": "Atari",
                    "source_revision_or_record": "hf_sha_abc123",
                },
            ]
        },
    )


def test_materializer_plan_lists_selected_sources_without_downloading(tmp_path: Path):
    _write_candidates(tmp_path)
    payload = build_or_execute(_args(tmp_path))
    assert payload["status"] == "planned"
    assert payload["execute"] is False
    assert payload["selected_source_ids"] == ["hf_aux", "zenodo_aux"]
    providers = {row["id"]: row["provider"] for row in payload["plans"]}
    assert providers == {"hf_aux": "huggingface_dataset", "zenodo_aux": "zenodo"}
    hf = next(row for row in payload["plans"] if row["id"] == "hf_aux")
    assert hf["repo_id"] == "org/name"
    assert hf["revision"] == "abc123"


def test_materializer_executes_file_url_zenodo_fixture_and_writes_split_manifests(tmp_path: Path):
    source_file = tmp_path / "remote/source.zip"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("fixture-data", encoding="utf-8")
    metadata = tmp_path / "remote/metadata.json"
    metadata.write_text(
        json.dumps({"id": 1, "files": [{"key": "source.zip", "size": source_file.stat().st_size, "links": {"self": source_file.as_uri()}}]}),
        encoding="utf-8",
    )
    _write_candidates(tmp_path, metadata_url=metadata.as_uri())
    payload = build_or_execute(_args(tmp_path, source_id=["zenodo_aux"], execute=True))
    assert payload["status"] == "pass"
    execution = payload["executions"][0]
    assert execution["download_count"] == 1
    namespace = tmp_path / "outputs/aux/zenodo_aux"
    assert (namespace / "raw/source.zip").exists()
    for split in ["train", "val", "test"]:
        assert (namespace / split / "manifest.json").exists()
    assert (namespace / "materialization_summary.json").exists()


def test_materializer_replaces_invalid_existing_zenodo_file_using_size_and_checksum(tmp_path: Path):
    source_file = tmp_path / "remote/source.zip"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("fixture-data", encoding="utf-8")
    digest = hashlib.md5(source_file.read_bytes()).hexdigest()  # noqa: S324 - validates upstream Zenodo md5 format in tests
    metadata = tmp_path / "remote/metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "id": 1,
                "files": [
                    {
                        "key": "source.zip",
                        "size": source_file.stat().st_size,
                        "checksum": f"md5:{digest}",
                        "links": {"self": source_file.as_uri()},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_candidates(tmp_path, metadata_url=metadata.as_uri())
    existing = tmp_path / "outputs/aux/zenodo_aux/raw/source.zip"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("partial", encoding="utf-8")

    payload = build_or_execute(_args(tmp_path, source_id=["zenodo_aux"], execute=True))

    assert payload["status"] == "pass"
    download = payload["executions"][0]["downloads"][0]
    assert download["status"] == "downloaded"
    assert download["replaced_invalid_existing"]
    assert download["validation"]["valid"] is True
    assert existing.read_text(encoding="utf-8") == "fixture-data"
    assert list(existing.parent.glob("source.zip.invalid-*"))


def test_materializer_blocks_bad_zenodo_checksum_without_split_manifests(tmp_path: Path):
    source_file = tmp_path / "remote/source.zip"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("fixture-data", encoding="utf-8")
    metadata = tmp_path / "remote/metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "id": 1,
                "files": [
                    {
                        "key": "source.zip",
                        "size": source_file.stat().st_size,
                        "checksum": "md5:00000000000000000000000000000000",
                        "links": {"self": source_file.as_uri()},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_candidates(tmp_path, metadata_url=metadata.as_uri())

    payload = build_or_execute(_args(tmp_path, source_id=["zenodo_aux"], execute=True))

    assert payload["status"] == "blocked"
    assert any(item["code"] == "download_validation_failed" for item in payload["findings"])
    namespace = tmp_path / "outputs/aux/zenodo_aux"
    assert not (namespace / "train/manifest.json").exists()
    assert list((namespace / "raw").glob("source.zip.part-*"))


def test_materializer_blocks_unsupported_manual_provider(tmp_path: Path):
    write_json(
        tmp_path / "artifacts/sources/aux.json",
        {"candidates": [{"id": "manual", "selection_status": "selected_candidate", "source_url": "https://example.test/manual", "domain": "Other"}]},
    )
    payload = build_or_execute(_args(tmp_path, execute=True))
    assert payload["status"] == "blocked"
    assert any(item["code"] == "unsupported_provider" for item in payload["findings"])
