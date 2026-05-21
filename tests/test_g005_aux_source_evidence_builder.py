from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from build_g005_aux_source_evidence import build_evidence
from build_g005_aux_namespace_manifest import build_manifest


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "aux_candidates": "artifacts/sources/aux.json",
        "namespace_root": "outputs/aux",
        "source_id": None,
        "required_splits": ["train", "val", "test"],
        "max_files": None,
        "output": "artifacts/aux/source_evidence.json",
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_candidates(root: Path) -> None:
    write_json(
        root / "artifacts/sources/aux.json",
        {
            "candidates": [
                {"id": "minerl", "selection_status": "selected_candidate", "domain": "Minecraft human demonstrations", "source_url": "https://example.test/minerl", "license_id": "mit"},
                {"id": "atari", "selection_status": "review_required", "domain": "Atari", "source_url": "https://example.test/atari", "license_id": "cc"},
            ]
        },
    )


def _write_source(root: Path) -> None:
    for split in ["train", "val", "test"]:
        path = root / "outputs/aux/minerl" / split / "sample.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{{\"split\":\"{split}\"}}\n", encoding="utf-8")


def test_g005_aux_source_evidence_passes_for_materialized_source_with_splits(tmp_path: Path):
    _write_candidates(tmp_path)
    _write_source(tmp_path)
    payload = build_evidence(_args(tmp_path))
    assert payload["status"] == "pass"
    assert payload["materialized_source_ids"] == ["minerl"]
    source = payload["aux_sources"][0]
    assert source["materialized"] is True
    assert source["d2e_heldout_overlap_count"] == 0
    assert source["action_head"]["namespace"] == "minerl"
    assert source["split_hashes"]["train"]["sha256"]
    assert source["provenance_sha256"]


def test_g005_aux_source_evidence_blocks_missing_namespace(tmp_path: Path):
    _write_candidates(tmp_path)
    payload = build_evidence(_args(tmp_path))
    assert payload["status"] == "blocked"
    assert any(item["code"] == "aux_namespace_missing" for item in payload["findings"])
    assert payload["aux_sources"][0]["template_only"] is True


def test_g005_aux_source_evidence_blocks_missing_required_split(tmp_path: Path):
    _write_candidates(tmp_path)
    path = tmp_path / "outputs/aux/minerl/train/sample.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    payload = build_evidence(_args(tmp_path))
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "blocked"
    assert "missing_aux_split_files" in codes


def test_namespace_completion_ready_requires_split_hashes(tmp_path: Path):
    _write_candidates(tmp_path)
    _write_source(tmp_path)
    source_evidence = build_evidence(_args(tmp_path))
    source_path = tmp_path / "artifacts/aux/source_evidence.json"
    write_json(source_path, source_evidence)
    eval_hashes = tmp_path / "artifacts/aux/eval_hashes.json"
    write_json(
        eval_hashes,
        {
            "temporal": {"sha256": "temporal"},
            "heldout_recording": {"sha256": "heldout_recording"},
            "heldout_game": {"sha256": "heldout_game"},
        },
    )
    payload = build_manifest(
        aux_candidates_path=str(tmp_path / "artifacts/sources/aux.json"),
        source_evidence_paths=[str(source_path)],
        eval_manifest_hashes_path=str(eval_hashes),
        completion_ready_requested=True,
        allow_template=False,
    )
    assert payload["completion_ready"] is True
    assert payload["aux_sources"][0]["split_hashes"]["test"]["sha256"]
