from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from build_g005_aux_loader_manifest import build_manifest


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "action_registry": "artifacts/aux/action_registry.json",
        "archive_inventory": "artifacts/aux/archive_inventory.json",
        "materialization_integrity": "artifacts/aux/integrity.json",
        "required_splits": ["train", "val", "test"],
        "output": "artifacts/aux/loader_manifest.json",
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_pass_fixture(root: Path) -> None:
    write_json(
        root / "artifacts/aux/action_registry.json",
        {
            "status": "pass",
            "action_heads": [
                {
                    "id": "aux_a",
                    "namespace": "aux_a",
                    "type": "atari_discrete",
                    "adapter": "atari_discrete_action_adapter",
                    "transfer_role": "control",
                    "d2e_endpoint_claims_allowed": [],
                }
            ],
        },
    )
    write_json(
        root / "artifacts/aux/archive_inventory.json",
        {
            "status": "pass",
            "aux_sources": [
                {
                    "id": "aux_a",
                    "raw_file_count": 1,
                    "action_candidate_member_count": 1,
                    "files": [
                        {
                            "path": "raw/source.zip",
                            "bytes": 12,
                            "archive_type": "zip",
                            "action_candidate_members": [{"path": "actions.jsonl", "bytes": 3}],
                        }
                    ],
                }
            ],
        },
    )
    write_json(
        root / "artifacts/aux/integrity.json",
        {
            "status": "pass",
            "aux_sources": [
                {
                    "id": "aux_a",
                    "split_manifests": [
                        {"split": "train", "referenced_files_exist": True},
                        {"split": "val", "referenced_files_exist": True},
                        {"split": "test", "referenced_files_exist": True},
                    ],
                }
            ],
        },
    )


def test_loader_manifest_passes_with_registry_inventory_and_integrity(tmp_path: Path):
    _write_pass_fixture(tmp_path)
    payload = build_manifest(_args(tmp_path))
    assert payload["status"] == "pass"
    assert payload["source_specific_action_heads"] is True
    assert payload["no_aux_in_d2e_heldout"] is True
    assert payload["selected_aux_source_ids"] == ["aux_a"]
    stage = payload["loader_stages"][0]
    assert stage["action_head"]["namespace"] == "aux_a"
    assert stage["loader_contract"]["train_manifest"] == "outputs/aux_examples/aux_a/train.jsonl"
    assert stage["loader_contract"]["example_builder_command"] == [
        "uv",
        "run",
        "python",
        "scripts/build_g005_aux_examples.py",
        "--source-id",
        "aux_a",
    ]
    assert stage["action_candidate_members"] == [{"archive": "raw/source.zip", "path": "actions.jsonl", "bytes": 3}]


def test_loader_manifest_blocks_when_integrity_or_inventory_not_pass(tmp_path: Path):
    _write_pass_fixture(tmp_path)
    write_json(tmp_path / "artifacts/aux/integrity.json", {"status": "blocked", "error_count": 2, "aux_sources": []})
    payload = build_manifest(_args(tmp_path))
    assert payload["status"] == "blocked"
    codes = {item["code"] for item in payload["findings"]}
    assert "materialization_integrity_not_pass" in codes
    assert "loader_manifest_missing_integrity_sources" in codes


def test_loader_manifest_blocks_missing_split_rows(tmp_path: Path):
    _write_pass_fixture(tmp_path)
    write_json(tmp_path / "artifacts/aux/integrity.json", {"status": "pass", "aux_sources": [{"id": "aux_a", "split_manifests": [{"split": "train"}]}]})
    payload = build_manifest(_args(tmp_path))
    assert payload["status"] == "blocked"
    assert any(item["code"] == "loader_source_missing_integrity_split_rows" for item in payload["findings"])
