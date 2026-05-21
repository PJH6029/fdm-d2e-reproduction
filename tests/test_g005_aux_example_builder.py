from __future__ import annotations

import json
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from build_g005_aux_examples import build_examples


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "action_registry": "artifacts/aux/action_registry.json",
        "namespace_root": "outputs/aux",
        "examples_root": "outputs/aux_examples",
        "source_id": None,
        "required_splits": ["train", "val", "test"],
        "max_examples_per_source": None,
        "allow_incomplete_raw": False,
        "output": "artifacts/aux/examples.json",
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_registry(root: Path, *, include_unsupported: bool = False) -> None:
    heads = [
        {
            "id": "atari_head_zenodo_v4",
            "namespace": "atari_head_zenodo_v4",
            "type": "atari_discrete",
            "adapter": "atari_head_zip_csv_action_adapter",
        }
    ]
    if include_unsupported:
        heads.append(
            {
                "id": "minerl_2019_zenodo_v2",
                "namespace": "minerl_2019_zenodo_v2",
                "type": "minecraft_keyboard_mouse",
                "adapter": "minerl_action_dict_adapter",
            }
        )
    write_json(root / "artifacts/aux/action_registry.json", {"schema": "g005_aux_action_registry.v1", "status": "pass", "action_heads": heads})


def _write_atari_zip(root: Path) -> None:
    raw = root / "outputs/aux/atari_head_zenodo_v4/raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "action_enums.txt").write_text("0 NOOP\n1 FIRE\n3 RIGHT\n", encoding="utf-8")
    with zipfile.ZipFile(raw / "bank_heist.zip", "w") as archive:
        archive.writestr("bank_heist/demo_sequence.tar.bz2", b"not-a-real-tar-needed-for-ref-only")
        archive.writestr(
            "bank_heist/demo_sequence.txt",
            "frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions\n"
            "RZ_1_1,0,0,50,0,3,1.0,2.0\n"
            "RZ_1_2,0,1,50,1,1,1.5,2.5\n"
            "RZ_1_3,0,2,50,0,0,2.0,3.0\n",
        )


def test_builds_atari_head_aux_examples_with_source_specific_contract(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    _write_atari_zip(tmp_path)

    payload = build_examples(_args(tmp_path, source_id=["atari_head_zenodo_v4"]))

    assert payload["status"] == "pass"
    assert payload["total_examples"] == 3
    assert payload["selected_source_ids"] == ["atari_head_zenodo_v4"]
    source = payload["sources"][0]
    assert source["status"] == "pass"
    assert sum(source["split_counts"].values()) == 3
    rows = []
    for split in ["train", "val", "test"]:
        path = tmp_path / f"outputs/aux_examples/atari_head_zenodo_v4/{split}.jsonl"
        assert path.exists()
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    assert len(rows) == 3
    first = rows[0]
    assert first["source_id"] == "atari_head_zenodo_v4"
    assert first["action_head_namespace"] == "atari_head_zenodo_v4"
    assert first["action"]["type"] == "atari_discrete"
    assert first["action"]["action_id"] in {0, 1, 3}
    assert first["frame_or_state_ref"].startswith("nested-archive://outputs/aux/atari_head_zenodo_v4/raw/bank_heist.zip!")
    assert first["provenance"]["csv_member"] == "bank_heist/demo_sequence.txt"


def test_blocks_default_when_selected_source_adapter_is_not_supported(tmp_path: Path) -> None:
    _write_registry(tmp_path, include_unsupported=True)
    _write_atari_zip(tmp_path)

    payload = build_examples(_args(tmp_path))

    assert payload["status"] == "blocked"
    codes = {item["code"] for item in payload["findings"]}
    assert "unsupported_aux_example_adapter" in codes
    supported = next(row for row in payload["sources"] if row["source_id"] == "atari_head_zenodo_v4")
    assert supported["status"] == "pass"


def test_reports_invalid_zip_as_warning_when_incomplete_raw_is_allowed(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    raw = tmp_path / "outputs/aux/atari_head_zenodo_v4/raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "partial.zip").write_bytes(b"not a complete zip")

    payload = build_examples(_args(tmp_path, source_id=["atari_head_zenodo_v4"], allow_incomplete_raw=True))

    assert payload["status"] == "blocked"
    codes = {item["code"] for item in payload["findings"]}
    assert "aux_raw_zip_invalid" in codes
    assert "aux_examples_empty" in codes
