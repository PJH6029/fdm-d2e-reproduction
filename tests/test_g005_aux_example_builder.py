from __future__ import annotations

import json
import io
import pickle
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
import build_g005_aux_examples as aux_examples_module
from build_g005_aux_examples import build_examples, _split_for_sequence


SPLITS = ("train", "val", "test")


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "action_registry": "artifacts/aux/action_registry.json",
        "namespace_root": "outputs/aux",
        "examples_root": "outputs/aux_examples",
        "source_id": None,
        "required_splits": list(SPLITS),
        "max_examples_per_source": None,
        "allow_incomplete_raw": False,
        "output": "artifacts/aux/examples.json",
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_registry(root: Path, heads: list[dict]) -> None:
    write_json(root / "artifacts/aux/action_registry.json", {"schema": "g005_aux_action_registry.v1", "status": "pass", "action_heads": heads})


def _head(source_id: str, adapter: str, head_type: str = "atari_discrete") -> dict:
    return {"id": source_id, "namespace": source_id, "type": head_type, "adapter": adapter}


def _sequence_for_split(prefix: str, split: str, *, salt: str = "") -> str:
    for idx in range(10000):
        candidate = f"{prefix}_{split}_{idx}"
        key = f"{salt}{candidate}" if salt else candidate
        if _split_for_sequence(key, SPLITS) == split:
            return candidate
    raise AssertionError(f"could not find sequence for {split}")


def _write_atari_zip(root: Path) -> None:
    raw = root / "outputs/aux/atari_head_zenodo_v4/raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "action_enums.txt").write_text("0 NOOP\n1 FIRE\n3 RIGHT\n", encoding="utf-8")
    with zipfile.ZipFile(raw / "bank_heist.zip", "w") as archive:
        for split in SPLITS:
            seq = _sequence_for_split("atari_demo", split)
            archive.writestr(f"bank_heist/{seq}.tar.bz2", b"not-a-real-tar-needed-for-ref-only")
            archive.writestr(
                f"bank_heist/{seq}.txt",
                "frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions\n"
                f"{seq}_1,0,0,50,0,3,1.0,2.0\n",
            )


def test_builds_atari_head_aux_examples_with_source_specific_contract(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_head("atari_head_zenodo_v4", "atari_head_zip_csv_action_adapter")])
    _write_atari_zip(tmp_path)

    payload = build_examples(_args(tmp_path, source_id=["atari_head_zenodo_v4"]))

    assert payload["status"] == "pass"
    assert payload["total_examples"] == 3
    source = payload["sources"][0]
    assert source["status"] == "pass"
    assert source["split_counts"] == {"train": 1, "val": 1, "test": 1}
    rows = []
    for split in SPLITS:
        path = tmp_path / f"outputs/aux_examples/atari_head_zenodo_v4/{split}.jsonl"
        assert path.exists()
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    first = rows[0]
    assert first["source_id"] == "atari_head_zenodo_v4"
    assert first["action_head_namespace"] == "atari_head_zenodo_v4"
    assert first["action"]["type"] == "atari_discrete"
    assert first["action"]["action_id"] in {0, 1, 3}
    assert first["frame_or_state_ref"].startswith("nested-archive://outputs/aux/atari_head_zenodo_v4/raw/bank_heist.zip!")
    assert first["provenance"]["csv_member"].startswith("bank_heist/")


def _write_minerl_zip(root: Path) -> None:
    source_id = "minerl_2019_zenodo_v2"
    raw = root / f"outputs/aux/{source_id}/raw"
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw / "MineRLTreechop-v0.zip", "w") as archive:
        for split in SPLITS:
            member_stem = _sequence_for_split("trajectory", split, salt="MineRLTreechop-v0.zip:trajectories/")
            archive.writestr(
                f"trajectories/{member_stem}.json",
                json.dumps(
                    {
                        "actions": [
                            {"camera": [1.0, -1.0], "forward": 1, "attack": 0},
                            {"camera": [0.0, 0.0], "jump": 1},
                        ]
                    }
                ),
            )


def test_builds_minerl_zip_action_dict_examples(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_head("minerl_2019_zenodo_v2", "minerl_action_dict_adapter", "minecraft_keyboard_mouse")])
    _write_minerl_zip(tmp_path)

    payload = build_examples(_args(tmp_path, source_id=["minerl_2019_zenodo_v2"]))

    assert payload["status"] == "pass"
    source = payload["sources"][0]
    assert source["split_counts"] == {"train": 2, "val": 2, "test": 2}
    train_rows = [json.loads(line) for line in (tmp_path / "outputs/aux_examples/minerl_2019_zenodo_v2/train.jsonl").read_text().splitlines()]
    assert train_rows[0]["action"]["type"] == "minecraft_keyboard_mouse"
    assert "camera" in train_rows[0]["action"]["raw_action"]
    assert train_rows[0]["frame_or_state_ref"].startswith("zip-json://outputs/aux/minerl_2019_zenodo_v2/raw/MineRLTreechop-v0.zip!")


def _write_minerl_npz_zip(root: Path) -> None:
    source_id = "minerl_2019_zenodo_v2"
    raw = root / f"outputs/aux/{source_id}/raw"
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw / "MineRLTreechop-v0.zip", "w") as archive:
        for split in SPLITS:
            for idx in range(10000):
                sequence = f"npz_trajectory_{split}_{idx}"
                if _split_for_sequence(f"MineRLTreechop-v0.zip:MineRLTreechop-v0/{sequence}", SPLITS) == split:
                    break
            else:
                raise AssertionError(f"could not find npz sequence for {split}")
            member_root = f"MineRLTreechop-v0/{sequence}"
            archive.writestr(f"{member_root}/metadata.json", json.dumps({"duration_steps": 2}))
            buffer = io.BytesIO()
            np.savez(
                buffer,
                **{
                    "action$forward": np.asarray([1, 0], dtype=np.int64),
                    "action$attack": np.asarray([0, 1], dtype=np.int64),
                    "action$camera": np.asarray([[1.5, -2.0], [0.0, 0.25]], dtype=np.float32),
                    "reward": np.asarray([0.0, 1.0], dtype=np.float32),
                },
            )
            archive.writestr(f"{member_root}/rendered.npz", buffer.getvalue())


def test_builds_minerl_rendered_npz_actions(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_head("minerl_2019_zenodo_v2", "minerl_action_dict_adapter", "minecraft_keyboard_mouse")])
    _write_minerl_npz_zip(tmp_path)

    payload = build_examples(_args(tmp_path, source_id=["minerl_2019_zenodo_v2"]))

    assert payload["status"] == "pass"
    source = payload["sources"][0]
    assert source["json_member_count"] == 3
    assert source["npz_member_count"] == 3
    assert source["split_counts"] == {"train": 2, "val": 2, "test": 2}
    rows = [json.loads(line) for line in (tmp_path / "outputs/aux_examples/minerl_2019_zenodo_v2/train.jsonl").read_text().splitlines()]
    assert rows[0]["frame_or_state_ref"].startswith("zip-npz://outputs/aux/minerl_2019_zenodo_v2/raw/MineRLTreechop-v0.zip!")
    assert rows[0]["action"]["raw_action"]["forward"] in {0, 1}
    assert isinstance(rows[0]["action"]["raw_action"]["camera"], list)
    assert rows[0]["provenance"]["npz_member"].endswith("rendered.npz")


def _install_fake_array_record(monkeypatch, tmp_path: Path) -> None:
    package_root = tmp_path / "fake_array_record_pkg"
    module_dir = package_root / "array_record/python"
    module_dir.mkdir(parents=True)
    (package_root / "array_record/__init__.py").write_text("")
    (module_dir / "__init__.py").write_text("")
    (module_dir / "array_record_module.py").write_text(
        "import pickle\n"
        "class ArrayRecordReader:\n"
        "    def __init__(self, path):\n"
        "        self.records = pickle.loads(open(path, 'rb').read())\n"
        "        self.index = 0\n"
        "        self.closed = False\n"
        "    def num_records(self):\n"
        "        return len(self.records)\n"
        "    def read(self):\n"
        "        if self.index >= len(self.records):\n"
        "            raise IndexError(f'Out of range of num_records: {len(self.records)}')\n"
        "        record = self.records[self.index]\n"
        "        self.index += 1\n"
        "        return pickle.dumps(record)\n"
        "    def close(self):\n"
        "        self.closed = True\n"
    )
    monkeypatch.syspath_prepend(str(package_root))
    for name in list(sys.modules):
        if name.startswith("array_record"):
            del sys.modules[name]


def _write_pdoom_records(root: Path) -> None:
    source_id = "p_doom_atari_breakout_hf"
    raw = root / f"outputs/aux/{source_id}/raw"
    write_json(raw / "metadata.json", {"num_actions": 4})
    for split, actions in {"train": [0, 1], "val": [2], "test": [3]}.items():
        split_dir = raw / split
        split_dir.mkdir(parents=True, exist_ok=True)
        record = {"sequence_length": len(actions), "raw_video": b"", "actions": actions}
        (split_dir / "data_0000.array_record").write_bytes(pickle.dumps([record]))


def test_builds_pdoom_array_record_examples_with_optional_reader(tmp_path: Path, monkeypatch) -> None:
    _install_fake_array_record(monkeypatch, tmp_path)
    _write_registry(tmp_path, [_head("p_doom_atari_breakout_hf", "p_doom_array_record_action_adapter")])
    _write_pdoom_records(tmp_path)

    payload = build_examples(_args(tmp_path, source_id=["p_doom_atari_breakout_hf"]))

    assert payload["status"] == "pass"
    source = payload["sources"][0]
    assert source["split_counts"] == {"train": 2, "val": 1, "test": 1}
    row = json.loads((tmp_path / "outputs/aux_examples/p_doom_atari_breakout_hf/test.jsonl").read_text().strip())
    assert row["action"]["type"] == "atari_discrete"
    assert row["action"]["action_id"] == 3
    assert row["action"]["action_enum"] == "LEFT"
    assert row["frame_or_state_ref"].startswith("array-record://outputs/aux/p_doom_atari_breakout_hf/raw/test/data_0000.array_record#")


def test_pdoom_blocks_without_array_record_dependency(tmp_path: Path, monkeypatch) -> None:
    for name in list(sys.modules):
        if name.startswith("array_record"):
            del sys.modules[name]
    monkeypatch.syspath_prepend(str(tmp_path / "empty"))
    monkeypatch.setattr(
        aux_examples_module,
        "_load_array_record_reader",
        lambda: (None, ModuleNotFoundError("No module named 'array_record'")),
    )
    _write_registry(tmp_path, [_head("p_doom_atari_breakout_hf", "p_doom_array_record_action_adapter")])
    _write_pdoom_records(tmp_path)

    payload = build_examples(_args(tmp_path, source_id=["p_doom_atari_breakout_hf"]))

    assert payload["status"] == "blocked"
    assert any(item["code"] == "array_record_dependency_missing" for item in payload["findings"])


def test_reports_invalid_zip_as_warning_when_incomplete_raw_is_allowed(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_head("atari_head_zenodo_v4", "atari_head_zip_csv_action_adapter")])
    raw = tmp_path / "outputs/aux/atari_head_zenodo_v4/raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "partial.zip").write_bytes(b"not a complete zip")

    payload = build_examples(_args(tmp_path, source_id=["atari_head_zenodo_v4"], allow_incomplete_raw=True))

    assert payload["status"] == "blocked"
    codes = {item["code"] for item in payload["findings"]}
    assert "aux_raw_zip_invalid" in codes
    assert "aux_examples_empty" in codes
