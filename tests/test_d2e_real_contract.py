import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fdm_d2e.data.d2e_real import (
    build_real_manifests,
    build_recording_refs,
    normalize_owa_event,
    prepare_real_dataset,
    split_recordings,
)


FAKE_FILES = [
    "Apex_Legends/0805_01.mcap",
    "Apex_Legends/0805_01.mkv",
    "Apex_Legends/0805_02.mcap",
    "Apex_Legends/0805_02.mkv",
    "Brotato/0901_01.mcap",
    "Brotato/0901_01.mkv",
    "Brotato/ignored_only.mcap",
    "README.md",
]


class D2ERealContractTests(unittest.TestCase):
    def test_build_recording_refs_requires_video_mcap_pairs(self):
        refs = build_recording_refs("open-world-agents/D2E-480p", FAKE_FILES)
        self.assertEqual([r.pair_id for r in refs], ["Apex_Legends/0805_01", "Apex_Legends/0805_02", "Brotato/0901_01"])
        self.assertTrue(refs[0].video_url.endswith("/Apex_Legends/0805_01.mkv"))
        self.assertTrue(refs[0].mcap_url.endswith("/Apex_Legends/0805_01.mcap"))

    def test_split_recordings_keeps_heldout_recording(self):
        refs = build_recording_refs("open-world-agents/D2E-480p", FAKE_FILES)
        split = split_recordings(refs, train_fraction=0.8, min_heldout=1)
        self.assertEqual(len(split["train"]), 2)
        self.assertEqual(len(split["heldout"]), 1)

    def test_build_real_manifests_validates_v2_contracts(self):
        prepared = build_real_manifests(
            {"hf_repo_id": "open-world-agents/D2E-480p", "max_recordings": 2, "train_fraction": 0.5},
            files=FAKE_FILES,
        )
        manifest = prepared["manifest"]
        self.assertEqual(manifest["schema"], "data_manifest.v2")
        self.assertEqual(manifest["license"], "cc-by-nc-4.0")
        self.assertEqual(manifest["source_contract"]["default_bin_ms"], 50)
        self.assertEqual(manifest["splits"], {"train": 1, "heldout": 1})
        self.assertEqual(prepared["recording_manifest"]["num_recordings"], 2)
        self.assertEqual(set(prepared["split_manifest"]["splits"]), {"train", "heldout"})
        self.assertEqual(prepared["sequence_pack"]["schema"], "sequence_pack.v2")
        self.assertEqual(len(prepared["sequence_pack"]["sequences"]), 2)
        self.assertEqual(prepared["sequence_pack"]["sequences"][0]["event_source"]["type"], "mcap")

    def test_prepare_real_dataset_writes_manifest_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            prepared = prepare_real_dataset({"output_dir": td, "max_recordings": 2}, files=FAKE_FILES)
            data_dir = Path(td) / "data"
            self.assertTrue((data_dir / "manifest.v2.json").exists())
            self.assertTrue((data_dir / "recording_manifest.json").exists())
            self.assertTrue((data_dir / "split_manifest.json").exists())
            self.assertTrue((data_dir / "sample_sequence_pack.v2.json").exists())
            self.assertEqual(len(prepared["manifest"]["recordings"]), 2)

    def test_normalize_owa_event_without_owa_imports(self):
        key = normalize_owa_event("keyboard", {"event_type": "press", "vk": 87}, 123)
        move = normalize_owa_event("mouse/raw", {"dx": 5, "dy": -2, "button_flags": 0}, 124)
        button = normalize_owa_event("mouse/raw", {"dx": 0, "dy": 0, "button_flags": 1}, 125)
        self.assertEqual(key["type"], "keyboard")
        self.assertEqual(move["type"], "mouse_move")
        self.assertEqual(button["type"], "mouse_button")


if __name__ == "__main__":
    unittest.main()
