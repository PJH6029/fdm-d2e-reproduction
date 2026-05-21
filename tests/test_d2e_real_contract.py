import tempfile
import unittest
from pathlib import Path
import sys
import urllib.error
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fdm_d2e.data.d2e_real import (
    D2ERecordingRef,
    _ppm_features,
    build_real_manifests,
    build_recording_refs,
    build_window_records,
    choose_action_dense_window_start,
    download_recording_ref,
    normalize_owa_event,
    normalize_owa_events,
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
        move = normalize_owa_event("mouse/raw", {"last_x": 5, "last_y": -2, "button_flags": 0}, 124)
        button = normalize_owa_event("mouse/raw", {"dx": 0, "dy": 0, "button_flags": 1}, 125)
        self.assertEqual(key["type"], "keyboard")
        self.assertEqual(move["type"], "mouse_move")
        self.assertEqual(button["type"], "mouse_button")
        self.assertEqual(button["button"], "left")
        self.assertEqual(button["event_type"], "press")

    def test_normalize_raw_mouse_preserves_move_button_and_wheel(self):
        rows = normalize_owa_events("mouse/raw", {"last_x": 3, "last_y": -1, "button_flags": 0x0001 | 0x0400, "button_data": 65416}, 200)
        self.assertEqual([row["type"] for row in rows], ["mouse_move", "mouse_button", "scroll"])
        self.assertEqual(rows[1]["button"], "left")
        self.assertEqual(rows[2]["dy"], -1.0)

    def test_build_window_records_bins_real_decoded_actions(self):
        ref = D2ERecordingRef(
            repo_id="open-world-agents/D2E-480p",
            revision="main",
            game="Apex_Legends",
            recording_id="0805_01",
            video_path="Apex_Legends/0805_01.mkv",
            mcap_path="Apex_Legends/0805_01.mcap",
            video_url="https://example.test/0805_01.mkv",
            mcap_url="https://example.test/0805_01.mcap",
        )
        events = [
            {"type": "screen", "timestamp_ns": 1_000_000_000, "pts_ns": 1_000_000_000},
            {"type": "keyboard", "event_type": "press", "key": "87", "timestamp_ns": 1_005_000_000},
            {"type": "mouse_move", "dx": 2, "dy": -2, "timestamp_ns": 1_055_000_000},
        ]
        frames = [
            {"frame_index": 0, "path": "frame0.ppm", "features": [0.1, 0.2, 0.3, 0.2]},
            {"frame_index": 1, "path": "frame1.ppm", "features": [0.4, 0.5, 0.6, 0.5]},
        ]
        records = build_window_records(ref, events, split="train", bin_ms=50, frame_features=frames)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["ground_truth_tokens"], ["KEY_PRESS_87"])
        self.assertEqual(records[1]["ground_truth_tokens"], ["MOUSE_DX_P2", "MOUSE_DY_N2"])
        self.assertEqual(records[0]["next_frame_features"], frames[1]["features"])
        self.assertAlmostEqual(records[0]["frame_delta_features"][0], 0.3)

    def test_choose_action_dense_window_skips_noop_prefix(self):
        events = [
            {"type": "screen", "timestamp_ns": 0},
            {"type": "keyboard", "timestamp_ns": 1_000_000_000},
            {"type": "mouse_move", "timestamp_ns": 2_000_000_000},
            {"type": "mouse_move", "timestamp_ns": 2_010_000_000},
        ]
        self.assertEqual(choose_action_dense_window_start(events, duration_ns=50_000_000), 2_000_000_000)

    def test_ppm_feature_extraction_uses_real_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            ppm = Path(td) / "tiny.ppm"
            ppm.write_bytes(b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 0, 255]))
            features = _ppm_features(ppm)
            self.assertAlmostEqual(features[0], 0.5)
            self.assertAlmostEqual(features[1], 0.0)
            self.assertAlmostEqual(features[2], 0.5)

    def test_download_recording_ref_retries_transient_url_errors(self):
        ref = D2ERecordingRef(
            repo_id="open-world-agents/D2E-480p",
            revision="main",
            game="Apex_Legends",
            recording_id="0805_01",
            video_path="Apex_Legends/0805_01.mkv",
            mcap_path="Apex_Legends/0805_01.mcap",
            video_url="https://example.test/0805_01.mkv",
            mcap_url="https://example.test/0805_01.mcap",
        )

        class FakeResponse:
            def __init__(self, chunks):
                self.chunks = list(chunks)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, _size):
                if self.chunks:
                    return self.chunks.pop(0)
                return b""

        calls = [
            urllib.error.URLError(ConnectionResetError("connection reset by peer")),
            FakeResponse([b"abc", b"def"]),
        ]

        def fake_urlopen(_req, timeout):
            self.assertEqual(timeout, 120)
            item = calls.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch("urllib.request.urlopen", side_effect=fake_urlopen),
            mock.patch("time.sleep") as sleep,
        ):
            result = download_recording_ref(
                ref,
                td,
                kinds=("video",),
                max_attempts=2,
                retry_backoff_s=0.0,
            )

            video_path = Path(result["video"])
            self.assertEqual(video_path.read_bytes(), b"abcdef")
            self.assertFalse(video_path.with_name(f"{video_path.name}.part").exists())
            sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
