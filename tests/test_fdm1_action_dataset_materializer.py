from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fdm_d2e.data.fdm1_action_dataset import build_alignment_summary, materialize_action_slot_records, write_action_slot_dataset
from fdm_d2e.io_utils import read_json, read_jsonl, write_jsonl
from fdm_d2e.tokenization.fdm1_actions import ActionSlotTokenizer


def _window_records() -> list[dict]:
    return [
        {
            "schema": "d2e_window_record.v1",
            "sequence_id": "Toy/0001#000000",
            "recording_id": "Toy/0001",
            "game": "ToyGame",
            "split": "train_core",
            "timestamp_ns": 0,
            "bin_index": 0,
            "frame": {"path": "toy.mkv#frame=0", "index": 0, "features": [0.1], "grid8": [0.0] * 192},
            "events": [
                {"type": "mouse_move", "dx": 5, "dy": -9, "timestamp_ns": 10_000_000},
                {"type": "keyboard", "event_type": "press", "vk": 87, "timestamp_ns": 20_000_000},
            ],
            "eval_split_tags": [],
        },
        {
            "schema": "d2e_window_record.v1",
            "sequence_id": "Toy/0001#000001",
            "recording_id": "Toy/0001",
            "game": "ToyGame",
            "split": "eval",
            "eval_split_tags": ["temporal"],
            "timestamp_ns": 50_000_000,
            "bin_index": 1,
            "frame": {"path": "toy.mkv#frame=1", "index": 1, "features": [0.2]},
            "events": [
                {"type": "mouse_button", "event_type": "press", "button": "left", "x": 427, "y": 240, "timestamp_ns": 60_000_000},
            ],
        },
    ]


def test_materialize_action_slot_records_preserves_50ms_video_and_idm_mask_contract():
    rows = materialize_action_slot_records(_window_records(), tokenizer=ActionSlotTokenizer(k_event_slots=4), bin_ms=50, frame_fps=20)

    assert rows[0]["schema"] == "fdm1_action_slot_record.v1"
    assert rows[0]["video_bin"]["sample_policy"] == "center_frame_per_50ms_bin"
    assert rows[0]["video_bin"]["frame_fps"] == 20
    assert rows[0]["action_tokens"][:3] == ["MOUSE_MOVE_BIN_P05_N07", "KEY_DOWN_87", "NO_ACTION"]
    assert rows[0]["idm_masked_action_tokens"] == ["MOUSE_MOVE_BIN_P05_N07", "MASK_ACTION", "MASK_ACTION", "MASK_ACTION", "MASK_ACTION"]
    assert rows[0]["click_position_target"] == "NEXT_CLICK_POSITION_BIN_16_9"
    assert rows[1]["event_slots"][0] == "MOUSE_LEFT_DOWN"


def test_write_action_slot_dataset_emits_packed_splits_and_summaries(tmp_path: Path):
    result = write_action_slot_dataset(_window_records(), output_dir=tmp_path, source_paths=[], tokenizer=ActionSlotTokenizer(k_event_slots=4))

    assert (tmp_path / "action_slots.jsonl").exists()
    assert (tmp_path / "splits" / "train_core.jsonl").exists()
    assert (tmp_path / "splits" / "target_temporal.jsonl").exists()
    summary = read_json(tmp_path / "dataset_summary.json")
    alignment = read_json(tmp_path / "alignment_summary.json")
    overflow = read_json(tmp_path / "overflow_summary.json")
    pack = read_json(tmp_path / "sequence_pack.json")
    assert summary["records"] == 2
    assert summary["split_counts"]["train_core"] == 1
    assert alignment["status"] == "pass"
    assert overflow["overflow_events"] == 0
    assert pack["schema"] == "fdm1_action_sequence_pack.v1"
    assert len(read_jsonl(tmp_path / "splits" / "target_all_eval.jsonl")) == 1
    assert result["summary"]["dataset_fingerprint"] == pack["dataset_fingerprint"]


def test_alignment_summary_fails_events_outside_50ms_bin():
    rows = _window_records()
    rows[0] = {**rows[0], "events": [{"type": "keyboard", "event_type": "press", "vk": 65, "timestamp_ns": 70_000_000}]}
    summary = build_alignment_summary(rows, bin_ms=50, frame_fps=20)
    assert summary["status"] == "fail"
    assert summary["event_outside_bin_count"] == 1


def test_materialize_fdm1_action_dataset_cli_smoke(tmp_path: Path):
    input_path = tmp_path / "records.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, _window_records())

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_fdm1_action_dataset.py",
            "--input-records",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--k-event-slots",
            "4",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "materialized FDM-1 action slots" in completed.stdout
    rows = read_jsonl(output_dir / "action_slots.jsonl")
    assert rows[0]["event_slots"][0] == "KEY_DOWN_87"
    cli_summary = json.loads((output_dir / "dataset_summary.json").read_text())
    assert cli_summary["records"] == 2
    assert cli_summary["streaming"] is True
