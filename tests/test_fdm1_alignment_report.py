from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fdm_d2e.data.fdm1_action_dataset import write_action_slot_dataset
from fdm_d2e.data.fdm1_alignment_report import audit_action_slot_alignment, build_alignment_report
from fdm_d2e.io_utils import read_json, write_jsonl
from fdm_d2e.tokenization.fdm1_actions import ActionSlotTokenizer


def _window_records() -> list[dict]:
    return [
        {
            "schema": "d2e_window_record.v1",
            "sequence_id": f"Toy/0001#{idx:06d}",
            "recording_id": "Toy/0001",
            "game": "ToyGame",
            "split": "train_core",
            "timestamp_ns": idx * 50_000_000,
            "bin_index": idx,
            "frame": {"path": f"toy.mkv#frame={idx}", "index": idx, "features": [float(idx)]},
            "events": [{"type": "mouse_move", "dx": idx, "dy": -idx, "timestamp_ns": idx * 50_000_000 + 1_000_000}] if idx else [],
            "eval_split_tags": [],
        }
        for idx in range(3)
    ]


def test_alignment_report_renders_human_timeline(tmp_path: Path):
    dataset = write_action_slot_dataset(_window_records(), output_dir=tmp_path / "slots", tokenizer=ActionSlotTokenizer(k_event_slots=4))
    markdown = tmp_path / "alignment.md"
    audit_path = tmp_path / "alignment.json"
    audit = build_alignment_report(dataset["records"], markdown_path=markdown, audit_path=audit_path)

    assert audit["status"] == "pass"
    text = markdown.read_text()
    assert "FDM-1 action-slot alignment visual check" in text
    assert "MOUSE_MOVE_BIN" in text
    assert read_json(audit_path)["error_count"] == 0


def test_alignment_audit_catches_mask_length_mismatch():
    rows = write_action_slot_dataset(_window_records(), output_dir=Path("/tmp/fdm1-alignment-test"), tokenizer=ActionSlotTokenizer(k_event_slots=4))["records"]
    rows[0] = {**rows[0], "idm_masked_action_tokens": ["MOUSE_MOVE_BIN_Z00_Z00"]}
    audit = audit_action_slot_alignment(rows)
    assert audit["status"] == "fail"
    assert "mask/token length mismatch" in audit["errors"][0]


def test_build_fdm1_action_alignment_report_cli(tmp_path: Path):
    dataset = write_action_slot_dataset(_window_records(), output_dir=tmp_path / "slots", tokenizer=ActionSlotTokenizer(k_event_slots=4))
    action_slots = tmp_path / "input_action_slots.jsonl"
    write_jsonl(action_slots, dataset["records"])
    markdown = tmp_path / "report.md"
    audit = tmp_path / "audit.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_fdm1_action_alignment_report.py",
            "--action-slots",
            str(action_slots),
            "--markdown-out",
            str(markdown),
            "--audit-out",
            str(audit),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "built FDM-1 action alignment visual check" in completed.stdout
    assert read_json(audit)["status"] == "pass"
    assert "Toy/0001#000001" in markdown.read_text()
