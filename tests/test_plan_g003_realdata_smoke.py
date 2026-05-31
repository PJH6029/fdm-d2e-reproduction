from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.plan_g003_realdata_smoke import build_plan


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _fixture(root: Path) -> dict:
    universe = {
        "recordings": [
            {
                "source_id": "d2e_480p",
                "resolution_tier": "480p",
                "status": "included",
                "game": "Toy",
                "recording_id": "rec0",
                "repo_id": "open-world-agents/D2E-480p",
                "files": {"mcap": {"path": "Toy/rec0.mcap"}, "video": {"path": "Toy/rec0.mkv"}},
                "requested_revision": "main",
                "resolved_revision": "rev",
                "cross_resolution_key": "Toy/rec0",
            }
        ]
    }
    _write_json(root / "artifacts/sources/universe.json", universe)
    _write_json(root / "configs/data/extract.json", {"data_universe": "artifacts/sources/universe.json"})
    _write_json(root / "configs/data/finalization.json", {"paths": {}})
    _write_json(root / "configs/eval/completion.json", {"paths": {"dataset_summary": "old"}})
    return {
        "extract_config": "configs/data/extract.json",
        "finalization_config": "configs/data/finalization.json",
        "completion_config": "configs/eval/completion.json",
        "output_dir": ".omx/tmp/smoke/window_records",
        "summary_out": ".omx/tmp/smoke/decode_summary.json",
        "cache_dir": ".omx/tmp/smoke/cache",
        "action_output_dir": ".omx/tmp/smoke/action_slots",
        "artifacts_dir": ".omx/tmp/smoke/artifacts",
        "max_recordings": 1,
        "max_bins_per_recording": 8,
        "event_limit": 2000,
        "video_mode": "remote",
        "force": True,
    }


def test_build_realdata_smoke_plan_selects_first_d2e_480p_row(tmp_path: Path, monkeypatch):
    cfg = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    plan = build_plan(cfg)
    assert plan["status"] == "planned"
    assert plan["selected_rows"][0]["universe_row_id"] == "d2e_480p:Toy/rec0"
    command = " ".join(plan["extract_command"])
    assert "--max-recordings 1" in command
    assert "--video-mode remote" in command
    assert "Tiny real-D2E smoke" in plan["claim_boundary"]


def test_cli_writes_plan_and_shell():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/plan_g003_realdata_smoke.py",
            "--output",
            "/tmp/fdm1_g003_realdata_smoke_plan_test.json",
            "--shell-out",
            "/tmp/fdm1_g003_realdata_smoke_test.sh",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "planned"
    assert payload["selected_rows"]
    assert Path("/tmp/fdm1_g003_realdata_smoke_test.sh").read_text().startswith("#!/usr/bin/env bash")
