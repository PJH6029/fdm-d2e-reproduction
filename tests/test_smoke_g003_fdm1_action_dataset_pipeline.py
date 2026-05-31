from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.smoke_g003_fdm1_action_dataset_pipeline import run_smoke


def test_run_smoke_passes_full_synthetic_post_extraction_path(tmp_path: Path):
    summary = run_smoke(tmp_path / "smoke", force=True)
    assert summary["status"] == "pass"
    assert summary["finalization_status"] == "pass"
    assert summary["evidence_bundle_status"] == "pass"
    assert summary["monitor_status"] == "pass"
    assert summary["handoff_status"] == "ready_to_checkpoint"
    assert "Synthetic smoke only" in summary["claim_boundary"]


def test_smoke_cli_writes_summary(tmp_path: Path):
    root = tmp_path / "smoke"
    summary = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_g003_fdm1_action_dataset_pipeline.py",
            "--root",
            str(root),
            "--summary-out",
            str(summary),
            "--force",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "G003 synthetic pipeline smoke: status=pass" in completed.stdout
    data = json.loads(summary.read_text())
    assert data["status"] == "pass"
