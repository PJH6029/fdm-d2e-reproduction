from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fdm_d2e.io_utils import read_json, write_json


def _fixture(root: Path) -> Path:
    paths = {
        "dataset_summary": "outputs/slots/dataset_summary.json",
        "action_slots": "outputs/slots/action_slots.jsonl",
        "train_core_slots": "outputs/slots/splits/train_core.jsonl",
        "fitted_mouse_bins": "artifacts/bins.json",
        "visual_alignment_report": "artifacts/visual.md",
    }
    for rel in paths.values():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if rel.endswith(".json") or rel.endswith(".jsonl") else "FDM-1 action-slot alignment visual check\n")
    write_json(root / paths["dataset_summary"], {"dataset_fingerprint": "fp", "output_hashes": {"all": "a" * 64, "train_core": "b" * 64}})
    audit_path = root / "artifacts/audit.json"
    write_json(audit_path, {"schema": "fdm1_g003_action_dataset_completion_audit.v1", "status": "pass", "error_count": 0})
    cfg = {
        "paths": paths,
        "omit_sha256_artifact_keys": ["action_slots", "train_core_slots"],
        "output_path": "artifacts/audit.json",
    }
    cfg_path = root / "configs/completion.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cfg_path, cfg)
    return cfg_path


def test_build_fdm1_g003_evidence_bundle_cli_stages_small_files_and_large_hashes(tmp_path: Path):
    cfg_path = _fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_fdm1_g003_evidence_bundle.py",
            "--completion-config",
            str(cfg_path),
            "--root",
            str(tmp_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "built FDM-1 G003 evidence bundle" in completed.stdout
    manifest = read_json(tmp_path / "artifacts/sources/fdm1_g003_evidence_bundle_manifest.json")
    assert manifest["status"] == "pass"
    assert any(item["key"] == "dataset_summary" and item["copied"] for item in manifest["copied_artifacts"])
    large_by_key = {item["key"]: item for item in manifest["large_artifacts_not_copied"]}
    assert large_by_key["action_slots"]["sha256"] == "a" * 64
    assert large_by_key["train_core_slots"]["sha256"] == "b" * 64
