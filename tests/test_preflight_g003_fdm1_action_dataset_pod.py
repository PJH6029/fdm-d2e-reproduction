from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts import preflight_g003_fdm1_action_dataset_pod as preflight


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _fixture(root: Path, *, bad_split: bool = False) -> None:
    (root / "ROADMAP.md").write_text("# roadmap\n", encoding="utf-8")
    _write_json(
        root / "configs/data/fdm1_d2e_480p_full_corpus_extract.yaml",
        {
            "canonical_roadmap": "ROADMAP.md",
            "split_mode": "wrong" if bad_split else "fdm1-g002",
            "source_ids": ["d2e_480p"],
            "resolution_tiers": ["480p"],
            "bin_ms": 50,
            "frame_fps": 20,
            "output_dir": "outputs/data/fdm1_d2e_480p_window_records",
            "cache_dir": "cache/d2e",
        },
    )
    _write_json(root / "configs/data/fdm1_g003_action_dataset_finalization.yaml", {"k_event_slots": 8})
    _write_json(
        root / "configs/eval/fdm1_g003_action_dataset_completion.yaml",
        {
            "expected_recording_variants": 459,
            "expected_split_mode": "fdm1-g002",
            "required_source_ids": ["d2e_480p"],
            "required_resolution_tiers": ["480p"],
        },
    )
    _write_json(root / "artifacts/sources/fdm1_d2e_g002_validation.json", {"status": "pass"})
    for rel in [
        "artifacts/sources/fdm1_d2e_game_metadata.json",
        "artifacts/sources/fdm1_d2e_recording_level_split_manifest.json",
        "artifacts/sources/fdm1_d2e_heldout_game_split_manifest.json",
        "artifacts/sources/fdm1_d2e_pseudo_label_split_manifest.json",
        "artifacts/sources/fdm1_d2e_scale_split_manifest.json",
        "configs/tokenization/fdm1_action_slots.json",
    ]:
        _write_json(root / rel, {"ok": True, "padding": "x" * 40})


def _args(root: Path, **overrides):
    data = {
        "root": str(root),
        "extract_config": "configs/data/fdm1_d2e_480p_full_corpus_extract.yaml",
        "finalization_config": "configs/data/fdm1_g003_action_dataset_finalization.yaml",
        "completion_config": "configs/eval/fdm1_g003_action_dataset_completion.yaml",
        "pid_file": "outputs/cluster/fdm1_g003_action_dataset_pipeline.pid",
        "output": "artifacts/cluster/preflight.json",
        "expected_branch": "",
        "min_free_gb": 0.0,
        "require_clean": False,
        "require_pod": False,
        "require_cache_dir": False,
        "allow_active_pid": False,
        "allow_blocked": False,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_preflight_ready_with_reset_g003_fixture(tmp_path: Path):
    _fixture(tmp_path)
    payload = preflight.build_preflight(_args(tmp_path))
    assert payload["status"] == "ready"
    assert payload["disk"]["free_gb"] >= 0
    assert payload["artifacts"]["roadmap"]["exists"] is True


def test_preflight_blocks_wrong_split_mode(tmp_path: Path):
    _fixture(tmp_path, bad_split=True)
    payload = preflight.build_preflight(_args(tmp_path))
    assert payload["status"] == "blocked"
    assert any(f["code"] == "unexpected_config_value" for f in payload["findings"])


def test_preflight_require_pod_blocks_local(tmp_path: Path):
    _fixture(tmp_path)
    payload = preflight.build_preflight(_args(tmp_path, require_pod=True))
    assert payload["status"] == "blocked"
    assert any(f["code"] == "not_inside_kubernetes_pod" for f in payload["findings"])


def test_preflight_cli_writes_blocked_output_when_allowed(tmp_path: Path):
    _fixture(tmp_path, bad_split=True)
    output = tmp_path / "preflight.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/preflight_g003_fdm1_action_dataset_pod.py",
            "--root",
            str(tmp_path),
            "--expected-branch",
            "",
            "--output",
            str(output),
            "--allow-blocked",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "status=blocked" in completed.stdout
    assert json.loads(output.read_text())["status"] == "blocked"


def test_preflight_cache_stat_error_is_warning_unless_required(tmp_path: Path, monkeypatch):
    _fixture(tmp_path)
    monkeypatch.setattr(preflight, "_safe_exists", lambda path: (False, "PermissionError: denied") if str(path).endswith("cache/d2e") else (path.exists(), None))
    payload = preflight.build_preflight(_args(tmp_path))
    assert payload["status"] == "ready"
    assert any(f["code"] == "cache_dir_stat_error" for f in payload["findings"])

    payload = preflight.build_preflight(_args(tmp_path, require_cache_dir=True))
    assert payload["status"] == "blocked"
    assert any(f["code"] == "missing_cache_dir" for f in payload["findings"])
