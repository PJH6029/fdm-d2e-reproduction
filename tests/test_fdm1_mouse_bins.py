from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fdm_d2e.data.fdm1_mouse_bins import build_fitted_mouse_bins, collect_mouse_magnitude_histogram
from fdm_d2e.io_utils import read_json, write_jsonl
from fdm_d2e.tokenization.fdm1_actions import fit_signed_exponential_boundaries_from_histogram


def _records() -> list[dict]:
    return [
        {"split": "train_core", "events": [{"type": "mouse_move", "dx": 1, "dy": -2}, {"type": "keyboard", "vk": 87}]},
        {"split": "train_core", "events": [{"type": "mouse_move", "dx": 8, "dy": -13}]},
        {"split": "eval", "events": [{"type": "mouse_move", "dx": 999, "dy": -999}]},
    ]


def test_fit_signed_exponential_boundaries_from_histogram_is_monotonic():
    boundaries = fit_signed_exponential_boundaries_from_histogram({1: 10, 2: 5, 10: 2, 100: 1})
    assert len(boundaries) == 24
    assert all(a < b for a, b in zip(boundaries, boundaries[1:]))
    assert boundaries[:4] == (1.0, 2.0, 3.0, 4.0)


def test_collect_mouse_magnitude_histogram_uses_train_split_only(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, _records())
    collected = collect_mouse_magnitude_histogram([path], split="train_core")
    assert collected["records_used"] == 2
    assert collected["histogram"] == {1: 1, 2: 1, 8: 1, 13: 1}
    assert 999 not in collected["histogram"]


def test_build_fitted_mouse_bins_writes_summary_and_config(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    bins = tmp_path / "bins.json"
    config = tmp_path / "fitted_config.json"
    write_jsonl(path, _records())
    result = build_fitted_mouse_bins([path], bins_output_path=bins, fitted_config_path=config)

    assert result["summary"]["status"] == "pass"
    assert read_json(bins)["mouse_events"] == 2
    fitted = read_json(config)
    assert fitted["mouse_move"]["positive_boundaries_default"][:4] == [1, 2, 3, 4]
    assert fitted["fitted_from"]["split"] == "train_core"


def test_build_fdm1_mouse_bins_cli(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    bins = tmp_path / "bins.json"
    config = tmp_path / "fitted_config.json"
    write_jsonl(path, _records())
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_fdm1_mouse_bins.py",
            "--input-records",
            str(path),
            "--bins-output",
            str(bins),
            "--fitted-config-output",
            str(config),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "fit FDM-1 mouse bins" in completed.stdout
    assert read_json(bins)["status"] == "pass"
