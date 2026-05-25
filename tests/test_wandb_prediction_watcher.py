from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "watch_wandb_prediction.py"
    spec = importlib.util.spec_from_file_location("watch_wandb_prediction", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_env_file_does_not_override_existing(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_PROJECT=from_file\nWANDB_ENTITY=team\n", encoding="utf-8")
    monkeypatch.setenv("WANDB_PROJECT", "existing")

    loaded = module._load_env_file(env_file)

    assert loaded == ["WANDB_PROJECT", "WANDB_ENTITY"]
    assert os.environ["WANDB_PROJECT"] == "existing"
    assert os.environ["WANDB_ENTITY"] == "team"


def test_prediction_status_sums_part_and_canonical_sizes(tmp_path: Path) -> None:
    module = _load_module()
    parts = tmp_path / "prediction_parts"
    (parts / "part_000").mkdir(parents=True)
    (parts / "part_001").mkdir(parents=True)
    (parts / "part_000" / "predictions.jsonl").write_bytes(b"abc")
    (parts / "part_001" / "predictions.jsonl").write_bytes(b"defg")
    (parts / "part_000" / "pseudolabels.jsonl").write_bytes(b"hi")
    predictions = tmp_path / "predictions.jsonl"
    pseudolabels = tmp_path / "pseudolabels.jsonl"
    predictions.write_bytes(b"canonical")

    status = module._prediction_status(parts, predictions, pseudolabels)

    assert status["part_prediction_count"] == 2
    assert status["part_prediction_bytes"] == 7
    assert status["part_pseudolabel_count"] == 1
    assert status["part_pseudolabel_bytes"] == 2
    assert status["predictions"]["bytes"] == 9
    assert status["pseudolabels"]["exists"] is False
