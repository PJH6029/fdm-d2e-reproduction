from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "log_wandb_artifacts.py"
    spec = importlib.util.spec_from_file_location("log_wandb_artifacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_env_file_does_not_override_existing(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_PROJECT=from_file\nWANDB_ENTITY=team\nBAD\n", encoding="utf-8")
    monkeypatch.setenv("WANDB_PROJECT", "existing")

    loaded = module._load_env_file(env_file)

    assert loaded == ["WANDB_PROJECT", "WANDB_ENTITY"]
    assert os.environ["WANDB_PROJECT"] == "existing"
    assert os.environ["WANDB_ENTITY"] == "team"


def test_flatten_numeric_skips_strings_and_caps() -> None:
    module = _load_module()
    out: dict[str, float | int] = {}

    module._flatten_numeric("root", {"a": 1, "b": True, "c": "secret", "d": {"x": 2.5, "y": 3}}, out, max_items=3)

    assert out == {"root/a": 1, "root/b": 1, "root/d/x": 2.5}


def test_artifact_file_metadata_reports_missing(tmp_path: Path) -> None:
    module = _load_module()
    existing = tmp_path / "metrics.json"
    existing.write_text("{}", encoding="utf-8")

    assert module._artifact_file_metadata(existing)["bytes"] == 2
    assert module._artifact_file_metadata(tmp_path / "missing.json")["exists"] is False
