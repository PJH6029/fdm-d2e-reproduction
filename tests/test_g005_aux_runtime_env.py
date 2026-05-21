from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from validate_g005_aux_runtime_env import validate_runtime_env


_FAKE_MODULE_PREFIXES = ("array_record", "huggingface_hub", "torch")


@pytest.fixture(autouse=True)
def _restore_runtime_dependency_modules():
    saved = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in _FAKE_MODULE_PREFIXES)
    }
    yield
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in _FAKE_MODULE_PREFIXES):
            del sys.modules[name]
    sys.modules.update(saved)


def _args(root: Path, **overrides) -> Namespace:
    data = {"root": str(root), "action_registry": "artifacts/aux/action_registry.json", "output": "artifacts/aux/runtime_env.json", "allow_fail": False}
    data.update(overrides)
    return Namespace(**data)


def _write_registry(root: Path, adapters: list[str]) -> None:
    write_json(
        root / "artifacts/aux/action_registry.json",
        {
            "schema": "g005_aux_action_registry.v1",
            "status": "pass",
            "action_heads": [
                {"id": f"source_{idx}", "namespace": f"source_{idx}", "type": "source_specific", "adapter": adapter}
                for idx, adapter in enumerate(adapters)
            ],
        },
    )


def _install_fake_modules(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "fake_modules"
    for package in ["array_record/python", "huggingface_hub", "torch"]:
        path = root / package
        path.mkdir(parents=True, exist_ok=True)
        parts = package.split("/")
        accum = root
        for part in parts:
            accum = accum / part
            init = accum / "__init__.py"
            init.parent.mkdir(parents=True, exist_ok=True)
            init.write_text("__version__ = 'test'\n", encoding="utf-8")
    (root / "array_record/python/array_record_module.py").write_text(
        "class ArrayRecordReader:\n    pass\n__version__ = 'test'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(root))
    for name in list(sys.modules):
        if name.startswith(("array_record", "huggingface_hub", "torch")):
            del sys.modules[name]


def test_runtime_env_passes_when_selected_adapter_dependencies_exist(tmp_path: Path, monkeypatch) -> None:
    _install_fake_modules(monkeypatch, tmp_path)
    _write_registry(tmp_path, ["atari_head_zip_csv_action_adapter", "minerl_action_dict_adapter", "p_doom_array_record_action_adapter"])

    payload = validate_runtime_env(_args(tmp_path))

    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    modules = {row["module"] for row in payload["checks"]}
    assert "array_record.python.array_record_module" in modules
    assert payload["selected_adapters"]["p_doom_array_record_action_adapter"] == ["source_2"]


def test_runtime_env_blocks_missing_array_record_for_pdoom(tmp_path: Path, monkeypatch) -> None:
    _install_fake_modules(monkeypatch, tmp_path)
    for name in list(sys.modules):
        if name.startswith("array_record"):
            del sys.modules[name]
    # Remove fake array_record but leave torch/huggingface_hub available.
    import shutil

    shutil.rmtree(tmp_path / "fake_modules/array_record")
    _write_registry(tmp_path, ["p_doom_array_record_action_adapter"])

    payload = validate_runtime_env(_args(tmp_path))

    assert payload["status"] == "blocked"
    assert any(item["code"] == "missing_runtime_dependency" and item["module"] == "array_record.python.array_record_module" for item in payload["findings"])


def test_runtime_env_blocks_unknown_adapter_requirements(tmp_path: Path, monkeypatch) -> None:
    _install_fake_modules(monkeypatch, tmp_path)
    _write_registry(tmp_path, ["new_unknown_adapter"])

    payload = validate_runtime_env(_args(tmp_path))

    assert payload["status"] == "blocked"
    assert any(item["code"] == "unknown_aux_adapter_runtime_requirements" for item in payload["findings"])
