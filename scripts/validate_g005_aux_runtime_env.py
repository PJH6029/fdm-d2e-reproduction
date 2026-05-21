#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json


DEFAULT_OUTPUT = "artifacts/aux/g005_aux_runtime_env.json"
DEFAULT_ACTION_REGISTRY = "artifacts/aux/g005_aux_action_registry.json"

ADAPTER_REQUIREMENTS = {
    "atari_head_zip_csv_action_adapter": [],
    "minerl_action_dict_adapter": [],
    "p_doom_array_record_action_adapter": [
        {
            "module": "array_record.python.array_record_module",
            "symbol": "ArrayRecordReader",
            "package_hint": "array-record",
            "reason": "p-doom Atari Breakout files are ArrayRecord streams containing pickled raw_video/actions records.",
        }
    ],
}
BASE_REQUIREMENTS = [
    {"module": "huggingface_hub", "symbol": None, "package_hint": "huggingface_hub", "reason": "G005 materialization can pull selected Hugging Face dataset sources."},
    {"module": "torch", "symbol": None, "package_hint": "torch", "reason": "G005 D2E+aux pretraining/finetuning uses the torch training stack."},
]


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _module_check(requirement: dict[str, Any]) -> dict[str, Any]:
    module_name = str(requirement["module"])
    symbol = requirement.get("symbol")
    row = {**requirement, "available": False, "version": None, "error": None}
    try:
        module = importlib.import_module(module_name)
        if symbol:
            getattr(module, str(symbol))
        row["available"] = True
        row["version"] = getattr(module, "__version__", None)
    except Exception as exc:  # pragma: no cover - import failures are environment-specific
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _registry_adapters(action_registry: dict[str, Any]) -> dict[str, list[str]]:
    adapters: dict[str, list[str]] = {}
    for row in action_registry.get("action_heads", []) or []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        adapters.setdefault(str(row.get("adapter") or ""), []).append(str(row["id"]))
    return {key: sorted(value) for key, value in sorted(adapters.items()) if key}


def validate_runtime_env(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    findings: list[dict[str, Any]] = []
    action_registry_path = _path(root, args.action_registry)
    action_registry = _load_json(action_registry_path) if action_registry_path.exists() else None
    adapters: dict[str, list[str]] = {}
    requirements: list[dict[str, Any]] = []

    if action_registry is None:
        findings.append({"severity": "error", "code": "missing_action_registry", "path": args.action_registry})
    elif action_registry.get("status") != "pass":
        findings.append({"severity": "error", "code": "action_registry_not_pass", "status": action_registry.get("status")})
        adapters = _registry_adapters(action_registry)
    else:
        adapters = _registry_adapters(action_registry)

    seen_modules: set[tuple[str, str | None]] = set()
    for requirement in BASE_REQUIREMENTS:
        key = (str(requirement["module"]), requirement.get("symbol"))
        if key not in seen_modules:
            requirements.append(requirement)
            seen_modules.add(key)
    for adapter, source_ids in adapters.items():
        adapter_requirements = ADAPTER_REQUIREMENTS.get(adapter)
        if adapter_requirements is None:
            findings.append({"severity": "error", "code": "unknown_aux_adapter_runtime_requirements", "adapter": adapter, "source_ids": source_ids})
            continue
        for requirement in adapter_requirements:
            key = (str(requirement["module"]), requirement.get("symbol"))
            if key in seen_modules:
                continue
            requirements.append({**requirement, "required_by_adapter": adapter, "required_by_source_ids": source_ids})
            seen_modules.add(key)

    checks = [_module_check(requirement) for requirement in requirements]
    for check in checks:
        if not check.get("available"):
            findings.append(
                {
                    "severity": "error",
                    "code": "missing_runtime_dependency",
                    "module": check.get("module"),
                    "symbol": check.get("symbol"),
                    "package_hint": check.get("package_hint"),
                    "required_by_adapter": check.get("required_by_adapter"),
                    "required_by_source_ids": check.get("required_by_source_ids"),
                    "error": check.get("error"),
                }
            )
    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g005_aux_runtime_env.v1",
        "status": "pass" if not errors else "blocked",
        "root": str(root),
        "python_executable": sys.executable,
        "action_registry": args.action_registry,
        "selected_adapters": adapters,
        "checks": checks,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "Runtime dependency preflight only; it does not materialize aux data, launch G005 training, checkpoint goals, or prove D2E+aux quality.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate optional runtime dependencies needed by selected G005 auxiliary adapters.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--action-registry", default=DEFAULT_ACTION_REGISTRY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = validate_runtime_env(args)
    write_json(_path(Path(args.root).resolve(), args.output), payload)
    print(f"g005 aux runtime env: status={payload['status']} errors={payload['error_count']} output={args.output}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
