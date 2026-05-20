from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML without requiring PyYAML.

    The checked-in .yaml files intentionally use JSON syntax. If a user provides
    true YAML and PyYAML is installed, it will be used as a convenience.
    """
    p = Path(path)
    text = p.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ValueError(f"{p} is not JSON-compatible YAML and PyYAML is unavailable") from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"{p} must load to an object")
        return loaded
