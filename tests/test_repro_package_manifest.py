from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_repro_package_manifest import DEFAULT_PATTERNS, iter_paths


def test_default_repro_manifest_patterns_cover_repo_native_reproduction_surface() -> None:
    paths = {str(path) for path in iter_paths(DEFAULT_PATTERNS)}
    expected = set()
    for pattern in [
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "docs/*.md",
        "notes/*.md",
        "configs/**/*.json",
        "configs/**/*.yaml",
        "schemas/*.json",
        "scripts/*.py",
        "scripts/*.sh",
        "src/fdm_d2e/**/*.py",
        "tests/test_*.py",
    ]:
        expected.update(str(path) for path in Path().glob(pattern) if path.is_file())

    missing = sorted(expected - paths)
    assert missing == []
