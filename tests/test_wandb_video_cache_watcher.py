from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "watch_wandb_video_cache.py"
    spec = importlib.util.spec_from_file_location("watch_wandb_video_cache", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_status_sums_manifest_rows_bytes_chunks(tmp_path: Path) -> None:
    module = _load_module()
    cache_dir = tmp_path / "target"
    cache_dir.mkdir()
    (cache_dir / "a.manifest.json").write_text(
        json.dumps({"rows": 3, "bytes": 100, "chunks": [{"path": "a"}, {"path": "b"}]}),
        encoding="utf-8",
    )
    (cache_dir / "b.manifest.json").write_text(
        json.dumps({"rows": 5, "bytes": 200, "chunks": [{"path": "c"}]}),
        encoding="utf-8",
    )
    chunk_dir = cache_dir / "a.manifest"
    chunk_dir.mkdir()
    (chunk_dir / "chunk_000000.pt").write_bytes(b"abc")

    status = module._cache_status(cache_dir)

    assert status == {
        "manifest_count": 2,
        "readable_manifest_count": 2,
        "rows": 8,
        "bytes": 300,
        "chunks": 3,
        "chunk_file_count": 1,
        "chunk_file_bytes": 3,
    }
