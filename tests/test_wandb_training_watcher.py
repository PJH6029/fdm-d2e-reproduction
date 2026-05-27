from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "watch_wandb_training.py"
    spec = importlib.util.spec_from_file_location("watch_wandb_training", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_env_file_does_not_override_existing(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_PROJECT=from_file\nWANDB_ENTITY=team\n# ignored\nBAD\n", encoding="utf-8")
    monkeypatch.setenv("WANDB_PROJECT", "existing")

    loaded = module._load_env_file(env_file)

    assert loaded == ["WANDB_PROJECT", "WANDB_ENTITY"]
    assert os.environ["WANDB_PROJECT"] == "existing"
    assert os.environ["WANDB_ENTITY"] == "team"


def test_latest_gpu_rows_parses_latest_timestamp(tmp_path: Path) -> None:
    module = _load_module()
    monitor = tmp_path / "gpu.csv"
    with monitor.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "index", "name", "utilization.gpu [%]", "utilization.memory [%]", "memory.used [MiB]", "memory.total [MiB]", "power.draw [W]"])
        writer.writerow(["t1", "0", "NVIDIA H200", "10 %", "2 %", "100 MiB", "143771 MiB", "250 W"])
        writer.writerow(["t2", "0", "NVIDIA H200", "55 %", "11 %", "200 MiB", "143771 MiB", "300 W"])
        writer.writerow(["t2", "1", "NVIDIA H200", "65 %", "12 %", "220 MiB", "143771 MiB", "310 W"])

    rows = module._latest_gpu_rows(monitor)

    assert [row["index"] for row in rows] == ["0", "1"]
    assert rows[0]["utilization_gpu_pct"] == 55.0
    assert rows[1]["power_draw_w"] == 310.0


def test_latest_gpu_rows_keeps_same_poll_block_with_slight_timestamp_skew(tmp_path: Path) -> None:
    module = _load_module()
    monitor = tmp_path / "gpu.csv"
    with monitor.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "index",
                "name",
                "utilization.gpu [%]",
                "utilization.memory [%]",
                "memory.used [MiB]",
                "memory.total [MiB]",
                "power.draw [W]",
            ]
        )
        writer.writerow(["2026/05/27 15:34:20.000", "0", "NVIDIA H200", "1 %", "0 %", "10 MiB", "143771 MiB", "80 W"])
        writer.writerow(["2026/05/27 15:34:20.004", "1", "NVIDIA H200", "2 %", "0 %", "10 MiB", "143771 MiB", "81 W"])
        writer.writerow(["2026/05/27 15:34:50.071", "0", "NVIDIA H200", "70 %", "5 %", "1000 MiB", "143771 MiB", "300 W"])
        writer.writerow(["2026/05/27 15:34:50.075", "1", "NVIDIA H200", "80 %", "6 %", "1100 MiB", "143771 MiB", "310 W"])

    rows = module._latest_gpu_rows(monitor)

    assert [row["index"] for row in rows] == ["0", "1"]
    assert [row["utilization_gpu_pct"] for row in rows] == [70.0, 80.0]
