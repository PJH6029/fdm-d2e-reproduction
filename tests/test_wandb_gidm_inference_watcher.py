from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "watch_wandb_gidm_inference.py"
    spec = importlib.util.spec_from_file_location("watch_wandb_gidm_inference", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mcap_status_counts_final_zero_and_temp_outputs(tmp_path: Path) -> None:
    module = _load_module()
    pred = tmp_path / "predicted_mcap"
    pred.mkdir()
    (pred / "a.mcap").write_bytes(b"mcap")
    (pred / "b.mcap").write_bytes(b"")
    (pred / "c.mcap.tmp.123.0").write_bytes(b"partial")

    status = module._mcap_status(pred)

    assert status["final_mcap_count"] == 1
    assert status["zero_final_mcap_count"] == 1
    assert status["temp_output_count"] == 1
    assert status["final_mcap_bytes"] == 4
    assert status["temp_output_bytes"] == 7


def test_latest_gpu_rows_parses_epoch_prefixed_monitor(tmp_path: Path) -> None:
    module = _load_module()
    csv_path = tmp_path / "gpu.csv"
    csv_path.write_text(
        "1,2026/05/25 18:00:00.000,0,10,2,3000,143771,100.0\n"
        "2,2026/05/25 18:01:00.000,0,20,3,4000,143771,120.0\n"
        "2,2026/05/25 18:01:00.000,1,30,4,5000,143771,130.0\n",
        encoding="utf-8",
    )

    rows = module._latest_gpu_rows(csv_path)

    assert rows[0]["index"] == "0"
    assert rows[0]["utilization_gpu_pct"] == 20.0
    assert rows[1]["index"] == "1"
    assert rows[1]["memory_used_mib"] == 5000.0


def test_latest_gpu_rows_parses_named_monitor(tmp_path: Path) -> None:
    module = _load_module()
    csv_path = tmp_path / "gpu.csv"
    csv_path.write_text(
        "timestamp, index, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W]\n"
        "2026/05/25 18:01:00.000, 2, H200, 42 %, 7 %, 8000 MiB, 143771 MiB, 250 W\n",
        encoding="utf-8",
    )

    rows = module._latest_gpu_rows(csv_path)

    assert rows == [
        {
            "timestamp": "2026/05/25 18:01:00.000",
            "index": "2",
            "utilization_gpu_pct": 42.0,
            "utilization_memory_pct": 7.0,
            "memory_used_mib": 8000.0,
            "memory_total_mib": 143771.0,
            "power_draw_w": 250.0,
        }
    ]
