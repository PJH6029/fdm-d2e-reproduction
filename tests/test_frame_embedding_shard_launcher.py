from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_frame_embedding_shards import build_shard_plan, summarize_gpu_monitor


def _compact_fields(value: float) -> dict:
    return {
        "grid8": [value for _ in range(8 * 8 * 3)],
        "luma16": [value for _ in range(16 * 16)],
    }


def _row(idx: int) -> dict:
    return {
        "sequence_id": f"seq-{idx}",
        "recording_id": "rec-a",
        "timestamp_ns": idx * 50_000_000,
        "bin_index": idx,
        "game": "test",
        "split": "target",
        "frame": {"path": f"/missing/frame_{idx:04d}.ppm", "features": [idx], **_compact_fields(0.1 + idx)},
        "next_frame_features": [idx + 1],
        "next_frame_grid8": [0.2 + idx for _ in range(8 * 8 * 3)],
        "next_frame_luma16": [0.2 + idx for _ in range(16 * 16)],
        "prior_action_tokens": [],
        "ground_truth_tokens": ["NOOP"],
        "eval_split_tags": ["temporal"],
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_build_shard_plan_assigns_contiguous_skip_ranges(tmp_path: Path) -> None:
    specs = build_shard_plan(
        total_rows=5,
        shard_count=2,
        start_row=10,
        output_dir=tmp_path / "out",
        artifact_dir=tmp_path / "artifacts",
        artifact_prefix="demo",
        devices=["0", "1"],
    )
    assert [(spec.index, spec.skip_rows, spec.max_rows, spec.cuda_visible_devices) for spec in specs] == [
        (0, 10, 3, "0"),
        (1, 13, 2, "1"),
    ]


def test_shard_launcher_materializes_and_combines_dummy_stat_rows(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join(json.dumps(_row(idx), sort_keys=True) for idx in range(4)) + "\n")
    summary_path = tmp_path / "artifacts" / "run_summary.json"
    combined_path = tmp_path / "combined.jsonl"

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_frame_embedding_shards.py"),
            "--input-path",
            str(input_path),
            "--output-dir",
            str(tmp_path / "shards"),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--summary-out",
            str(summary_path),
            "--combined-output-path",
            str(combined_path),
            "--artifact-prefix",
            "unit",
            "--total-rows",
            "4",
            "--shard-count",
            "2",
            "--backend",
            "dummy-stat",
            "--frame-source",
            "compact-luma",
            "--frame-offsets",
            "0,2",
            "--image-size",
            "8",
            "--no-summary-features",
            "--no-gpu-monitor",
            "--progress-rows",
            "1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "pass"
    assert summary["rows_written"] == 4
    assert summary["combined_rows"] == 4
    assert [item["skip_rows"] for item in summary["shards"]] == [0, 2]
    assert [item["max_rows"] for item in summary["shards"]] == [2, 2]
    assert [item["source_rows_skipped"] for item in summary["shards"]] == [0, 2]
    assert [row["sequence_id"] for row in _read_jsonl(combined_path)] == ["seq-0", "seq-1", "seq-2", "seq-3"]


def test_summarize_gpu_monitor_parses_nvidia_smi_csv(tmp_path: Path) -> None:
    path = tmp_path / "gpu.csv"
    path.write_text(
        "timestamp, index, name, utilization.gpu [%], utilization.memory [%]\n"
        "2026/05/28 10:00:00.000, 0, H200, 25 %, 10 %\n"
        "2026/05/28 10:00:01.000, 0, H200, 75 %, 12 %\n"
        "2026/05/28 10:00:01.000, 1, H200, 50 %, 11 %\n"
    )
    summary = summarize_gpu_monitor(path)
    assert summary["available"] is True
    assert summary["by_index"]["0"]["samples"] == 2
    assert summary["by_index"]["0"]["mean"] == 50.0
    assert summary["by_index"]["1"]["max"] == 50.0
