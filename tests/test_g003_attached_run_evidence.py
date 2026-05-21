from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from attach_g003_gpu_monitor import run_monitor
from build_g003_attached_train_run_summary import build_summary


def test_build_attached_train_run_summary_passes_with_integrated_evidence(tmp_path: Path):
    write_json(tmp_path / "artifacts/idm/integrated.json", {"idm_nproc_per_node": 4})
    write_json(tmp_path / "artifacts/idm/summary.json", {"schema": "streaming_idm_train_summary.v1"})
    write_json(
        tmp_path / "outputs/idm/checkpoint_metadata.json",
        {
            "train_records": 10,
            "target_records": 4,
            "checkpoint_path": "outputs/idm/checkpoint.pt",
            "metrics_path": "outputs/idm/metrics.json",
            "label_quality_report_path": "outputs/idm/label_quality_report.json",
            "statistical_comparison_path": "outputs/idm/statistical_comparison.json",
            "convergence_report_path": "outputs/idm/convergence_report.json",
            "convergence_plateau_met": True,
            "distributed": {"enabled": True, "world_size": 4},
        },
    )
    write_json(tmp_path / "artifacts/idm/monitor_meta.json", {"samples": 2})
    gpu = tmp_path / "artifacts/idm/gpu.csv"
    gpu.parent.mkdir(parents=True, exist_ok=True)
    gpu.write_text(
        "sample_unix,parent_pid,timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw\n"
        "1,2,now,0,H200,90,10,1,80,200\n"
        "1,2,now,1,H200,91,11,1,80,201\n"
        "1,2,now,2,H200,92,12,1,80,202\n"
        "1,2,now,3,H200,93,13,1,80,203\n"
    )
    metrics = tmp_path / "outputs/idm/metrics.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text("{}\n")
    args = Namespace(
        integrated_run_evidence=str(tmp_path / "artifacts/idm/integrated.json"),
        idm_summary=str(tmp_path / "artifacts/idm/summary.json"),
        checkpoint_metadata=str(tmp_path / "outputs/idm/checkpoint_metadata.json"),
        metrics=str(metrics),
        gpu_monitor=str(gpu),
        attached_monitor_metadata=str(tmp_path / "artifacts/idm/monitor_meta.json"),
        nproc_per_node=4,
        expected_gpus=4,
    )
    payload = build_summary(args)
    assert payload["exit_code"] == 0
    assert payload["nproc_per_node"] == 4
    assert payload["attached_monitor_samples"] == 2
    assert payload["gpu_monitor_status"]["covers_expected_gpus"] is True
    assert not payload["findings"]


def test_build_attached_train_run_summary_fails_when_core_artifacts_missing(tmp_path: Path):
    args = Namespace(
        integrated_run_evidence=str(tmp_path / "missing_integrated.json"),
        idm_summary=str(tmp_path / "missing_summary.json"),
        checkpoint_metadata=str(tmp_path / "missing_metadata.json"),
        metrics=str(tmp_path / "missing_metrics.json"),
        gpu_monitor=str(tmp_path / "missing_gpu.csv"),
        attached_monitor_metadata=str(tmp_path / "missing_monitor.json"),
        nproc_per_node=4,
        expected_gpus=4,
    )
    payload = build_summary(args)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["exit_code"] == 2
    assert "missing_integrated_run_evidence" in codes
    assert "missing_gpu_monitor" in codes
    assert "missing_attached_monitor_metadata" in codes
    assert "missing_metrics" in codes


def test_build_attached_train_run_summary_rejects_partial_gpu_monitor(tmp_path: Path):
    write_json(tmp_path / "artifacts/idm/integrated.json", {"idm_nproc_per_node": 4})
    write_json(tmp_path / "artifacts/idm/summary.json", {"schema": "streaming_idm_train_summary.v1"})
    write_json(tmp_path / "outputs/idm/checkpoint_metadata.json", {"distributed": {"enabled": True, "world_size": 4}})
    write_json(tmp_path / "artifacts/idm/monitor_meta.json", {"samples": 1})
    gpu = tmp_path / "artifacts/idm/gpu.csv"
    gpu.parent.mkdir(parents=True, exist_ok=True)
    gpu.write_text("sample_unix,parent_pid,timestamp,index\n1,2,now,0\n")
    metrics = tmp_path / "outputs/idm/metrics.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text("{}\n")
    args = Namespace(
        integrated_run_evidence=str(tmp_path / "artifacts/idm/integrated.json"),
        idm_summary=str(tmp_path / "artifacts/idm/summary.json"),
        checkpoint_metadata=str(tmp_path / "outputs/idm/checkpoint_metadata.json"),
        metrics=str(metrics),
        gpu_monitor=str(gpu),
        attached_monitor_metadata=str(tmp_path / "artifacts/idm/monitor_meta.json"),
        nproc_per_node=4,
        expected_gpus=4,
    )
    payload = build_summary(args)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["exit_code"] == 2
    assert "gpu_monitor_does_not_cover_expected_gpus" in codes


def test_attach_gpu_monitor_collects_one_fake_sample(tmp_path: Path):
    fake_smi = tmp_path / "nvidia-smi"
    fake_smi.write_text(
        "#!/usr/bin/env bash\n"
        "cat <<'EOF'\n"
        "2026/05/21 14:00:00, 0, H200, 90, 10, 1024, 81559, 200\n"
        "2026/05/21 14:00:00, 1, H200, 91, 11, 1024, 81559, 201\n"
        "2026/05/21 14:00:00, 2, H200, 92, 12, 1024, 81559, 202\n"
        "2026/05/21 14:00:00, 3, H200, 93, 13, 1024, 81559, 203\n"
        "EOF\n"
    )
    fake_smi.chmod(0o755)
    pid_file = tmp_path / "parent.pid"
    pid_file.write_text(str(os.getpid()) + "\n")
    args = Namespace(
        pid_file=str(pid_file),
        output=str(tmp_path / "gpu.csv"),
        metadata_out=str(tmp_path / "monitor.json"),
        monitor_pid_file=str(tmp_path / "monitor.pid"),
        interval_seconds=0,
        max_samples=1,
        nvidia_smi_bin=str(fake_smi),
        truncate=True,
        force=False,
        keep_running_on_error=False,
    )
    payload = run_monitor(args)
    assert payload["exit_reason"] == "max_samples"
    assert payload["samples"] == 1
    rows = (tmp_path / "gpu.csv").read_text().strip().splitlines()
    assert len(rows) == 5


def test_attach_gpu_monitor_is_idempotent_with_live_monitor_pid(tmp_path: Path):
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        monitor_pid_file = tmp_path / "monitor.pid"
        monitor_pid_file.write_text(str(sleeper.pid) + "\n")
        pid_file = tmp_path / "parent.pid"
        pid_file.write_text(str(os.getpid()) + "\n")
        args = Namespace(
            pid_file=str(pid_file),
            output=str(tmp_path / "gpu.csv"),
            metadata_out=str(tmp_path / "monitor.json"),
            monitor_pid_file=str(monitor_pid_file),
            interval_seconds=0,
            max_samples=1,
            nvidia_smi_bin="missing-nvidia-smi",
            truncate=True,
            force=False,
            keep_running_on_error=False,
        )
        payload = run_monitor(args)
        assert payload["exit_reason"] == "existing_monitor_running"
        assert payload["existing_monitor_pid"] == sleeper.pid
        assert payload["samples"] == 0
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=5)
