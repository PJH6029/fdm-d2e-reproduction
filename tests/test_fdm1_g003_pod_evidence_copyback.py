from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.build_fdm1_g003_pod_evidence_copyback import build_plan, classify_completion_paths


def _config() -> dict:
    return {
        "output_path": "artifacts/sources/audit.json",
        "omit_sha256_artifact_keys": ["action_slots", "train_core_slots"],
        "paths": {
            "action_slots": "outputs/data/fdm1_action_slots/action_slots.jsonl",
            "train_core_slots": "outputs/data/fdm1_action_slots/splits/train_core.jsonl",
            "dataset_summary": "outputs/data/fdm1_action_slots/dataset_summary.json",
            "visual_alignment_audit": "artifacts/sources/visual.json",
        },
    }


def test_classify_completion_paths_excludes_large_slot_packs():
    small, large = classify_completion_paths(_config())
    assert "outputs/data/fdm1_action_slots/action_slots.jsonl" not in small
    assert "outputs/data/fdm1_action_slots/splits/train_core.jsonl" not in small
    assert "outputs/data/fdm1_action_slots/dataset_summary.json" in small
    assert "artifacts/sources/audit.json" in small
    assert {entry["output_hash_role"] for entry in large} == {"all", "train_core"}


def test_build_plan_uses_kubectl_tar_and_preserves_paths():
    plan = build_plan(_config(), namespace="p-production", pod="pod-a", kubeconfig="/tmp/kube.yaml", extra_paths=["artifacts/cluster/monitor.json"])
    assert plan["schema"] == "fdm1_g003_pod_evidence_copyback_plan.v1"
    assert "kubectl --kubeconfig /tmp/kube.yaml -n p-production exec pod-a" in plan["shell_command"]
    assert "tar -C /root/work/code/continuous-gui-poc/fdm-d2e-reproduction -cf -" in plan["shell_command"]
    assert "artifacts/cluster/monitor.json" in plan["copy_paths"]
    assert all(not path.endswith("action_slots.jsonl") for path in plan["copy_paths"])


def test_cli_writes_plan_and_shell(tmp_path: Path):
    cfg = tmp_path / "completion.json"
    cfg.write_text(json.dumps(_config()), encoding="utf-8")
    output = tmp_path / "plan.json"
    shell = tmp_path / "copy.sh"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_fdm1_g003_pod_evidence_copyback.py",
            "--completion-config",
            str(cfg),
            "--pod",
            "pod-a",
            "--output",
            str(output),
            "--shell-out",
            str(shell),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    summary = json.loads(completed.stdout)
    assert summary["status"] == "planned"
    data = json.loads(output.read_text())
    assert data["pod"] == "pod-a"
    assert shell.read_text().startswith("#!/usr/bin/env bash")
