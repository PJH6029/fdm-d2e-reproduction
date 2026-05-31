from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import launch_g003_fdm1_action_dataset_pod as launcher


def test_build_launch_commands_refuse_non_pod_and_duplicate_pid():
    commands = launcher.build_launch_commands(repo_dir="/repo", branch="branch", log_path="artifacts/logs/run.log", pid_path="outputs/cluster/run.pid")
    joined = "\n".join(commands)
    assert "KUBERNETES_SERVICE_HOST" in joined
    assert "refusing: this launch script must run inside the MLXP Kubernetes pod" in joined
    assert "existing G003 pipeline pid is still active" in joined
    assert "nohup bash scripts/run_g003_fdm1_action_dataset_pipeline.sh" in joined


def test_build_launch_commands_can_replace_existing_pid():
    commands = launcher.build_launch_commands(repo_dir="/repo", replace_existing=True)
    joined = "\n".join(commands)
    assert "stopping existing G003 launch pid" in joined
    assert "existing G003 pipeline pid is still active" not in joined


def test_parse_env_assignments_rejects_shell_unsafe_keys():
    try:
        launcher.parse_env_assignments(["BAD-KEY=value"])
    except ValueError as exc:
        assert "invalid environment key" in str(exc)
    else:  # pragma: no cover - defensive assertion for clearer failure
        raise AssertionError("expected invalid env key to be rejected")


def test_cli_writes_plan_and_shell_without_executing(tmp_path: Path):
    plan = tmp_path / "plan.json"
    shell = tmp_path / "launch.sh"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/launch_g003_fdm1_action_dataset_pod.py",
            "--repo-dir",
            "/repo",
            "--plan-out",
            str(plan),
            "--shell-out",
            str(shell),
            "--env",
            "EXTRACT_EXTRA_ARGS=--max-recordings 2",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = json.loads(completed.stdout)
    assert stdout["execute"] is False
    data = json.loads(plan.read_text())
    assert data["schema"] == "fdm1_g003_pod_launch_plan.v1"
    assert data["extra_env"] == {"EXTRACT_EXTRA_ARGS": "--max-recordings 2"}
    assert data["runtime_detection"]["inside_mlxp_pod"] is False
    shell_text = shell.read_text()
    assert shell_text.startswith("#!/usr/bin/env bash")
    assert "EXTRACT_EXTRA_ARGS='--max-recordings 2'" in shell_text
    assert "uv run python - <<'PY'" in shell_text
    assert "preflight_g003_fdm1_action_dataset_pod.py --require-pod" in shell_text


def test_execute_refuses_outside_pod(tmp_path: Path, monkeypatch):
    plan = tmp_path / "plan.json"
    shell = tmp_path / "launch.sh"
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    called = False

    def fake_execute(_shell):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(launcher, "execute_shell", fake_execute)
    rc = launcher.main(["--repo-dir", str(tmp_path / "missing-pod-repo"), "--plan-out", str(plan), "--shell-out", str(shell), "--execute"])
    assert rc == 2
    assert called is False
    assert plan.exists()
    assert shell.exists()


def test_detect_runtime_handles_inaccessible_default_path():
    runtime = launcher.detect_runtime()
    assert "inside_mlxp_pod" in runtime
    assert "repo_dir_error" in runtime
