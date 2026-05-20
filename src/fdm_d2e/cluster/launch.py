from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.io_utils import ensure_dir, write_json

DEFAULT_MLXP_REPO_PATH = "/root/work/code/continuous-gui-poc/fdm-d2e-reproduction"
DEFAULT_OUTPUT_DIR = "outputs/cluster"


@dataclass(frozen=True)
class LaunchCommand:
    gpu_count: int
    repo_path: str
    command: list[str]
    env: dict[str, str]
    report_path: str

    def shell(self) -> str:
        env_bits = [f"{key}={shlex.quote(value)}" for key, value in sorted(self.env.items())]
        return "cd " + shlex.quote(self.repo_path) + " && " + " ".join(env_bits + [shlex.join(self.command)])

    def to_json(self) -> dict[str, Any]:
        return {
            "gpu_count": self.gpu_count,
            "repo_path": self.repo_path,
            "command": self.command,
            "env": self.env,
            "report_path": self.report_path,
            "shell": self.shell(),
        }


def launcher_env(gpu_count: int) -> dict[str, str]:
    if gpu_count < 1:
        raise ValueError("gpu_count must be >= 1")
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(str(idx) for idx in range(gpu_count)),
        "NCCL_DEBUG": "WARN",
        "TOKENIZERS_PARALLELISM": "false",
        "UV_LINK_MODE": "copy",
    }


def build_torchrun_command(
    *,
    gpu_count: int,
    script: str,
    script_args: Sequence[str] = (),
    repo_path: str = DEFAULT_MLXP_REPO_PATH,
    report_path: str | None = None,
) -> LaunchCommand:
    """Build the command shape used inside the MLXP PVC checkout.

    Single-GPU runs use `uv run python`; multi-GPU runs use `uv run torchrun`
    so G4/G5 training scripts can later inherit a stable launcher contract.
    """

    report = report_path or f"{DEFAULT_OUTPUT_DIR}/gpu_smoke_{gpu_count}.json"
    base_args = [script, "--expected-gpus", str(gpu_count), "--report", report]
    base_args.extend(script_args)
    if gpu_count == 1:
        command = ["uv", "run", "python", *base_args]
    else:
        command = ["uv", "run", "torchrun", "--standalone", "--nproc-per-node", str(gpu_count), *base_args]
    return LaunchCommand(gpu_count=gpu_count, repo_path=repo_path, command=command, env=launcher_env(gpu_count), report_path=report)


def build_gpu_smoke_matrix(
    gpu_counts: Iterable[int] = (1, 2, 4),
    *,
    repo_path: str = DEFAULT_MLXP_REPO_PATH,
    script: str = "scripts/cluster_gpu_smoke.py",
) -> list[LaunchCommand]:
    return [build_torchrun_command(gpu_count=int(count), script=script, repo_path=repo_path) for count in gpu_counts]


def write_gpu_smoke_matrix(
    path: str | Path,
    gpu_counts: Iterable[int] = (1, 2, 4),
    *,
    repo_path: str = DEFAULT_MLXP_REPO_PATH,
) -> dict[str, Any]:
    commands = build_gpu_smoke_matrix(gpu_counts, repo_path=repo_path)
    payload = {
        "schema": "cluster_gpu_smoke_matrix.v1",
        "repo_path": repo_path,
        "mode": "dry_run_launcher_contract",
        "commands": [cmd.to_json() for cmd in commands],
    }
    write_json(path, payload)
    return payload


def execute_launch_command(command: LaunchCommand, *, allow_cpu: bool = False, timeout_seconds: int = 300) -> dict[str, Any]:
    cmd = list(command.command)
    if allow_cpu and "--allow-cpu" not in cmd:
        cmd.append("--allow-cpu")
    result = subprocess.run(
        cmd,
        cwd=command.repo_path if Path(command.repo_path).exists() else None,
        env={**os.environ, **command.env},
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "gpu_count": command.gpu_count,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "report_path": command.report_path,
    }


def write_local_cpu_probe(path: str | Path, *, expected_gpus: int = 1) -> dict[str, Any]:
    ensure_dir(Path(path).parent)
    command = build_torchrun_command(gpu_count=expected_gpus, script="scripts/cluster_gpu_smoke.py", repo_path=str(Path.cwd()))
    result = execute_launch_command(command, allow_cpu=True)
    payload = {"schema": "cluster_local_cpu_probe.v1", "command": command.to_json(), "result": result}
    write_json(path, payload)
    return payload
