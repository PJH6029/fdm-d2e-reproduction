#!/usr/bin/env python3
"""Build an audited in-pod launch plan for G003 action-slot materialization.

The default mode is intentionally non-executing: it writes a JSON launch plan and
an executable shell script that can be copied into an MLXP pod after a reservation
exists.  ``--execute`` is guarded so it only runs from inside the Kubernetes pod
workspace, preventing accidental local launches or unapproved production side
effects from this helper.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_REPO_DIR = "/root/work/code/continuous-gui-poc/fdm-d2e-reproduction"
DEFAULT_BRANCH = "research/fdm1-d2e-ultragoal"
DEFAULT_PLAN_OUT = "artifacts/cluster/fdm1_g003_action_dataset_pod_launch_plan.json"
DEFAULT_SHELL_OUT = "artifacts/cluster/fdm1_g003_action_dataset_pod_launch.sh"
DEFAULT_LOG = "artifacts/logs/fdm1_g003_action_dataset_pipeline.log"
DEFAULT_PID = "outputs/cluster/fdm1_g003_action_dataset_pipeline.pid"
DEFAULT_LAUNCH_CONTEXT = "artifacts/cluster/fdm1_g003_action_dataset_pod_launch_context.json"
DEFAULT_PIPELINE = "bash scripts/run_g003_fdm1_action_dataset_pipeline.sh"
DEFAULT_PREFLIGHT_MIN_FREE_GB = 100.0


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _safe_is_dir(path: Path) -> tuple[bool, str | None]:
    try:
        return path.is_dir(), None
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def detect_runtime(repo_dir: str = DEFAULT_REPO_DIR) -> dict[str, Any]:
    repo = Path(repo_dir)
    repo_exists, repo_error = _safe_is_dir(repo)
    return {
        "hostname": platform.node(),
        "cwd": str(Path.cwd()),
        "repo_dir": repo_dir,
        "repo_dir_exists": repo_exists,
        "repo_dir_error": repo_error,
        "kubernetes_service_host": bool(os.environ.get("KUBERNETES_SERVICE_HOST")),
        "mlxp_reservation_id": os.environ.get("MLXP_RESERVATION_ID") or os.environ.get("RESERVATION_ID"),
        "inside_mlxp_pod": bool(os.environ.get("KUBERNETES_SERVICE_HOST")) and repo_exists,
    }


def _assignment_key(key: str) -> str:
    if not key:
        raise ValueError("environment assignment key must not be empty")
    if not (key[0].isalpha() or key[0] == "_"):
        raise ValueError(f"invalid environment key {key!r}")
    if not all(ch.isalnum() or ch == "_" for ch in key):
        raise ValueError(f"invalid environment key {key!r}")
    return key


def parse_env_assignments(items: list[str] | None) -> dict[str, str]:
    extra_env: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        extra_env[_assignment_key(key)] = value
    return extra_env


def build_launch_commands(
    *,
    repo_dir: str = DEFAULT_REPO_DIR,
    branch: str = DEFAULT_BRANCH,
    log_path: str = DEFAULT_LOG,
    pid_path: str = DEFAULT_PID,
    launch_context_path: str = DEFAULT_LAUNCH_CONTEXT,
    pipeline_command: str = DEFAULT_PIPELINE,
    sync: bool = True,
    pull: bool = True,
    background: bool = True,
    extra_env: dict[str, str] | None = None,
    replace_existing: bool = False,
    preflight_min_free_gb: float = DEFAULT_PREFLIGHT_MIN_FREE_GB,
) -> list[str]:
    env = dict(extra_env or {})
    log_dir = Path(log_path).parent
    pid_dir = Path(pid_path).parent
    launch_context_dir = Path(launch_context_path).parent
    commands = [
        "set -euo pipefail",
        "export PATH=\"$HOME/.local/bin:$PATH\"",
        "if [[ -z \"${KUBERNETES_SERVICE_HOST:-}\" ]]; then echo 'refusing: this launch script must run inside the MLXP Kubernetes pod' >&2; exit 2; fi",
        f"if [[ ! -d {q(repo_dir)} ]]; then echo 'refusing: repo dir not found: {q(repo_dir)}' >&2; exit 2; fi",
        f"cd {q(repo_dir)}",
    ]
    if pull:
        commands.extend(
            [
                f"git fetch origin {q(branch)}",
                f"git checkout {q(branch)}",
                "git pull --ff-only",
            ]
        )
    if sync:
        commands.append("uv sync --extra d2e --extra train --extra test")
    commands.append(
        f"uv run python scripts/preflight_g003_fdm1_action_dataset_pod.py --require-pod --expected-branch {q(branch)} --min-free-gb {preflight_min_free_gb:g}"
    )
    commands.extend(
        [
            f"mkdir -p {q(log_dir)} {q(pid_dir)} {q(launch_context_dir)} artifacts/sources artifacts/reports",
            "git rev-parse HEAD > artifacts/sources/fdm1_g003_pod_launch_commit.txt",
        ]
    )
    if replace_existing:
        commands.append(f"if [[ -s {q(pid_path)} ]] && kill -0 \"$(cat {q(pid_path)})\" 2>/dev/null; then echo 'stopping existing G003 launch pid' $(cat {q(pid_path)}); kill \"$(cat {q(pid_path)})\"; sleep 5; fi")
    else:
        commands.append(f"if [[ -s {q(pid_path)} ]] && kill -0 \"$(cat {q(pid_path)})\" 2>/dev/null; then echo 'refusing: existing G003 pipeline pid is still active:' $(cat {q(pid_path)}) >&2; exit 3; fi")
    env_prefix = " ".join(f"{_assignment_key(key)}={q(value)}" for key, value in sorted(env.items()))
    pipeline = pipeline_command
    if env_prefix:
        pipeline = f"{env_prefix} {pipeline}"
    if background:
        commands.append(f"nohup {pipeline} > {q(log_path)} 2>&1 & echo $! > {q(pid_path)}")
        commands.append(f"echo launched $(cat {q(pid_path)}) log={q(log_path)}")
    else:
        commands.append(f"{pipeline} 2>&1 | tee {q(log_path)}")
    commands.extend(
        [
            "uv run python - <<'PY'",
            "import json, os, subprocess, time",
            f"path = {str(launch_context_path)!r}",
            "data = {",
            "  'schema': 'fdm1_g003_pod_launch_context.v1',",
            "  'created_at_unix': time.time(),",
            "  'hostname': os.uname().nodename,",
            f"  'branch': {branch!r},",
            "  'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),",
            f"  'pid_path': {pid_path!r},",
            f"  'log_path': {log_path!r},",
            "  'pid': open(" + repr(pid_path) + ").read().strip() if os.path.exists(" + repr(pid_path) + ") else None,",
            "  'claim_boundary': 'Launch context only; G003 completion requires completion audit pass and OMX checkpoint.',",
            "}",
            "os.makedirs(os.path.dirname(path), exist_ok=True)",
            "with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)",
            "print(json.dumps(data, ensure_ascii=False, indent=2))",
            "PY",
        ]
    )
    return commands


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    extra_env = parse_env_assignments(args.env)
    commands = build_launch_commands(
        repo_dir=args.repo_dir,
        branch=args.branch,
        log_path=args.log_path,
        pid_path=args.pid_path,
        launch_context_path=args.launch_context_path,
        pipeline_command=args.pipeline_command,
        sync=not args.skip_sync,
        pull=not args.skip_pull,
        background=not args.foreground,
        extra_env=extra_env,
        replace_existing=args.replace_existing,
        preflight_min_free_gb=args.preflight_min_free_gb,
    )
    return {
        "schema": "fdm1_g003_pod_launch_plan.v1",
        "created_at": utc_now_iso(),
        "repo_dir": args.repo_dir,
        "branch": args.branch,
        "log_path": args.log_path,
        "pid_path": args.pid_path,
        "launch_context_path": args.launch_context_path,
        "pipeline_command": args.pipeline_command,
        "background": not args.foreground,
        "sync": not args.skip_sync,
        "pull": not args.skip_pull,
        "replace_existing": args.replace_existing,
        "preflight_min_free_gb": args.preflight_min_free_gb,
        "extra_env": extra_env,
        "runtime_detection": detect_runtime(args.repo_dir),
        "commands": commands,
        "post_launch_checks": [
            f"tail -n 80 {args.log_path}",
            f"cat {args.pid_path}",
            "uv run python scripts/monitor_g003_fdm1_action_dataset_pod.py --refresh-audit --build-bundle-if-pass",
            "uv run python scripts/validate_fdm1_g003_action_dataset_completion.py --config configs/eval/fdm1_g003_action_dataset_completion.yaml --allow-fail",
            "uv run python scripts/build_fdm1_g003_evidence_bundle.py --completion-config configs/eval/fdm1_g003_action_dataset_completion.yaml",
        ],
        "claim_boundary": "Launches the G003 action-slot materialization pipeline only; completion still requires the audit/evidence bundle to pass and be checkpointed.",
    }


def write_plan(plan: dict[str, Any], *, plan_out: str | Path, shell_out: str | Path) -> None:
    plan_path = Path(plan_out)
    shell_path = Path(shell_out)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    shell_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shell_path.write_text("#!/usr/bin/env bash\n" + "\n".join(plan["commands"]) + "\n", encoding="utf-8")
    shell_path.chmod(0o755)


def execute_shell(shell_out: str | Path) -> int:
    return subprocess.call(["bash", str(shell_out)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or execute the in-pod launch plan for G003 FDM-1 action dataset materialization.")
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--plan-out", default=DEFAULT_PLAN_OUT)
    parser.add_argument("--shell-out", default=DEFAULT_SHELL_OUT)
    parser.add_argument("--log-path", default=DEFAULT_LOG)
    parser.add_argument("--pid-path", default=DEFAULT_PID)
    parser.add_argument("--launch-context-path", default=DEFAULT_LAUNCH_CONTEXT)
    parser.add_argument("--pipeline-command", default=DEFAULT_PIPELINE)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--replace-existing", action="store_true", help="Stop an active pid from the same pid file before relaunching. Default refuses duplicates.")
    parser.add_argument("--preflight-min-free-gb", type=float, default=DEFAULT_PREFLIGHT_MIN_FREE_GB)
    parser.add_argument("--env", action="append", help="Extra environment assignment, e.g. EXTRACT_EXTRA_ARGS=--max-recordings 2")
    parser.add_argument("--execute", action="store_true", help="Execute the generated shell script. Refuses unless this process is inside the MLXP pod workspace.")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
    except ValueError as exc:
        print(f"error: {exc}", flush=True)
        return 2
    write_plan(plan, plan_out=args.plan_out, shell_out=args.shell_out)
    response = {"plan_out": args.plan_out, "shell_out": args.shell_out, "execute": args.execute, "background": plan["background"], "inside_mlxp_pod": plan["runtime_detection"]["inside_mlxp_pod"]}
    print(json.dumps(response, ensure_ascii=False, indent=2))
    if args.execute:
        if not plan["runtime_detection"]["inside_mlxp_pod"]:
            print("refusing --execute outside the MLXP pod workspace; rerun inside the reserved pod", flush=True)
            return 2
        return execute_shell(args.shell_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
