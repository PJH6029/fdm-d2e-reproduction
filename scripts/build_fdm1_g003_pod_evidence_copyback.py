#!/usr/bin/env python3
"""Generate a safe copyback plan for small G003 pod evidence artifacts.

The plan excludes large action-slot JSONL packs by reading the completion config's
`omit_sha256_artifact_keys`.  It emits a tar-over-kubectl command that preserves
repo-relative paths and leaves large PVC outputs in place, represented by the
write-time hashes in `dataset_summary`/evidence bundle.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json

DEFAULT_COMPLETION_CONFIG = "configs/eval/fdm1_g003_action_dataset_completion.yaml"
DEFAULT_OUTPUT = "artifacts/cluster/fdm1_g003_pod_evidence_copyback_plan.json"
DEFAULT_SHELL_OUT = "artifacts/cluster/fdm1_g003_pod_evidence_copyback.sh"
DEFAULT_REPO_DIR = "/root/work/code/continuous-gui-poc/fdm-d2e-reproduction"
DEFAULT_NAMESPACE = "p-production"
DEFAULT_LOCAL_ROOT = "."
DEFAULT_EXTRA_PATHS = [
    "artifacts/cluster/fdm1_g003_action_dataset_preflight.json",
    "artifacts/cluster/fdm1_g003_action_dataset_pod_launch_context.json",
    "artifacts/cluster/fdm1_g003_action_dataset_pod_monitor.json",
    "artifacts/cluster/fdm1_g003_checkpoint_handoff.json",
    "artifacts/sources/fdm1_g003_evidence_bundle_manifest.json",
    "artifacts/sources/fdm1_g003_evidence_bundle",
    "artifacts/sources/fdm1_g003_action_dataset_finalization_summary.json",
    "artifacts/logs/fdm1_g003_action_dataset_pipeline.log",
    "artifacts/reports/fdm1_g003_action_alignment_visual_check.md",
]


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = str(value).strip().lstrip("/")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def classify_completion_paths(config: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    paths = {str(k): str(v) for k, v in dict(config.get("paths", {})).items()}
    omit = set(map(str, config.get("omit_sha256_artifact_keys", [])))
    small: list[str] = []
    large: list[dict[str, Any]] = []
    for key, rel in sorted(paths.items()):
        if key in omit:
            role = "all" if key == "action_slots" else key.removesuffix("_slots") if key.endswith("_slots") else None
            large.append({"key": key, "path": rel, "output_hash_role": role})
        else:
            small.append(rel)
    audit = config.get("output_path")
    if audit:
        small.append(str(audit))
    return _dedupe_keep_order(small), large


def build_plan(
    completion_config: dict[str, Any],
    *,
    namespace: str = DEFAULT_NAMESPACE,
    pod: str,
    remote_repo_dir: str = DEFAULT_REPO_DIR,
    local_root: str = DEFAULT_LOCAL_ROOT,
    kubeconfig: str | None = None,
    extra_paths: list[str] | None = None,
) -> dict[str, Any]:
    small, large = classify_completion_paths(completion_config)
    copy_paths = _dedupe_keep_order(small + list(extra_paths or []))
    kubectl = ["kubectl"]
    if kubeconfig:
        kubectl.extend(["--kubeconfig", kubeconfig])
    kubectl.extend(["-n", namespace, "exec", pod, "--", "tar", "-C", remote_repo_dir, "-cf", "-"])
    kubectl.extend(copy_paths)
    shell_command = " ".join(q(part) for part in kubectl) + " | " + f"tar -C {q(local_root)} -xf -"
    return {
        "schema": "fdm1_g003_pod_evidence_copyback_plan.v1",
        "canonical_roadmap": "ROADMAP.md",
        "namespace": namespace,
        "pod": pod,
        "remote_repo_dir": remote_repo_dir,
        "local_root": local_root,
        "kubeconfig": kubeconfig,
        "copy_paths": copy_paths,
        "large_artifacts_not_copied": large,
        "shell_command": shell_command,
        "post_copy_checks": [
            "uv run python scripts/monitor_g003_fdm1_action_dataset_pod.py --output artifacts/cluster/fdm1_g003_action_dataset_pod_monitor.local.json",
            "uv run python scripts/build_fdm1_g003_checkpoint_handoff.py --allow-blocked",
            "uv run python -m json.tool artifacts/sources/fdm1_g003_action_dataset_completion_audit.json",
        ],
        "claim_boundary": "Copies small G003 evidence only. Large JSONL action-slot packs remain on the MLXP PVC and must be represented by output hashes, not committed raw.",
    }


def write_shell(plan: dict[str, Any], shell_out: str | Path) -> None:
    path = Path(shell_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "#!/usr/bin/env bash\nset -euo pipefail\n" + plan["shell_command"] + "\n"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate or run a copyback plan for small G003 pod evidence artifacts.")
    parser.add_argument("--completion-config", default=DEFAULT_COMPLETION_CONFIG)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--remote-repo-dir", default=DEFAULT_REPO_DIR)
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--kubeconfig")
    parser.add_argument("--extra-path", action="append", default=[])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--shell-out", default=DEFAULT_SHELL_OUT)
    parser.add_argument("--execute", action="store_true", help="Execute the generated copyback shell locally after writing it.")
    args = parser.parse_args(argv)
    plan = build_plan(
        load_config(args.completion_config),
        namespace=args.namespace,
        pod=args.pod,
        remote_repo_dir=args.remote_repo_dir,
        local_root=args.local_root,
        kubeconfig=args.kubeconfig,
        extra_paths=DEFAULT_EXTRA_PATHS + list(args.extra_path or []),
    )
    write_json(args.output, plan)
    write_shell(plan, args.shell_out)
    print(json.dumps({"status": "planned", "output": args.output, "shell_out": args.shell_out, "copy_paths": len(plan["copy_paths"]), "large_artifacts_not_copied": len(plan["large_artifacts_not_copied"])}, ensure_ascii=False, indent=2))
    if args.execute:
        return subprocess.call(["bash", str(args.shell_out)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
