#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import sha256_file, write_json
from fdm_d2e.reporting.g003_completion import write_g003_full_idm_completion_audit


DEFAULT_OUTPUT = "artifacts/fdm/g004_launch_readiness.json"


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _file_status(root: Path, rel_path: str | None) -> dict[str, Any]:
    if not rel_path:
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    path = _path(root, rel_path)
    if not path.exists() or not path.is_file():
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    return {"path": rel_path, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _goal_status(root: Path, goals_path: str, goal_id: str) -> str:
    payload = _load_json(_path(root, goals_path)) or {}
    for goal in payload.get("goals", []) or []:
        if str(goal.get("id")) == goal_id:
            return str(goal.get("status"))
    return "missing"


def _refresh_or_load_g003_audit(args: argparse.Namespace, root: Path) -> dict[str, Any] | None:
    if args.skip_refresh_g003_audit:
        return _load_json(_path(root, args.g003_audit))
    config = load_config(_path(root, args.g003_completion_config))
    return write_g003_full_idm_completion_audit(config, root=root, output_path=args.g003_audit)


def _gpu_report(expected_gpus: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return {"checked": True, "available": False, "reason": "nvidia-smi_not_found", "count": 0, "expected": expected_gpus}
    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("GPU ")]
    return {
        "checked": True,
        "available": proc.returncode == 0 and len(lines) >= expected_gpus,
        "returncode": proc.returncode,
        "count": len(lines),
        "expected": expected_gpus,
        "stderr": proc.stderr.strip(),
    }


def plan_launch(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    findings: list[dict[str, Any]] = []

    g003_audit = _refresh_or_load_g003_audit(args, root)
    if g003_audit is None:
        findings.append({"severity": "error", "code": "missing_g003_audit", "path": args.g003_audit})
    elif g003_audit.get("status") != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "g003_audit_not_pass",
                "status": g003_audit.get("status"),
                "error_count": g003_audit.get("error_count"),
            }
        )

    g003_goal_status = _goal_status(root, args.goals_path, args.g003_goal_id)
    if not args.allow_precheckpoint and g003_goal_status != "complete":
        findings.append(
            {
                "severity": "error",
                "code": "g003_goal_not_checkpointed_complete",
                "goal_id": args.g003_goal_id,
                "actual": g003_goal_status,
            }
        )

    fdm_config_status = _file_status(root, args.fdm_config)
    predict_config_status = _file_status(root, args.idm_predict_config)
    run_script_status = _file_status(root, args.g004_run_script)
    if not fdm_config_status["exists"]:
        findings.append({"severity": "error", "code": "missing_fdm_config", "path": args.fdm_config})
        fdm_config: dict[str, Any] = {}
    else:
        fdm_config = load_config(_path(root, args.fdm_config))
    if not run_script_status["exists"]:
        findings.append({"severity": "error", "code": "missing_g004_run_script", "path": args.g004_run_script})

    required_inputs = {
        "source_idm_metadata": str(fdm_config.get("source_idm_metadata", "")),
        "records_path": str(fdm_config.get("records_path", "")),
        "target_records_path": str(fdm_config.get("target_records_path", "")),
        "data_universe": str(fdm_config.get("data_universe", "")),
        "split_contract": str(fdm_config.get("split_contract", "")),
    }
    input_artifacts = {name: _file_status(root, rel_path) for name, rel_path in required_inputs.items() if rel_path}
    for name, artifact in input_artifacts.items():
        if not artifact["exists"]:
            findings.append({"severity": "error", "code": "missing_required_g004_input", "input": name, "path": artifact["path"]})

    labels_path = str(fdm_config.get("labels_path", args.fdm_labels))
    label_artifact = _file_status(root, labels_path)
    pseudolabel_mode = "reuse_existing_labels" if label_artifact["exists"] else "generate_with_trained_g003_idm"
    if not label_artifact["exists"] and not predict_config_status["exists"]:
        findings.append(
            {
                "severity": "error",
                "code": "missing_pseudolabels_and_predict_config",
                "labels_path": labels_path,
                "idm_predict_config": args.idm_predict_config,
            }
        )

    run_summary = _load_json(_path(root, args.g004_run_summary))
    if run_summary and run_summary.get("exit_code") == 0:
        findings.append({"severity": "warning", "code": "existing_successful_g004_run_summary", "path": args.g004_run_summary})
    elif run_summary and run_summary.get("exit_code") not in {None, 0}:
        findings.append({"severity": "warning", "code": "existing_failed_g004_run_summary", "path": args.g004_run_summary, "exit_code": run_summary.get("exit_code")})

    gpu = _gpu_report(int(args.expected_gpus)) if args.check_gpus else {"checked": False, "expected": int(args.expected_gpus)}
    if args.check_gpus and not gpu.get("available"):
        findings.append({"severity": "error", "code": "insufficient_visible_gpus", "gpu_report": gpu})

    errors = [item for item in findings if item.get("severity") == "error"]
    command = (
        f"CONFIG={args.fdm_config} "
        f"IDM_PREDICT_CONFIG={args.idm_predict_config} "
        f"NPROC_PER_NODE={args.nproc_per_node} "
        f"EXPECTED_GPUS={args.expected_gpus} "
        f"bash {args.g004_run_script}"
    )
    payload = {
        "schema": "g004_launch_readiness.v1",
        "status": "ready" if not errors else "blocked",
        "root": str(root),
        "recommended_command": command,
        "claim_boundary": "This planner only proves whether launching G004 is safe; it does not launch training, checkpoint G003/G004, or prove G004 completion.",
        "g003_goal_status": g003_goal_status,
        "g003_audit_status": (g003_audit or {}).get("status"),
        "g003_audit_error_count": (g003_audit or {}).get("error_count"),
        "allow_precheckpoint": bool(args.allow_precheckpoint),
        "pseudolabel_mode": pseudolabel_mode,
        "artifacts": {
            "fdm_config": fdm_config_status,
            "idm_predict_config": predict_config_status,
            "g004_run_script": run_script_status,
            "labels_path": label_artifact,
            "required_inputs": input_artifacts,
            "g004_run_summary": _file_status(root, args.g004_run_summary),
        },
        "gpu": gpu,
        "findings": findings,
        "error_count": len(errors),
    }
    write_json(_path(root, args.output), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan a fail-closed launch of the G004 D2E-only FDM 4xH200 run.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--goals-path", default=".omx/ultragoal/goals.json")
    parser.add_argument("--g003-goal-id", default="G003-d2e-only-idm")
    parser.add_argument("--g003-completion-config", default="configs/eval/g003_full_idm_completion.yaml")
    parser.add_argument("--g003-audit", default="artifacts/idm/g003_full_idm_completion_audit.json")
    parser.add_argument("--skip-refresh-g003-audit", action="store_true", help="Use the existing G003 audit instead of refreshing it.")
    parser.add_argument("--allow-precheckpoint", action="store_true", help="Allow planning before the G003 OMX checkpoint is complete; never use for terminal handoff.")
    parser.add_argument("--fdm-config", default="configs/model/fdm_streaming_d2e_full_compact.yaml")
    parser.add_argument("--idm-predict-config", default="configs/model/idm_streaming_d2e_full_compact_predict_fdm_train.yaml")
    parser.add_argument("--fdm-labels", default="outputs/idm_streaming_d2e_full_compact/fdm_train_core_pseudolabels/pseudolabels.jsonl")
    parser.add_argument("--g004-run-script", default="scripts/run_g004_d2e_full_fdm_4xh200.sh")
    parser.add_argument("--g004-run-summary", default="artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--check-gpus", action="store_true", help="Require currently visible GPUs before reporting ready.")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = plan_launch(args)
    print(f"g004 launch readiness: status={payload['status']} errors={payload['error_count']} output={args.output}")
    return 0 if payload["status"] == "ready" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
