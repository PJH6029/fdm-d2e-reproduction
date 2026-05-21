#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.evaluation_readiness import write_g006_evaluation_readiness
from fdm_d2e.reporting.final_eval import build_g006_final_artifacts
from fdm_d2e.reporting.g006_completion import write_g006_completion_audit


DEFAULT_SUMMARY_OUT = "artifacts/eval/g006_finalization_summary.json"


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rel_or_abs(root: Path, value: str | Path) -> str:
    p = _path(root, value)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    findings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    if args.skip_build:
        build_summary = _load_json(_path(root, args.build_summary_out))
        actions.append(
            {
                "name": "g006_final_artifact_build",
                "status": (build_summary or {}).get("status"),
                "output": _rel_or_abs(root, args.build_summary_out),
                "reason": "skip_build",
                "exists": build_summary is not None,
            }
        )
        if build_summary is None:
            findings.append({"severity": "error", "code": "missing_build_summary", "path": args.build_summary_out})
    else:
        build_summary = build_g006_final_artifacts(load_config(_path(root, args.build_config)), root=root)
        write_json(_path(root, args.build_summary_out), build_summary)
        actions.append(
            {
                "name": "g006_final_artifact_build",
                "status": build_summary.get("status"),
                "output": _rel_or_abs(root, args.build_summary_out),
                "statuses": build_summary.get("statuses"),
            }
        )
    if build_summary is not None and build_summary.get("status") != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "g006_final_artifact_build_not_pass",
                "status": build_summary.get("status"),
                "statuses": build_summary.get("statuses"),
            }
        )

    readiness = write_g006_evaluation_readiness(
        load_config(_path(root, args.readiness_config)),
        root=root,
        output_path=args.readiness_output,
    )
    actions.append(
        {
            "name": "g006_evaluation_readiness",
            "output": _rel_or_abs(root, args.readiness_output),
            "status": readiness.get("status"),
            "error_count": readiness.get("error_count"),
        }
    )
    if readiness.get("status") != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "g006_evaluation_readiness_not_pass",
                "error_count": readiness.get("error_count"),
            }
        )

    audit = write_g006_completion_audit(
        load_config(_path(root, args.g006_completion_config)),
        root=root,
        output_path=args.g006_audit_output,
    )
    actions.append(
        {
            "name": "g006_completion_audit",
            "output": _rel_or_abs(root, args.g006_audit_output),
            "status": audit.get("status"),
            "error_count": audit.get("error_count"),
        }
    )
    if audit.get("status") != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "g006_completion_audit_not_pass",
                "error_count": audit.get("error_count"),
            }
        )

    status = "pass" if not findings else "fail"
    payload = {
        "schema": "g006_evaluation_finalization.v1",
        "status": status,
        "root": str(root),
        "build_summary_status": (build_summary or {}).get("status"),
        "build_summary_statuses": (build_summary or {}).get("statuses"),
        "readiness_status": readiness.get("status"),
        "readiness_error_count": readiness.get("error_count"),
        "g006_audit_status": audit.get("status"),
        "g006_audit_error_count": audit.get("error_count"),
        "actions": actions,
        "findings": findings,
        "claim_boundary": "Finalizes G006 endpoint statistics, failure analysis, claim taxonomy, readiness, and completion audits from completed G003/G004/G005 evidence; it does not checkpoint G006 or weaken prerequisite goals.",
    }
    write_json(_path(root, args.summary_out), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize/audit G006 evaluation, failure-analysis, and claim-taxonomy evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="Use the existing build summary instead of rebuilding endpoint/failure/taxonomy artifacts.")
    parser.add_argument("--build-config", default="configs/eval/g006_final_artifacts.yaml")
    parser.add_argument("--build-summary-out", default="artifacts/eval/g006_final_artifact_build_summary.json")
    parser.add_argument("--readiness-config", default="configs/eval/g006_evaluation_readiness.yaml")
    parser.add_argument("--readiness-output", default="artifacts/eval/g006_evaluation_readiness_audit.json")
    parser.add_argument("--g006-completion-config", default="configs/eval/g006_completion.yaml")
    parser.add_argument("--g006-audit-output", default="artifacts/eval/g006_completion_audit.json")
    args = parser.parse_args()
    payload = finalize(args)
    print(f"g006 finalization: status={payload['status']} findings={len(payload['findings'])} output={args.summary_out}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
