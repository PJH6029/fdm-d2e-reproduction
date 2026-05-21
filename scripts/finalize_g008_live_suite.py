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
from fdm_d2e.reporting.g008_completion import write_g008_live_suite_completion_audit
from fdm_d2e.rollout.live_suite import run_live_suite_validation


DEFAULT_SUMMARY_OUT = "artifacts/harness/g008_live_open_game_suite_finalization_summary.json"
DEFAULT_EVIDENCE_VALIDATION_OUT = "artifacts/harness/g008_live_open_game_suite_evidence_validation.json"


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


def _gate_status(report: dict[str, Any] | None) -> str | None:
    if not report:
        return None
    gate = report.get("quality_gate")
    if not isinstance(gate, dict):
        return None
    status = gate.get("status")
    return str(status) if status is not None else None


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    suite_config = load_config(_path(root, args.suite_config))
    completion_config = load_config(_path(root, args.g008_completion_config))
    evidence_validation_out = args.evidence_validation_output or dict(completion_config.get("paths", {})).get(
        "evidence_validation",
        DEFAULT_EVIDENCE_VALIDATION_OUT,
    )
    protocol_out = args.protocol_output or suite_config.get(
        "output_path",
        "artifacts/harness/g008_live_open_game_suite_protocol.json",
    )

    findings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    protocol_report = run_live_suite_validation(suite_config, None, root=root)
    write_json(_path(root, protocol_out), protocol_report)
    actions.append(
        {
            "name": "protocol_report",
            "output": _rel_or_abs(root, protocol_out),
            "status": _gate_status(protocol_report),
        }
    )

    evidence_report: dict[str, Any] | None = None
    evidence_path = args.evidence
    if not evidence_path:
        findings.append({"severity": "error", "code": "missing_live_evidence", "path": None})
        actions.append(
            {
                "name": "evidence_validation",
                "status": "skipped_missing_evidence",
                "output": _rel_or_abs(root, evidence_validation_out),
            }
        )
    else:
        evidence = _load_json(_path(root, evidence_path))
        if evidence is None:
            findings.append({"severity": "error", "code": "missing_live_evidence", "path": evidence_path})
            actions.append(
                {
                    "name": "evidence_validation",
                    "status": "missing_evidence_file",
                    "input": evidence_path,
                    "output": _rel_or_abs(root, evidence_validation_out),
                }
            )
        else:
            evidence_report = run_live_suite_validation(suite_config, evidence, root=root)
            write_json(_path(root, evidence_validation_out), evidence_report)
            gate_status = _gate_status(evidence_report)
            actions.append(
                {
                    "name": "evidence_validation",
                    "input": evidence_path,
                    "output": _rel_or_abs(root, evidence_validation_out),
                    "status": gate_status,
                    "findings_count": evidence_report.get("quality_gate", {}).get("findings_count"),
                }
            )
            if gate_status != "pass":
                findings.append(
                    {
                        "severity": "error",
                        "code": "live_evidence_validation_not_pass",
                        "status": gate_status,
                        "findings_count": evidence_report.get("quality_gate", {}).get("findings_count"),
                    }
                )

    audit = write_g008_live_suite_completion_audit(
        completion_config,
        root=root,
        output_path=args.g008_audit_output,
    )
    actions.append(
        {
            "name": "g008_completion_audit",
            "output": _rel_or_abs(root, args.g008_audit_output),
            "status": audit.get("status"),
            "error_count": audit.get("error_count"),
        }
    )
    if audit.get("status") != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "g008_completion_audit_not_pass",
                "error_count": audit.get("error_count"),
            }
        )

    status = "pass" if not findings else "fail"
    payload = {
        "schema": "g008_live_suite_finalization.v1",
        "status": status,
        "root": str(root),
        "protocol_output": _rel_or_abs(root, protocol_out),
        "protocol_status": _gate_status(protocol_report),
        "evidence_path": evidence_path,
        "evidence_validation_output": _rel_or_abs(root, evidence_validation_out),
        "evidence_validation_status": _gate_status(evidence_report),
        "g008_audit_status": audit.get("status"),
        "g008_audit_error_count": audit.get("error_count"),
        "actions": actions,
        "findings": findings,
        "claim_boundary": "Finalizes G008 live open-source graphical-game suite evidence only after an explicit live evidence JSON is validated; it does not checkpoint G008 and protocol-only evidence cannot pass.",
    }
    write_json(_path(root, args.summary_out), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize/audit G008 live open-source graphical-game suite evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--suite-config", default="configs/harness/g008_live_open_game_suite.yaml")
    parser.add_argument("--g008-completion-config", default="configs/eval/g008_live_suite_completion.yaml")
    parser.add_argument("--g008-audit-output", default="artifacts/harness/g008_live_suite_completion_audit.json")
    parser.add_argument("--evidence", help="Live suite evidence JSON to validate; required for a passing finalization.")
    parser.add_argument("--evidence-validation-output", help="Override completion-config evidence_validation path.")
    parser.add_argument("--protocol-output", help="Override protocol report output path.")
    args = parser.parse_args()
    payload = finalize(args)
    print(f"g008 finalization: status={payload['status']} findings={len(payload['findings'])} output={args.summary_out}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
