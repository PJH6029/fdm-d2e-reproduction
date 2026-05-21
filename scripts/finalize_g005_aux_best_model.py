#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.g005_completion import write_g005_aux_completion_audit
from build_g005_aux_namespace_manifest import build_manifest


DEFAULT_SUMMARY_OUT = "artifacts/aux/g005_aux_finalization_summary.json"


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


def _build_namespace_if_requested(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    output = _path(root, args.namespace_manifest_output)
    if args.skip_namespace_build:
        payload = _load_json(output)
        return {"status": "skipped", "reason": "skip_namespace_build", "path": str(output), "exists": payload is not None, "payload": payload}
    if output.exists() and not args.force_namespace:
        payload = _load_json(output)
        return {"status": "existing", "reason": "existing_manifest", "path": str(output), "exists": payload is not None, "completion_ready": (payload or {}).get("completion_ready"), "payload": payload}
    if not args.source_evidence and not args.allow_template_namespace:
        return {"status": "blocked_missing_source_evidence", "path": str(output), "exists": False}
    if not args.eval_manifest_hashes and not args.allow_template_namespace:
        return {"status": "blocked_missing_eval_manifest_hashes", "path": str(output), "exists": False}
    try:
        payload = build_manifest(
            aux_candidates_path=str(_path(root, args.aux_candidates)),
            source_evidence_paths=[str(_path(root, item)) for item in args.source_evidence],
            eval_manifest_hashes_path=str(_path(root, args.eval_manifest_hashes)) if args.eval_manifest_hashes else None,
            completion_ready_requested=bool(args.completion_ready),
            allow_template=bool(args.allow_template_namespace),
        )
    except SystemExit as exc:
        return {"status": "fail", "path": str(output), "error": str(exc)}
    write_json(output, payload)
    return {"status": "built", "path": str(output), "exists": True, "completion_ready": payload.get("completion_ready"), "payload": payload}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    findings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    namespace = _build_namespace_if_requested(args, root)
    actions.append({"name": "namespace_manifest", **{k: v for k, v in namespace.items() if k != "payload"}})
    if namespace.get("status") in {"blocked_missing_source_evidence", "blocked_missing_eval_manifest_hashes", "fail"}:
        findings.append({"severity": "error", "code": "namespace_manifest_not_ready", "status": namespace.get("status"), "error": namespace.get("error")})
    elif namespace.get("payload") is not None and namespace.get("completion_ready") is not True:
        findings.append({"severity": "error", "code": "namespace_manifest_not_completion_ready", "completion_ready": namespace.get("completion_ready")})

    run_summary_path = _path(root, args.run_summary)
    run_summary = _load_json(run_summary_path)
    if run_summary is None:
        findings.append({"severity": "error", "code": "missing_run_summary", "path": args.run_summary})
    elif run_summary.get("exit_code") != 0:
        findings.append({"severity": "error", "code": "run_summary_exit_nonzero", "actual": run_summary.get("exit_code")})
    actions.append({"name": "run_summary", "path": args.run_summary, "exists": run_summary is not None, "exit_code": (run_summary or {}).get("exit_code")})

    audit = write_g005_aux_completion_audit(load_config(_path(root, args.g005_completion_config)), root=root, output_path=args.g005_audit_output)
    actions.append({"name": "g005_completion_audit", "output": args.g005_audit_output, "status": audit.get("status"), "error_count": audit.get("error_count")})
    if audit.get("status") != "pass":
        findings.append({"severity": "error", "code": "g005_completion_audit_not_pass", "error_count": audit.get("error_count")})

    status = "pass" if not findings else "fail"
    payload = {
        "schema": "g005_aux_finalization.v1",
        "status": status,
        "root": str(root),
        "actions": actions,
        "namespace_manifest_path": _rel_or_abs(root, args.namespace_manifest_output),
        "namespace_manifest_status": namespace.get("status"),
        "namespace_completion_ready": namespace.get("completion_ready"),
        "run_summary": run_summary,
        "g005_audit_status": audit.get("status"),
        "g005_audit_error_count": audit.get("error_count"),
        "findings": findings,
        "claim_boundary": "Finalizes artifacts for the G005 D2E+aux best-model lane after D2E-only gates and aux training/ablation evidence exist; it does not checkpoint G005 or weaken D2E-only prerequisites.",
    }
    write_json(_path(root, args.summary_out), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize/audit artifacts for G005 D2E+aux best-model evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--g005-completion-config", default="configs/eval/g005_aux_completion.yaml")
    parser.add_argument("--g005-audit-output", default="artifacts/aux/g005_aux_completion_audit.json")
    parser.add_argument("--run-summary", default="artifacts/aux/g005_d2e_aux_train_run.json")
    parser.add_argument("--namespace-manifest-output", default="artifacts/aux/g005_aux_namespace_manifest.json")
    parser.add_argument("--aux-candidates", default="artifacts/sources/aux_game_action_dataset_candidates.json")
    parser.add_argument("--source-evidence", action="append", default=[], help="JSON source materialization evidence; repeatable.")
    parser.add_argument("--eval-manifest-hashes", help="JSON D2E eval-manifest hash evidence.")
    parser.add_argument("--completion-ready", action="store_true", help="Request completion_ready=true when building namespace manifest; builder remains fail-closed.")
    parser.add_argument("--allow-template-namespace", action="store_true", help="Allow a non-terminal template namespace manifest.")
    parser.add_argument("--skip-namespace-build", action="store_true")
    parser.add_argument("--force-namespace", action="store_true")
    args = parser.parse_args()
    payload = finalize(args)
    print(f"g005 finalization: status={payload['status']} findings={len(payload['findings'])} output={args.summary_out}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
