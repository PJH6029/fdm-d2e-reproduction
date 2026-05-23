#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.claim_audit import audit_claim_boundaries
from fdm_d2e.reporting.g009_completion import write_g009_completion_audit
from fdm_d2e.reporting.quality_gates import write_final_quality_gate_audit
from build_external_artifact_manifest import DEFAULT_SOURCE_ARTIFACTS, DEFAULT_STORAGE_ROOT, build_external_manifest
from build_repro_package_manifest import DEFAULT_PATTERNS, classify, iter_paths, sha256_file


DEFAULT_SUMMARY_OUT = "artifacts/reproducibility/g009_finalization_summary.json"


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _rel_or_abs(root: Path, value: str | Path) -> str:
    p = _path(root, value)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def _bootstrap_patterns(root: Path, patterns: list[str]) -> list[str]:
    """Drop explicit missing self-generated artifacts for the seed manifest.

    The first final-quality audit needs a package manifest to exist, but the
    final-quality and G009 audit files may be generated later in this same
    finalizer. Globs are already optional in build_repro_package_manifest.py, so
    this only filters explicit self-generated paths during the bootstrap pass.
    """

    generated = {
        "artifacts/reproducibility/final_quality_gate_audit.json",
        "artifacts/reproducibility/g009_completion_audit.json",
        "artifacts/reproducibility/g009_finalization_summary.json",
    }
    filtered = []
    for pattern in patterns:
        if pattern in generated and not _path(root, pattern).exists():
            continue
        filtered.append(pattern)
    return filtered


def _build_package_manifest(root: Path, *, output: str | Path, patterns: list[str]) -> dict[str, Any]:
    with _pushd(root):
        entries = [
            {"path": str(path), "kind": classify(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in iter_paths(patterns)
        ]
    payload = {
        "schema": "repro_package_manifest.v1",
        "generated_at_utc": "deterministic-manifest-no-wall-clock",
        "entry_count": len(entries),
        "entries": entries,
        "notes": [
            "D2E-derived artifacts are research/non-commercial and must follow upstream terms.",
            "This manifest supports a scaled reproduction report, not an FDM-1 parity claim.",
        ],
    }
    write_json(_path(root, output), payload)
    return payload


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    findings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    patterns = list(args.package_patterns or DEFAULT_PATTERNS)

    with _pushd(root):
        claim = audit_claim_boundaries(goals_path=args.goals_path)
    write_json(_path(root, args.claim_audit_output), claim)
    actions.append({"name": "claim_boundary_audit", "output": _rel_or_abs(root, args.claim_audit_output), "status": claim.get("status")})
    if claim.get("status") != "pass":
        findings.append({"severity": "error", "code": "claim_boundary_audit_not_pass", "status": claim.get("status")})

    final_quality_config = load_config(_path(root, args.final_quality_config))
    external_source_artifacts = getattr(args, "external_source_artifacts", DEFAULT_SOURCE_ARTIFACTS)
    external_storage_root = getattr(args, "external_storage_root", DEFAULT_STORAGE_ROOT)
    external_manifest_output = getattr(args, "external_manifest_output", "artifacts/reproducibility/external_artifact_manifest.json")
    with _pushd(root):
        external_manifest = build_external_manifest(
            config=final_quality_config,
            source_paths=[Path(source) for source in external_source_artifacts],
            storage_root=external_storage_root,
        )
    write_json(_path(root, external_manifest_output), external_manifest)
    actions.append(
        {
            "name": "external_artifact_manifest",
            "output": _rel_or_abs(root, external_manifest_output),
            "status": external_manifest.get("status"),
            "entry_count": external_manifest.get("entry_count"),
            "error_count": external_manifest.get("error_count"),
        }
    )
    if external_manifest.get("status") != "pass":
        findings.append({"severity": "error", "code": "external_artifact_manifest_not_pass", "error_count": external_manifest.get("error_count")})

    package: dict[str, Any] | None = None
    try:
        package = _build_package_manifest(root, output=args.package_manifest_output, patterns=_bootstrap_patterns(root, patterns))
        actions.append(
            {
                "name": "package_manifest_bootstrap",
                "output": _rel_or_abs(root, args.package_manifest_output),
                "entry_count": package.get("entry_count"),
            }
        )
    except FileNotFoundError as exc:
        findings.append({"severity": "error", "code": "package_manifest_build_failed", "error": str(exc)})
        actions.append({"name": "package_manifest_bootstrap", "status": "failed", "error": str(exc)})

    final_quality = write_final_quality_gate_audit(
        final_quality_config,
        root=root,
        output_path=args.final_quality_output,
    )
    actions.append(
        {
            "name": "final_quality_gate_audit",
            "output": _rel_or_abs(root, args.final_quality_output),
            "status": final_quality.get("status"),
            "error_count": final_quality.get("error_count"),
        }
    )
    if final_quality.get("status") != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "final_quality_gate_audit_not_pass",
                "error_count": final_quality.get("error_count"),
            }
        )

    g009 = write_g009_completion_audit(
        load_config(_path(root, args.g009_completion_config)),
        root=root,
        output_path=args.g009_audit_output,
    )
    actions.append(
        {
            "name": "g009_completion_audit_pre_final_manifest",
            "output": _rel_or_abs(root, args.g009_audit_output),
            "status": g009.get("status"),
            "error_count": g009.get("error_count"),
        }
    )

    final_package = None
    if package is not None:
        try:
            final_package = _build_package_manifest(root, output=args.package_manifest_output, patterns=patterns)
            actions.append(
                {
                    "name": "package_manifest_final",
                    "output": _rel_or_abs(root, args.package_manifest_output),
                    "entry_count": final_package.get("entry_count"),
                }
            )
        except FileNotFoundError as exc:
            findings.append({"severity": "error", "code": "final_package_manifest_build_failed", "error": str(exc)})
            actions.append({"name": "package_manifest_final", "status": "failed", "error": str(exc)})

    final_g009 = write_g009_completion_audit(
        load_config(_path(root, args.g009_completion_config)),
        root=root,
        output_path=args.g009_audit_output,
    )
    actions.append(
        {
            "name": "g009_completion_audit",
            "output": _rel_or_abs(root, args.g009_audit_output),
            "status": final_g009.get("status"),
            "error_count": final_g009.get("error_count"),
        }
    )
    if final_g009.get("status") != "pass":
        findings.append({"severity": "error", "code": "g009_completion_audit_not_pass", "error_count": final_g009.get("error_count")})

    status = "pass" if not [item for item in findings if item.get("severity") == "error"] else "fail"
    payload = {
        "schema": "g009_report_package_finalization.v1",
        "status": status,
        "root": str(root),
        "claim_audit_status": claim.get("status"),
        "final_quality_status": final_quality.get("status"),
        "final_quality_error_count": final_quality.get("error_count"),
        "package_manifest_entry_count": (final_package or package or {}).get("entry_count"),
        "g009_audit_status": final_g009.get("status"),
        "g009_audit_error_count": final_g009.get("error_count"),
        "actions": actions,
        "findings": findings,
        "claim_boundary": "Finalizes report/package evidence by refreshing claim audit, final quality gates, package manifest, and G009 completion audit; it does not checkpoint G009 or complete the aggregate goal.",
    }
    write_json(_path(root, args.summary_out), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize/audit G009 final report and reproducibility package evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--goals-path", default=".omx/ultragoal/goals.json")
    parser.add_argument("--claim-audit-output", default="artifacts/reproducibility/claim_boundary_audit.json")
    parser.add_argument("--final-quality-config", default="configs/eval/final_quality_gates.yaml")
    parser.add_argument("--final-quality-output", default="artifacts/reproducibility/final_quality_gate_audit.json")
    parser.add_argument("--external-manifest-output", default="artifacts/reproducibility/external_artifact_manifest.json")
    parser.add_argument("--external-storage-root", default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--external-source-artifacts", nargs="*", default=DEFAULT_SOURCE_ARTIFACTS)
    parser.add_argument("--package-manifest-output", default="artifacts/reproducibility/package_manifest.json")
    parser.add_argument("--package-patterns", nargs="*", help="Override package-manifest path/glob patterns; mainly for tests.")
    parser.add_argument("--g009-completion-config", default="configs/eval/g009_completion.yaml")
    parser.add_argument("--g009-audit-output", default="artifacts/reproducibility/g009_completion_audit.json")
    args = parser.parse_args()
    payload = finalize(args)
    print(f"g009 finalization: status={payload['status']} findings={len(payload['findings'])} output={args.summary_out}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
