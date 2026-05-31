#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import sha256_file, write_json


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(key: str, rel_path: str) -> str:
    suffix = Path(rel_path).suffix or ".artifact"
    return f"{key}{suffix}"


def build_bundle(
    completion_config: dict[str, Any],
    *,
    root: str | Path = ".",
    output_dir: str | Path = "artifacts/sources/fdm1_g003_evidence_bundle",
    manifest_out: str | Path = "artifacts/sources/fdm1_g003_evidence_bundle_manifest.json",
    require_pass: bool = True,
) -> dict[str, Any]:
    root_path = Path(root)
    bundle_dir = root_path / output_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)
    paths = {str(k): str(v) for k, v in dict(completion_config.get("paths", {})).items()}
    omit_keys = set(map(str, completion_config.get("omit_sha256_artifact_keys", [])))
    audit_path = root_path / str(completion_config.get("output_path", "artifacts/sources/fdm1_g003_action_dataset_completion_audit.json"))
    audit = _load_json(audit_path) if audit_path.exists() else None
    if require_pass and (audit is None or audit.get("status") != "pass"):
        raise RuntimeError(f"completion audit is not pass: {audit_path}")
    dataset_summary = _load_json(root_path / paths["dataset_summary"]) if "dataset_summary" in paths and (root_path / paths["dataset_summary"]).exists() else {}
    output_hashes = dataset_summary.get("output_hashes", {}) if isinstance(dataset_summary, dict) else {}
    copied: list[dict[str, Any]] = []
    large: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for key, rel in sorted(paths.items()):
        src = root_path / rel
        if not src.exists() or not src.is_file():
            missing.append({"key": key, "path": rel})
            continue
        entry = {"key": key, "path": rel, "bytes": src.stat().st_size}
        if key in omit_keys:
            role = None
            if key == "action_slots":
                role = "all"
            elif key.endswith("_slots"):
                role = key.removesuffix("_slots")
            entry.update({"copied": False, "output_hash_role": role, "sha256": output_hashes.get(role) if role else None})
            large.append(entry)
            continue
        dst = bundle_dir / _safe_name(key, rel)
        shutil.copy2(src, dst)
        entry.update({"copied": True, "bundle_path": str(dst.relative_to(root_path)), "sha256": sha256_file(dst)})
        copied.append(entry)
    if audit_path.exists():
        dst = bundle_dir / "completion_audit.json"
        shutil.copy2(audit_path, dst)
        copied.append({"key": "completion_audit", "path": str(audit_path.relative_to(root_path)), "copied": True, "bundle_path": str(dst.relative_to(root_path)), "bytes": dst.stat().st_size, "sha256": sha256_file(dst)})
    manifest = {
        "schema": "fdm1_g003_evidence_bundle_manifest.v1",
        "canonical_roadmap": "ROADMAP.md",
        "status": "pass" if not missing and (audit is None or audit.get("status") == "pass") else "fail",
        "completion_audit_status": audit.get("status") if audit else "missing",
        "bundle_dir": str(Path(output_dir)),
        "manifest_path": str(Path(manifest_out)),
        "copied_artifacts": copied,
        "large_artifacts_not_copied": large,
        "missing_artifacts": missing,
        "dataset_fingerprint": dataset_summary.get("dataset_fingerprint") if isinstance(dataset_summary, dict) else None,
        "claim_boundary": "This bundle stages small G003 action-dataset evidence only; large JSONL packs remain on PVC and are represented by write-time output hashes.",
    }
    write_json(root_path / manifest_out, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage small G003 action dataset evidence and manifest large slot-pack hashes.")
    parser.add_argument("--completion-config", default="configs/eval/fdm1_g003_action_dataset_completion.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/sources/fdm1_g003_evidence_bundle")
    parser.add_argument("--manifest-out", default="artifacts/sources/fdm1_g003_evidence_bundle_manifest.json")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    manifest = build_bundle(
        load_config(args.completion_config),
        root=args.root,
        output_dir=args.output_dir,
        manifest_out=args.manifest_out,
        require_pass=not args.allow_fail,
    )
    print(
        "built FDM-1 G003 evidence bundle: "
        f"status={manifest['status']} copied={len(manifest['copied_artifacts'])} large={len(manifest['large_artifacts_not_copied'])} missing={len(manifest['missing_artifacts'])}"
    )
    return 0 if manifest["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
