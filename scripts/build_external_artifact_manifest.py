#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json


DEFAULT_SOURCE_ARTIFACTS = [
    "artifacts/idm/g003_full_idm_completion_audit.json",
    "artifacts/fdm/g004_full_fdm_completion_audit.json",
    "artifacts/idm/idm_streaming_d2e_full_compact_fdm_train_core_pseudolabels_summary.json",
    "artifacts/fdm/g004_safe_restart_live_snapshot.json",
]
DEFAULT_STORAGE_ROOT = (
    "mlxp-pvc://p-production/prod-rsv-jeonghunpark-20260521-76e25a"
    "/root/work/code/continuous-gui-poc/fdm-d2e-reproduction"
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _external_paths_from_config(config: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for gate in config.get("goal_gates", []):
        for value in gate.get("external_artifact_paths", []):
            path = str(value)
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _walk_objects(data: Any) -> Iterator[dict[str, Any]]:
    if isinstance(data, dict):
        yield data
        for value in data.values():
            yield from _walk_objects(value)
    elif isinstance(data, list):
        for item in data:
            yield from _walk_objects(item)


def _candidate_strength(candidate: dict[str, Any]) -> tuple[int, int, int]:
    has_sha = bool(candidate.get("sha256"))
    has_fingerprint = bool(candidate.get("fingerprint"))
    return (1 if has_sha else 0, 1 if has_fingerprint else 0, int(candidate.get("bytes") or 0))


def _collect_candidates(source_paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    for source_path in source_paths:
        payload = _load_json(source_path)
        if payload is None:
            continue
        source_rel = str(source_path)
        for obj in _walk_objects(payload):
            path = obj.get("path")
            if path:
                candidate = {
                    "path": str(path),
                    "exists": bool(obj.get("exists", True)),
                    "bytes": int(obj.get("bytes") or 0),
                    "sha256": obj.get("sha256"),
                    "source_artifact": source_rel,
                }
                candidates.setdefault(str(path), []).append(candidate)
        pseudo_label_path = payload.get("pseudo_label_path")
        prediction_fingerprint = payload.get("prediction_fingerprint")
        if pseudo_label_path and prediction_fingerprint:
            candidates.setdefault(str(pseudo_label_path), []).append(
                {
                    "path": str(pseudo_label_path),
                    "exists": True,
                    "bytes": 0,
                    "fingerprint": str(prediction_fingerprint),
                    "fingerprint_type": "streaming_idm_prediction_fingerprint",
                    "records": payload.get("records"),
                    "source_artifact": source_rel,
                }
            )
    return candidates


def _merge_candidate(path: str, candidates: list[dict[str, Any]], *, storage_root: str) -> dict[str, Any]:
    best = max(candidates, key=_candidate_strength)
    # Preserve a fingerprint from any same-path candidate when the strongest
    # evidence is a size-only artifact snapshot.
    fingerprint = best.get("fingerprint")
    fingerprint_type = best.get("fingerprint_type")
    if not fingerprint:
        for candidate in candidates:
            if candidate.get("fingerprint"):
                fingerprint = candidate.get("fingerprint")
                fingerprint_type = candidate.get("fingerprint_type")
                break
    bytes_value = int(best.get("bytes") or 0)
    if bytes_value <= 0:
        bytes_value = max(int(candidate.get("bytes") or 0) for candidate in candidates)
    source_artifacts = sorted({str(candidate.get("source_artifact")) for candidate in candidates if candidate.get("source_artifact")})
    return {
        "path": path,
        "exists": bool(best.get("exists", True)),
        "bytes": bytes_value,
        "sha256": best.get("sha256"),
        "fingerprint": fingerprint,
        "fingerprint_type": fingerprint_type,
        "storage_uri": f"{storage_root.rstrip('/')}/{path}",
        "source_artifact": source_artifacts[0] if source_artifacts else None,
        "source_artifacts": source_artifacts,
        "proof": "sha256" if best.get("sha256") else ("fingerprint" if fingerprint else "size_only"),
    }


def build_external_manifest(
    *,
    config: dict[str, Any],
    source_paths: list[Path],
    storage_root: str,
) -> dict[str, Any]:
    requested = _external_paths_from_config(config)
    candidates = _collect_candidates(source_paths)
    entries: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for path in requested:
        path_candidates = candidates.get(path, [])
        if not path_candidates:
            findings.append({"severity": "error", "code": "missing_external_artifact_candidate", "path": path})
            continue
        entry = _merge_candidate(path, path_candidates, storage_root=storage_root)
        if entry["bytes"] <= 0 or not (entry.get("sha256") or entry.get("fingerprint")):
            findings.append(
                {
                    "severity": "error",
                    "code": "weak_external_artifact_candidate",
                    "path": path,
                    "bytes": entry["bytes"],
                    "has_sha256": bool(entry.get("sha256")),
                    "has_fingerprint": bool(entry.get("fingerprint")),
                }
            )
        entries.append(entry)
    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "external_artifact_manifest.v1",
        "status": "pass" if not errors else "fail",
        "storage_root": storage_root,
        "entry_count": len(entries),
        "entries": entries,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": (
            "Large full-corpus JSONL artifacts are retained on the MLXP PVC; this manifest "
            "records path/size/hash-or-fingerprint evidence instead of placing multi-GB/TB "
            "files in git."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build external-artifact manifest for PVC-resident full-corpus evidence.")
    parser.add_argument("--config", default="configs/eval/final_quality_gates.yaml")
    parser.add_argument("--output", default="artifacts/reproducibility/external_artifact_manifest.json")
    parser.add_argument("--storage-root", default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--source-artifacts", nargs="*", default=DEFAULT_SOURCE_ARTIFACTS)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = build_external_manifest(
        config=config,
        source_paths=[Path(path) for path in args.source_artifacts],
        storage_root=args.storage_root,
    )
    write_json(Path(args.output), payload)
    print(f"external artifact manifest: status={payload['status']} entries={payload['entry_count']} output={args.output}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
