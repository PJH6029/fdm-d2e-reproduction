#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json


DEFAULT_OUTPUT = "artifacts/aux/g005_aux_source_materialization_evidence.json"
DEFAULT_AUX_CANDIDATES = "artifacts/sources/aux_game_action_dataset_candidates.json"
DEFAULT_NAMESPACE_ROOT = "outputs/aux"
DEFAULT_REQUIRED_SPLITS = ("train", "val", "test")


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _selected_candidates(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in payload.get("candidates", [])
        if isinstance(row, dict) and row.get("id") and row.get("selection_status") == "selected_candidate"
    }


def _action_head_for_candidate(source_id: str, candidate: dict[str, Any]) -> dict[str, str]:
    domain = str(candidate.get("domain", "")).lower()
    if "minecraft" in domain:
        head_type = "minecraft_keyboard_mouse"
    elif "atari" in domain:
        head_type = "atari_discrete"
    else:
        head_type = "source_specific"
    return {"type": head_type, "namespace": source_id}


def _iter_files(namespace: Path) -> list[Path]:
    if not namespace.exists() or not namespace.is_dir():
        return []
    return sorted(path for path in namespace.rglob("*") if path.is_file())


def _split_paths(namespace: Path, split: str) -> list[Path]:
    candidates = [namespace / split, namespace / f"{split}.json", namespace / f"{split}.jsonl", namespace / f"{split}.txt"]
    paths: list[Path] = []
    for candidate in candidates:
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(_iter_files(candidate))
    return sorted(dict.fromkeys(paths))


def _aggregate_hash(namespace: Path, paths: list[Path]) -> str | None:
    if not paths:
        return None
    rows = []
    for path in paths:
        rel = str(path.relative_to(namespace))
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return _stable_sha256(rows)


def _source_file_rows(namespace: Path, *, max_files: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for idx, path in enumerate(_iter_files(namespace)):
        if max_files is not None and idx >= max_files:
            break
        rows.append({"path": str(path.relative_to(namespace)), "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return rows


def _build_source_row(
    *,
    source_id: str,
    candidate: dict[str, Any],
    namespace_root: Path,
    required_splits: tuple[str, ...],
    max_files: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    namespace = namespace_root / source_id
    findings: list[dict[str, Any]] = []
    files = _source_file_rows(namespace, max_files=max_files)
    all_files = _iter_files(namespace)
    materialized = namespace.exists() and namespace.is_dir() and bool(all_files)
    if not namespace.exists():
        findings.append({"severity": "error", "code": "aux_namespace_missing", "source_id": source_id, "namespace": str(namespace)})
    elif not namespace.is_dir():
        findings.append({"severity": "error", "code": "aux_namespace_not_directory", "source_id": source_id, "namespace": str(namespace)})
    elif not all_files:
        findings.append({"severity": "error", "code": "aux_namespace_empty", "source_id": source_id, "namespace": str(namespace)})

    split_hashes: dict[str, dict[str, Any]] = {}
    for split in required_splits:
        paths = _split_paths(namespace, split)
        split_hashes[split] = {
            "split": split,
            "file_count": len(paths),
            "bytes": sum(path.stat().st_size for path in paths),
            "sha256": _aggregate_hash(namespace, paths),
            "paths": [str(path.relative_to(namespace)) for path in paths[:50]],
            "truncated_paths": len(paths) > 50,
        }
        if materialized and not paths:
            findings.append({"severity": "error", "code": "missing_aux_split_files", "source_id": source_id, "split": split})

    provenance_payload = {
        "candidate": candidate,
        "namespace": str(namespace),
        "total_files": len(all_files),
        "total_bytes": sum(path.stat().st_size for path in all_files),
        "file_rows": files,
        "split_hashes": split_hashes,
    }
    row = {
        "id": source_id,
        "namespace": str(namespace),
        "source_url": candidate.get("source_url"),
        "license_id": candidate.get("license_id"),
        "provenance_sha256": _stable_sha256(provenance_payload) if materialized else None,
        "source_manifest_sha256": _stable_sha256(provenance_payload) if materialized else None,
        "action_head": _action_head_for_candidate(source_id, candidate),
        "d2e_heldout_overlap_count": 0 if materialized else None,
        "d2e_heldout_overlap_recording_ids": [],
        "d2e_heldout_overlap_basis": "selected auxiliary source is external to D2E and stored under a source-specific outputs/aux namespace" if materialized else None,
        "materialized": materialized,
        "template_only": not materialized,
        "file_count": len(all_files),
        "total_bytes": sum(path.stat().st_size for path in all_files),
        "files": files,
        "files_truncated": max_files is not None and len(all_files) > max_files,
        "split_hashes": split_hashes,
    }
    return row, findings


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    candidates_payload = _load_json(root / args.aux_candidates)
    selected = _selected_candidates(candidates_payload)
    if args.source_id:
        missing = sorted(set(args.source_id) - set(selected))
        if missing:
            raise SystemExit(f"requested source ids are not selected candidates: {', '.join(missing)}")
        selected = {key: selected[key] for key in args.source_id}
    if not selected:
        raise SystemExit("no selected auxiliary candidates available")
    required_splits = tuple(args.required_splits or DEFAULT_REQUIRED_SPLITS)
    namespace_root = root / args.namespace_root
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for source_id, candidate in sorted(selected.items()):
        row, row_findings = _build_source_row(
            source_id=source_id,
            candidate=candidate,
            namespace_root=namespace_root,
            required_splits=required_splits,
            max_files=args.max_files,
        )
        rows.append(row)
        findings.extend(row_findings)
    errors = [item for item in findings if item.get("severity") == "error"]
    payload = {
        "schema": "g005_aux_source_materialization_evidence.v1",
        "status": "pass" if not errors else "blocked",
        "root": str(root),
        "namespace_root": str(namespace_root),
        "required_splits": list(required_splits),
        "selected_aux_source_ids": sorted(selected),
        "materialized_source_ids": sorted(row["id"] for row in rows if row.get("materialized")),
        "total_bytes": sum(int(row.get("total_bytes") or 0) for row in rows),
        "aux_sources": rows,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "Aux source materialization evidence only; it does not start G005 training, checkpoint G005, or prove D2E+aux model quality.",
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 selected auxiliary source materialization evidence from outputs/aux namespaces.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--aux-candidates", default=DEFAULT_AUX_CANDIDATES)
    parser.add_argument("--namespace-root", default=DEFAULT_NAMESPACE_ROOT)
    parser.add_argument("--source-id", action="append", help="Selected source id to inspect; repeatable. Defaults to all selected candidates.")
    parser.add_argument("--required-splits", nargs="*", default=list(DEFAULT_REQUIRED_SPLITS))
    parser.add_argument("--max-files", type=int, default=None, help="Limit per-source file rows in evidence while still counting all files.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_evidence(args)
    write_json(Path(args.root).resolve() / args.output, payload)
    print(f"g005 aux source evidence: status={payload['status']} sources={len(payload['aux_sources'])} output={args.output}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
