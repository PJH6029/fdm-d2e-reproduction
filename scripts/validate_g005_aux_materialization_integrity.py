#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdm_d2e.io_utils import sha256_file, write_json
from materialize_g005_aux_sources import _hf_repo_id, _parse_checksum, _provider, _zenodo_files_from_metadata, _hash_file


DEFAULT_OUTPUT = "artifacts/aux/g005_aux_materialization_integrity.json"
DEFAULT_AUX_CANDIDATES = "artifacts/sources/aux_game_action_dataset_candidates.json"
DEFAULT_NAMESPACE_ROOT = "outputs/aux"
DEFAULT_MATERIALIZATION_SUMMARY = "artifacts/aux/g005_aux_materialization_execute_summary.json"
DEFAULT_SPLITS = ("train", "val", "test")


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {"schema": "unexpected_json", "payload_type": type(payload).__name__}


def _read_json_url(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - explicit dataset metadata URLs only
        return json.loads(response.read().decode("utf-8"))


def _selected_candidates(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in payload.get("candidates", [])
        if isinstance(row, dict) and row.get("id") and row.get("selection_status") == "selected_candidate"
    }


def _execution_by_source(summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not summary:
        return {}
    return {
        str(row["id"]): row
        for row in summary.get("executions", []) or []
        if isinstance(row, dict) and row.get("id")
    }


def _file_validation(path: Path, *, expected_size: int | None = None, checksum: Any = None, expected_sha256: str | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "bytes": 0,
            "sha256": None,
            "valid": False,
            "findings": [{"severity": "error", "code": "materialized_file_missing", "path": str(path)}],
        }
    size = path.stat().st_size
    if expected_size is not None and size != int(expected_size):
        findings.append({"severity": "error", "code": "materialized_file_size_mismatch", "expected_size_bytes": int(expected_size), "actual_size_bytes": size})
    checksum_report: dict[str, Any] | None = None
    parsed = _parse_checksum(checksum)
    if parsed is not None:
        algorithm, expected_digest = parsed
        actual = _hash_file(path, algorithm)
        checksum_report = {"algorithm": algorithm, "expected": expected_digest, "actual": actual}
        if actual.lower() != expected_digest.lower():
            findings.append({"severity": "error", "code": "materialized_file_checksum_mismatch", **checksum_report})
    sha256 = sha256_file(path)
    if expected_sha256 and sha256.lower() != str(expected_sha256).lower():
        findings.append({"severity": "error", "code": "materialized_file_sha256_mismatch", "expected": expected_sha256, "actual": sha256})
    return {
        "path": str(path),
        "exists": True,
        "bytes": size,
        "sha256": sha256,
        "checksum": checksum_report,
        "valid": not any(item.get("severity") == "error" for item in findings),
        "findings": findings,
    }


def _resolve_manifest_source_path(namespace: Path, raw_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    raw_candidate = raw_dir / path
    if raw_candidate.exists():
        return raw_candidate
    namespace_candidate = namespace / path
    if namespace_candidate.exists():
        return namespace_candidate
    return raw_candidate


def _validate_split_manifests(namespace: Path, splits: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    raw_dir = namespace / "raw"
    for split in splits:
        manifest = namespace / split / "manifest.json"
        row: dict[str, Any] = {"split": split, "path": str(manifest), "exists": manifest.exists() and manifest.is_file()}
        if not manifest.exists() or not manifest.is_file():
            findings.append({"severity": "error", "code": "split_manifest_missing", "split": split, "path": str(manifest)})
            row.update({"source_file_count": 0, "referenced_files_exist": False})
            rows.append(row)
            continue
        try:
            payload = _load_json(manifest)
        except json.JSONDecodeError as exc:
            findings.append({"severity": "error", "code": "split_manifest_invalid_json", "split": split, "path": str(manifest), "error": str(exc)})
            row.update({"source_file_count": 0, "referenced_files_exist": False})
            rows.append(row)
            continue
        source_files = payload.get("source_files", []) if isinstance(payload, dict) else []
        missing_refs = []
        for item in source_files:
            ref = item.get("path") if isinstance(item, dict) else item
            resolved = _resolve_manifest_source_path(namespace, raw_dir, ref)
            if resolved is None or not resolved.exists() or not resolved.is_file():
                missing_refs.append(str(ref))
        if not source_files:
            findings.append({"severity": "error", "code": "split_manifest_empty_source_files", "split": split, "path": str(manifest)})
        if missing_refs:
            findings.append({"severity": "error", "code": "split_manifest_missing_referenced_files", "split": split, "missing": missing_refs[:20], "missing_truncated": len(missing_refs) > 20})
        row.update({"source_file_count": len(source_files), "referenced_files_exist": not missing_refs})
        rows.append(row)
    return rows, findings


def _validate_zenodo_source(source_id: str, candidate: dict[str, Any], namespace: Path, summary: dict[str, Any] | None, splits: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    metadata_url = candidate.get("metadata_api_url")
    expected_files: list[dict[str, Any]] = []
    if not metadata_url:
        findings.append({"severity": "error", "code": "missing_metadata_api_url", "source_id": source_id})
    else:
        try:
            expected_files = _zenodo_files_from_metadata(_read_json_url(str(metadata_url)))
        except Exception as exc:  # pragma: no cover - surfaced in artifact
            findings.append({"severity": "error", "code": "metadata_read_failed", "source_id": source_id, "metadata_api_url": metadata_url, "error": str(exc)})
    raw_dir = namespace / "raw"
    file_rows = []
    for item in expected_files:
        filename = Path(str(item.get("filename") or "")).name
        if not filename:
            findings.append({"severity": "error", "code": "metadata_file_missing_name", "source_id": source_id, "file": item})
            continue
        size = int(item.get("size_bytes") or 0) or None
        validation = _file_validation(raw_dir / filename, expected_size=size, checksum=item.get("checksum"))
        file_rows.append({"filename": filename, "expected_size_bytes": size, "checksum": item.get("checksum"), **validation})
        findings.extend({**finding, "source_id": source_id, "filename": filename} for finding in validation.get("findings", []))
    split_rows, split_findings = _validate_split_manifests(namespace, splits)
    findings.extend({**finding, "source_id": source_id} for finding in split_findings)
    return {
        "id": source_id,
        "provider": "zenodo",
        "namespace": str(namespace),
        "summary_status": summary.get("status") if isinstance(summary, dict) else None,
        "expected_file_count": len(expected_files),
        "validated_file_count": sum(1 for row in file_rows if row.get("valid")),
        "files": file_rows,
        "split_manifests": split_rows,
    }, findings


def _validate_hf_source(source_id: str, candidate: dict[str, Any], namespace: Path, summary: dict[str, Any] | None, splits: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    raw_dir = namespace / "raw"
    repo_id = _hf_repo_id(candidate)
    if not isinstance(summary, dict):
        findings.append({"severity": "error", "code": "missing_source_materialization_summary", "source_id": source_id})
        summary = {}
    if summary.get("repo_id") and repo_id and summary.get("repo_id") != repo_id:
        findings.append({"severity": "error", "code": "hf_repo_id_mismatch", "source_id": source_id, "expected": repo_id, "actual": summary.get("repo_id")})
    listed_files = summary.get("files", []) if isinstance(summary.get("files"), list) else []
    file_rows = []
    for item in listed_files:
        if not isinstance(item, dict) or not item.get("path"):
            findings.append({"severity": "error", "code": "hf_summary_file_row_malformed", "source_id": source_id, "file": item})
            continue
        validation = _file_validation(raw_dir / str(item["path"]), expected_size=int(item.get("bytes") or 0) or None, expected_sha256=item.get("sha256"))
        file_rows.append({"relative_path": str(item["path"]), **validation})
        findings.extend({**finding, "source_id": source_id, "relative_path": str(item["path"])} for finding in validation.get("findings", []))
    actual_raw_files = [path for path in raw_dir.rglob("*") if path.is_file()] if raw_dir.exists() and raw_dir.is_dir() else []
    if summary.get("file_count") is not None and int(summary.get("file_count") or 0) != len(actual_raw_files):
        findings.append({"severity": "error", "code": "hf_file_count_mismatch", "source_id": source_id, "expected": summary.get("file_count"), "actual": len(actual_raw_files)})
    if summary.get("files_truncated") is True:
        findings.append({"severity": "warning", "code": "hf_validation_uses_truncated_file_listing", "source_id": source_id})
    split_rows, split_findings = _validate_split_manifests(namespace, splits)
    findings.extend({**finding, "source_id": source_id} for finding in split_findings)
    return {
        "id": source_id,
        "provider": "huggingface_dataset",
        "namespace": str(namespace),
        "summary_status": summary.get("status"),
        "repo_id": repo_id,
        "summary_repo_id": summary.get("repo_id"),
        "summary_file_count": summary.get("file_count"),
        "actual_raw_file_count": len(actual_raw_files),
        "validated_listed_file_count": sum(1 for row in file_rows if row.get("valid")),
        "files": file_rows,
        "split_manifests": split_rows,
    }, findings


def build_integrity(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    aux_candidates = _load_json(_path(root, args.aux_candidates))
    selected = _selected_candidates(aux_candidates)
    if args.source_id:
        missing = sorted(set(args.source_id) - set(selected))
        if missing:
            raise SystemExit(f"requested source ids are not selected candidates: {', '.join(missing)}")
        selected = {key: selected[key] for key in args.source_id}
    if not selected:
        raise SystemExit("no selected auxiliary candidates available")
    top_summary = _load_json_if_exists(_path(root, args.materialization_summary))
    execution_by_id = _execution_by_source(top_summary)
    namespace_root = _path(root, args.namespace_root)
    splits = tuple(args.required_splits or DEFAULT_SPLITS)
    sources: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    if top_summary is None:
        findings.append({"severity": "error", "code": "missing_materialization_summary", "path": args.materialization_summary})
    elif top_summary.get("status") != "pass":
        findings.append({"severity": "error", "code": "materialization_summary_not_pass", "path": args.materialization_summary, "status": top_summary.get("status")})
    for source_id, candidate in sorted(selected.items()):
        namespace = namespace_root / source_id
        source_summary = execution_by_id.get(source_id) or _load_json_if_exists(namespace / "materialization_summary.json")
        if not isinstance(source_summary, dict):
            findings.append({"severity": "error", "code": "missing_source_materialization_summary", "source_id": source_id, "path": str(namespace / "materialization_summary.json")})
        elif source_summary.get("status") != "pass":
            findings.append({"severity": "error", "code": "source_materialization_summary_not_pass", "source_id": source_id, "status": source_summary.get("status")})
        provider = _provider(candidate)
        if provider == "zenodo":
            row, source_findings = _validate_zenodo_source(source_id, candidate, namespace, source_summary, splits)
        elif provider == "huggingface_dataset":
            row, source_findings = _validate_hf_source(source_id, candidate, namespace, source_summary, splits)
        else:
            row = {"id": source_id, "provider": provider, "namespace": str(namespace), "summary_status": source_summary.get("status") if isinstance(source_summary, dict) else None}
            source_findings = [{"severity": "error", "code": "unsupported_provider", "source_id": source_id, "provider": provider}]
        sources.append(row)
        findings.extend(source_findings)
    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g005_aux_materialization_integrity.v1",
        "status": "pass" if not errors else "blocked",
        "root": str(root),
        "materialization_summary": str(args.materialization_summary),
        "namespace_root": str(namespace_root),
        "selected_source_ids": sorted(selected),
        "required_splits": list(splits),
        "aux_sources": sources,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "Materialization integrity only; it validates downloaded source bytes and source-level split manifests before source evidence, but does not start G005 training, checkpoint G005, or prove D2E+aux model quality.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate selected G005 auxiliary materialization bytes/manifests before source evidence is accepted.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--aux-candidates", default=DEFAULT_AUX_CANDIDATES)
    parser.add_argument("--namespace-root", default=DEFAULT_NAMESPACE_ROOT)
    parser.add_argument("--materialization-summary", default=DEFAULT_MATERIALIZATION_SUMMARY)
    parser.add_argument("--source-id", action="append", help="Selected source id to validate; repeatable. Defaults to all selected candidates.")
    parser.add_argument("--required-splits", nargs="*", default=list(DEFAULT_SPLITS))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_integrity(args)
    write_json(Path(args.root).resolve() / args.output, payload)
    print(f"g005 aux materialization integrity: status={payload['status']} sources={len(payload['aux_sources'])} output={args.output}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
