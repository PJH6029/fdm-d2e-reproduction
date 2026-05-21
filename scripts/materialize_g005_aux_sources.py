#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import sha256_file, write_json


DEFAULT_OUTPUT = "artifacts/aux/g005_aux_materialization_plan.json"
DEFAULT_AUX_CANDIDATES = "artifacts/sources/aux_game_action_dataset_candidates.json"
DEFAULT_NAMESPACE_ROOT = "outputs/aux"
DEFAULT_SPLITS = ("train", "val", "test")


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_url(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - explicit user/repo URLs only
        return json.loads(response.read().decode("utf-8"))


def _selected_candidates(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in payload.get("candidates", [])
        if isinstance(row, dict) and row.get("id") and row.get("selection_status") == "selected_candidate"
    }


def _provider(candidate: dict[str, Any]) -> str:
    source_url = str(candidate.get("source_url") or "").lower()
    metadata_url = str(candidate.get("metadata_api_url") or "").lower()
    if "zenodo.org" in source_url or "zenodo.org" in metadata_url:
        return "zenodo"
    if "huggingface.co/datasets" in source_url or "huggingface.co/api/datasets" in metadata_url:
        return "huggingface_dataset"
    return "manual"


def _hf_repo_id(candidate: dict[str, Any]) -> str | None:
    for key in ("source_url", "metadata_api_url"):
        value = str(candidate.get(key) or "")
        if "huggingface.co" not in value:
            continue
        parsed = urllib.parse.urlparse(value)
        parts = [part for part in parsed.path.split("/") if part]
        if "datasets" in parts:
            idx = parts.index("datasets")
            if len(parts) >= idx + 3:
                return "/".join(parts[idx + 1 : idx + 3])
        if parsed.netloc == "huggingface.co" and len(parts) >= 2:
            return "/".join(parts[:2])
    return None


def _hf_revision(candidate: dict[str, Any]) -> str | None:
    value = str(candidate.get("source_revision_or_record") or "")
    if value.startswith("hf_sha_"):
        return value.removeprefix("hf_sha_")
    return None


def _zenodo_files_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in metadata.get("files", []) or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or item.get("filename") or item.get("id") or "")
        links = item.get("links") if isinstance(item.get("links"), dict) else {}
        url = links.get("self") or links.get("download") or item.get("download_url")
        size = item.get("size") or item.get("filesize")
        checksum = item.get("checksum")
        rows.append({"filename": key, "url": url, "size_bytes": size, "checksum": checksum})
    return rows


def _parse_checksum(value: Any) -> tuple[str, str] | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        algorithm, digest = text.split(":", 1)
        algorithm = algorithm.lower().strip()
        digest = digest.lower().strip()
    else:
        digest = text.lower()
        algorithm = "sha256" if len(digest) == 64 else "md5" if len(digest) == 32 else ""
    if algorithm not in {"md5", "sha1", "sha256", "sha512"} or not digest:
        return None
    return algorithm, digest


def _hash_file(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_validation(path: Path, *, expected_size: int | None, checksum: Any) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    actual_size = path.stat().st_size if path.exists() and path.is_file() else None
    if expected_size is not None and actual_size != int(expected_size):
        findings.append(
            {
                "severity": "error",
                "code": "download_size_mismatch",
                "expected_size_bytes": int(expected_size),
                "actual_size_bytes": actual_size,
            }
        )
    parsed_checksum = _parse_checksum(checksum)
    checksum_report: dict[str, Any] | None = None
    if parsed_checksum is not None and path.exists() and path.is_file():
        algorithm, expected_digest = parsed_checksum
        actual_digest = _hash_file(path, algorithm)
        checksum_report = {"algorithm": algorithm, "expected": expected_digest, "actual": actual_digest}
        if actual_digest.lower() != expected_digest.lower():
            findings.append({"severity": "error", "code": "download_checksum_mismatch", **checksum_report})
    return {
        "valid": not any(item.get("severity") == "error" for item in findings),
        "expected_size_bytes": expected_size,
        "actual_size_bytes": actual_size,
        "checksum": checksum_report,
        "findings": findings,
    }


def _candidate_plan(source_id: str, candidate: dict[str, Any], namespace_root: Path) -> dict[str, Any]:
    provider = _provider(candidate)
    namespace = namespace_root / source_id
    plan = {
        "id": source_id,
        "provider": provider,
        "namespace": str(namespace),
        "raw_dir": str(namespace / "raw"),
        "source_url": candidate.get("source_url"),
        "metadata_api_url": candidate.get("metadata_api_url"),
        "license_id": candidate.get("license_id"),
        "expected_size_bytes": candidate.get("size_bytes") or candidate.get("estimated_size_gib"),
        "source_revision_or_record": candidate.get("source_revision_or_record"),
        "valid_for_training_now": candidate.get("valid_for_training_now"),
        "execute_supported": provider in {"zenodo", "huggingface_dataset"},
    }
    if provider == "huggingface_dataset":
        plan.update({"repo_id": _hf_repo_id(candidate), "repo_type": "dataset", "revision": _hf_revision(candidate)})
    return plan


def _invalid_backup_path(dest: Path) -> Path:
    suffix = f".invalid-{os.getpid()}"
    candidate = dest.with_name(dest.name + suffix)
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = dest.with_name(dest.name + f"{suffix}-{counter}")
    return candidate


def _download_url(url: str, dest: Path, *, expected_size: int | None = None, checksum: Any = None) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.is_file() and dest.stat().st_size > 0:
        validation = _download_validation(dest, expected_size=expected_size, checksum=checksum)
        if validation["valid"]:
            return {
                "path": str(dest),
                "status": "existing",
                "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "validation": validation,
                "findings": [],
            }
        backup = _invalid_backup_path(dest)
        dest.replace(backup)
        replaced_invalid_existing = str(backup)
    else:
        replaced_invalid_existing = None
    parsed = urllib.parse.urlparse(url)
    tmp = dest.with_name(dest.name + f".part-{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    if parsed.scheme == "file":
        src = Path(urllib.request.url2pathname(parsed.path))
        shutil.copyfile(src, tmp)
    else:
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:  # noqa: S310 - explicit dataset URLs only
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    validation = _download_validation(tmp, expected_size=expected_size, checksum=checksum)
    if validation["valid"]:
        tmp.replace(dest)
        return {
            "path": str(dest),
            "status": "downloaded",
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "validation": validation,
            "replaced_invalid_existing": replaced_invalid_existing,
            "findings": [],
        }
    return {
        "path": str(dest),
        "status": "invalid_download",
        "tmp_path": str(tmp),
        "bytes": tmp.stat().st_size if tmp.exists() else 0,
        "sha256": sha256_file(tmp) if tmp.exists() else None,
        "validation": validation,
        "replaced_invalid_existing": replaced_invalid_existing,
        "findings": [{"severity": "error", "code": "download_validation_failed", "path": str(dest), "url": url, "validation": validation}],
    }


def _write_split_manifests(namespace: Path, files: list[dict[str, Any]], splits: tuple[str, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in splits:
        split_dir = namespace / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = split_dir / "manifest.json"
        payload = {
            "schema": "g005_aux_source_split_manifest.v1",
            "split": split,
            "source_files": files,
            "note": "Source-level manifest used for materialization/provenance gating; training loaders may create finer example-level splits later.",
        }
        write_json(manifest_path, payload)
        output[split] = {"path": str(manifest_path), "bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path)}
    return output


def _execute_zenodo(source_id: str, candidate: dict[str, Any], namespace_root: Path, splits: tuple[str, ...], max_bytes: int | None) -> dict[str, Any]:
    metadata_url = candidate.get("metadata_api_url")
    if not metadata_url:
        return {"id": source_id, "status": "blocked", "findings": [{"severity": "error", "code": "missing_metadata_api_url"}]}
    metadata = _read_json_url(str(metadata_url))
    files = _zenodo_files_from_metadata(metadata)
    findings = []
    if not files:
        findings.append({"severity": "error", "code": "zenodo_metadata_has_no_files"})
    namespace = namespace_root / source_id
    raw_dir = namespace / "raw"
    downloads = []
    total_requested = 0
    for row in files:
        if not row.get("url") or not row.get("filename"):
            findings.append({"severity": "error", "code": "zenodo_file_missing_url_or_name", "file": row})
            continue
        size = int(row.get("size_bytes") or 0)
        if max_bytes is not None and total_requested + size > max_bytes:
            findings.append({"severity": "warning", "code": "max_bytes_skipped_file", "filename": row.get("filename"), "size_bytes": size})
            continue
        total_requested += size
        download = _download_url(
            str(row["url"]),
            raw_dir / Path(str(row["filename"])).name,
            expected_size=size if size > 0 else None,
            checksum=row.get("checksum"),
        )
        downloads.append(download)
        findings.extend(download.get("findings", []))
    valid_downloads = [item for item in downloads if item.get("status") != "invalid_download"]
    split_manifests = _write_split_manifests(namespace, valid_downloads, splits) if valid_downloads else {}
    errors = [item for item in findings if item.get("severity") == "error"]
    summary = {
        "id": source_id,
        "provider": "zenodo",
        "status": "pass" if valid_downloads and not errors else "blocked",
        "namespace": str(namespace),
        "download_count": len(valid_downloads),
        "attempted_download_count": len(downloads),
        "downloaded_bytes": sum(int(item.get("bytes") or 0) for item in valid_downloads),
        "downloads": downloads,
        "split_manifests": split_manifests,
        "metadata_record_id": metadata.get("id") if isinstance(metadata, dict) else None,
        "findings": findings,
    }
    write_json(namespace / "materialization_summary.json", summary)
    return summary


def _execute_hf(source_id: str, candidate: dict[str, Any], namespace_root: Path, splits: tuple[str, ...], allow_patterns: list[str] | None, ignore_patterns: list[str] | None) -> dict[str, Any]:
    repo_id = _hf_repo_id(candidate)
    if not repo_id:
        return {"id": source_id, "provider": "huggingface_dataset", "status": "blocked", "findings": [{"severity": "error", "code": "missing_huggingface_repo_id"}]}
    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:  # pragma: no cover - depends on optional env
        return {"id": source_id, "provider": "huggingface_dataset", "status": "blocked", "findings": [{"severity": "error", "code": "huggingface_hub_unavailable", "error": str(exc)}]}
    namespace = namespace_root / source_id
    raw_dir = namespace / "raw"
    revision = _hf_revision(candidate)
    api = HfApi()
    info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(raw_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    file_rows = []
    for path in sorted(Path(local_dir).rglob("*")):
        if path.is_file():
            file_rows.append({"path": str(path.relative_to(raw_dir)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    split_manifests = _write_split_manifests(namespace, file_rows, splits) if file_rows else {}
    summary = {
        "id": source_id,
        "provider": "huggingface_dataset",
        "status": "pass" if file_rows else "blocked",
        "namespace": str(namespace),
        "repo_id": repo_id,
        "revision": revision,
        "resolved_sha": getattr(info, "sha", None),
        "local_dir": str(local_dir),
        "file_count": len(file_rows),
        "downloaded_bytes": sum(int(item.get("bytes") or 0) for item in file_rows),
        "files": file_rows[:200],
        "files_truncated": len(file_rows) > 200,
        "split_manifests": split_manifests,
        "findings": [] if file_rows else [{"severity": "error", "code": "huggingface_snapshot_empty"}],
    }
    write_json(namespace / "materialization_summary.json", summary)
    return summary


def build_or_execute(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    candidates_payload = _load_json(root / args.aux_candidates)
    selected = _selected_candidates(candidates_payload)
    if args.source_id:
        missing = sorted(set(args.source_id) - set(selected))
        if missing:
            raise SystemExit(f"requested source ids are not selected candidates: {', '.join(missing)}")
        selected = {key: selected[key] for key in args.source_id}
    namespace_root = root / args.namespace_root
    splits = tuple(args.splits or DEFAULT_SPLITS)
    plans = [_candidate_plan(source_id, candidate, namespace_root) for source_id, candidate in sorted(selected.items())]
    executions = []
    findings: list[dict[str, Any]] = []
    if args.execute:
        for source_id, candidate in sorted(selected.items()):
            provider = _provider(candidate)
            if provider == "zenodo":
                result = _execute_zenodo(source_id, candidate, namespace_root, splits, args.max_bytes)
            elif provider == "huggingface_dataset":
                result = _execute_hf(source_id, candidate, namespace_root, splits, args.allow_patterns, args.ignore_patterns)
            else:
                result = {"id": source_id, "provider": provider, "status": "blocked", "findings": [{"severity": "error", "code": "unsupported_provider"}]}
            executions.append(result)
            findings.extend(result.get("findings", []))
    errors = [item for item in findings if item.get("severity") == "error"]
    status = "planned" if not args.execute else ("pass" if executions and not errors and all(item.get("status") == "pass" for item in executions) else "blocked")
    return {
        "schema": "g005_aux_materialization_plan.v1",
        "status": status,
        "root": str(root),
        "execute": bool(args.execute),
        "namespace_root": str(namespace_root),
        "selected_source_ids": sorted(selected),
        "splits": list(splits),
        "plans": plans,
        "executions": executions,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "Materializes selected G005 auxiliary source files only; it does not train, checkpoint G005, or prove D2E+aux model quality.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan or execute selected G005 auxiliary source materialization into outputs/aux namespaces.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--aux-candidates", default=DEFAULT_AUX_CANDIDATES)
    parser.add_argument("--namespace-root", default=DEFAULT_NAMESPACE_ROOT)
    parser.add_argument("--source-id", action="append", help="Selected source id to materialize; repeatable. Defaults to all selected candidates.")
    parser.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="Actually download/materialize sources. Without this, only writes a plan.")
    parser.add_argument("--max-bytes", type=int, help="Optional per-run Zenodo download byte cap for staged materialization.")
    parser.add_argument("--allow-patterns", action="append", help="Hugging Face snapshot allow pattern; repeatable.")
    parser.add_argument("--ignore-patterns", action="append", help="Hugging Face snapshot ignore pattern; repeatable.")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_or_execute(args)
    write_json(Path(args.root).resolve() / args.output, payload)
    print(f"g005 aux materialization: status={payload['status']} execute={payload['execute']} sources={len(payload['selected_source_ids'])} output={args.output}")
    return 0 if payload["status"] in {"planned", "pass"} or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
