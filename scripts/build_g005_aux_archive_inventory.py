#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import sha256_file, write_json


DEFAULT_AUX_CANDIDATES = "artifacts/sources/aux_game_action_dataset_candidates.json"
DEFAULT_NAMESPACE_ROOT = "outputs/aux"
DEFAULT_OUTPUT = "artifacts/aux/g005_aux_archive_inventory.json"
ACTION_HINTS = (
    "action",
    "actions",
    "action_enums",
    "button",
    "buttons",
    "keyboard",
    "mouse",
    "joystick",
    "event",
    "events",
    "label",
    "labels",
    "trajectory",
    "trajectories",
    "demonstration",
    "demonstrations",
)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selected_candidates(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in payload.get("candidates", [])
        if isinstance(row, dict) and row.get("id") and row.get("selection_status") == "selected_candidate"
    }


def _raw_files(namespace: Path) -> list[Path]:
    raw = namespace / "raw"
    if not raw.exists() or not raw.is_dir():
        return []
    return sorted(path for path in raw.rglob("*") if path.is_file())


def _is_action_candidate(name: str) -> bool:
    lower = name.lower()
    if lower.endswith(".array_record"):
        return True
    return any(hint in lower for hint in ACTION_HINTS)


def _trim_members(rows: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], bool]:
    return rows[:limit], len(rows) > limit


def _inspect_zip(path: Path, max_members: int) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
    members = [
        {
            "path": info.filename,
            "bytes": int(info.file_size),
            "compressed_bytes": int(info.compress_size),
            "action_candidate": _is_action_candidate(info.filename),
        }
        for info in infos
    ]
    sample, truncated = _trim_members(members, max_members)
    action_members = [row for row in members if row["action_candidate"]]
    action_sample, action_truncated = _trim_members(action_members, max_members)
    return {
        "archive_type": "zip",
        "member_count": len(members),
        "sample_members": sample,
        "sample_members_truncated": truncated,
        "action_candidate_members": action_sample,
        "action_candidate_members_truncated": action_truncated,
    }


def _inspect_tar(path: Path, max_members: int) -> dict[str, Any]:
    with tarfile.open(path) as tf:
        infos = [info for info in tf.getmembers() if info.isfile()]
    members = [
        {
            "path": info.name,
            "bytes": int(info.size),
            "action_candidate": _is_action_candidate(info.name),
        }
        for info in infos
    ]
    sample, truncated = _trim_members(members, max_members)
    action_members = [row for row in members if row["action_candidate"]]
    action_sample, action_truncated = _trim_members(action_members, max_members)
    return {
        "archive_type": "tar",
        "member_count": len(members),
        "sample_members": sample,
        "sample_members_truncated": truncated,
        "action_candidate_members": action_sample,
        "action_candidate_members_truncated": action_truncated,
    }


def _regular_file_inventory(path: Path) -> dict[str, Any]:
    return {
        "archive_type": "regular_file",
        "member_count": 1,
        "sample_members": [{"path": path.name, "bytes": path.stat().st_size, "action_candidate": _is_action_candidate(path.name)}],
        "sample_members_truncated": False,
        "action_candidate_members": [{"path": path.name, "bytes": path.stat().st_size, "action_candidate": True}] if _is_action_candidate(path.name) else [],
        "action_candidate_members_truncated": False,
    }


def _inspect_file(path: Path, *, namespace: Path, max_members: int, hash_files: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    row: dict[str, Any] = {
        "path": str(path.relative_to(namespace)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path) if hash_files else None,
    }
    try:
        if zipfile.is_zipfile(path):
            row.update(_inspect_zip(path, max_members))
        elif tarfile.is_tarfile(path):
            row.update(_inspect_tar(path, max_members))
        else:
            row.update(_regular_file_inventory(path))
    except (OSError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        row.update(
            {
                "archive_type": "unreadable_archive",
                "member_count": 0,
                "sample_members": [],
                "sample_members_truncated": False,
                "action_candidate_members": [],
                "action_candidate_members_truncated": False,
                "error": str(exc),
            }
        )
        findings.append({"severity": "error", "code": "archive_unreadable", "path": row["path"], "error": str(exc)})
    return row, findings


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
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

    namespace_root = root / args.namespace_root
    sources: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for source_id, candidate in sorted(selected.items()):
        namespace = namespace_root / source_id
        raw_files = _raw_files(namespace)
        if not namespace.exists():
            findings.append({"severity": "error", "code": "aux_namespace_missing", "source_id": source_id, "namespace": str(namespace)})
        elif not raw_files:
            findings.append({"severity": "error", "code": "aux_raw_files_missing", "source_id": source_id, "raw_dir": str(namespace / "raw")})
        file_rows: list[dict[str, Any]] = []
        action_candidate_count = 0
        for path in raw_files:
            row, row_findings = _inspect_file(path, namespace=namespace, max_members=int(args.max_members), hash_files=bool(args.hash_files))
            file_rows.append(row)
            action_candidate_count += len(row.get("action_candidate_members") or [])
            for item in row_findings:
                findings.append({**item, "source_id": source_id})
        if raw_files and action_candidate_count == 0:
            findings.append(
                {
                    "severity": "warning",
                    "code": "no_action_candidate_members_detected",
                    "source_id": source_id,
                    "note": "Inventory hints are heuristic; source-specific loaders may still find action labels inside nested formats.",
                }
            )
        sources.append(
            {
                "id": source_id,
                "namespace": str(namespace),
                "source_url": candidate.get("source_url"),
                "license_id": candidate.get("license_id"),
                "raw_file_count": len(raw_files),
                "raw_total_bytes": sum(path.stat().st_size for path in raw_files),
                "action_candidate_member_count": action_candidate_count,
                "files": file_rows,
            }
        )
    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g005_aux_archive_inventory.v1",
        "status": "pass" if not errors else "blocked",
        "root": str(root),
        "namespace_root": str(namespace_root),
        "selected_source_ids": sorted(selected),
        "hash_files": bool(args.hash_files),
        "max_members_per_list": int(args.max_members),
        "aux_sources": sources,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "Archive inventory only; it helps implement source-specific loaders but does not start G005 training, checkpoint G005, or prove D2E+aux model quality.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory materialized G005 auxiliary source archives for source-specific loader implementation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--aux-candidates", default=DEFAULT_AUX_CANDIDATES)
    parser.add_argument("--namespace-root", default=DEFAULT_NAMESPACE_ROOT)
    parser.add_argument("--source-id", action="append", help="Selected source id to inspect; repeatable. Defaults to all selected candidates.")
    parser.add_argument("--max-members", type=int, default=200)
    parser.add_argument("--hash-files", action="store_true", help="Hash raw files. Disabled by default to avoid expensive reads of multi-GB archives.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_inventory(args)
    write_json(Path(args.root).resolve() / args.output, payload)
    print(f"g005 aux archive inventory: status={payload['status']} sources={len(payload['aux_sources'])} output={args.output}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
