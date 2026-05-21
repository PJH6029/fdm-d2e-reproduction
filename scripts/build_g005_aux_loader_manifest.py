#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import stable_hash_json, write_json


DEFAULT_OUTPUT = "artifacts/aux/g005_aux_loader_manifest.json"
DEFAULT_ACTION_REGISTRY = "artifacts/aux/g005_aux_action_registry.json"
DEFAULT_ARCHIVE_INVENTORY = "artifacts/aux/g005_aux_archive_inventory.json"
DEFAULT_MATERIALIZATION_INTEGRITY = "artifacts/aux/g005_aux_materialization_integrity.json"
DEFAULT_REQUIRED_SPLITS = ("train", "val", "test")


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {"schema": "unexpected_json", "payload_type": type(payload).__name__}


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _by_id(rows: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        return {}
    return {str(row["id"]): row for row in rows if isinstance(row, dict) and row.get("id") is not None}


def _adapter_stage(head: dict[str, Any], inventory: dict[str, Any] | None, required_splits: tuple[str, ...]) -> dict[str, Any]:
    source_id = str(head.get("id"))
    action_members = []
    raw_files = []
    if inventory:
        for file_row in inventory.get("files", []) or []:
            if isinstance(file_row, dict):
                raw_files.append({"path": file_row.get("path"), "bytes": file_row.get("bytes"), "archive_type": file_row.get("archive_type")})
                for member in file_row.get("action_candidate_members", []) or []:
                    if isinstance(member, dict):
                        action_members.append({"archive": file_row.get("path"), "path": member.get("path"), "bytes": member.get("bytes")})
    return {
        "source_id": source_id,
        "action_head": {"namespace": head.get("namespace"), "type": head.get("type"), "adapter": head.get("adapter")},
        "raw_files": raw_files,
        "action_candidate_members": action_members,
        "required_splits": list(required_splits),
        "loader_contract": {
            "input_namespace": f"outputs/aux/{source_id}/raw",
            "output_namespace": f"outputs/aux_examples/{source_id}",
            "train_manifest": f"outputs/aux_examples/{source_id}/train.jsonl",
            "val_manifest": f"outputs/aux_examples/{source_id}/val.jsonl",
            "test_manifest": f"outputs/aux_examples/{source_id}/test.jsonl",
            "action_head_namespace": source_id,
            "adapter": head.get("adapter"),
            "example_builder_command": [
                "uv",
                "run",
                "python",
                "scripts/build_g005_aux_examples.py",
                "--source-id",
                source_id,
            ],
            "must_emit_fields": ["source_id", "source_sequence_id", "frame_or_state_ref", "action", "action_head_namespace", "split", "provenance"],
        },
        "training_role": head.get("transfer_role"),
        "claim_boundary": "Loader outputs remain source-specific auxiliary examples and cannot be counted as D2E heldout/eval examples.",
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    action_registry = _load_optional_json(_path(root, args.action_registry))
    archive_inventory = _load_optional_json(_path(root, args.archive_inventory))
    materialization_integrity = _load_optional_json(_path(root, args.materialization_integrity))
    required_splits = tuple(args.required_splits or DEFAULT_REQUIRED_SPLITS)
    findings: list[dict[str, Any]] = []

    if not action_registry:
        findings.append({"severity": "error", "code": "missing_action_registry", "path": args.action_registry})
        heads: dict[str, dict[str, Any]] = {}
    elif action_registry.get("status") != "pass":
        findings.append({"severity": "error", "code": "action_registry_not_pass", "status": action_registry.get("status")})
        heads = _by_id(action_registry.get("action_heads"))
    else:
        heads = _by_id(action_registry.get("action_heads"))
    if not heads:
        findings.append({"severity": "error", "code": "action_registry_has_no_heads"})

    inventory_by_id = _by_id((archive_inventory or {}).get("aux_sources"))
    if not archive_inventory:
        findings.append({"severity": "error", "code": "missing_archive_inventory", "path": args.archive_inventory})
    elif archive_inventory.get("status") != "pass":
        findings.append({"severity": "error", "code": "archive_inventory_not_pass", "status": archive_inventory.get("status"), "error_count": archive_inventory.get("error_count")})

    integrity_by_id = _by_id((materialization_integrity or {}).get("aux_sources"))
    if not materialization_integrity:
        findings.append({"severity": "error", "code": "missing_materialization_integrity", "path": args.materialization_integrity})
    elif materialization_integrity.get("status") != "pass":
        findings.append({"severity": "error", "code": "materialization_integrity_not_pass", "status": materialization_integrity.get("status"), "error_count": materialization_integrity.get("error_count")})

    selected_ids = sorted(heads)
    inventory_ids = set(inventory_by_id)
    integrity_ids = set(integrity_by_id)
    missing_inventory = sorted(set(selected_ids) - inventory_ids)
    missing_integrity = sorted(set(selected_ids) - integrity_ids)
    if missing_inventory:
        findings.append({"severity": "error", "code": "loader_manifest_missing_inventory_sources", "missing": missing_inventory})
    if missing_integrity:
        findings.append({"severity": "error", "code": "loader_manifest_missing_integrity_sources", "missing": missing_integrity})

    stages = []
    for source_id in selected_ids:
        head = heads[source_id]
        inv = inventory_by_id.get(source_id)
        integ = integrity_by_id.get(source_id)
        if inv and int(inv.get("raw_file_count") or 0) <= 0:
            findings.append({"severity": "error", "code": "loader_source_has_no_raw_files", "source_id": source_id})
        if inv and int(inv.get("action_candidate_member_count") or 0) <= 0:
            findings.append({"severity": "warning", "code": "loader_source_has_no_action_member_hints", "source_id": source_id})
        if integ:
            split_rows = integ.get("split_manifests", []) if isinstance(integ.get("split_manifests"), list) else []
            split_names = {str(row.get("split")) for row in split_rows if isinstance(row, dict) and row.get("split")}
            missing_splits = sorted(set(required_splits) - split_names)
            if missing_splits:
                findings.append({"severity": "error", "code": "loader_source_missing_integrity_split_rows", "source_id": source_id, "missing": missing_splits})
        stages.append(_adapter_stage(head, inv, required_splits))

    errors = [item for item in findings if item.get("severity") == "error"]
    payload = {
        "schema": "g005_aux_loader_manifest.v1",
        "status": "pass" if not errors else "blocked",
        "root": str(root),
        "selected_aux_source_ids": selected_ids,
        "source_specific_action_heads": True,
        "no_aux_in_d2e_heldout": True,
        "no_cross_source_action_collapse": True,
        "required_splits": list(required_splits),
        "inputs": {
            "action_registry": args.action_registry,
            "archive_inventory": args.archive_inventory,
            "materialization_integrity": args.materialization_integrity,
        },
        "loader_stages": stages,
        "training_sequence": [
            "materialize selected aux raw sources under outputs/aux/<source_id>/raw",
            "validate materialization integrity and source-level split manifests",
            "build source-specific auxiliary example manifests under outputs/aux_examples/<source_id>/",
            "pretrain shared visual-temporal backbone with source-specific action heads",
            "finetune/evaluate on D2E-only train/eval manifests without aux in heldouts",
        ],
        "manifest_fingerprint": stable_hash_json({"sources": selected_ids, "stages": stages, "required_splits": list(required_splits)}),
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "Aux loader manifest only; it plans source-specific loader outputs and cannot train, checkpoint G005, or support D2E+aux model-quality claims by itself.",
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the G005 source-specific auxiliary loader manifest after materialization/inventory/integrity evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--action-registry", default=DEFAULT_ACTION_REGISTRY)
    parser.add_argument("--archive-inventory", default=DEFAULT_ARCHIVE_INVENTORY)
    parser.add_argument("--materialization-integrity", default=DEFAULT_MATERIALIZATION_INTEGRITY)
    parser.add_argument("--required-splits", nargs="*", default=list(DEFAULT_REQUIRED_SPLITS))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_manifest(args)
    write_json(Path(args.root).resolve() / args.output, payload)
    print(f"g005 aux loader manifest: status={payload['status']} stages={len(payload['loader_stages'])} output={args.output}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
