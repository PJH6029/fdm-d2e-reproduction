#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import sha256_file, write_json


REQUIRED_SPLITS = ("temporal", "heldout_recording", "heldout_game")
DEFAULT_OUTPUT = "artifacts/aux/d2e_eval_manifest_hashes.json"
DEFAULT_SPLIT_PATHS = {
    "temporal": "artifacts/sources/d2e_full_temporal_split_manifest.json",
    "heldout_recording": "artifacts/sources/d2e_full_heldout_recording_split_manifest.json",
    "heldout_game": "artifacts/sources/d2e_full_heldout_game_split_manifest.json",
}


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _manifest_status(root: Path, rel_path: str | Path) -> dict[str, Any]:
    path = _path(root, rel_path)
    if not path.exists() or not path.is_file():
        return {"path": str(rel_path), "exists": False, "bytes": 0, "sha256": None}
    return {"path": str(rel_path), "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _split_paths_from_args(args: argparse.Namespace, prefix: str) -> dict[str, str]:
    return {
        "temporal": getattr(args, f"{prefix}_temporal"),
        "heldout_recording": getattr(args, f"{prefix}_heldout_recording"),
        "heldout_game": getattr(args, f"{prefix}_heldout_game"),
    }


def build_hash_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    d2e_only_paths = _split_paths_from_args(args, "d2e_only")
    d2e_aux_paths = _split_paths_from_args(args, "d2e_aux") if args.d2e_aux_temporal else dict(d2e_only_paths)
    findings: list[dict[str, Any]] = []
    splits: dict[str, dict[str, Any]] = {}
    for split in REQUIRED_SPLITS:
        d2e_only = _manifest_status(root, d2e_only_paths[split])
        d2e_aux = _manifest_status(root, d2e_aux_paths[split])
        same_hash = bool(d2e_only["sha256"] and d2e_only["sha256"] == d2e_aux["sha256"])
        if not d2e_only["exists"]:
            findings.append({"severity": "error", "code": "missing_d2e_only_eval_manifest", "split": split, "path": d2e_only["path"]})
        if not d2e_aux["exists"]:
            findings.append({"severity": "error", "code": "missing_d2e_aux_eval_manifest", "split": split, "path": d2e_aux["path"]})
        if d2e_only["exists"] and d2e_aux["exists"] and not same_hash:
            item = {
                "severity": "warning" if args.allow_mismatch else "error",
                "code": "d2e_aux_eval_manifest_hash_mismatch",
                "split": split,
                "d2e_only_path": d2e_only["path"],
                "d2e_aux_path": d2e_aux["path"],
                "d2e_only_manifest_sha256": d2e_only["sha256"],
                "d2e_aux_manifest_sha256": d2e_aux["sha256"],
            }
            findings.append(item)
        splits[split] = {
            "split": split,
            "sha256": d2e_only["sha256"],
            "d2e_only_manifest_path": d2e_only["path"],
            "d2e_only_manifest_bytes": d2e_only["bytes"],
            "d2e_only_manifest_sha256": d2e_only["sha256"],
            "d2e_aux_manifest_path": d2e_aux["path"],
            "d2e_aux_manifest_bytes": d2e_aux["bytes"],
            "d2e_aux_manifest_sha256": d2e_aux["sha256"],
            "same_hash": same_hash,
        }
    errors = [item for item in findings if item.get("severity") == "error"]
    payload = {
        "schema": "g005_d2e_eval_manifest_hashes.v1",
        "status": "pass" if not errors else "fail",
        "root": str(root),
        "required_splits": list(REQUIRED_SPLITS),
        "same_d2e_eval_manifests": all(row["same_hash"] for row in splits.values()),
        "splits": splits,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "D2E eval-manifest hashes for G005 namespace evidence only; this does not launch aux training, checkpoint G005, or prove D2E+aux model quality.",
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 D2E eval-manifest hash evidence for D2E-only vs D2E+aux ablations.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--d2e-only-temporal", default=DEFAULT_SPLIT_PATHS["temporal"])
    parser.add_argument("--d2e-only-heldout-recording", default=DEFAULT_SPLIT_PATHS["heldout_recording"])
    parser.add_argument("--d2e-only-heldout-game", default=DEFAULT_SPLIT_PATHS["heldout_game"])
    parser.add_argument("--d2e-aux-temporal", help="Optional aux eval manifest path; defaults to D2E-only temporal manifest for byte-identical evidence.")
    parser.add_argument("--d2e-aux-heldout-recording", help="Optional aux eval manifest path; defaults with --d2e-aux-temporal group.")
    parser.add_argument("--d2e-aux-heldout-game", help="Optional aux eval manifest path; defaults with --d2e-aux-temporal group.")
    parser.add_argument("--allow-mismatch", action="store_true", help="Write non-terminal evidence if aux manifest hashes differ.")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    if args.d2e_aux_temporal and not (args.d2e_aux_heldout_recording and args.d2e_aux_heldout_game):
        parser.error("--d2e-aux-temporal requires --d2e-aux-heldout-recording and --d2e-aux-heldout-game")
    payload = build_hash_manifest(args)
    write_json(_path(Path(args.root).resolve(), args.output), payload)
    print(f"g005 eval manifest hashes: status={payload['status']} splits={len(payload['splits'])} output={args.output}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
