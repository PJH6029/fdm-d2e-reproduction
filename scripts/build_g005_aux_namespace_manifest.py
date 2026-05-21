#!/usr/bin/env python3
"""Build the G005 auxiliary namespace manifest from materialization evidence.

The completion audit intentionally requires a separate namespace manifest so
D2E+aux claims cannot be inferred from checkpoint metadata alone. This builder
keeps that manifest reproducible: selected auxiliary sources come from the
candidate plan, materialized source facts come from explicit per-source evidence
JSON files, and D2E eval split hashes come from an explicit hash manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_SPLITS = ("temporal", "heldout_recording", "heldout_game")


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stable_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_candidates(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in payload.get("candidates", [])
        if isinstance(row, dict) and row.get("selection_status") == "selected_candidate" and row.get("id")
    }


def _load_source_evidence(paths: list[str]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = _load_json(path)
        rows = payload if isinstance(payload, list) else payload.get("aux_sources", payload.get("sources", [payload]))
        if not isinstance(rows, list):
            raise SystemExit(f"source evidence must be a JSON object/list with source rows: {path}")
        for row in rows:
            if not isinstance(row, dict) or not row.get("id"):
                raise SystemExit(f"source evidence row missing id in {path}")
            source_id = str(row["id"])
            if source_id in evidence:
                raise SystemExit(f"duplicate source evidence id={source_id}")
            evidence[source_id] = row
    return evidence


def _action_head_for_candidate(source_id: str, candidate: dict[str, Any]) -> dict[str, str]:
    domain = str(candidate.get("domain", "")).lower()
    if "minecraft" in domain:
        head_type = "minecraft_keyboard_mouse"
    elif "atari" in domain:
        head_type = "atari_discrete"
    else:
        head_type = "source_specific"
    return {"type": head_type, "namespace": source_id}


def _template_source_row(source_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": source_id,
        "namespace": f"outputs/aux/{source_id}/TEMPLATE_NOT_MATERIALIZED/",
        "source_url": candidate.get("source_url"),
        "license_id": candidate.get("license_id"),
        "provenance_sha256": f"candidate-plan-sha256:{_stable_sha256(candidate)}",
        "action_head": _action_head_for_candidate(source_id, candidate),
        "d2e_heldout_overlap_count": None,
        "d2e_heldout_overlap_recording_ids": [],
        "materialized": False,
        "template_only": True,
    }


def _normalize_source_row(source_id: str, candidate: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    merged.setdefault("id", source_id)
    merged.setdefault("source_url", candidate.get("source_url"))
    merged.setdefault("license_id", candidate.get("license_id"))
    merged.setdefault("namespace", f"outputs/aux/{source_id}/")
    merged.setdefault("action_head", _action_head_for_candidate(source_id, candidate))
    merged.setdefault("d2e_heldout_overlap_recording_ids", [])
    merged.setdefault("materialized", True)
    if "provenance_sha256" not in merged and merged.get("source_manifest_sha256"):
        merged["provenance_sha256"] = merged["source_manifest_sha256"]
    return merged


def _load_eval_hashes(path: str | None, *, allow_template: bool) -> dict[str, dict[str, Any]]:
    if not path:
        if not allow_template:
            raise SystemExit("--eval-manifest-hashes is required unless --allow-template is set")
        return {
            split: {
                "d2e_only_manifest_sha256": None,
                "d2e_aux_manifest_sha256": None,
                "same_hash": False,
                "template_only": True,
            }
            for split in REQUIRED_SPLITS
        }
    payload = _load_json(path)
    raw = payload.get("splits", payload) if isinstance(payload, dict) else payload
    if isinstance(raw, list):
        rows = {str(row["split"]): row for row in raw if isinstance(row, dict) and row.get("split")}
    elif isinstance(raw, dict):
        rows = {str(split): row for split, row in raw.items() if isinstance(row, dict)}
    else:
        raise SystemExit("eval hash manifest must be a JSON object/list")
    normalized: dict[str, dict[str, Any]] = {}
    for split in REQUIRED_SPLITS:
        row = dict(rows.get(split, {}))
        row.setdefault("d2e_only_manifest_sha256", row.get("sha256"))
        row.setdefault("d2e_aux_manifest_sha256", row.get("sha256"))
        row["same_hash"] = bool(row.get("same_hash") is True or (row.get("d2e_only_manifest_sha256") and row.get("d2e_only_manifest_sha256") == row.get("d2e_aux_manifest_sha256")))
        normalized[split] = row
    return normalized


def _completion_ready(sources: list[dict[str, Any]], eval_hashes: dict[str, dict[str, Any]], requested: bool) -> bool:
    if not requested:
        return False
    for row in sources:
        if row.get("template_only") or not row.get("materialized"):
            return False
        if not row.get("provenance_sha256") or not row.get("namespace"):
            return False
        try:
            overlap_count = int(row.get("d2e_heldout_overlap_count", -1))
        except (TypeError, ValueError):
            overlap_count = -1
        if overlap_count != 0:
            return False
        action_head = row.get("action_head")
        if not isinstance(action_head, dict) or action_head.get("namespace") != row.get("id") or not action_head.get("type"):
            return False
    return all(
        bool(row.get("same_hash") and row.get("d2e_only_manifest_sha256") and row.get("d2e_aux_manifest_sha256"))
        for row in eval_hashes.values()
    )


def build_manifest(
    *,
    aux_candidates_path: str,
    source_evidence_paths: list[str],
    eval_manifest_hashes_path: str | None,
    completion_ready_requested: bool,
    allow_template: bool,
) -> dict[str, Any]:
    aux_candidates = _load_json(aux_candidates_path)
    selected = _selected_candidates(aux_candidates)
    source_evidence = _load_source_evidence(source_evidence_paths)
    missing = sorted(set(selected) - set(source_evidence))
    extra = sorted(set(source_evidence) - set(selected))
    if missing and not allow_template:
        raise SystemExit(f"missing materialized source evidence for selected aux ids: {', '.join(missing)}")
    if extra:
        raise SystemExit(f"source evidence contains ids not selected in candidate plan: {', '.join(extra)}")

    sources = []
    for source_id, candidate in sorted(selected.items()):
        if source_id in source_evidence:
            sources.append(_normalize_source_row(source_id, candidate, source_evidence[source_id]))
        else:
            sources.append(_template_source_row(source_id, candidate))
    eval_hashes = _load_eval_hashes(eval_manifest_hashes_path, allow_template=allow_template)
    ready = _completion_ready(sources, eval_hashes, completion_ready_requested)
    return {
        "schema": "g005_aux_namespace_manifest.v1",
        "source_namespace": "d2e_aux",
        "completion_ready": ready,
        "selected_aux_source_ids": sorted(selected),
        "claim_boundary": {
            "no_aux_in_d2e_heldout": True,
            "no_d2e_aux_claim_before_d2e_only_gates": True,
            "summary": "This manifest only supports G005 completion when completion_ready=true and validate_g005_aux_completion.py passes.",
        },
        "training_policy": {
            "source_specific_action_heads": True,
            "namespace_rule": "All selected aux examples stay under outputs/aux/<dataset_id>/... and use source-specific action heads before D2E finetuning.",
        },
        "d2e_eval_manifests": {
            "same_as_d2e_only": all(row.get("same_hash") is True for row in eval_hashes.values()),
            "required_splits": list(REQUIRED_SPLITS),
            "splits": eval_hashes,
        },
        "aux_sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 aux namespace manifest from source/eval evidence.")
    parser.add_argument("--aux-candidates", default="artifacts/sources/aux_game_action_dataset_candidates.json")
    parser.add_argument("--source-evidence", action="append", default=[], help="JSON source evidence object/list; repeatable.")
    parser.add_argument("--eval-manifest-hashes", help="JSON mapping/list for temporal, heldout_recording, heldout_game eval manifest hashes.")
    parser.add_argument("--output", default="artifacts/aux/g005_aux_namespace_manifest.json")
    parser.add_argument("--completion-ready", action="store_true", help="Request completion_ready=true; still fail-closed if evidence is incomplete.")
    parser.add_argument("--allow-template", action="store_true", help="Allow missing evidence and write a non-terminal template manifest.")
    args = parser.parse_args()
    payload = build_manifest(
        aux_candidates_path=args.aux_candidates,
        source_evidence_paths=args.source_evidence,
        eval_manifest_hashes_path=args.eval_manifest_hashes,
        completion_ready_requested=args.completion_ready,
        allow_template=args.allow_template,
    )
    _write_json(args.output, payload)
    print(
        "wrote {output} sources={sources} completion_ready={ready}".format(
            output=args.output,
            sources=len(payload["aux_sources"]),
            ready=payload["completion_ready"],
        )
    )
    if args.completion_ready and not payload["completion_ready"]:
        print("completion_ready requested but evidence was incomplete; wrote fail-closed manifest", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
