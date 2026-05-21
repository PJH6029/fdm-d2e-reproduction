#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import stable_hash_json, write_json


DEFAULT_OUTPUT = "artifacts/aux/g005_aux_action_registry.json"
DEFAULT_AUX_CANDIDATES = "artifacts/sources/aux_game_action_dataset_candidates.json"


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selected_candidates(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in payload.get("candidates", [])
        if isinstance(row, dict) and row.get("id") and row.get("selection_status") == "selected_candidate"
    }


def _domain_family(candidate: dict[str, Any]) -> str:
    domain = str(candidate.get("domain") or "").lower()
    source_id = str(candidate.get("id") or "").lower()
    name = str(candidate.get("name") or "").lower()
    text = " ".join([domain, source_id, name])
    if "minecraft" in text or "minerl" in text:
        return "minecraft_keyboard_mouse"
    if "atari" in text or "breakout" in text:
        return "atari_discrete"
    return "source_specific"


def _head_spec(source_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
    family = _domain_family({**candidate, "id": source_id})
    if family == "minecraft_keyboard_mouse":
        controls = [
            {"name": "camera_delta_x", "kind": "continuous_axis", "units": "degrees_or_source_native"},
            {"name": "camera_delta_y", "kind": "continuous_axis", "units": "degrees_or_source_native"},
            {"name": "buttons", "kind": "multi_binary", "examples": ["attack", "forward", "back", "left", "right", "jump", "sneak", "sprint"]},
        ]
        adapter = "minerl_action_dict_adapter"
        transfer_role = "high_transfer_first_person_keyboard_mouse_like_pretraining"
    elif family == "atari_discrete":
        controls = [{"name": "atari_action_id", "kind": "categorical", "enum_source": "source_specific_action_enums_or_dataset_card"}]
        adapter = "atari_discrete_action_adapter"
        transfer_role = "discrete_control_auxiliary_or_negative_transfer_control"
    else:
        controls = [{"name": "source_action_token", "kind": "source_specific"}]
        adapter = "custom_source_action_adapter"
        transfer_role = "source_specific_auxiliary_control"
    return {
        "id": source_id,
        "namespace": source_id,
        "type": family,
        "action_space_family": family,
        "adapter": adapter,
        "transfer_role": transfer_role,
        "source_url": candidate.get("source_url"),
        "license_id": candidate.get("license_id"),
        "supervision": candidate.get("supervision"),
        "action_signal": candidate.get("action_signal"),
        "controls": controls,
        "d2e_endpoint_claims_allowed": [],
        "claim_boundary": "This auxiliary head may train a shared backbone or source-specific head only; D2E keyboard/mouse endpoint claims require D2E eval evidence after finetuning.",
        "registry_fingerprint": stable_hash_json({"id": source_id, "family": family, "controls": controls, "source_url": candidate.get("source_url")}),
    }


def build_registry(args: argparse.Namespace) -> dict[str, Any]:
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
    heads = [_head_spec(source_id, candidate) for source_id, candidate in sorted(selected.items())]
    return {
        "schema": "g005_aux_action_registry.v1",
        "status": "pass",
        "root": str(root),
        "selected_aux_source_ids": sorted(selected),
        "source_specific_action_heads": True,
        "no_cross_source_action_collapse": True,
        "d2e_endpoint_claim_boundary": {
            "no_aux_source_directly_claims_d2e_keyboard_mouse": True,
            "d2e_claims_require_d2e_finetune_and_eval": True,
            "summary": "Auxiliary action spaces remain source-specific. Only D2E keyboard/mouse endpoints can support D2E desktop-action claims after D2E finetune/eval.",
        },
        "action_heads": heads,
        "claim_boundary": "Action registry only; it does not materialize sources, train G005, checkpoint goals, or prove D2E+aux model quality.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a source-specific action-head registry for selected G005 auxiliary datasets.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--aux-candidates", default=DEFAULT_AUX_CANDIDATES)
    parser.add_argument("--source-id", action="append", help="Selected source id to include; repeatable. Defaults to all selected candidates.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_registry(args)
    write_json(Path(args.root).resolve() / args.output, payload)
    print(f"g005 aux action registry: status={payload['status']} heads={len(payload['action_heads'])} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
