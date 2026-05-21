#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.data.splits import build_generalization_split_contract
from fdm_d2e.io_utils import read_json, write_json


def write_report(path: Path, contract: dict[str, Any]) -> None:
    leak = contract["leakage_report"]
    rec = contract["manifests"]["heldout_recording"]["splits"]
    game = contract["manifests"]["heldout_game"]["splits"]
    temporal_count = len(contract["manifests"]["temporal"]["splits"]["recordings"])
    lines = [
        "# D2E Full Generalization Split Contract",
        "",
        f"- Dataset fingerprint: `{contract['dataset_fingerprint']}`",
        f"- Source recording groups: `{contract['source_recording_groups']}`",
        f"- Recording variants: `{contract['recording_variants']}`",
        f"- Games: `{len(contract['games'])}`",
        f"- Leakage status: `{leak['status']}`",
        "",
        "## Split counts",
        "",
        f"- Temporal groups with prefix/tail policy: `{temporal_count}`",
        f"- Heldout-recording train rows: `{len(rec['train'])}`",
        f"- Heldout-recording heldout rows: `{len(rec['heldout_recording'])}`",
        f"- Heldout-recording groups: `{len(rec['heldout_recording_keys'])}`",
        f"- Heldout-game train rows: `{len(game['train'])}`",
        f"- Heldout-game heldout rows: `{len(game['heldout_game'])}`",
        f"- Heldout games: `{', '.join(game['heldout_games'])}`",
        "",
        "## Leakage checks",
        "",
        "| Check | Pass |",
        "| --- | --- |",
    ]
    for name, passed in leak["checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(
        [
            "",
            "## Contract boundary",
            "",
            "This contract assigns recording/resolution variants and policies. It does not decode frames or create training windows; downstream ingestion must apply the temporal prefix/tail policy at window construction time.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe generalization split manifests from a data-universe manifest.")
    parser.add_argument("--manifest", default="artifacts/sources/d2e_full_data_universe_manifest.json")
    parser.add_argument("--output-dir", default="artifacts/sources")
    parser.add_argument("--report", default="docs/d2e_full_split_contract.md")
    parser.add_argument("--temporal-train-fraction", type=float, default=0.8)
    parser.add_argument("--heldout-recording-fraction", type=float, default=0.2)
    parser.add_argument("--heldout-game-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default="fdm-d2e-full-v1")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    contract = build_generalization_split_contract(
        manifest,
        temporal_train_fraction=args.temporal_train_fraction,
        heldout_recording_fraction=args.heldout_recording_fraction,
        heldout_game_fraction=args.heldout_game_fraction,
        seed=args.seed,
    )
    out = Path(args.output_dir)
    write_json(out / "d2e_full_split_contract.json", contract)
    write_json(out / "d2e_full_temporal_split_manifest.json", contract["manifests"]["temporal"])
    write_json(out / "d2e_full_heldout_recording_split_manifest.json", contract["manifests"]["heldout_recording"])
    write_json(out / "d2e_full_heldout_game_split_manifest.json", contract["manifests"]["heldout_game"])
    write_json(out / "d2e_full_split_leakage_report.json", contract["leakage_report"])
    write_report(Path(args.report), contract)
    print(
        "built split contract: "
        f"groups={contract['source_recording_groups']} variants={contract['recording_variants']} "
        f"games={len(contract['games'])} leakage={contract['leakage_report']['status']}"
    )


if __name__ == "__main__":
    main()
