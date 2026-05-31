#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.data.fdm1_g002 import (
    build_game_metadata,
    build_heldout_game_manifest,
    build_pseudo_label_split_manifest,
    build_recording_level_split_manifest,
    build_scale_split_manifest,
    fetch_text,
    readme_url_for_revision,
    validate_g002_contract,
)
from fdm_d2e.io_utils import read_json, sha256_file, stable_hash_json, write_json


def _source_revision(universe: dict[str, Any], source_id: str) -> str:
    for source in universe.get("d2e_sources", []):
        if source.get("source_id") == source_id:
            return str(source.get("resolved_revision") or source.get("requested_revision") or "main")
    return "main"


def _write_report(path: Path, *, bundle: dict[str, Any], validation: dict[str, Any]) -> None:
    universe = bundle["data_universe"]
    game_metadata = bundle["game_metadata"]
    recording = bundle["recording_level_split"]
    heldout = bundle["heldout_game_split"]
    pseudo = bundle["pseudo_label_split"]
    scale = bundle["scale_split"]
    source_480p = next((s for s in universe.get("d2e_sources", []) if s.get("source_id") == "d2e_480p"), {})
    lines = [
        "# G002 FDM-1/D2E data universe and split contract",
        "",
        "**Canonical roadmap:** `ROADMAP.md`.",
        "",
        "## Dataset pin",
        "",
        f"- Primary dataset: `open-world-agents/D2E-480p`",
        f"- Pinned revision: `{source_480p.get('resolved_revision')}`",
        f"- License: `{source_480p.get('license')}`",
        f"- Games: `{universe.get('coverage', {}).get('games_count')}`",
        f"- 480p recording pairs: `{source_480p.get('paired_recordings')}`",
        f"- Published D2E-480p README summary hours: `{game_metadata.get('totals', {}).get('published_summary_hours_from_readme_text')}`",
        f"- Published D2E-480p per-game table hour sum: `{game_metadata.get('totals', {}).get('published_hours_from_readme_table_sum')}`",
        f"- Dataset fingerprint: `{universe.get('dataset_fingerprint')}`",
        "",
        "## Split artifacts",
        "",
        "| Split | Manifest | Counts |",
        "| --- | --- | --- |",
        f"| Recording-level in-distribution 80/10/10 | `artifacts/sources/fdm1_d2e_recording_level_split_manifest.json` | `{json.dumps(recording.get('counts', {}), sort_keys=True)}` |",
        f"| Held-out game category coverage | `artifacts/sources/fdm1_d2e_heldout_game_split_manifest.json` | `{json.dumps(heldout.get('counts', {}), sort_keys=True)}` |",
        f"| Pseudo-label simulation | `artifacts/sources/fdm1_d2e_pseudo_label_split_manifest.json` | `{json.dumps(pseudo.get('counts', {}), sort_keys=True)}` |",
        f"| Data scale | `artifacts/sources/fdm1_d2e_scale_split_manifest.json` | `{', '.join(scale.get('scales', {}).keys())}` |",
        "",
        "## Held-out game roles",
        "",
        "| Role | Game |",
        "| --- | --- |",
    ]
    for role, game in heldout.get("heldout_selection", {}).get("selected_by_role", {}).items():
        lines.append(f"| `{role}` | `{game}` |")
    lines.extend([
        "",
        "## Per-game metadata and coarse action statistics",
        "",
        "| Game | Category | Tags | README hours | 480p recs | Decoded events | 50ms windows | Events/hour |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in game_metadata.get("games", []):
        stats = row.get("coarse_action_statistics", {})
        lines.append(
            "| {game} | {cat} | {tags} | {hours} | {recs} | {events} | {windows} | {eph} |".format(
                game=row["game"],
                cat=row["primary_category"],
                tags=", ".join(row.get("tags", [])),
                hours=row.get("published_hours_480p_readme"),
                recs=row.get("recording_count_480p"),
                events=stats.get("decoded_events"),
                windows=stats.get("window_records_50ms"),
                eph=stats.get("decoded_events_per_hour"),
            )
        )
    lines.extend([
        "",
        "## Validation",
        "",
        f"- Status: `{validation['status']}`",
        f"- Error count: `{validation['error_count']}`",
        f"- Findings: `{json.dumps(validation.get('findings', []), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Claim boundary",
        "",
        "G002 proves only dataset pinning, all-game inventory, coarse pre-tokenization action/window statistics, and leakage-safe split manifests. It does not prove decoding correctness, action-tokenization correctness, model training, baseline wins, harness stability, or FDM-1 parity.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FDM-1 reset G002 data universe/split bundle from the canonical D2E manifest.")
    parser.add_argument("--universe", default="artifacts/sources/d2e_full_data_universe_manifest.json")
    parser.add_argument("--decode-summary", default="artifacts/sources/d2e_full_corpus_decode_summary.json")
    parser.add_argument("--output-dir", default="artifacts/sources")
    parser.add_argument("--report", default="docs/fdm1_d2e_g002_data_contract.md")
    parser.add_argument("--readme", default=None, help="Optional local D2E-480p README. If omitted, fetches from the pinned HF revision.")
    parser.add_argument("--primary-source-id", default="d2e_480p")
    args = parser.parse_args()

    universe = read_json(args.universe)
    decode_summary = read_json(args.decode_summary) if Path(args.decode_summary).exists() else None
    revision = _source_revision(universe, args.primary_source_id)
    if args.readme:
        readme_text = Path(args.readme).read_text(encoding="utf-8")
        readme_url = str(Path(args.readme))
    else:
        readme_url = readme_url_for_revision(revision)
        readme_text = fetch_text(readme_url)

    game_metadata = build_game_metadata(
        universe,
        readme_text=readme_text,
        readme_revision=revision,
        readme_url=readme_url,
        decode_summary=decode_summary,
        primary_source_id=args.primary_source_id,
    )
    recording = build_recording_level_split_manifest(universe, primary_source_id=args.primary_source_id)
    heldout = build_heldout_game_manifest(universe, game_metadata, primary_source_id=args.primary_source_id)
    pseudo = build_pseudo_label_split_manifest(recording)
    scale = build_scale_split_manifest(recording, game_metadata)
    bundle = {
        "schema": "fdm1_g002_data_contract_bundle.v1",
        "canonical_roadmap": "ROADMAP.md",
        "source_paths": {
            "data_universe": args.universe,
            "decode_summary": args.decode_summary if Path(args.decode_summary).exists() else None,
        },
        "source_hashes": {
            "data_universe_sha256": sha256_file(args.universe),
            "decode_summary_sha256": sha256_file(args.decode_summary) if Path(args.decode_summary).exists() else None,
        },
        "data_universe": universe,
        "game_metadata": game_metadata,
        "recording_level_split": recording,
        "heldout_game_split": heldout,
        "pseudo_label_split": pseudo,
        "scale_split": scale,
    }
    validation = validate_g002_contract(bundle)
    bundle["validation"] = validation
    bundle["fingerprint"] = stable_hash_json({k: v for k, v in bundle.items() if k not in {"data_universe", "fingerprint"}})

    out = Path(args.output_dir)
    write_json(out / "fdm1_d2e_game_metadata.json", game_metadata)
    write_json(out / "fdm1_d2e_recording_level_split_manifest.json", recording)
    write_json(out / "fdm1_d2e_heldout_game_split_manifest.json", heldout)
    write_json(out / "fdm1_d2e_pseudo_label_split_manifest.json", pseudo)
    write_json(out / "fdm1_d2e_scale_split_manifest.json", scale)
    # Omit the full universe from the compact bundle copy to avoid duplicating the 1.4MB manifest.
    compact_bundle = {k: v for k, v in bundle.items() if k != "data_universe"}
    write_json(out / "fdm1_d2e_g002_contract_bundle.json", compact_bundle)
    write_json(out / "fdm1_d2e_g002_validation.json", validation)
    _write_report(Path(args.report), bundle=bundle, validation=validation)
    print(
        "built fdm1 g002 contract: "
        f"status={validation['status']} games={game_metadata['totals']['games']} "
        f"recording_counts={recording['counts']} heldout_games={heldout['heldout_selection']['heldout_games']}"
    )
    if validation["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
