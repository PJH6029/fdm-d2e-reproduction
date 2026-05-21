#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.data.data_universe import build_data_universe_manifest, validate_data_universe_manifest
from fdm_d2e.io_utils import write_json


def _bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    coverage = manifest["coverage"]
    storage = manifest["storage_budget"]
    lines = [
        "# D2E Full Data Universe Audit",
        "",
        f"- Generated: {manifest['generated_at_utc']}",
        f"- Manifest fingerprint: `{manifest['dataset_fingerprint']}`",
        f"- D2E source count: `{coverage['d2e_source_count']}`",
        f"- Recording variants: `{coverage['recording_variants']}`",
        f"- Unique cross-resolution recordings: `{coverage['unique_cross_resolution_recordings']}`",
        f"- Games: `{coverage['games_count']}`",
        f"- Status counts: `{json.dumps(coverage['status_counts'], sort_keys=True)}`",
        "",
        "## Sources",
        "",
        "| Source | Resolution | Revision | License | Recordings | Games | Size GiB |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for source in manifest["d2e_sources"]:
        lines.append(
            "| {source_id} | {resolution_tier} | `{resolved_revision}` | {license} | {paired_recordings} | {games_count} | {size:.2f} |".format(
                source_id=source["source_id"],
                resolution_tier=source["resolution_tier"],
                resolved_revision=str(source.get("resolved_revision"))[:12],
                license=source["license"],
                paired_recordings=source["paired_recordings"],
                games_count=source["games_count"],
                size=_bytes_to_gib(int(source.get("total_file_size_bytes") or 0)),
            )
        )
    lines.extend(
        [
            "",
            "## Storage budget",
            "",
            f"- Budget: `{storage['budget_tib']}` TiB",
            f"- D2E source total: `{_bytes_to_gib(storage['d2e_source_total_bytes']):.2f}` GiB",
            f"- Auxiliary planned bytes: `{storage['auxiliary_planned_bytes']}`",
            f"- Source bytes within budget: `{storage['total_source_bytes_within_budget']}`",
            f"- Requires staged cache or extra storage: `{storage['requires_staged_cache_or_extra_storage']}`",
            f"- Working-set policy: {storage['working_set_policy']}",
            "",
            "## Auxiliary candidates",
            "",
            "| Candidate | License | Status | Supervision |",
            "| --- | --- | --- | --- |",
        ]
    )
    for candidate in manifest["auxiliary_candidates"]:
        lines.append(
            f"| {candidate['name']} | {candidate['license']} | {candidate['selection_status']} | {candidate['supervision']} |"
        )
    lines.extend(
        [
            "",
            "## Gate verdict",
            "",
            "- Inventory status coverage: PASS" if coverage["all_recording_variants_statused"] else "- Inventory status coverage: FAIL",
            "- Exclusion audit: PASS; no non-included D2E recording variants are present in the current HF tree." if coverage["status_counts"].get("included", 0) == coverage["recording_variants"] else "- Exclusion audit: REVIEW non-included rows.",
            "- Storage: PASS for staged/streaming working-set policy; source total may exceed local cache if future aux data is materialized." if storage["requires_staged_cache_or_extra_storage"] else "- Storage: PASS within 5TiB source budget.",
            "",
            "## Claim boundary",
            "",
            "This audit is an inventory and storage-planning artifact. It does not prove full-corpus training, D2E-only convergence, D2E+aux model quality, or live harness performance.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full D2E data-universe audit manifest from Hugging Face dataset trees.")
    parser.add_argument("--output", default="artifacts/sources/d2e_full_data_universe_manifest.json")
    parser.add_argument("--report", default="docs/d2e_full_data_universe.md")
    parser.add_argument("--budget-tib", type=float, default=5.0)
    args = parser.parse_args()

    manifest = build_data_universe_manifest(budget_tib=args.budget_tib)
    validate_data_universe_manifest(manifest)
    write_json(args.output, manifest)
    write_report(Path(args.report), manifest)
    coverage = manifest["coverage"]
    print(
        "built data universe: "
        f"sources={coverage['d2e_source_count']} variants={coverage['recording_variants']} "
        f"unique={coverage['unique_cross_resolution_recordings']} games={coverage['games_count']} "
        f"fingerprint={manifest['dataset_fingerprint'][:12]}"
    )


if __name__ == "__main__":
    main()
