#!/usr/bin/env python3
"""Build the G005 auxiliary gameplay/action dataset candidate plan.

This script is intentionally deterministic: it records source facts that were
manually verified from official source pages/API metadata, then writes the
machine-readable artifact and a Markdown handoff document. It does not download
auxiliary datasets or start training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GIB = 1024 ** 3
BUDGET_TIB = 5.0
D2E_SOURCE_GIB = 1881.96


def _gib(size_bytes: int) -> float:
    return round(size_bytes / GIB, 3)


CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "minerl_2019_zenodo_v2",
        "name": "MineRL 2019 Datasets backup",
        "domain": "Minecraft human demonstrations",
        "source_url": "https://zenodo.org/records/12659939",
        "metadata_api_url": "https://zenodo.org/api/records/12659939",
        "doi": "10.5281/zenodo.12659939",
        "official_project_url": "https://github.com/minerllabs/minerl",
        "publication_date": "2024-07-05",
        "source_revision_or_record": "zenodo_record_12659939",
        "size_bytes": 8_571_175_338,
        "published_size_note": "Zenodo files total 8.6 GB decimal across four MineRL-v0 zips.",
        "license": "MIT License",
        "license_id": "mit-license",
        "license_status": "usable_with_attribution_review",
        "selection_status": "selected_candidate",
        "valid_for_training_now": True,
        "supervision": "human players playing Minecraft; video feed and actions captured/stored",
        "action_signal": "state-action demonstrations with Minecraft camera/button/control fields via MineRL data tooling",
        "adapter_strategy": "Use a Minecraft-specific action head and map camera/buttons into a provenance-preserving aux token namespace; do not collapse into D2E keyboard/mouse metrics.",
        "g005_role": "high-transfer first-person keyboard/mouse-like auxiliary pretraining before D2E finetune/ablation",
        "why": "Closest small, license-permissive public game/action corpus among selected candidates; complements D2E with 3D first-person control.",
        "risks": [
            "MineRL-v0 environment/action schema differs from D2E desktop MCAP keyboard/mouse events",
            "needs adapter/token mapping and source-specific heads",
            "Minecraft IP/provenance notices must remain separate from D2E artifacts",
        ],
        "source_evidence": [
            "Zenodo description says the record backs up parts of MineRL 2019 and contains human Minecraft play with video feed and actions captured/stored.",
            "MineRL GitHub README links the Zenodo backup after original mirrors went down and describes data access for Minecraft agents.",
            "Zenodo API metadata reports license id mit-license and total file bytes 8571175338.",
        ],
    },
    {
        "id": "atari_head_zenodo_v4",
        "name": "Atari-HEAD human eye-tracking and demonstration dataset",
        "domain": "Atari human gameplay demonstrations",
        "source_url": "https://zenodo.org/records/3451402",
        "metadata_api_url": "https://zenodo.org/api/records/3451402",
        "doi": "10.5281/zenodo.3451402",
        "official_project_url": "https://arxiv.org/abs/1903.06754",
        "publication_date": "2019-09-19",
        "source_revision_or_record": "zenodo_record_3451402_v4",
        "size_bytes": 8_726_806_887,
        "published_size_note": "Zenodo files total 8.7 GB decimal and include action_enums.txt plus per-game zips.",
        "license": "Creative Commons Attribution 4.0 International",
        "license_id": "cc-by-4.0",
        "license_status": "usable_with_attribution_required",
        "selection_status": "selected_candidate",
        "valid_for_training_now": True,
        "supervision": "per-frame Atari action demonstrations with gaze/reward metadata",
        "action_signal": "ALE action integer/action enum per frame; gaze fields optional and not required for IDM/FDM action training",
        "adapter_strategy": "Use a discrete Atari action head and keep Atari metrics separated from D2E mouse/keyboard endpoint claims.",
        "g005_role": "small human-demonstration discrete-control auxiliary ablation and low-cost overfitting/control experiment",
        "why": "Human gameplay with explicit actions across 20 Atari games; useful for testing whether game-action pretraining helps heldout-game generalization.",
        "risks": [
            "Atari observations/actions are lower dimensional than D2E desktop video",
            "not keyboard/mouse desktop control",
            "CC-BY attribution requirements must be propagated in artifact metadata",
        ],
        "source_evidence": [
            "Zenodo v4 metadata includes action_enums.txt and per-game zips with cc-by-4.0 license id.",
            "Atari-HEAD paper reports 117 gameplay hours, 20 games, 8M action demonstrations, and 328M gaze samples.",
        ],
    },
    {
        "id": "p_doom_atari_breakout_hf",
        "name": "p-doom Atari-Breakout action-conditioned frames",
        "domain": "Atari Breakout agent gameplay/world-model data",
        "source_url": "https://huggingface.co/datasets/p-doom/atari-breakout-dataset",
        "metadata_api_url": "https://huggingface.co/api/datasets/p-doom/atari-breakout-dataset",
        "doi": None,
        "official_project_url": "https://github.com/p-doom/jasmine/tree/main/data/jasmine_data",
        "publication_date": "2025-10-31",
        "source_revision_or_record": "hf_sha_34c60197acabf0011a36affff183cc87c755edc4",
        "size_bytes": 823_000_000,
        "published_size_note": "Hugging Face file tree reports approximately 823 MB.",
        "license": "CC0 1.0",
        "license_id": "cc0-1.0",
        "license_status": "usable_public_domain_dedication",
        "selection_status": "selected_candidate",
        "valid_for_training_now": True,
        "supervision": "10M 84x84 frames and actions collected from Atari Breakout agent training",
        "action_signal": "action-conditioned video prediction records with train/val/test splits",
        "adapter_strategy": "Use only for discrete-action data-loader/provenance tests and a small auxiliary scaling control; do not treat as human demonstration evidence.",
        "g005_role": "tiny permissive adapter/prototype corpus and negative/low-transfer control for aux scaling curves",
        "why": "Very small, permissive, and already split; useful to validate auxiliary ingestion without consuming cluster storage.",
        "risks": [
            "agent-generated rather than human demonstrations",
            "single game only",
            "ArrayRecord/Grain reader integration needed before training",
        ],
        "source_evidence": [
            "Hugging Face dataset card lists license cc0-1.0, 10M frames/actions, 84x84 resolution, and train/val/test splits.",
            "Hugging Face API reports dataset sha 34c60197acabf0011a36affff183cc87c755edc4 and card license cc0-1.0.",
        ],
    },
    {
        "id": "openai_vpt_basalt_2022",
        "name": "OpenAI VPT BASALT 2022 contractor demonstrations",
        "domain": "Minecraft contractor demonstrations",
        "source_url": "https://github.com/openai/Video-Pre-Training",
        "metadata_api_url": None,
        "doi": None,
        "official_project_url": "https://github.com/openai/Video-Pre-Training",
        "publication_date": "2022-07-26",
        "source_revision_or_record": "openai_vpt_repository_dataset_index_links",
        "size_bytes": None,
        "estimated_size_gib": 559.0,
        "published_size_note": "Repository README states around 150 GB per BASALT task; four task indexes imply roughly 600 GB decimal before cache expansion.",
        "license": "repository code is MIT; data/license provenance requires explicit review before training use",
        "license_id": "review_required",
        "license_status": "license_provenance_review_required",
        "selection_status": "high_value_review_required_not_selected",
        "valid_for_training_now": False,
        "supervision": "MP4 recordings with associated JSONL action files/indexes for BASALT tasks",
        "action_signal": "keyboard/mouse-like Minecraft action JSONL paired with video demonstrations",
        "adapter_strategy": "If license review passes, use as the highest-priority Minecraft aux corpus with source-specific tokenizer and D2E finetune ablation.",
        "g005_role": "best high-transfer candidate after license/provenance review; not part of current selected working set",
        "why": "Closest public analog to VPT/FDM-style IDM labeling and behavioral cloning, with video+action contractor demonstrations.",
        "risks": [
            "dataset terms are not as clearly packaged as Zenodo/HF candidates",
            "around 150 GB per task and larger cache/checkpoint footprint",
            "README notes action JSONL stripping/caveats for BASALT competition context",
            "Minecraft IP disclaimer must be preserved",
        ],
        "source_evidence": [
            "OpenAI VPT README documents IDM demo videos/actions and BASALT 2022 demonstration datasets around 150GB per task.",
            "OpenAI VPT README says repository code is MIT but also includes a Minecraft IP disclaimer; dataset rights require separate review.",
        ],
    },
]


def build_payload() -> dict[str, Any]:
    selected = [c for c in CANDIDATES if c["selection_status"] == "selected_candidate"]
    review = [c for c in CANDIDATES if c["selection_status"] == "high_value_review_required_not_selected"]
    for candidate in CANDIDATES:
        if candidate.get("size_bytes") is not None:
            candidate["size_gib"] = _gib(int(candidate["size_bytes"]))
        else:
            candidate["size_gib"] = float(candidate["estimated_size_gib"])

    selected_total_gib = round(sum(float(c["size_gib"]) for c in selected), 3)
    review_total_gib = round(sum(float(c["size_gib"]) for c in review), 3)
    budget_gib = BUDGET_TIB * 1024
    return {
        "schema": "aux_game_action_dataset_candidates.v1",
        "generated_at_utc": "deterministic-manifest-no-wall-clock",
        "purpose": "G005 preparation only: public game/action datasets that may be mixed with D2E after D2E-only hard gates finish.",
        "user_decision": {
            "d2e_aux_may_be_primary": True,
            "interpretation": "D2E+aux may be the best/primary final model, but D2E-only gates and D2E-only vs D2E+aux ablations remain mandatory and separately reported.",
        },
        "claim_boundary": {
            "no_d2e_aux_claim_before_d2e_only_gates": True,
            "no_training_started_by_this_artifact": True,
            "no_live_harness_claim_by_this_artifact": True,
            "summary": "This is a source-selection and storage plan, not model-quality or harness evidence.",
        },
        "storage_policy": {
            "cap_tib": BUDGET_TIB,
            "cap_gib": budget_gib,
            "d2e_source_estimate_gib": D2E_SOURCE_GIB,
            "selected_candidate_total_gib": selected_total_gib,
            "selected_plus_d2e_gib": round(D2E_SOURCE_GIB + selected_total_gib, 3),
            "fits_cap_with_selected_candidates": D2E_SOURCE_GIB + selected_total_gib <= budget_gib,
            "review_required_candidate_total_gib": review_total_gib,
            "selected_plus_review_plus_d2e_gib": round(D2E_SOURCE_GIB + selected_total_gib + review_total_gib, 3),
            "fits_cap_if_high_value_review_passes": D2E_SOURCE_GIB + selected_total_gib + review_total_gib <= budget_gib,
            "cache_warning": "Source files fit the 5TiB cap, but decoded caches/checkpoints must be staged and garbage-collected during G005.",
        },
        "training_policy": {
            "artifact_namespaces": [
                "outputs/aux/<dataset_id>/...",
                "outputs/idm_aux/<run_id>/...",
                "outputs/fdm_aux/<run_id>/...",
                "artifacts/aux/<run_id>/...",
            ],
            "recommended_curriculum": [
                "Validate each selected aux loader with source-specific train/val/test split and provenance hash.",
                "Pretrain shared visual-temporal backbone on selected aux with source-specific action heads.",
                "Finetune on D2E-only train split; never use aux examples inside D2E heldout splits.",
                "Compare D2E-only vs D2E+aux under the same temporal, heldout-recording, and heldout-game D2E eval manifests.",
            ],
            "action_mapping_rule": "Keep source-specific action heads/tokens and report source-specific metrics; only D2E keyboard/mouse endpoints can support D2E desktop-action claims.",
        },
        "candidates": CANDIDATES,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = []
    for c in payload["candidates"]:
        rows.append(
            "| {id} | {status} | {license_id} | {size:.3f} | {domain} | {role} |".format(
                id=c["id"],
                status=c["selection_status"],
                license_id=c["license_id"],
                size=float(c["size_gib"]),
                domain=c["domain"],
                role=c["g005_role"].replace("|", "/"),
            )
        )
    selected = payload["storage_policy"]["selected_candidate_total_gib"]
    selected_plus = payload["storage_policy"]["selected_plus_d2e_gib"]
    review_plus = payload["storage_policy"]["selected_plus_review_plus_d2e_gib"]
    text = f"""# Auxiliary Game-Action Dataset Plan (G005 preparation)

This document records the current G005 auxiliary-data decision after the user answered
`d2e_aux_may_be_primary`: **D2E+aux may become the best/primary final model**, but
**D2E-only hard gates remain mandatory**. This plan does not start training, does not
checkpoint G005, and does not weaken the G003/G004 D2E-only requirements.

Machine-readable artifact: `artifacts/sources/aux_game_action_dataset_candidates.json`.

## Claim boundary

- No D2E+aux result may be claimed until G003 and G004 finish and the same D2E eval
  manifests are used for D2E-only vs D2E+aux comparison.
- No auxiliary source may be mixed into D2E heldout recordings/games.
- Source-specific action heads are required for non-D2E action spaces; Atari metrics do
  not support D2E keyboard/mouse endpoint claims.
- This artifact is source-selection/storage evidence only, not model-quality evidence.

## Storage gate

- 5TiB source/cache budget expressed as `{payload['storage_policy']['cap_gib']:.0f}` GiB.
- D2E source estimate: `{payload['storage_policy']['d2e_source_estimate_gib']:.2f}` GiB.
- Selected Tier-A aux source files: `{selected:.3f}` GiB.
- D2E + selected Tier-A: `{selected_plus:.3f}` GiB; fits budget: `{payload['storage_policy']['fits_cap_with_selected_candidates']}`.
- If the high-value VPT/BASALT candidate passes license review, D2E + Tier-A + VPT/BASALT
  source estimate: `{review_plus:.3f}` GiB; fits source budget: `{payload['storage_policy']['fits_cap_if_high_value_review_passes']}`.
- Decoded frame caches and checkpoints must still be staged/garbage-collected in G005.

## Candidate table

| Candidate | Status | License ID | Size GiB | Domain | G005 role |
| --- | --- | --- | ---: | --- | --- |
""" + "\n".join(rows) + """

## Recommended G005 training curriculum

1. Build source-specific loaders under separate namespaces (`outputs/aux/<dataset_id>/...`).
2. Pretrain a shared visual-temporal backbone with source-specific action heads/tokens.
3. Finetune on D2E-only train split; keep D2E eval manifests unchanged.
4. Report D2E-only vs D2E+aux ablations for temporal, heldout-recording, and heldout-game
   splits, including non-significant or negative results.
5. If VPT/BASALT license review passes, treat it as the highest-transfer Minecraft
   candidate; otherwise keep it excluded.

## Source evidence summary

- MineRL 2019 Zenodo: backup of human Minecraft demonstrations with video feed and actions;
  Zenodo API reports `mit-license` and 8,571,175,338 bytes.
- MineRL GitHub: official package page links the Zenodo backup and documents Minecraft
  environments/data access.
- Atari-HEAD Zenodo/arXiv: v4 includes `action_enums.txt`, per-game zips, `cc-by-4.0`, and
  the paper reports 117 hours, 20 games, and 8M action demonstrations.
- p-doom Atari Breakout Hugging Face: dataset card reports `cc0-1.0`, 10M 84x84 frames and
  actions, and train/val/test splits.
- OpenAI VPT repository: documents IDM demos and BASALT 2022 video/action datasets around
  150GB per task; data/license provenance must be reviewed before selection.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build G005 auxiliary dataset candidate manifest and handoff doc.")
    parser.add_argument("--output", default="artifacts/sources/aux_game_action_dataset_candidates.json")
    parser.add_argument("--doc-output", default="docs/auxiliary_data_plan.md")
    args = parser.parse_args()
    payload = build_payload()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.doc_output))
    print(
        "wrote {output} candidates={candidates} selected_gib={selected}".format(
            output=output,
            candidates=len(payload["candidates"]),
            selected=payload["storage_policy"]["selected_candidate_total_gib"],
        )
    )
    print(f"wrote {args.doc_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
