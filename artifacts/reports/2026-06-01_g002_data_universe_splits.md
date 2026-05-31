# G002 progress report — D2E universe and split manifests

**Date:** 2026-06-01 KST  
**Branch:** `research/fdm1-d2e-ultragoal`  
**Active OMX story:** `G002-d2e-universe-and-split-manifests`  
**Canonical roadmap:** `ROADMAP.md`

## Outcome

Built the FDM-1 reset data contract for D2E-480p as the primary training source. The contract reuses the existing all-game HF inventory as source input, fetches the pinned D2E-480p README for current published hours, records game categories and coarse pre-tokenization action/window statistics, and emits ROADMAP-required split manifests.

## Evidence created

- `src/fdm_d2e/data/fdm1_g002.py` — deterministic G002 metadata/split builders and validator.
- `scripts/build_fdm1_g002_data_contract.py` — reproducible builder for the G002 bundle and report.
- `artifacts/sources/fdm1_d2e_game_metadata.json` — 29-game taxonomy, published hours, recording counts, coarse decoded-event/50ms-window stats.
- `artifacts/sources/fdm1_d2e_recording_level_split_manifest.json` — Split A recording-level 80/10/10 manifest (`train=368`, `val=46`, `test=45`).
- `artifacts/sources/fdm1_d2e_heldout_game_split_manifest.json` — Split B category-coverage held-out games: Battlefield_6_Open_Beta, Minecraft_1.21.8, Stardew_Valley, MapleStory_Worlds_Southperry, PEAK.
- `artifacts/sources/fdm1_d2e_pseudo_label_split_manifest.json` — Split C pseudo-label simulation subsets (`D_IDM_LABELED_A=187`, `D_PSEUDO_B=107`, `D_FDM_GT_EVAL=74`).
- `artifacts/sources/fdm1_d2e_scale_split_manifest.json` — Split D nested 1/5/10/25/50/100% training scales.
- `artifacts/sources/fdm1_d2e_g002_validation.json` — `status=pass`, `error_count=0`.
- `docs/fdm1_d2e_g002_data_contract.md` — human-readable G002 report.
- `configs/data/fdm1_d2e_g002_contract.yaml` — path config for downstream G003+ work.

## Dataset pin

- Primary source: `open-world-agents/D2E-480p`.
- Pinned revision: `f075f7e25df6f6d385840a836f86bf92dfb877ff`.
- License: `cc-by-nc-4.0`.
- Inventory: 29 games, 459 paired 480p recordings, 918 paired 480p+Original recording variants in the existing all-game universe manifest.
- README hour note: the pinned README prose reports 268.7 hours while its per-game table sums to 269.8 hours; the manifest logs both and uses per-game table hours for scale allocation.

## Validation evidence

```text
uv run python scripts/build_fdm1_g002_data_contract.py
# built fdm1 g002 contract: status=pass games=29 recording_counts={'test': 45, 'train': 368, 'val': 46} heldout_games=['Battlefield_6_Open_Beta', 'MapleStory_Worlds_Southperry', 'Minecraft_1.21.8', 'PEAK', 'Stardew_Valley']

uv run python -m py_compile src/fdm_d2e/data/fdm1_g002.py scripts/build_fdm1_g002_data_contract.py
uv run pytest tests/test_fdm1_g002_contract.py tests/test_data_universe.py tests/test_split_contract.py -q
# 7 passed
python3 -m json.tool artifacts/sources/fdm1_d2e_g002_validation.json
python3 -m json.tool artifacts/sources/fdm1_d2e_g002_contract_bundle.json
```

## Claim boundary

G002 proves dataset pinning, game metadata, coarse pre-tokenization action/window statistics, and split-manifest construction. It does **not** prove G003 action-token correctness, model training, baseline improvement, 4xH200 scaling, stable harness execution, or FDM-1 parity.

## Next action

Start G003 by adapting/implementing the 50ms action-token materializer around these FDM-1 split manifests: fixed K slots, keyboard press/release, binned mouse movement, mouse buttons, scroll, click-position auxiliary target, detokenization, overflow accounting, and alignment checks.
