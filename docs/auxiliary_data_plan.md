# Auxiliary Game-Action Dataset Plan (G005 preparation)

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

- 5TiB source/cache budget expressed as `5120` GiB.
- D2E source estimate: `1881.96` GiB.
- Selected Tier-A aux source files: `16.876` GiB.
- D2E + selected Tier-A: `1898.836` GiB; fits budget: `True`.
- If the high-value VPT/BASALT candidate passes license review, D2E + Tier-A + VPT/BASALT
  source estimate: `2457.836` GiB; fits source budget: `True`.
- Decoded frame caches and checkpoints must still be staged/garbage-collected in G005.

## Candidate table

| Candidate | Status | License ID | Size GiB | Domain | G005 role |
| --- | --- | --- | ---: | --- | --- |
| minerl_2019_zenodo_v2 | selected_candidate | mit-license | 7.983 | Minecraft human demonstrations | high-transfer first-person keyboard/mouse-like auxiliary pretraining before D2E finetune/ablation |
| atari_head_zenodo_v4 | selected_candidate | cc-by-4.0 | 8.127 | Atari human gameplay demonstrations | small human-demonstration discrete-control auxiliary ablation and low-cost overfitting/control experiment |
| p_doom_atari_breakout_hf | selected_candidate | cc0-1.0 | 0.766 | Atari Breakout agent gameplay/world-model data | tiny permissive adapter/prototype corpus and negative/low-transfer control for aux scaling curves |
| openai_vpt_basalt_2022 | high_value_review_required_not_selected | review_required | 559.000 | Minecraft contractor demonstrations | best high-transfer candidate after license/provenance review; not part of current selected working set |

## Recommended G005 training curriculum

1. Build source-specific loaders under separate namespaces (`outputs/aux/<dataset_id>/...`).
2. Pretrain a shared visual-temporal backbone with source-specific action heads/tokens.
3. Finetune on D2E-only train split; keep D2E eval manifests unchanged.
4. Report D2E-only vs D2E+aux ablations for temporal, heldout-recording, and heldout-game
   splits, including non-significant or negative results.
5. If VPT/BASALT license review passes, treat it as the highest-transfer Minecraft
   candidate; otherwise keep it excluded.


## G005 completion audit

Before checkpointing `G005-aux-data-best-model` complete, run:

```bash
uv run python scripts/validate_g005_aux_completion.py
```

During preparation this may be run with `--allow-fail`, but a terminal G005
checkpoint requires `artifacts/aux/g005_aux_completion_audit.json` to report
`status == pass`. The audit checks G003/G004 prerequisite goal state, selected
aux provenance/storage policy, separated aux namespaces, D2E-only vs D2E+aux
ablation coverage on temporal/heldout-recording/heldout-game splits, no aux
leakage into D2E heldout data, target split tags, prediction coverage, and run
evidence.

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
