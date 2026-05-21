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

## G005 namespace manifest requirement

The terminal G005 checkpoint must include `artifacts/aux/g005_aux_namespace_manifest.json`
with schema `g005_aux_namespace_manifest.v1`. The manifest must prove:

- every selected auxiliary dataset is materialized under `outputs/aux/<dataset_id>/...`;
- every selected source records `source_url`, `license_id`, `provenance_sha256`,
  source-specific split hashes, and `d2e_heldout_overlap_count == 0`;
- every non-D2E action space keeps a source-specific `action_head` namespace;
- temporal, heldout-recording, and heldout-game D2E eval manifests are byte-identical
  between D2E-only and D2E+aux ablations; and
- `completion_ready == true` only after G003/G004 D2E-only gates are complete and the
  above evidence is populated.

Build the manifest from explicit materialization/eval evidence rather than editing it by hand:

```bash
uv run python scripts/materialize_g005_aux_sources.py \
  --output artifacts/aux/g005_aux_materialization_plan.json
# Review the plan first. Add --execute only when the selected aux downloads should
# be materialized into outputs/aux/<dataset_id>/raw plus source split manifests.
uv run python scripts/materialize_g005_aux_sources.py \
  --execute \
  --output artifacts/aux/g005_aux_materialization_plan.json
uv run python scripts/build_g005_aux_action_registry.py \
  --output artifacts/aux/g005_aux_action_registry.json
uv run python scripts/build_g005_aux_archive_inventory.py \
  --output artifacts/aux/g005_aux_archive_inventory.json
uv run python scripts/build_g005_eval_manifest_hashes.py \
  --output artifacts/aux/d2e_eval_manifest_hashes.json
uv run python scripts/build_g005_aux_source_evidence.py \
  --output artifacts/aux/g005_aux_source_materialization_evidence.json
uv run python scripts/build_g005_aux_namespace_manifest.py   --source-evidence artifacts/aux/<source>_materialization.json   --eval-manifest-hashes artifacts/aux/d2e_eval_manifest_hashes.json   --completion-ready
```

`build_g005_eval_manifest_hashes.py` hashes the temporal, heldout-recording, and
heldout-game D2E split manifests and writes `same_hash=true` evidence for the
D2E-only vs D2E+aux comparison. By default the D2E+aux paths are the exact same
files as D2E-only; if separate aux eval manifests are supplied, hash mismatches
fail unless `--allow-mismatch` is used for non-terminal diagnostics.

`build_g005_aux_source_evidence.py` scans `outputs/aux/<dataset_id>/...` for
selected sources, hashes materialized files plus source-specific train/val/test
splits, records action-head namespaces, and writes a combined source-evidence
file. A `blocked` output is expected before the selected auxiliary datasets are
actually materialized.

`materialize_g005_aux_sources.py` is the safe entry point for that materialization
step. Its default mode is plan-only and records provider/namespace/download
strategy without network transfer. `--execute` currently supports selected Zenodo
records via their API file links and Hugging Face datasets via
`huggingface_hub.snapshot_download`, then writes `raw/` files and deterministic
train/val/test source-level manifests under each selected `outputs/aux/<dataset_id>/`
namespace. The resulting evidence is still source/provenance evidence only; it
does not authorize G005 training, G005 checkpointing, or D2E+aux model-quality
claims. Zenodo downloads are written through `.part-<pid>` files, verified against
published size/checksum metadata when available, and only then atomically moved
into `raw/`; invalid existing files are preserved as `.invalid-*` backups and
redownloaded rather than silently accepted as source evidence.

`build_g005_aux_archive_inventory.py` inspects the materialized `raw/` archives
after download and records archive/member counts plus heuristic action-label
member hints. It deliberately does not parse/train from the data; it is a
loader-implementation artifact so the next G005 lane can write source-specific
MineRL/Atari/Breakout adapters from observed archive structure rather than
guessing.

`build_g005_aux_action_registry.py` records the source-specific action heads that
must remain separate during auxiliary pretraining (`minecraft_keyboard_mouse`,
`atari_discrete`, etc.). The G005 completion audit requires this registry and
rejects collapsed/shared auxiliary action heads or any direct auxiliary claim on
D2E keyboard/mouse endpoints.

For unattended cluster source staging, start the materializer in the background
and then run the non-mutating watcher:

```bash
nohup uv run python scripts/materialize_g005_aux_sources.py \
  --execute \
  --output artifacts/aux/g005_aux_materialization_execute_summary.json \
  > artifacts/aux/g005_aux_materialization_execute.log 2>&1 &
echo $! > outputs/cluster/g005_aux_materialization.pid

nohup uv run python scripts/watch_g005_aux_materialization.py \
  --pid-file outputs/cluster/g005_aux_materialization.pid \
  --output artifacts/aux/g005_aux_materialization_watcher_summary.json \
  --allow-fail \
  > artifacts/aux/g005_aux_materialization_watcher.log 2>&1 &
echo $! > outputs/cluster/g005_aux_materialization_watcher.pid

uv run python scripts/monitor_g005_aux_materialization.py \
  --output artifacts/aux/g005_aux_materialization_progress.json
uv run python scripts/validate_g005_aux_materialization_integrity.py \
  --output artifacts/aux/g005_aux_materialization_integrity.json \
  --allow-fail
```

The monitor is non-mutating progress telemetry for partial downloads. The
integrity validator is a fail-closed post-download gate that checks materialized
raw files against Zenodo size/checksum metadata when available, validates Hugging
Face summary-listed files, and requires source-level train/val/test manifests
whose references resolve to real files. The watcher waits for source
materialization, runs the integrity gate, then rebuilds source evidence,
namespace-manifest readiness, and the fail-closed G005 launch-readiness report.
None of these scripts starts G005 training or checkpoints OMX/Codex state; until
G003/G004 D2E-only prerequisites pass, the watcher's expected terminal status is
`g005_launch_not_ready`.

Before launching any D2E+aux training run, run the fail-closed readiness planner:

```bash
uv run python scripts/plan_g005_launch.py \
  --source-evidence artifacts/aux/<source>_materialization.json \
  --eval-manifest-hashes artifacts/aux/d2e_eval_manifest_hashes.json \
  --require-eval-manifest-hashes
```

The planner requires completed G003/G004 D2E-only goal checkpoints and passing
G003/G004 audits by default. `--allow-precheckpoint` is diagnostic-only and must
not be used as launch authorization or claim evidence.

For long unattended G005 cluster runs, write the training parent PID to
`outputs/cluster/g005_d2e_aux_best.pid` and start the non-mutating watcher:

```bash
nohup uv run python scripts/watch_g005_then_finalize.py \
  --pid-file outputs/cluster/g005_d2e_aux_best.pid \
  --source-evidence artifacts/aux/<source>_materialization.json \
  --eval-manifest-hashes artifacts/aux/d2e_eval_manifest_hashes.json \
  --completion-ready \
  > artifacts/aux/g005_postrun_watcher.log 2>&1 &
echo $! > outputs/cluster/g005_postrun_watcher.pid
```

While the parent is alive the watcher only writes `waiting_active_parent`
telemetry. After the parent exits it runs the non-mutating G005 finalizer. It
does not mutate OMX/Codex goal state and cannot complete G005 by itself.

To avoid losing time after a long D2E-only G004 run, a separate fail-closed
readiness chain can be started after launching G004:

```bash
nohup uv run python scripts/watch_g004_then_plan_g005.py \
  --source-evidence artifacts/aux/<source>_materialization.json \
  --eval-manifest-hashes artifacts/aux/d2e_eval_manifest_hashes.json \
  --require-eval-manifest-hashes \
  --require-namespace-ready \
  --output artifacts/aux/g004_to_g005_readiness_chain_summary.json \
  > artifacts/aux/g004_to_g005_readiness_chain.log 2>&1 &
echo $! > outputs/cluster/g004_to_g005_readiness_chain.pid
```

This chain watches the G004 parent, reuses the G004 post-run watcher when it
already reports `finalized_pass`, otherwise runs the non-mutating G004 finalizer,
requires `g004_audit_status == pass`, and then runs `scripts/plan_g005_launch.py`.
It **does not** launch G005 training by default because the aux runner/source
materialization must remain source-specific and evidence-backed. A `g005_launch_ready`
status only means G005 can be handed to an explicit aux training lane without
weakening G003/G004 D2E-only gates.

After the D2E+aux training/ablation run finishes, run the G005 finalizer before
checkpointing:

```bash
uv run python scripts/finalize_g005_aux_best_model.py \
  --source-evidence artifacts/aux/<source>_materialization.json \
  --eval-manifest-hashes artifacts/aux/d2e_eval_manifest_hashes.json \
  --completion-ready
```

The finalizer builds or reuses the namespace manifest, requires the G005 run
summary, runs `scripts/validate_g005_aux_completion.py` through the same
completion-audit implementation, and writes
`artifacts/aux/g005_aux_finalization_summary.json`. It does not mutate OMX
state; checkpoint `G005-aux-data-best-model` only after the finalizer and
`artifacts/aux/g005_aux_completion_audit.json` report pass.

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

## G005 completion audit

Before checkpointing `G005-aux-data-best-model` complete, run:

```bash
uv run python scripts/validate_g005_aux_completion.py
```

During preparation this may be run with `--allow-fail`, but a terminal G005
checkpoint requires `artifacts/aux/g005_aux_completion_audit.json` to report
`status == pass`. The audit rejects missing G003/G004 D2E-only prerequisites,
missing namespace-manifest evidence, selected-source/provenance mismatches,
non-identical D2E eval manifests, auxiliary overlap with D2E heldouts, missing
D2E-only vs D2E+aux run IDs, prediction-count mismatch, and non-zero run exits.
