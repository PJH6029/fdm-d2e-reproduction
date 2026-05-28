# Ultragoal Operating Notes

Persistent user preferences and non-negotiable research constraints for the D2E/FDM reproduction ultragoal.

## Workflow preferences

- Use the requested progression: `$deep-interview` → `$ralplan` → `$ultragoal`, with sub-goals implemented by `$ralph` or `$team` when useful.
- Prefer `uv` for Python dependency resolution, script execution, testing, training launch wrappers, and environment workflows whenever practical.
- Commit regularly after coherent, verified milestones; do not batch the ultragoal into one huge final commit.
- Follow the Lore commit protocol for commit messages.
- Keep notes/evidence in repo-local artifacts and OMX state so future sessions can resume without relying on chat context.

## Research bar

- This is serious FDM-1-style reproduction research on the D2E dataset, not a smoke-path proof of concept.
- Do **not** claim FDM-1 parity, non-game-domain transfer, robotics/car transfer, commercial-game live control, or success from weak smoke-only evidence.
- Train/evaluate on real D2E data and preserve reusable checkpoints, configs, dataset fingerprints, logs, predictions, and reproducibility artifacts.
- Success requires trained IDM/FDM models to meaningfully beat smoke/baseline methods under predeclared metrics with a strong statistical bar.
- Trained policies must eventually execute stable action sequences in desktop/game harnesses, not only offline metrics.
- Final package must include method, baseline comparison, ablation/scaling curves, failure analysis, reproducible training pipeline, and trained checkpoints.

## 2026-05-28 course correction: FDM-1 recipe fidelity is global

- Keep the existing renewed ultragoal stories and ordering, but reinterpret every implementation choice through this mission: **reproduce the publicly described FDM-1 IDM/FDM training recipe on the D2E dataset**, then beat D2E paper/repo G-IDM targets with evidence-bound D2E training/evaluation.
- Do not choose arbitrary architectures/objectives merely because they are easy to implement or locally improve a metric. Novel exploration is allowed only to fill unpublished FDM-1 details in service of the public recipe.
- Public FDM-1 recipe anchors to preserve:
  - train/use a video encoder for high-compression screen video representations, with masked/self-supervised compression-style objectives where possible;
  - IDM is non-causal and predicts actions from all frames plus masked action tokens using a masked-diffusion/iterative unmasking objective and inference schedule;
  - FDM is an autoregressive next-action model over **interleaved frame and action tokens**, not a screenshot/VLM/CoT/tool-use proxy;
  - action tokens include key press/release and mouse deltas; mouse movement should be represented as discrete binned X/Y components, with click-position/trajectory auxiliary targets considered when feasible.
- Existing supervised MLP/luma-conv, heuristic, table, and post-hoc ensemble branches are now diagnostic baselines or failure evidence only. Future completion evidence for G005+ must include an FDM-1 recipe-alignment manifest/audit proving that the candidate uses the recipe-shaped architecture/objective, in addition to metric gates.
- No FDM-1 parity claim: because FDM-1 internals are not public, record which details are directly public, which are inferred, and which are novel approximations.


## 2026-05-29T01:48:58+09:00 KST — User reaffirmed FDM-1 recipe fidelity as the renewed mission

- Global instruction for all remaining ultragoal execution: preserve the existing G001–G012 goal list/statuses, but treat G005+ as **FDM-1 public IDM/FDM training-recipe reproduction on D2E** rather than arbitrary architecture/objective search.
- Mission target: train the FDM-1-shaped IDM/FDM recipe on D2E and produce artifacts that beat D2E paper/released G-IDM targets; exact-split G-IDM remains the follow-up baseline after paper targets are beaten.
- Novel exploration is allowed only to approximate unpublished FDM-1 internals while preserving recipe shape: video encoder/compression-style screen-video tokens; non-causal masked-diffusion IDM over action-token sequences with iterative confidence unmasking; autoregressive FDM over interleaved frame/action data; key press/release plus binned mouse/click/trajectory-style tokens.
- Branches using arbitrary supervised MLPs, tables, heuristic calibration, or non-interleaved proxy objectives remain diagnostic/failure evidence unless explicitly rebuilt behind the recipe-alignment manifest/audit gates.

## Current ultragoal gates

- Active aggregate Codex/OMX ultragoal covers G001–G009 in `.omx/ultragoal/goals.json`; do not complete the aggregate Codex goal until all stories are complete.
- Current completed gates:
  - `G001-data-universe-audit`: full D2E-480p + D2E-Original universe manifest and storage/license report.
  - `G002-split-leakage-contract`: temporal, heldout-recording, and heldout-game split/leakage contract.
  - `G003-d2e-only-idm`: full-corpus D2E-only IDM train/eval, canonical/accel64 promotion, split stats, and completion audit checkpointed in OMX.
  - `G004-d2e-only-fdm-4xh200`: full-corpus D2E-only FDM 4×H200 training/evaluation from G003 IDM pseudo-labels, finalization, split statistics, GPU monitor evidence, and completion audit checkpointed in OMX.
  - `G005-aux-data-best-model`: D2E+aux action-prior candidate over selected public aux action datasets, full D2E target eval predictions, D2E-only-vs-D2E+aux ablation, namespace/provenance audit, finalization, and completion audit checkpointed in OMX.
  - `G006-evaluation-failure-analysis`: final endpoint statistics, failure analysis, claim taxonomy, readiness audit, and completion audit checkpointed in OMX; unavailable `no_shared_clusters` tests are documented non-rejections, not wins.
  - `G007-runtime-sdk-adapter`: reusable SDK/action decoder/safety adapter/latency logger/deterministic replay contract only; this is not G008 live-game success.
- `G008-live-game-suite`: complete and checkpointed in OMX at commit `0cbcb8e`. Terminal artifacts: `artifacts/harness/g008_repo_live_suite/run_summary.json` status=pass episodes=15, `artifacts/harness/g008_live_open_game_suite_evidence_validation.json` quality_gate.status=pass findings_count=0, `artifacts/harness/g008_live_suite_completion_audit.json` status=pass error_count=0, and `artifacts/harness/g008_live_open_game_suite_finalization_summary.json` status=pass. Scope is repo-local open-source Tk graphical mini-games via live X11/xdotool; not commercial-game control.
- Final completion audit now exists (`configs/eval/final_quality_gates.yaml`, `scripts/validate_final_quality_gates.py`, `artifacts/reproducibility/final_quality_gate_audit.json`) and must pass before aggregate goal completion; current expected status is fail while G009 remains incomplete.
- G006 evaluation artifacts are terminal/pass: `artifacts/eval/final_endpoint_statistics.json`, `artifacts/eval/final_failure_analysis.json`, `artifacts/eval/final_claim_taxonomy.json`, `artifacts/eval/g006_evaluation_readiness_audit.json`, `artifacts/eval/g006_final_artifact_build_summary.json`, and `artifacts/eval/g006_completion_audit.json`.
- Pending gates:
  - `G009-report-repro-package`.
- D2E-only gates must finish before D2E+aux or runtime success claims. D2E+aux may become the primary/best final model, but D2E-only results/ablations remain mandatory and separately reported.
- Strong FDM/IDM evidence should include keyboard, mouse movement, and mouse-button endpoints; mouse-button claims must report precision/F1 and no-button false-positive rate, not only positive-class accuracy.

## Dataset and auxiliary-data policy

- Final success requires full usable D2E consumption across 480p and available original/FHD/QHD sources; audited exclusions are allowed only with retry/failure logs and impact counts.
- The project is not limited to D2E if public gameplay/action-event datasets fit within the 5TiB storage cap and have valid provenance/license/action-event utility.
- D2E+aux may be the primary/best model for final harness and report claims (`d2e_aux_may_be_primary`), provided D2E-only hard gates and comparisons are preserved.
- Latest deep-interview decision: `d2e_aux_may_be_primary`; D2E+aux may be the best/primary final model only after D2E-only hard gates, with separate D2E-only vs D2E+aux ablation evidence.

## Resource / cluster policy

- Use the `mlxp-reservation-api` skill for SNUPI MLXP GPU reservation workflows.
- User authorized cluster GPU reservations/scheduling without further confirmation during this ultragoal; use up to 4×H200 and design multi-GPU-capable training paths.
- Treat sustained 4×H200 GPU idle time as a serious execution blocker/risk across the ultragoal, not merely expected overhead. For every cluster training run, monitor GPU utilization, label CPU/IO-only phases explicitly, and prefer sharded/parallel materialization, tensor-cache, prediction-worker, and recovery paths that get DDP training onto GPUs sooner while preserving audit/reproducibility artifacts.
- Canonical goal-wide GPU-utilization rule is persisted in `notes/gpu-utilization-operating-rule.md`; use it during every G004–G009 launch/monitor/recovery handoff.
- Cluster workflow: edit locally, push, then pull in the pod PVC path `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction` before running GPU jobs.
- Docker registry username: `pjh6029`; auth is already configured.

## Current cluster handoff

- Current reservation/pod: `rsv-jeonghunpark-20260521-76e25a` / `prod-rsv-jeonghunpark-20260521-76e25a` in namespace `p-production`.
- Pod repo path: `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`.
- G003 is complete in OMX; do not treat older G003 run/watcher notes as active blockers.
- G004 is complete and checkpointed in OMX. Terminal evidence: `artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json` exit_code=0, `artifacts/fdm/g004_d2e_full_fdm_finalization_summary.json` status=pass, `artifacts/fdm/g004_full_fdm_completion_audit.json` status=pass/error_count=0, `artifacts/eval/g004_split_statistical_comparisons_summary.json` status=pass, and `artifacts/fdm/g004_d2e_full_fdm_4xh200_gpu_monitor.csv`.
- Raw G004 checkpoint/predictions/train-target JSONLs/caches remain on the MLXP PVC. Local git carries small audit/evidence artifacts and handoff notes only.
- GPU-utilization follow-up: `artifacts/fdm/g004_gpu_rank_imbalance_diagnosis.json` shows GPU0 could idle in the completed `bfe61db` run because cache shards were assigned by path modulo. Local/origin hardening switches future/recovery cache assignment to deterministic `greedy_rows`; use it for G005 and later training runs.
- G003 progress monitor exists (`scripts/monitor_g003_progress.py`, `docs/g003_progress_monitoring.md`) for non-mutating shard/PID/stale-progress summaries; monitor output is progress evidence only, not G003 completion.
- G003 live health audit exists (`scripts/audit_g003_live_health.py`) for non-mutating parent/extractor/watcher/GPU-monitor process-topology summaries; use it for handoff/recovery evidence but not for completion claims.
- Historical G003 accel64 process notes for parent PID `251593` are no longer current; keep them only as provenance for the completed G003 checkpoint. Do not block G004 on old G003 worker/PID state.
- G003 resume planner exists (`scripts/plan_g003_resume.py`, `artifacts/idm/g003_resume_plan.json`) but should not be used unless the completed G003 evidence is later found corrupt or missing.
- G004 FDM training now requires explicit train-core pseudo-labels from the completed G003 IDM checkpoint (`scripts/predict_idm_streaming.py` + `configs/model/idm_streaming_d2e_full_compact_predict_fdm_train.yaml`) and evaluates on untouched `target_all_eval`; do not revert to target_all_eval recording-tail training for completion evidence. The G004 model feature mode is `summary_causal_compact_grid8_time_prior_action` to avoid next-frame inverse-dynamics leakage and include previous-action context.
- G004 post-run watcher exists (`scripts/watch_g004_then_finalize.py`) and consumes `outputs/cluster/g004_d2e_full_fdm_4xh200.pid`; it runs the non-mutating finalizer after the parent exits but never checkpoints OMX/Codex state.
- G003→G004 chain watcher exists (`scripts/watch_g003_then_launch_g004.py`) and may run in the pod with `--launch --start-g004-watcher`; it launches G004 only after G003 finalization and G003 audit pass, then starts the G004 post-run watcher. It never checkpoints OMX/Codex state. As of 2026-05-22 14:09 KST, the pod still had a stale `outputs/cluster/g003_to_g004_chain_watcher.pid` and `artifacts/fdm/g003_to_g004_chain_summary.json`, but no running watcher process; after G003 is promoted and OMX-checkpointed, start a fresh current watcher with `--require-g003-goal-checkpoint`.
- G005 launch planner/watcher exists (`scripts/plan_g005_launch.py`, `scripts/watch_g005_then_finalize.py`) for D2E+aux best-model preparation; both preserve G003/G004 D2E-only gates and never mutate OMX/Codex state.
- G005 is complete in OMX. Its final candidate used 16-worker CPU/IO prediction over G004 recovery parts; future neural/visual aux candidates, if added, must still follow the goal-wide GPU-utilization rule.
- G004→G005 readiness chain exists (`scripts/watch_g004_then_plan_g005.py`) and can be started after G004 launch; it waits for G004 finalization/audit pass, then records G005 readiness only. It does not launch G005 training because aux source materialization/eval-hash evidence must be explicit.
- G005 eval-manifest hash builder exists (`scripts/build_g005_eval_manifest_hashes.py`, `artifacts/aux/d2e_eval_manifest_hashes.json`) and proves byte-identical temporal/heldout-recording/heldout-game D2E eval manifests for D2E-only vs D2E+aux comparisons; it does not materialize aux sources or start training.
- G005 aux action registry exists (`scripts/build_g005_aux_action_registry.py`, `artifacts/aux/g005_aux_action_registry.json`) and is required by the G005 completion audit. It records source-specific action heads and forbids collapsed/shared aux actions or direct aux claims on D2E keyboard/mouse endpoints.
- G005 aux loader manifest exists (`scripts/build_g005_aux_loader_manifest.py`, `artifacts/aux/g005_aux_loader_manifest.json`) and combines action registry, archive inventory, and materialization-integrity evidence into source-specific `outputs/aux_examples/<dataset_id>/{train,val,test}.jsonl` contracts for later real auxiliary pretraining. It blocks until materialization/integrity pass.
- G005 aux example builder exists (`scripts/build_g005_aux_examples.py`, `artifacts/aux/g005_aux_examples_summary.json`) and supports `atari_head_zip_csv_action_adapter`, `minerl_action_dict_adapter`, and `p_doom_array_record_action_adapter` source-specific train/val/test JSONL outputs. The p-doom adapter requires package `array-record` (project `d2e` extra) for `array_record.python.array_record_module.ArrayRecordReader`; without it the source remains fail-closed. G005 completion requires this summary to pass for all selected aux sources.
- G005 aux runtime preflight exists (`scripts/validate_g005_aux_runtime_env.py`, `artifacts/aux/g005_aux_runtime_env.json`) and is now required by launch/completion readiness. It fail-closes selected-source adapters when optional runtime deps are absent, especially `array-record` for p-doom ArrayRecord streams. It is dependency readiness only, not materialization/training/completion evidence.
- G005 aux source evidence builder exists (`scripts/build_g005_aux_source_evidence.py`, `artifacts/aux/g005_aux_source_materialization_evidence.json`) and scans `outputs/aux/<dataset_id>/...` for selected-source materialization, source-specific split hashes, action-head namespace, and D2E-heldout-overlap evidence; current expected status is blocked until aux source files are materialized.
- G005 aux materializer exists (`scripts/materialize_g005_aux_sources.py`, `artifacts/aux/g005_aux_materialization_plan.json`): default mode is plan-only; `--execute` downloads selected Zenodo/Hugging Face aux sources into `outputs/aux/<dataset_id>/raw` and writes source-level train/val/test manifests. Zenodo downloads use atomic `.part-<pid>` files and size/checksum validation before split manifests are written. It is materialization/provenance evidence only and does not authorize G005 training or D2E+aux claims.
- G005 aux materialization monitor exists (`scripts/monitor_g005_aux_materialization.py`, `artifacts/aux/g005_aux_materialization_progress.json`) for non-mutating partial-download telemetry: raw byte counts, partial/complete/missing source ids, PID state, and split-manifest readiness. It is progress telemetry only.
- G005 aux materialization integrity validator exists (`scripts/validate_g005_aux_materialization_integrity.py`, `artifacts/aux/g005_aux_materialization_integrity.json`) and is now called by the materialization watcher before source evidence. It validates post-download raw bytes/manifests and blocks source evidence if size/checksum/manifests are incomplete.
- G005 aux archive inventory builder exists (`scripts/build_g005_aux_archive_inventory.py`, `artifacts/aux/g005_aux_archive_inventory.json`) to inspect materialized raw archives and find heuristic action-label member names before writing source-specific loaders; it is not training or completion evidence.
- G005 aux materialization watcher exists (`scripts/watch_g005_aux_materialization.py`, `artifacts/aux/g005_aux_materialization_watcher_summary.json`): after the materializer exits, it runs the materialization-integrity validator, rebuilds source evidence, auxiliary example manifests, runtime dependency readiness, namespace readiness, and fail-closed G005 launch readiness. It never starts G005 training or checkpoints OMX/Codex state.
- Watcher PID hygiene: `watch_g003_then_finalize.py`, `watch_g003_then_launch_g004.py`, `watch_g004_then_finalize.py`, `watch_g004_then_plan_g005.py`, `watch_g005_aux_materialization.py`, and `watch_g005_then_finalize.py` self-write their Python PID files. Do not overwrite those pid files with shell `$!` from `uv run`; `$!` can be the uv wrapper PID and can cause false duplicate-watch blockers on restart.
- G005 aux source materialization artifacts remain useful provenance, but G005 itself is now complete in OMX. Do not rely on older watcher summaries that reported `g005_launch_not_ready` before G003/G004/G005 terminal evidence existed.
- G006 is complete in OMX. Readiness planner/watcher still exist (`scripts/plan_g006_readiness.py`, `scripts/watch_g006_then_finalize.py`) for recovery/rebuild, but do not rerun unless G006 artifacts are corrupt or intentionally refreshed.
- G008 readiness planner exists (`scripts/plan_g008_readiness.py`, `artifacts/harness/g008_readiness_plan.json`) for live open-source graphical-game collection prep; current terminal G008 readiness/evidence has passed, but keep the planner for future harness recovery or external-game expansion.
- G009 readiness planner exists (`scripts/plan_g009_readiness.py`, `artifacts/reproducibility/g009_readiness_plan.json`) for final report/repro package prep; next active work is G009 final report/reproducibility packaging now that G001-G008 are complete.
- Runtime adapter contract evidence: `artifacts/runtime/g007_runtime_replay_adapter_contract.json`; commits `34cddb7` and `e858114`; OMX checkpointed `G007-runtime-sdk-adapter` complete locally.
- `G003-d2e-only-idm` is already checkpointed complete in OMX. Do not repeat the G003 checkpoint unless reconciling a ledger corruption.
- Streaming IDM metadata now records config/data/split/source provenance (`checkpoint_metadata.json`, `resolved_config.json`); ensure the pod checkout includes this before the G003 extraction reaches training.
- Commit `6974f38` adds automatic split-stat generation to future G003/G004 run wrappers. The old accel64 G003 parent PID `251593` is historical; current live monitoring should focus on G004 parent PID `262618`.

## 2026-05-28 G005 FDM-1-recipe prefix probes

- Commit `a3a7f7d` added the public FDM-1 recipe manifest/audit and recipe-aligned IDM/FDM scaffold configs. Commit `b4e98ea` added a Torch prefix trainer for a non-causal masked-diffusion IDM over fixed action-token slots with iterative unmasking and D2E paper-metric export.
- MLXP reservation `rsv-jeonghunpark-20260528-a131d9` ran three non-terminal shard_0 prefix probes on 1×H200 and was cancelled afterward to avoid idle GPU. Evidence is under `artifacts/idm/g005_idm_masked_diffusion_prefix20k*.{json,csv,log}` plus `artifacts/idm/g005_idm_masked_diffusion_prefix_reservation_context.json`.
- Prefix results are **not** G005 completion evidence:
  - unweighted: training path passed on CUDA but decoded no action; keyboard/button/mouse metrics effectively zero and no-button FPR 0.
  - no-op-weighted (`noop_loss_weight=0.05`): emitted mouse movement but still key/button recall 0.
  - category/key-heavy: forced key emission (`KEY_PRESS_65`/`KEY_PRESS_68`) and raised paper keyboard only to ~0.008 while overfitting/overfiring; button remained 0.
- Diagnosis: the current single-vocabulary fixed-slot decoder is recipe-shaped but too crude for sparse D2E key/button events. Continue within the FDM-1 public recipe by improving token factorization/slot design and video-token conditioning, not by reverting to the old supervised per-head or heuristic branches as completion candidates.

## 2026-05-28 G005 factorized masked-diffusion prefix probe

- Commit `bcc8a41` added a typed/factorized masked action-plane IDM path to keep the public FDM-1 masked-diffusion recipe while separating mouse-axis, keyboard, and mouse-button token factors.
- MLXP reservation `rsv-jeonghunpark-20260528-5d6e98` ran `idm_factorized_masked_diffusion_d2e_prefix20k_h200` on shard_0 with 20k train / 2k target rows. `rsv-jeonghunpark-20260528-333a1d` was only used to recopy persistent PVC artifacts after the first pod was cancelled; both reservations are cancelled.
- Evidence: `artifacts/idm/g005_idm_factorized_masked_diffusion_prefix20k_h200_*`. The factorized path trains on CUDA and improves paper-compatible keyboard from the key-heavy fixed-slot probe (~0.008) to ~0.0188 on this tiny prefix, but it still overpredicts common keys and gets mouse-button 0.0. It is **not** G005 completion evidence.
- Next G005 branch should keep this factorized recipe path but add calibration/threshold selection, stronger video-token conditioning, and/or better key/button temporal context before any expensive full-corpus 4xH200 promotion.

## 2026-05-28 KST — G005 factorized masked-diffusion per-token prefix probe

- Ran a fresh 1×H200 D2E shard-0 prefix probe at commit `be74324` with `calibrate_per_token_thresholds=true` on the public FDM-1-shaped typed masked-diffusion IDM path.
- Evidence copied locally under `artifacts/idm/g005_idm_factorized_masked_diffusion_prefix20k_per_token_h200_*`; W&B artifact run: `artifacts/idm/g005_idm_factorized_masked_diffusion_prefix20k_per_token_h200_wandb_status.json`; reservation `rsv-jeonghunpark-20260528-cfcbdc` was cancelled after evidence copy.
- Result remains non-terminal and below G005 target: keyboard key accuracy `0.019203072491598656`, mouse-button accuracy/F1 `0.0`, no-button FPR `0.0`. Per-token calibration suppresses button false positives but does not recover recall and regresses keyboard versus the previous global-threshold prefix probe (~`0.0239`).
- Next recipe-aligned branch: keep the FDM-1-shaped noncausal masked-diffusion IDM objective, but improve the video encoder bootstrap by consuming compact luma-window frame tokens (`compact_luma_window`) rather than adding supervised/state-action shortcuts. Old state/action-prior branches are diagnostic only unless explicitly reworked into the FDM-1 recipe shape.

## 2026-05-28 KST — G005 compact luma-window masked-diffusion prefix probe

- Ran a second 1×H200 prefix probe at commit `7ede19c` on `outputs/data/d2e_luma_window5_corpus_shards_accel64/shard_0`, using the same FDM-1-shaped typed masked-diffusion IDM objective but conditioning on 5×16×16 compact luma-window video tokens.
- Evidence copied locally under `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_prefix20k_h200_*`; W&B artifact run: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_prefix20k_h200_wandb_status.json`; reservation `rsv-jeonghunpark-20260528-91d7a9` was cancelled after evidence copy.
- Result remains non-terminal: keyboard key accuracy `0.015781922525107604`, mouse-button accuracy/F1 `0.0`, no-button FPR `0.033846153846153845`. Luma-window video tokens alone did not recover button recall or paper-target keyboard accuracy.
- Next recipe-aligned branch implemented locally: a compact luma-window CNN video encoder feeding the same masked action-token planes (`configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_prefix20k.yaml`). Run this bounded H200 probe before considering larger/full-corpus masked-diffusion IDM launches.

## 2026-05-28 KST — G005 compact luma-window CNN masked-diffusion prefix probe

- Ran a third 1×H200 prefix probe at commit `673e011` with a compact luma-window CNN video encoder feeding the same typed masked action-token planes.
- Evidence copied locally under `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_prefix20k_h200_*`; W&B artifact run: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_prefix20k_h200_wandb_status.json`; reservation `rsv-jeonghunpark-20260528-4df5dc` was cancelled after evidence copy.
- Result is still non-terminal: keyboard key accuracy `0.020192307692307693`, mouse-button accuracy/F1 `0.0`, no-button FPR `0.0`. CNN video encoder recovers some keyboard performance versus flat luma-window (`0.01578`) but remains below the compact-frame factorized prefix (`~0.0239`) and below all G005 targets.
- Next branch should explicitly address sparse mouse-button recall inside the recipe-aligned typed masked-diffusion objective (for example, separate button down/up denoising head or recall-constrained button calibration) before any full-corpus 4×H200 G005 launch.

## 2026-05-28 KST — G005 button-event masked-diffusion prefix probe

- Ran a fourth 1×H200 prefix probe at commit `9d5f082` with compact luma-window CNN video encoder plus auxiliary sparse button-event denoising inside the typed masked action-token objective.
- Evidence copied locally under `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_event_prefix20k_h200_*`; W&B artifact run: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_event_prefix20k_h200_wandb_status.json`; reservation `rsv-jeonghunpark-20260528-29dc14` was cancelled after evidence copy.
- Result is negative/non-terminal: keyboard key accuracy `0.012219717641475857`, mouse-button accuracy/F1 `0.0`, no-button FPR `0.0`. Button-event calibration saturated: thresholds <= `0.45` predicted every row (`FPR=1.0`), thresholds >= `0.50` predicted none, so bounded-FPR calibration selected zero-recall behavior.
- Next branch: add probability-quantile/dynamic threshold candidates for sparse button event/token calibration so the calibration grid can choose between saturated coarse thresholds before any more scaling or full-corpus launch.

## 2026-05-28 KST — Course correction reaffirmed during G005

- User reaffirmed a global mission correction: keep existing renewed ultragoal IDs/status, but G005+ must use the **public FDM-1 IDM/FDM architecture and training recipe on D2E**, not arbitrary architecture/objective choices. Novel work is allowed only to approximate unpublished FDM-1 internals and improve metric performance while staying recipe-faithful.
- Public recipe anchors currently in force: video encoder/compression-style screen-video tokens; non-causal masked-diffusion IDM over masked action tokens with iterative unmasking; FDM autoregressive next-action prediction over interleaved frame/action tokens; key press/release and binned mouse/action tokens. No FDM-1 parity claim.
- Existing supervised MLP/state/table/heuristic branches remain diagnostic baselines or negative evidence only unless rebuilt into the public recipe shape and passing `fdm1_recipe_alignment` gates.

## 2026-05-28 KST — G005 dynamic button-event masked-diffusion prefix probe

- Copied terminal 1×H200 evidence for commit `fd65d03` from reservation `rsv-jeonghunpark-20260528-aac42f` / pod `prod-rsv-jeonghunpark-20260528-aac42f`; reservation was cancelled after copy to avoid idle GPU.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_event_dynamic_prefix20k_h200_*`, including run, GPU monitor, W&B status, resolved config, paper metrics, summary, reservation context, and diagnosis.
- Result is negative/non-terminal: dynamic calibration chose button-event threshold `0.42894818050956723`, with calibration recall `0.1724` at FPR `0.0917`, but target strict no-button FPR rose to `0.5482`, strict mouse-button F1 remained `0.0`, and exact true-positive button examples stayed `0`.
- Diagnosis: threshold quantiles overfit prefix calibration and do not transfer to target rows. Next recipe-faithful G005 branch needs split-robust calibration/abstention or target-invariant button ranking diagnostics before any larger/full 4×H200 promotion.

## 2026-05-28 KST — Next G005 branch: joint button-event/token-confidence gate

- Implemented a recipe-faithful follow-up branch for the typed masked-diffusion IDM: button-event calibration can now jointly require an event-head threshold and a calibrated minimum button-token probability before forcing a mouse-button token.
- Motivation from `fd65d03`/`923d3cc`: event probability alone transferred badly from calibration to target; the new joint gate preserves the FDM-1-shaped masked action-token objective while adding split-safe abstention against button false positives.
- Candidate config: `configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_button_event_joint_prefix20k.yaml`. It remains a prefix candidate, not completion evidence, until H200 metrics show target FPR/recall transfer.

## 2026-05-28 KST — G005 joint button-event/token-confidence H200 prefix probe

- Ran commit `04f03e5` on 1×H200 reservation `rsv-jeonghunpark-20260528-6e8e55` / pod `prod-rsv-jeonghunpark-20260528-6e8e55`; copied evidence locally and cancelled the reservation after artifact copy.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_event_joint_prefix20k_h200_*`, including run, GPU monitor, W&B status, resolved config, paper metrics, summary, reservation context, and diagnosis.
- Result is partial but non-terminal: calibration selected event threshold `0.496727854013443` plus min button-token probability `0.5808834896812439`; target no-button FPR improved from prior dynamic event-only `0.5482` to `0.2621`, and strict button F1 became nonzero (`0.00351`) with one exact true positive, but this still fails the `<=0.10` FPR gate and is far below paper-target performance.
- Next branch should remain recipe-faithful and add fold/recording-robust calibration or confidence-budgeted iterative unmasking before any full-corpus 4×H200 promotion.

## 2026-05-28 KST — Next G005 branch: confidence-budgeted button unmasking

- Implemented a recipe-faithful follow-up branch after joint-gate partial improvement: forced mouse-button unmasking can now be capped by a train/calibration-label event-rate prior while selecting highest-confidence unlabeled target candidates.
- Candidate config: `configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_button_event_budget_prefix20k.yaml`. It keeps the FDM-1-shaped video-token + noncausal masked action-token IDM and adds confidence-budgeted iterative unmasking as an inferred/novel public-recipe approximation.
- Intended bounded-probe success criterion before scaling: target no-button FPR near or below `0.10` while preserving nonzero button recall; still not G005 completion without full-corpus paper-target win.

## 2026-05-28 KST — G005 confidence-budgeted button-unmasking H200 prefix probe

- Ran commit `c0bf3b3` on 1×H200 reservation `rsv-jeonghunpark-20260528-4187c7` / pod `prod-rsv-jeonghunpark-20260528-4187c7`; copied evidence locally and cancelled the reservation after artifact copy.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_event_budget_prefix20k_h200_*`, including run, GPU monitor, W&B status, resolved config, paper metrics, compacted summary, reservation context, and diagnosis.
- Result is negative/non-terminal: the train-label confidence budget limited event-head forced unmasking to `58` target rows, but target predicted mouse-button examples still reached `710`, no-button FPR worsened to `0.3569`, and strict button F1 was only `0.00526`.
- Diagnosis: direct per-token button predictions now dominate overfire; next branch must route **all** mouse-button emission through a confidence budget or direct-button abstention gate, while keeping the FDM-1-shaped masked action-token recipe.

## 2026-05-28 KST — Next G005 branch: all-button confidence budget

- Implemented follow-up after the event-budget negative probe: `button_event_budget_applies_to_all_buttons=true` makes the confidence budget gate direct per-token mouse-button emissions as well as event-head forced emissions.
- Candidate config: `configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_button_event_allbudget_prefix20k.yaml`. This stays within the public FDM-1-shaped masked action-token IDM and treats the budget as confidence-based iterative unmasking/abstention rather than a supervised shortcut.
- Bounded H200 probe should verify whether target no-button FPR drops near `<=0.10` without eliminating all button recall.

## 2026-05-28 KST — G005 all-button confidence-budget H200 prefix probe

- Ran commit `bb5a34c` on 1×H200 reservation `rsv-jeonghunpark-20260528-93de5c` / pod `prod-rsv-jeonghunpark-20260528-93de5c`; copied evidence locally and cancelled the reservation after artifact copy.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_event_allbudget_prefix20k_h200_*`, including run, GPU monitor, W&B status, resolved config, paper metrics, compacted summary, reservation context, and diagnosis.
- Result is useful but non-terminal: all-button confidence budgeting reduced target no-button FPR to `0.02923` (passes the local FPR gate), but exact mouse-button true positives fell to `0`, strict button F1 stayed `0.0`, and paper-compatible mouse-button accuracy was `0.0`.
- Next branch should keep the all-button budget for FPR control but recover recall via train-only budget multiplier/score sweep or better button-token ranking. Do not promote to full 4×H200 until prefix has both `<=0.10` FPR and nonzero useful recall.

## 2026-05-28 KST — Next G005 branch: calibration-selected button budget multiplier

- Implemented a recipe-faithful continuation after the all-button confidence-budget probe: the typed masked-diffusion IDM can now sweep mouse-button confidence-budget multipliers on held-out train/calibration rows and select the multiplier under a calibration no-button FPR cap, then apply the selected budget to unlabeled target confidence rankings without target labels.
- Candidate config: `configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_button_event_multibudget_prefix20k.yaml`. It preserves compact luma-window CNN video tokens, noncausal masked action-token denoising, iterative unmasking gates, and all-button budgeted mouse-button emissions.
- Intended bounded H200 probe criterion before scaling: keep target no-button FPR near `<=0.10` while recovering nonzero/useful mouse-button recall. This remains prefix exploration, not G005 completion evidence.

## 2026-05-28 KST — G005 multibudget H200 prefix probe

- Ran commit `1453ce8` on 1×H200 reservation `rsv-jeonghunpark-20260528-d97ff1` / pod `prod-rsv-jeonghunpark-20260528-d97ff1`; copied evidence locally and cancelled the reservation after artifact copy.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_event_multibudget_prefix20k_h200_*`, including run, GPU monitor, W&B status, resolved config, paper metrics, compacted summary, reservation context, and diagnosis. W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/w3j2ms43`.
- Result is negative/non-terminal: calibration selected budget multiplier `2.0` with held-out calibration recall `0.2069` at FPR `0.0536`, and target no-button FPR stayed low (`0.03436`), but target exact mouse-button TP remained `0`, strict button F1 `0.0`, and paper-compatible mouse-button accuracy `0.0`.
- Diagnosis: event/button confidence ranking remains split-fragile; budget size is no longer the only bottleneck. Next recipe-faithful branch should improve button-token ranking/conditioning itself (for example temporal button-state tokens, down/up transition factorization, or click-region auxiliary denoising) before scaling.

## 2026-05-28 KST — Next G005 branch: categorical button-transition token

- Implemented a recipe-faithful follow-up after multibudget target recall stayed zero: the factorized masked-diffusion IDM can now denoise the mouse-button plane as a categorical no-button-vs-button-transition token (`button_transition_softmax`) and use that class distribution for button-token probabilities/event-budget scores.
- Candidate config: `configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_button_class_multibudget_prefix20k.yaml`. It keeps compact luma-window CNN video tokens, noncausal masked action-token denoising, and calibration-only budget multiplier selection; the new categorical button slot is a novel approximation to FDM-1's discrete action-token recipe, not a parity claim.
- Intended bounded H200 probe criterion: improve split-robust target mouse-button TP/F1 while preserving no-button FPR `<=0.10`; still prefix exploration only.

## 2026-05-28 KST — G005 button-class H200 prefix probe

- Ran commit `baf4cee` on 1×H200 reservation `rsv-jeonghunpark-20260528-96293f` / pod `prod-rsv-jeonghunpark-20260528-96293f`; copied evidence locally and cancelled the reservation after artifact copy.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_class_multibudget_prefix20k_h200_*`, including run, GPU monitor, W&B status, resolved config, paper metrics, compacted summary, reservation context, and diagnosis. W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/0xv23lp0`.
- Result is negative/non-terminal: calibration selected multiplier `2.5`, but target threshold-candidate count was `0`, so target predicted button examples were `0`, strict button F1 `0.0`, and no-button FPR `0.0`; keyboard key accuracy was `0.01909`.
- Diagnosis: categorical button-transition ranking did not transfer because the held-out calibration event threshold/min-token gate was too strict for target scores. Next recipe-faithful branch should decouple the unlabeled target confidence budget from brittle absolute event thresholds, e.g. score all target rows for budget ranking while keeping calibration-selected budget size/FPR constraints.

## 2026-05-28 KST — Next G005 branch: relaxed target budget ranking

- Implemented a recipe-faithful follow-up after button-class target candidates collapsed to zero: `button_event_budget_rank_all_scores=true` lets the confidence budget rank all unlabeled target rows by event/button score instead of filtering first by brittle absolute event/min-token thresholds. Calibration still selects the budget multiplier/FPR cap on held-out train rows; target labels remain unused.
- Candidate config: `configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_button_class_relaxed_budget_prefix20k.yaml`. This preserves the public FDM-1-shaped video-token + noncausal masked action-token IDM, categorical button-transition token, and iterative confidence unmasking.
- Intended bounded H200 probe criterion: restore nonzero target button predictions/TPs without exceeding no-button FPR `<=0.10`.

## 2026-05-28 KST — G005 relaxed-budget H200 prefix probe

- Ran commit `5a443a4` on 1×H200 reservation `rsv-jeonghunpark-20260528-8be6cc` / pod `prod-rsv-jeonghunpark-20260528-8be6cc`; copied evidence locally and cancelled the reservation after artifact copy.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_class_relaxed_budget_prefix20k_h200_*`, including run, GPU monitor, W&B status, resolved config, paper metrics, compacted summary, reservation context, and diagnosis. W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/uh4slone`.
- Result is negative/non-terminal but diagnostic: relaxed target ranking restored target button predictions (`175`) and kept no-button FPR below gate (`0.08872`), but exact target true positives remained `0`, strict button F1 `0.0`, and paper-compatible mouse-button accuracy `0.0`.
- Diagnosis: the issue is now **ranking/semantic alignment**, not budget size or absolute threshold transfer. Next recipe-faithful branch should improve button-token semantics/conditioning itself, e.g. align left/right/down/up mapping diagnostics, add click-region/visual-change auxiliary denoising, or train across a broader prefix before promoting.

## 2026-05-28 KST — Next G005 diagnostic: button semantic/ranking alignment

- Added `src/fdm_d2e/eval/button_semantic_ranking_diagnostic.py` and `scripts/build_g005_button_semantic_ranking_diagnostic.py` to inspect prefix IDM mouse-button failures without changing training or calibration.
- The diagnostic reports exact vs semantic overlap, predicted/ground-truth token distributions, offset sweeps, target-label-only mapping hypotheses, and examples. Claim boundary: target labels are used only for failure analysis; mappings/offsets are not valid training/calibration evidence unless revalidated without target leakage.
- Use this on the latest relaxed-budget prefix predictions before the next recipe-faithful correction; expected question is whether target positives are near top-ranked predictions under a token mapping or temporal offset, or whether visual/action conditioning itself is insufficient.

## 2026-05-28 KST — Global FDM-1 recipe course correction is binding

- User issued a major global correction: keep the current renewed ultragoal/story IDs and statuses, but G005+ work must reproduce the **public FDM-1 IDM/FDM training recipe on D2E**, not choose arbitrary architectures/objectives for convenience.
- Binding public recipe anchors from the FDM-1 technical report: video encoder/compression-style screen-video tokens; non-causal masked-diffusion IDM over masked action tokens with iterative confidence unmasking; FDM autoregressive next-action prediction over interleaved frame/action tokens; discrete key press/release and mouse movement/click tokens. Unpublished internals may be approximated only by clearly marked, recipe-faithful exploration.
- Mission framing: D2E-trained FDM-1-shaped IDM/FDM artifacts must first beat D2E paper/released G-IDM targets. Existing renewed goals remain; old supervised/state/action-prior/heuristic branches are diagnostic only unless rebuilt behind `fdm1_recipe_alignment` gates.

## 2026-05-28 KST — G005 relaxed-budget button semantic diagnostic evidence

- Copied diagnostic evidence from 1×H200 diagnostic reservation `rsv-jeonghunpark-20260528-1080ce` / pod `prod-rsv-jeonghunpark-20260528-1080ce` and cancelled the reservation after artifact copy.
- Evidence files: `artifacts/idm/g005_button_semantic_ranking_relaxed_prefix20k_h200_diagnostic.json` and `artifacts/idm/g005_button_semantic_ranking_relaxed_prefix20k_h200_reservation_context.json`.
- Result: on 2,000 target rows, relaxed-budget masked-diffusion IDM predicted `175` mouse-button examples with no-button FPR `0.08871794871794872`, but exact TP `0`, semantic overlap `0`, and all predicted button tokens collapsed to `MOUSE_LEFT_DOWN` while ground truth included left/right/middle down/up events.
- Offset/mapping diagnostics do not rescue the branch: best offset has semantic overlap `1`, and best target-label-only one-to-one mapping gives only `1` exact TP (`F1≈0.0089`). This confirms the next G005 branch must improve recipe-faithful button-token semantic conditioning/ranking, not merely thresholds/budget size or target-label calibration.

## 2026-05-28 KST — Next G005 branch: focal button-transition + train-only prior correction

- Implemented a recipe-faithful follow-up after semantic diagnostic showed single-token mouse-button collapse: the masked-diffusion IDM button-transition slot can now use focal categorical loss and train-only conditional button-class prior correction.
- Candidate config: `configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_button_class_focal_prior_prefix20k.yaml`. It keeps compact luma-window CNN video tokens, noncausal masked action-token planes, categorical no-button/button-transition denoising, iterative confidence unmasking, and calibration-only budget selection; the new prior correction redistributes button-token mass using train labels only while preserving the model's event/no-event probability.
- Intended bounded H200 probe criterion: avoid `MOUSE_LEFT_DOWN` collapse, recover nonzero exact mouse-button TP/F1, and keep target no-button FPR near or below `0.10` before any full-corpus 4×H200 promotion.

## 2026-05-28 KST — G005 focal-prior button-class H200 prefix probe

- Ran commit `ac5a35f` on 1×H200 reservation `rsv-jeonghunpark-20260528-e44aec` / pod `prod-rsv-jeonghunpark-20260528-e44aec`; copied evidence locally and cancelled the reservation after artifact copy.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_class_focal_prior_prefix20k_h200_*`, including run, GPU monitor, W&B status, resolved config, paper metrics, compact summary, semantic diagnostic, reservation context, and diagnosis. W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/o3u03fjy`.
- Result remains negative/non-terminal: keyboard key accuracy improved to `0.020469596628537028` and target no-button FPR stayed within gate (`0.07384615384615385`), but exact mouse-button TP remained `0`, strict button F1 `0.0`, and semantic diagnostic still showed all predicted button tokens collapsed to `MOUSE_LEFT_DOWN` (`146` predictions, semantic overlap `0`).
- Diagnosis: focal loss plus train-only conditional prior correction is insufficient; the bottleneck is not just class-prior imbalance. Next recipe-faithful branch should improve video-conditioned button timing/semantics directly, e.g. add click/visual-change auxiliary denoising from luma temporal deltas or train a broader multi-shard prefix before relying on sparse button semantics.

## 2026-05-28 KST — Next G005 branch: larger focal-prior prefix scaling

- Implemented a larger recipe-faithful scaling candidate after 20k-row focal/prior still collapsed mouse-button semantics: `configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_button_class_focal_prior_prefix320k.yaml`.
- It keeps the public FDM-1-shaped video-token + noncausal masked action-token IDM, focal categorical mouse-button transition denoising, train-only conditional prior correction, and iterative confidence-budgeted unmasking, but increases training coverage to `320,000` rows and calibration to up to `10,000` held-out train rows.
- Intended bounded H200 probe criterion: determine whether sparse button semantic collapse is primarily a 20k-prefix coverage issue before making heavier architecture changes or full-corpus 4×H200 launches.

## 2026-05-28 KST — G005 prefix320k calibration throttle

- The first `prefix320k` H200 attempt at `0ad160b` completed training/checkpoint quickly but entered a CPU-bound one-row-at-a-time calibration path with sustained GPU idle; it was terminated before metric outputs to preserve GPU time.
- Updated the prefix320k config to keep `320,000` training rows but bound calibration to `2,000` held-out train rows and target evaluation to `5,000` rows (`button_event_budget_max_target_rows=5000`). This keeps the probe recipe-faithful while avoiding avoidable CPU-bound GPU reservation waste.

## 2026-05-28 KST — G005 focal-prior prefix320k H200 probe

- Ran bounded prefix320k config at commit `e137353` on 1×H200 reservation `rsv-jeonghunpark-20260528-979d0f` / pod `prod-rsv-jeonghunpark-20260528-979d0f`; copied evidence locally and cancelled the reservation after artifact copy.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_class_focal_prior_prefix320k_h200_*`, including run, GPU monitor, W&B status, resolved config, paper metrics, compact summary, semantic diagnostic, reservation context, and diagnosis. W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/hq7lrhqs`.
- Result remains negative/non-terminal: scaling to `320,000` training rows reduced target no-button FPR to `0.013382746551369158`, but keyboard key accuracy collapsed to `0.0`, exact mouse-button TP stayed `0`, strict button F1 `0.0`, and semantic diagnostic still had zero semantic overlap (`46` predictions all `MOUSE_RIGHT_DOWN` on the 2k diagnostic prefix).
- Diagnosis: the 20k failure was not simply coverage; larger data with current focal/prior button-class objective is still not enough. Next branch should change the recipe-faithful modeling signal, e.g. video-token pretraining/rawer frame features or batched calibration/prediction infrastructure before full 4×H200 scaling.


### 2026-05-28T23:40:43+09:00 KST — Global course correction: FDM-1 recipe fidelity for G005+

User correction is global for the renewed ultragoal. Preserve the existing goal list, but interpret G005+ as: train/evaluate the publicly inferable FDM-1 IDM/FDM recipe on D2E, not arbitrary architecture/objective search. Public FDM-1 anchors from https://si.inc/posts/fdm1/: video encoder/compression-style screen-video tokens; IDM is non-causal masked-diffusion over masked action tokens conditioned on all frames with iterative high-confidence unmasking; FDM is autoregressive next-action prediction over interleaved frame/action data; action space includes key press/release, scroll, exponentially binned mouse deltas, and next-click position/trajectory signals. D2E anchors from https://worv-ai.github.io/d2e/: OWA desktop/game demonstrations and Generalist-IDM next-event/temporal-offset baseline metrics. Novel work is allowed only to make this public FDM-1 recipe practical on D2E and beat D2E G-IDM targets; old supervised/table/heuristic branches are diagnostics unless rebuilt recipe-faithfully.

Current implementation follow-up: batched factorized masked-diffusion IDM calibration/prediction has been added locally so larger G005 recipe probes do not waste H200 time in one-row CPU-bound post-checkpoint loops. This supports continued FDM-1-shaped IDM exploration; it is not a G005 completion claim.


## 2026-05-29T00:02:04+09:00 KST — G005 batched prefix320k H200 probe evidence

- Committed batched calibration/prediction infra at `35f3881`, then ran the recipe-shaped factorized masked-diffusion IDM prefix320k config on 1×H200 reservation `rsv-jeonghunpark-20260528-222734` / pod `prod-rsv-jeonghunpark-20260528-222734`; copied redacted evidence locally and cancelled the reservation (`after_status=cancelled`).
- W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/1zgsn8ei`. Evidence files use prefix `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_button_class_focal_prior_prefix320k_batched_h200_*`, including compact summary, paper metrics, semantic diagnostic, GPU monitor, train log, resolved config, and redacted reservation context.
- Result is negative/non-terminal for G005: train rows `320000`, target rows `5000`, wall clock `566.8s`; no-button FPR `0.014000411776816966` is under 0.10 but keyboard key accuracy `0.0`, mouse-button strict F1 `0.0`, and semantic button overlap `0` remain zero.
- Interpretation: batching solved the previous post-checkpoint one-row CPU loop enough to finish the bounded probe, but the compact luma/focal-prior model still fails FDM-1-style action-token semantics. Next recipe-faithful branch should strengthen screen-video tokenization/pretraining and press/release span diffusion rather than further tuning this collapsed button-class head.


## 2026-05-29T00:07:39+09:00 KST — Next G005 recipe-faithful branch: masked video-token pretraining

- Added a stronger FDM-1-recipe-faithful IDM branch after the batched prefix320k negative result: compact luma-window encoder can now be pretrained with masked luma-token reconstruction before action-token denoising, approximating FDM-1's reported video-encoder masked compression objective on available D2E compact screen-video tokens.
- New config: `configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_video_pretrain_prefix320k.yaml`. It preserves the factorized noncausal masked-diffusion IDM, focal button-transition objective, train-only conditional prior correction, and batched calibration/prediction, but adds `video_encoder_pretrain_epochs=1` and `video_reconstruction_aux_weight=0.05`.
- This branch is still a bounded G005 probe, not a completion claim; intended H200 criterion is whether masked video-token pretraining improves keyboard/button semantics over the zero-overlap focal-prior prefix baseline while keeping no-button FPR under 0.10.


## 2026-05-29T00:24:19+09:00 KST — G005 video-pretrain prefix320k H200 probe

- Ran new FDM-1-recipe branch `7084ef6` on 1×H200 reservation `rsv-jeonghunpark-20260529-cf059a` / pod `prod-rsv-jeonghunpark-20260529-cf059a`; copied redacted evidence locally and cancelled the reservation (`after_status=cancelled`). W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/jikumwsq`.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_video_pretrain_prefix320k_h200_*` plus `artifacts/idm/g005_video_pretrain_prefix320k_reservation_context.json`. The large full run/summary JSONs are not committed; hashes are recorded in the compact summary and raw outputs remain on the PVC worktree.
- Result remains negative/non-terminal: masked luma reconstruction pretraining completed (`loss=0.015299568372561871`), train rows `320000`, target rows `5000`, wall clock `671.2s`; no-button FPR `0.014000411776816966` stayed low, but keyboard key accuracy `0.0`, mouse-button F1 `0.0`, and semantic button overlap `0` are still zero.
- Diagnosis: compact video-token pretraining alone is not enough. Next recipe-faithful branch should model full action-token spans/press-release timing with noncausal masked diffusion over temporal windows, not only per-frame button-class ranking.


## 2026-05-29T00:28:23+09:00 KST — Next G005 branch: temporal button action-token span diffusion

- Implemented the next recipe-faithful branch after masked video-token pretraining did not improve key/button metrics: the factorized masked-diffusion IDM can now train an auxiliary noncausal button span head over neighboring press/release action-token offsets.
- New config: `configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_video_pretrain_span_prefix320k.yaml`. It keeps masked luma video-token pretraining and adds `button_span_diffusion=true`, offsets `[-2,-1,0,1,2]`, `button_span_loss_weight=2.0`, and uses the current-offset span logits for button probabilities.
- Rationale: FDM-1's public IDM labels masked action-token sequences noncausally; D2E button failures look like press/release timing/type collapse, so the next bounded probe should learn temporal action-token spans rather than only isolated per-frame button classes.


## 2026-05-29T00:45:58+09:00 KST — G005 temporal span prefix320k H200 probe

- Ran temporal button action-token span branch `3e286ab` on 1×H200 reservation `rsv-jeonghunpark-20260529-818342` / pod `prod-rsv-jeonghunpark-20260529-818342`; copied redacted evidence locally and cancelled the reservation (`after_status=cancelled`). W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/l8q0m6ih`.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_video_pretrain_span_prefix320k_h200_*` plus `artifacts/idm/g005_span_prefix320k_reservation_context.json`.
- Result is still non-terminal but is the first positive mouse-button signal in this renewed branch: exact TP `1`, strict button F1 `0.005263157894736842`, no-button FPR `0.047971999176446366` under 0.10. However keyboard key accuracy remains `0.0` and 2k semantic diagnostic overlap remains `0`.
- Next action: keep span diffusion but add held-out span-aware calibration/recovery (optimize F-beta under no-button FPR <= 0.10) and add analogous key press/release span head before another full training launch.


## 2026-05-29T00:52:09+09:00 KST — Next G005 branch: key span + span-aware aggregation/calibration

- Implemented the follow-up to the weak temporal-button-span signal: factorized masked-diffusion IDM now supports a temporal key press/release span head and max/mean/offset span aggregation for key and button probabilities before held-out threshold/budget calibration.
- New config: `configs/model/idm_factorized_masked_diffusion_d2e_luma_window5_cnn_keyspan_calibrated_prefix320k.yaml`. It keeps masked luma video-token pretraining and button span diffusion, adds `key_span_diffusion=true`, offsets `[-2,-1,0,1,2]`, `key_probability_source=key_span`, and max span aggregation for both key and button probabilities.
- Rationale: previous span probe produced the first nonzero mouse-button TP while keyboard remained zero; this branch brings key press/release into the same FDM-1-shaped noncausal action-token span recipe and lets calibration rank high-confidence tokens across local temporal offsets.

## 2026-05-29 KST — Global correction reaffirmed: FDM-1 recipe fidelity remains binding

- Preserve the existing renewed ultragoal IDs/statuses, but interpret all G005+ implementation and evidence as reproducing the publicly inferable FDM-1 IDM/FDM training recipe on D2E, not arbitrary architecture/objective exploration.
- Binding anchors: video encoder/compression-style screen-video tokens; IDM as noncausal masked diffusion over masked action-token sequences with iterative confidence unmasking; FDM as autoregressive next-action prediction over interleaved frame/action tokens; action tokens covering key press/release, scroll, binned mouse deltas, click/trajectory signals.
- Novel exploration is allowed only to approximate unpublished FDM-1 internals faithfully on D2E and beat D2E paper/released G-IDM targets. Older heuristic/supervised/table branches are diagnostic unless rebuilt behind recipe-alignment gates.

## 2026-05-29T01:16:00+09:00 KST — G005 key-span calibrated prefix320k H200 probe

- Ran recipe-faithful key/button temporal span branch `45d7327` on 1×H200 reservation `rsv-jeonghunpark-20260529-eb3c31` / pod `prod-rsv-jeonghunpark-20260529-eb3c31`; copied redacted evidence locally and cancelled the reservation (`after_status=cancelled`). W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/ewiclkhm`.
- Evidence files: `artifacts/idm/g005_idm_factorized_masked_diffusion_luma_window5_cnn_keyspan_calibrated_prefix320k_h200_*` plus `artifacts/idm/g005_keyspan_prefix320k_reservation_context.json`.
- Result is negative/non-terminal: train rows `320000`, target rows `5000`, wall clock `826.9s`; no-button FPR stayed below gate (`0.04159`), but keyboard key accuracy `0.0`, mouse-button accuracy `0.0`, strict mouse-button F1 `0.0`, exact button TP `0`, and semantic button overlap `0`.
- Diagnosis: key-span max aggregation did not rescue the compact factorized luma branch and regressed from the prior one-TP button-span probe. Next recipe-faithful step should avoid threshold-only tweaks and move toward a sequence-level masked action-token IDM with explicit temporal token positions plus tensorized/cached feature loading so H200 work is GPU-dominant.

## 2026-05-29T01:36:00+09:00 KST — G005 temporal sequence masked-diffusion prefix80k H200 probe

- Ran new recipe-faithful temporal sequence IDM branch `aa481c7` on 1×H200 reservation `rsv-jeonghunpark-20260529-72170e` / pod `prod-rsv-jeonghunpark-20260529-72170e`; copied redacted evidence locally and cancelled the reservation (`after_status=cancelled`). W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/gjzroj4m`.
- Evidence files: `artifacts/idm/g005_idm_temporal_masked_diffusion_luma_window5_prefix80k_h200_*` plus `artifacts/idm/g005_temporal_prefix80k_reservation_context.json`.
- Result is negative/non-terminal: train rows `80000`, target rows `5000`, wall clock `312.2s`, temporal window `7`, vocab `148`; no-button FPR is `0.0` only because button predictions collapsed to `0`, while keyboard key accuracy `0.0`, mouse-button F1 `0.0`, exact button TP `0`, and semantic overlap `0`.
- Operationally, GPU utilization improved versus factorized keyspan (peak about `70%` on the monitor), so the temporal sequence path is a better recipe-faithful execution surface. Next branch should keep sequence masked diffusion but add split-safe non-noop confidence-budget/top-k unmasking plus focal/logit-adjusted sparse action-token loss before another H200 probe.

## 2026-05-29T02:03:24+09:00 KST — G005 temporal budget H200 probe is negative; family-budget branch added

- Evidence copied from reservation `rsv-jeonghunpark-20260529-085253` / pod `prod-rsv-jeonghunpark-20260529-085253`, then the reservation was cancelled after evidence copy. Redacted context: `artifacts/idm/g005_temporal_budget_prefix80k_reservation_context.json`.
- Probe commit/run: `f885eb0`, W&B run `itixp40j`, config `configs/model/idm_temporal_masked_diffusion_d2e_luma_window5_budget_prefix80k.yaml`.
- Terminal evidence: `artifacts/idm/g005_idm_temporal_masked_diffusion_luma_window5_budget_prefix80k_h200_run.json` status=pass and `artifacts/idm/g005_idm_temporal_masked_diffusion_luma_window5_budget_prefix80k_h200_compact_summary.json` status=`nonterminal_negative_probe`.
- Metrics on the bounded 80k/5k probe remain far below G005: keyboard key accuracy `0.0`, mouse-button accuracy/F1 `0.0`, no-button FPR `0.0002058884`, mouse-move Pearson X/Y `-0.000255/-0.025478`. Do **not** checkpoint G005.
- Diagnosis: the recipe-shaped temporal masked-diffusion surface runs end-to-end, but single global non-noop budget collapses to frequent small mouse-move tokens and does not learn key/button semantics. Added the next recipe-faithful branch: family-specific held-out-train confidence budgets for keyboard/mouse-button/mouse-move action-token families, preserving the public FDM-1 iterative unmasking shape.

## 2026-05-29T02:24:02+09:00 KST — G005 family-budget temporal probe is also negative

- Reservation `rsv-jeonghunpark-20260529-92048b` ran managed Codex pod `prod-rsv-jeonghunpark-20260529-92048b` on node 4 GPU 1, then was cancelled after evidence copy. A failed precursor reservation `rsv-jeonghunpark-20260529-b05f11` used raw `image_path` and exited with missing `SHELL_SESSION_ID`; that context is preserved separately as startup-failure evidence.
- Probe commit/config: `9db975d`, `configs/model/idm_temporal_masked_diffusion_d2e_luma_window5_family_budget_prefix80k.yaml`, W&B sidecar evidence in `artifacts/idm/g005_idm_temporal_masked_diffusion_luma_window5_family_budget_prefix80k_h200_wandb_status.json`.
- Terminal evidence: `artifacts/idm/g005_idm_temporal_masked_diffusion_luma_window5_family_budget_prefix80k_h200_run.json` status=pass and compact summary status=`nonterminal_negative_probe`.
- Metrics: keyboard key accuracy improved from `0.0` to `0.01165`, but mouse-button accuracy/F1 remain `0.0`, no-button FPR is `0.00453`, mouse-move Pearson X is `-0.01085`, Y is undefined. This is **not** a G005 paper-target win and must not be checkpointed.
- Next recipe-faithful direction: add sparse key/button event-presence auxiliary losses/heads inside the temporal masked-diffusion IDM representation and use them only to bias iterative action-token unmasking; also vectorize prediction before any larger full-corpus/4xH200 promotion.

## 2026-05-29T02:46:57+09:00 KST — G005 event-aux temporal probe is negative

- Commit/config: `1675273`, `configs/model/idm_temporal_masked_diffusion_d2e_luma_window5_event_aux_prefix80k.yaml`.
- Reservation `rsv-jeonghunpark-20260529-f72f38` ran managed Codex pod `prod-rsv-jeonghunpark-20260529-f72f38` and was cancelled after evidence copy. Redacted context: `artifacts/idm/g005_event_aux_prefix80k_reservation_context.json`.
- Terminal evidence: `artifacts/idm/g005_idm_temporal_masked_diffusion_luma_window5_event_aux_prefix80k_h200_run.json` status=pass; compact summary status=`nonterminal_negative_probe`.
- Metrics on target are all nonterminal: keyboard key accuracy `0.0`, mouse-button accuracy/F1 `0.0`, no-button FPR `0.0`, mouse movement Pearson undefined. Train-heldout calibration showed keyboard signal (`0.03466`) but target emission abstained/regressed. Do **not** checkpoint G005.
- Next recipe-faithful direction: bootstrap masked-diffusion IDM with video/action-token retrieval priors from train video embeddings, then use those priors as denoising/candidate bias without target-label calibration or post-hoc metric heuristics.

## 2026-05-29T03:13:00+09:00 KST — G005 retrieval-prior temporal probe is negative

- Ran the recipe-faithful temporal masked-diffusion IDM with train-only video/action retrieval priors at commit `3190b99`, config `configs/model/idm_temporal_masked_diffusion_d2e_luma_window5_retrieval_prior_prefix80k.yaml`, on reservation `rsv-jeonghunpark-20260529-64e750` / pod `prod-rsv-jeonghunpark-20260529-64e750`; copied evidence locally and cancelled the reservation after evidence copy (`reservation_status_after_cancel=cancelled`).
- Evidence files: `artifacts/idm/g005_idm_temporal_masked_diffusion_luma_window5_retrieval_prior_prefix80k_h200_*`, `artifacts/idm/g005_idm_temporal_masked_diffusion_luma_window5_retrieval_prior_prefix80k_summary.json`, and redacted `artifacts/idm/g005_retrieval_prior_prefix80k_reservation_context.json`.
- Result remains negative/non-terminal for `G005-g014-idm-full-paper-target`: bounded 80k train / 5k target probe, retrieval index rows `78000`, keyboard key accuracy `0.01042054335690361`, mouse-button accuracy/F1 `0.0`, mouse-move Pearson X `0.02968279912722567`, no-button FPR `0.0`; far below paper targets (`keyboard 0.73`, `button 0.957`, `Pearson X/Y 0.796/0.783`).
- Diagnosis: retrieval prior preserved low false positives but still collapsed action identity/semantics (button semantic overlap `0`, exact button TP `0`, all 143 strict button examples missed). Do **not** checkpoint G005; continue with a recipe-faithful branch that improves video-token/action-token alignment or sparse action identity learning, and vectorize post-checkpoint retrieval/prediction before larger/full-corpus H200 runs.

## 2026-05-29T03:18:00+09:00 KST — Next G005 branch: PAD-aware temporal action-token diffusion

- Implemented the next recipe-faithful branch after retrieval priors failed: temporal masked-diffusion IDM can now preserve `<FDM1_ACTION_PAD>` in unused fixed-width action slots and exclude those PAD positions from denoising loss instead of converting every padded slot to repeated `NOOP` targets.
- New config: `configs/model/idm_temporal_masked_diffusion_d2e_luma_window5_padaware_prefix80k.yaml`. It keeps video-token encoder/pretraining, noncausal temporal masked action-token diffusion, family confidence budgets, event auxiliaries, and train-only retrieval priors, but adds `preserve_pad_action_slots=true`, four bounded epochs, and recipe-alignment metadata for PAD-aware sparse action-token slots.
- Rationale: prior temporal probes trained on many repeated `NOOP` labels from padded slots, which is a plausible sparse-action identity collapse mechanism. PAD-aware training is still aligned with the public FDM-1 action-token-sequence recipe and should be tested before larger/full-corpus promotion.

## 2026-05-29T03:41:00+09:00 KST — G005 PAD-aware temporal probe is negative but improves keyboard

- Ran commit/config `fdf2ed6` / `configs/model/idm_temporal_masked_diffusion_d2e_luma_window5_padaware_prefix80k.yaml` on 1×H200 reservation `rsv-jeonghunpark-20260529-e11fb5`; copied evidence locally and cancelled the reservation after evidence copy (`reservation_status_after_cancel=cancelled`). W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/av941smb`.
- Evidence prefix: `artifacts/idm/g005_idm_temporal_masked_diffusion_luma_window5_padaware_prefix80k_h200_*`; compact summary status is `nonterminal_negative_probe`.
- Result is still far from G005 completion: keyboard key accuracy improved to `0.02465233881163085` (strict keyboard `0.07952468007312614`), but mouse-button accuracy/F1 remain `0.0`, diagnostic predicted button examples `0/50`, mouse Pearson X/Y `-0.03867/-0.01642`, and no-button FPR is `0.0` only because button emission abstained.
- Decision: PAD-aware action slots are useful directionally and should stay, but G005 must continue. Next branch should add recipe-faithful video/action-token identity learning (e.g. contrastive token-set matching or family-aware token-presence logits inside masked diffusion) and vectorize/shard retrieval/prediction before larger/full-corpus H200 promotion.

## 2026-05-29T03:47:00+09:00 KST — Next G005 branch: token-presence identity auxiliary

- Implemented the follow-up to PAD-aware temporal diffusion: temporal masked-diffusion IDM now supports a multi-label token-presence auxiliary over the same action-token vocabulary, trained from non-PAD/non-NOOP action-token sets and blended into center-token candidate scores before train-heldout family calibration.
- New config: `configs/model/idm_temporal_masked_diffusion_d2e_luma_window5_token_presence_prefix80k.yaml`. It keeps PAD-aware temporal masked action-token diffusion, event auxiliaries, family budgets, and train-only retrieval priors, but adds `temporal_token_presence_auxiliary=true`, `token_presence_aux_weight=0.6`, and token-presence candidate blending.
- Rationale: PAD-aware loss improved keyboard but button emission still abstained. This remains FDM-1-recipe-faithful because it learns identity over the action-token set from the masked-diffusion temporal representation, rather than adding target-label post-processing or arbitrary supervised shortcuts.

## 2026-05-29T04:14:00+09:00 KST — G005 token-presence temporal probe regressed

- Ran commit/config `dbe8b24` / `configs/model/idm_temporal_masked_diffusion_d2e_luma_window5_token_presence_prefix80k.yaml` on 1×H200 reservation `rsv-jeonghunpark-20260529-37c4bd`; copied evidence locally and cancelled the reservation after evidence copy (`reservation_status_after_cancel=cancelled`). W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/qsi17e52`.
- Evidence prefix: `artifacts/idm/g005_idm_temporal_masked_diffusion_luma_window5_token_presence_prefix80k_h200_*`; compact summary status is `nonterminal_negative_probe`.
- Result regressed from PAD-aware-only: keyboard key accuracy `0.00545950864422202`, strict keyboard `0.003656307129798903`, mouse-button accuracy/F1 still `0.0`, diagnostic predicted button examples `0/50`, mouse Pearson X/Y `-0.0014/-0.00045`. Calibration reported mouse-button family `skipped/no_family_candidates`.
- Decision: do not checkpoint G005. Keep PAD-aware slots, but reject this token-presence blend as configured. Next work should add diagnostics for per-family candidate/logit availability and/or a family-gated temporal button transition identity head before another H200 run.
