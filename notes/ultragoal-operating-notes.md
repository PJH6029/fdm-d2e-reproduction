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
