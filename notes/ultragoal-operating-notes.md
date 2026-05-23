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

## Current ultragoal gates

- Active aggregate Codex/OMX ultragoal covers G001–G009 in `.omx/ultragoal/goals.json`; do not complete the aggregate Codex goal until all stories are complete.
- Current completed gates:
  - `G001-data-universe-audit`: full D2E-480p + D2E-Original universe manifest and storage/license report.
  - `G002-split-leakage-contract`: temporal, heldout-recording, and heldout-game split/leakage contract.
  - `G003-d2e-only-idm`: full-corpus D2E-only IDM train/eval, canonical/accel64 promotion, split stats, and completion audit checkpointed in OMX.
  - `G007-runtime-sdk-adapter`: reusable SDK/action decoder/safety adapter/latency logger/deterministic replay contract only; this is not G008 live-game success.
- G008 live-suite protocol exists (`docs/live_open_game_suite.md`, `configs/harness/g008_live_open_game_suite.yaml`, `scripts/validate_live_game_suite.py`) and requires >=3 open-source graphical games, >=3 tasks, 5 seeds/task, video/replay/latency/failure logs, and statistical comparison; protocol/readiness evidence is not G008 completion.
- Final completion audit now exists (`configs/eval/final_quality_gates.yaml`, `scripts/validate_final_quality_gates.py`, `artifacts/reproducibility/final_quality_gate_audit.json`) and must pass before aggregate goal completion; current expected status is fail while G003-G009 remain incomplete.
- G006 evaluation readiness audit exists (`configs/eval/g006_evaluation_readiness.yaml`, `scripts/validate_g006_evaluation_readiness.py`, `artifacts/eval/g006_evaluation_readiness_audit.json`) and must pass before G006 checkpoint; current expected status is fail until final split-aware endpoint stats, failure analysis, and claim taxonomy exist.
- Current active gate:
  - `G004-d2e-only-fdm-4xh200`: D2E-only FDM 4×H200 run on the existing MLXP pod.
- Pending gates:
  - `G005-aux-data-best-model`, `G006-evaluation-failure-analysis`, `G008-live-game-suite`, `G009-report-repro-package`.
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
- Cluster workflow: edit locally, push, then pull in the pod PVC path `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction` before running GPU jobs.
- Docker registry username: `pjh6029`; auth is already configured.

## Current cluster handoff

- Current reservation/pod: `rsv-jeonghunpark-20260521-76e25a` / `prod-rsv-jeonghunpark-20260521-76e25a` in namespace `p-production`.
- Pod repo path: `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`.
- G003 is complete in OMX; do not treat older G003 run/watcher notes as active blockers.
- Current G004 run path: `bash scripts/run_g004_d2e_full_fdm_4xh200.sh` on pod checkout `d38a3b1`, parent PID `262618`, watcher PID `262772`.
- Latest G004 progress snapshot is committed at `artifacts/fdm/g004_deferred_materialization_progress_snapshot.json`; as of 2026-05-23T00:21:04Z materialization was actively growing (train parts `16` / `368,945,307,285` bytes; target parts `11` / `63,658,038,599` bytes), GPUs were still 0% because this was pre-DDP CPU/IO materialization, and no split summary/train history existed yet.
- Do not pull latest origin into the pod while G004 parent PID `262618` or child rank/materialization workers are alive. Monitor until split summary/cache/train-history appears, then verify GPU utilization during DDP training and final prediction.
- G003 progress monitor exists (`scripts/monitor_g003_progress.py`, `docs/g003_progress_monitoring.md`) for non-mutating shard/PID/stale-progress summaries; monitor output is progress evidence only, not G003 completion.
- G003 live health audit exists (`scripts/audit_g003_live_health.py`) for non-mutating parent/extractor/watcher/GPU-monitor process-topology summaries; use it for handoff/recovery evidence but not for completion claims.
- Historical G003 accel64 process notes for parent PID `251593` are no longer current; keep them only as provenance for the completed G003 checkpoint. Do not block G004 on old G003 worker/PID state.
- G003 resume planner exists (`scripts/plan_g003_resume.py`, `artifacts/idm/g003_resume_plan.json`) but should not be used unless the completed G003 evidence is later found corrupt or missing.
- G004 FDM training now requires explicit train-core pseudo-labels from the completed G003 IDM checkpoint (`scripts/predict_idm_streaming.py` + `configs/model/idm_streaming_d2e_full_compact_predict_fdm_train.yaml`) and evaluates on untouched `target_all_eval`; do not revert to target_all_eval recording-tail training for completion evidence. The G004 model feature mode is `summary_causal_compact_grid8_time_prior_action` to avoid next-frame inverse-dynamics leakage and include previous-action context.
- G004 post-run watcher exists (`scripts/watch_g004_then_finalize.py`) and consumes `outputs/cluster/g004_d2e_full_fdm_4xh200.pid`; it runs the non-mutating finalizer after the parent exits but never checkpoints OMX/Codex state.
- G003→G004 chain watcher exists (`scripts/watch_g003_then_launch_g004.py`) and may run in the pod with `--launch --start-g004-watcher`; it launches G004 only after G003 finalization and G003 audit pass, then starts the G004 post-run watcher. It never checkpoints OMX/Codex state. As of 2026-05-22 14:09 KST, the pod still had a stale `outputs/cluster/g003_to_g004_chain_watcher.pid` and `artifacts/fdm/g003_to_g004_chain_summary.json`, but no running watcher process; after G003 is promoted and OMX-checkpointed, start a fresh current watcher with `--require-g003-goal-checkpoint`.
- G005 launch planner/watcher exists (`scripts/plan_g005_launch.py`, `scripts/watch_g005_then_finalize.py`) for D2E+aux best-model preparation; both preserve G003/G004 D2E-only gates and never mutate OMX/Codex state.
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
- G005 aux materialization has finished: as of 2026-05-22 14:12 KST, materializer PID `49075` is not running, `artifacts/aux/g005_aux_materialization_progress.json` reports `status=pass`, `artifacts/aux/g005_aux_runtime_env.json` reports `status=pass`, and `artifacts/aux/g005_aux_namespace_manifest.json` reports `completion_ready=true`. The watcher summary still reports `g005_launch_not_ready`, and G005 completion audit remains fail-closed (`status=fail`, `error_count=51`) because G003/G004 and G005 training/ablation prerequisites are incomplete. This is parallel source staging only; do not interpret it as G005 launch/completion.
- G006 readiness planner/watcher exists (`scripts/plan_g006_readiness.py`, `scripts/watch_g006_then_finalize.py`) for final evaluation/failure-analysis handoff; both are non-mutating and require G003/G004/G005 evidence before finalization.
- G008 readiness planner exists (`scripts/plan_g008_readiness.py`, `artifacts/harness/g008_readiness_plan.json`) for live open-source graphical-game collection prep; it is non-mutating, does not launch games, and currently blocks as expected until D2E-only prerequisites/checkpoint metadata and live-game host binaries are available.
- G009 readiness planner exists (`scripts/plan_g009_readiness.py`, `artifacts/reproducibility/g009_readiness_plan.json`) for final report/repro package prep; it is non-mutating, does not refresh audits/packages, and currently blocks as expected until G003-G008 are complete.
- Runtime adapter contract evidence: `artifacts/runtime/g007_runtime_replay_adapter_contract.json`; commits `34cddb7` and `e858114`; OMX checkpointed `G007-runtime-sdk-adapter` complete locally.
- `G003-d2e-only-idm` is already checkpointed complete in OMX. Do not repeat the G003 checkpoint unless reconciling a ledger corruption.
- Streaming IDM metadata now records config/data/split/source provenance (`checkpoint_metadata.json`, `resolved_config.json`); ensure the pod checkout includes this before the G003 extraction reaches training.
- Commit `6974f38` adds automatic split-stat generation to future G003/G004 run wrappers. The old accel64 G003 parent PID `251593` is historical; current live monitoring should focus on G004 parent PID `262618`.
