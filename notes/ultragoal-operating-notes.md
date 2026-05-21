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
  - `G007-runtime-sdk-adapter`: reusable SDK/action decoder/safety adapter/latency logger/deterministic replay contract only; this is not G008 live-game success.
- G008 live-suite protocol exists (`docs/live_open_game_suite.md`, `configs/harness/g008_live_open_game_suite.yaml`, `scripts/validate_live_game_suite.py`) and requires >=3 open-source graphical games, >=3 tasks, 5 seeds/task, video/replay/latency/failure logs, and statistical comparison; protocol/readiness evidence is not G008 completion.
- Final completion audit now exists (`configs/eval/final_quality_gates.yaml`, `scripts/validate_final_quality_gates.py`, `artifacts/reproducibility/final_quality_gate_audit.json`) and must pass before aggregate goal completion; current expected status is fail while G003-G009 remain incomplete.
- G006 evaluation readiness audit exists (`configs/eval/g006_evaluation_readiness.yaml`, `scripts/validate_g006_evaluation_readiness.py`, `artifacts/eval/g006_evaluation_readiness_audit.json`) and must pass before G006 checkpoint; current expected status is fail until final split-aware endpoint stats, failure analysis, and claim taxonomy exist.
- Current active gate:
  - `G003-d2e-only-idm`: full-corpus D2E-only IDM extraction/training/evaluation.
- Pending gates:
  - `G004-d2e-only-fdm-4xh200`, `G005-aux-data-best-model`, `G006-evaluation-failure-analysis`, `G008-live-game-suite`, `G009-report-repro-package`.
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
- Cluster workflow: edit locally, push, then pull in the pod PVC path `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction` before running GPU jobs.
- Docker registry username: `pjh6029`; auth is already configured.

## Current cluster handoff

- Current reservation/pod: `rsv-jeonghunpark-20260521-76e25a` / `prod-rsv-jeonghunpark-20260521-76e25a` in namespace `p-production`.
- Pod repo path: `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`.
- Current G003 full-corpus IDM run path: `NUM_SHARDS=16 bash scripts/run_g003_d2e_full_idm_parallel.sh`.
- G003 progress monitor exists (`scripts/monitor_g003_progress.py`, `docs/g003_progress_monitoring.md`) for non-mutating shard/PID/stale-progress summaries; monitor output is progress evidence only, not G003 completion.
- G003 live health audit exists (`scripts/audit_g003_live_health.py`) for non-mutating parent/extractor/watcher/GPU-monitor process-topology summaries; use it for handoff/recovery evidence but not for completion claims.
- For the current already-running integrated G003 process, attached GPU monitor PID `31950` is collecting `artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv` in the pod without restarting PID `9289`; after the run exits and monitor metadata is written, use `scripts/build_g003_attached_train_run_summary.py` to synthesize the required fail-closed train-run summary.
- G003 resume planner exists (`scripts/plan_g003_resume.py`, `artifacts/idm/g003_resume_plan.json`) and defers while the original parent PID is active; use only for operator-reviewed recovery if G003 stops before all shard summaries exist.
- G004 FDM training now requires explicit train-core pseudo-labels from the completed G003 IDM checkpoint (`scripts/predict_idm_streaming.py` + `configs/model/idm_streaming_d2e_full_compact_predict_fdm_train.yaml`) and evaluates on untouched `target_all_eval`; do not revert to target_all_eval recording-tail training for completion evidence. The G004 model feature mode is `summary_causal_compact_grid8_time_prior_action` to avoid next-frame inverse-dynamics leakage and include previous-action context.
- G004 post-run watcher exists (`scripts/watch_g004_then_finalize.py`) and consumes `outputs/cluster/g004_d2e_full_fdm_4xh200.pid`; it runs the non-mutating finalizer after the parent exits but never checkpoints OMX/Codex state.
- G003→G004 chain watcher exists (`scripts/watch_g003_then_launch_g004.py`) and may run in the pod with `--launch --start-g004-watcher`; it launches G004 only after G003 finalization and G003 audit pass, then starts the G004 post-run watcher. It never checkpoints OMX/Codex state.
- G005 launch planner/watcher exists (`scripts/plan_g005_launch.py`, `scripts/watch_g005_then_finalize.py`) for D2E+aux best-model preparation; both preserve G003/G004 D2E-only gates and never mutate OMX/Codex state.
- G004→G005 readiness chain exists (`scripts/watch_g004_then_plan_g005.py`) and can be started after G004 launch; it waits for G004 finalization/audit pass, then records G005 readiness only. It does not launch G005 training because aux source materialization/eval-hash evidence must be explicit.
- G005 eval-manifest hash builder exists (`scripts/build_g005_eval_manifest_hashes.py`, `artifacts/aux/d2e_eval_manifest_hashes.json`) and proves byte-identical temporal/heldout-recording/heldout-game D2E eval manifests for D2E-only vs D2E+aux comparisons; it does not materialize aux sources or start training.
- G005 aux action registry exists (`scripts/build_g005_aux_action_registry.py`, `artifacts/aux/g005_aux_action_registry.json`) and is required by the G005 completion audit. It records source-specific action heads and forbids collapsed/shared aux actions or direct aux claims on D2E keyboard/mouse endpoints.
- G005 aux source evidence builder exists (`scripts/build_g005_aux_source_evidence.py`, `artifacts/aux/g005_aux_source_materialization_evidence.json`) and scans `outputs/aux/<dataset_id>/...` for selected-source materialization, source-specific split hashes, action-head namespace, and D2E-heldout-overlap evidence; current expected status is blocked until aux source files are materialized.
- G005 aux materializer exists (`scripts/materialize_g005_aux_sources.py`, `artifacts/aux/g005_aux_materialization_plan.json`): default mode is plan-only; `--execute` downloads selected Zenodo/Hugging Face aux sources into `outputs/aux/<dataset_id>/raw` and writes source-level train/val/test manifests. Zenodo downloads use atomic `.part-<pid>` files and size/checksum validation before split manifests are written. It is materialization/provenance evidence only and does not authorize G005 training or D2E+aux claims.
- G005 aux materialization monitor exists (`scripts/monitor_g005_aux_materialization.py`, `artifacts/aux/g005_aux_materialization_progress.json`) for non-mutating partial-download telemetry: raw byte counts, partial/complete/missing source ids, PID state, and split-manifest readiness. It is progress telemetry only.
- G005 aux materialization integrity validator exists (`scripts/validate_g005_aux_materialization_integrity.py`, `artifacts/aux/g005_aux_materialization_integrity.json`) and is now called by the materialization watcher before source evidence. It validates post-download raw bytes/manifests and blocks source evidence if size/checksum/manifests are incomplete.
- G005 aux archive inventory builder exists (`scripts/build_g005_aux_archive_inventory.py`, `artifacts/aux/g005_aux_archive_inventory.json`) to inspect materialized raw archives and find heuristic action-label member names before writing source-specific loaders; it is not training or completion evidence.
- G005 aux materialization watcher exists (`scripts/watch_g005_aux_materialization.py`, `artifacts/aux/g005_aux_materialization_watcher_summary.json`): after the materializer exits, it rebuilds source evidence, namespace readiness, and fail-closed G005 launch readiness. It never starts G005 training or checkpoints OMX/Codex state.
- Current pod also has a background G005 aux materialization execution started after commit `317e974`: PID `49075`, log `artifacts/aux/g005_aux_materialization_execute.log`, output `artifacts/aux/g005_aux_materialization_execute_summary.json`, namespace root `outputs/aux/`. After commit `934d57a`, the materialization watcher was started as Python PID `49401` with output `artifacts/aux/g005_aux_materialization_watcher_summary.json`. This is parallel source staging only while G003 runs; do not interpret it as G005 launch/completion.
- G006 readiness planner/watcher exists (`scripts/plan_g006_readiness.py`, `scripts/watch_g006_then_finalize.py`) for final evaluation/failure-analysis handoff; both are non-mutating and require G003/G004/G005 evidence before finalization.
- G008 readiness planner exists (`scripts/plan_g008_readiness.py`, `artifacts/harness/g008_readiness_plan.json`) for live open-source graphical-game collection prep; it is non-mutating, does not launch games, and currently blocks as expected until D2E-only prerequisites/checkpoint metadata and live-game host binaries are available.
- G009 readiness planner exists (`scripts/plan_g009_readiness.py`, `artifacts/reproducibility/g009_readiness_plan.json`) for final report/repro package prep; it is non-mutating, does not refresh audits/packages, and currently blocks as expected until G003-G008 are complete.
- Runtime adapter contract evidence: `artifacts/runtime/g007_runtime_replay_adapter_contract.json`; commits `34cddb7` and `e858114`; OMX checkpointed `G007-runtime-sdk-adapter` complete locally.
- Do not checkpoint `G003-d2e-only-idm` complete until all 918 source variants are decoded/merged, streaming IDM training finishes, reports/metrics are validated, and evidence summaries are committed.
- Streaming IDM metadata now records config/data/split/source provenance (`checkpoint_metadata.json`, `resolved_config.json`); ensure the pod checkout includes this before the G003 extraction reaches training.
- Commit `6974f38` adds automatic split-stat generation to future G003/G004 run wrappers, but the current G003 parent PID `9289` was already running when the pod fast-forwarded. If the active run exits without `artifacts/eval/g003_split_statistical_comparisons_summary.json`, run `uv run python scripts/build_split_statistical_comparisons.py --config configs/eval/g003_split_statistics.yaml` manually before G003 completion audit/checkpoint.
