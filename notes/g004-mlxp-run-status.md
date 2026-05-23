# G004 MLXP run status

- Reservation: `rsv-jeonghunpark-20260521-76e25a`
- Pod: `prod-rsv-jeonghunpark-20260521-76e25a`
- Repo path on pod: `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`
- Launch commit: `c090d6a81f28239096aad649d9f3809b26d86da0`
- Launch artifact: `artifacts/fdm/g004_launch_start.json`
- Initial monitor artifact: `artifacts/fdm/g004_initial_monitor_snapshot.json`
- Postrun watcher artifact: `artifacts/fdm/g004_postrun_watcher_summary.json`

## 2026-05-22 initial status

G004 D2E-only 4xH200 run was launched with `scripts/run_g004_d2e_full_fdm_4xh200.sh`.
The initial stage is full train-core IDM pseudo-label generation for FDM training.
The initial monitor snapshot shows 16 shard-parallel prediction workers active across the four H200 GPUs and writing `prediction_recovery_parts/part_*` outputs.

Claim boundary: this is launch/active-run evidence only. G004 is not complete until the run summary, split statistics, G004 completion audit, and OMX checkpoint all pass.

## 2026-05-22 transition status

The full train-core IDM prediction stage completed on the pod: `artifacts/idm/idm_streaming_d2e_full_compact_fdm_train_core_pseudolabels_summary.json` reports 19,211,006 train-core pseudo-label records.
The run then entered `torchrun` and rank-0 FDM split materialization. Current transition evidence is stored at `artifacts/fdm/g004_transition_to_fdm_materialization_snapshot.json`.

Claim boundary: G004 remains incomplete until FDM training/evaluation, split statistics, G004 audit, and OMX checkpoint pass.

## 2026-05-23 parallel restart status

The original G004 parent `260000` was still in pre-patch serial CPU/IO FDM materialization with all four H200 GPUs idle. It was terminated deliberately after restart evidence was captured, and its partial serial outputs were preserved on the pod under `outputs/fdm_streaming_d2e_full_compact_serial_backup_20260522T233926Z` plus artifact backup `artifacts/fdm/g004_serial_restart_backup_20260522T233926Z`.

The pod was fast-forwarded to commit `6e40ec6`, which adds parallel FDM materialization from IDM prediction parts, increases training-cache workers to 16, and enables 16-way final checkpoint prediction over target shards. A fresh G004 run was launched with parent PID `262210` and watcher PID `262232`; launch evidence is mirrored locally at `artifacts/fdm/g004_parallel_restart_launch.json`, and the restart decision evidence is mirrored at `artifacts/fdm/g004_parallel_restart_decision.json`.

At the first post-relaunch inspection, 16 spawned materialization workers were actively writing `outputs/fdm_streaming_d2e_full_compact/materialization_parts/train/part_*.jsonl`. This is still an offline CPU/IO materialization stage, so GPU utilization may remain near 0% until split materialization, monolith concat, stats/cache build, and DDP training begin.

Claim boundary: this is active-run/restart evidence only. G004 remains incomplete until `artifacts/fdm/g004_d2e_full_fdm_finalization_summary.json` reports pass and `artifacts/fdm/g004_full_fdm_completion_audit.json` reports `status=pass`, `error_count=0`, followed by OMX checkpointing with a fresh goal snapshot if the Codex goal tool is usable.

## 2026-05-23 GPU-idle restart status

After the aggregate Codex goal was restored, the user made GPU utilization an explicit ultragoal-wide operating constraint. The active G004 run on commit `6e40ec6` was still in CPU/IO materialization with all four H200 GPUs at 0% utilization, so it was terminated and backed up under `outputs/fdm_streaming_d2e_full_compact_gpu_idle_restart_backup_20260522T235357Z` plus `artifacts/fdm/g004_gpu_idle_restart_backup_20260522T235357Z`.

The pod was reset to commit `d38a3b1`, which persists the GPU-utilization rule and defers G004 audit-facing train/target monolith construction until after GPU-relevant sharded training/prediction work. Fresh G004 parent PID: `262618`; watcher PID: `262772`. Mirrored local evidence: `artifacts/fdm/g004_gpu_idle_restart_decision.json` and `artifacts/fdm/g004_gpu_idle_restart_launch.json`.

Claim boundary: this restart improves GPU wall-clock utilization strategy but is not G004 completion evidence. G004 remains incomplete until finalization and `g004_full_fdm_completion_audit.json` pass, followed by OMX checkpointing.

## 2026-05-23 deferred materialization progress

The current `d38a3b1` G004 run is still alive and making CPU/IO progress. Latest committed snapshot:
`artifacts/fdm/g004_deferred_materialization_progress_snapshot.json`.

At `2026-05-23T00:21:04Z`:

- Parent PID: `262618`; watcher PID: `262772`.
- Materialization workers: 16 spawned workers plus four torchrun ranks waiting behind the rank-0 materialization barrier.
- Train materialization parts: 16 files, `368,945,307,285` bytes and still growing.
- Target materialization parts: 11 files, `63,658,038,599` bytes and still growing.
- Sharded train/target symlinks, split summary, streaming stats, train cache, train history, checkpoint, predictions, wrapper summary, and finalizer output were not present yet.
- All four H200 GPUs were at 0% utilization, which is expected only for the active CPU/IO materialization phase. If bytes stop growing or if GPUs remain idle after split summary/cache completion, treat it as a blocker and diagnose immediately.

Do not launch another G004 run or pull newer local commits into the pod while PID `262618` and its children are alive. Continue monitoring until DDP training starts, then verify GPU utilization and train-history creation.

## 2026-05-23 DDP runtime failure and reuse fix

The `d38a3b1` run completed FDM split materialization, then failed before stats/cache/training with:
`UnboundLocalError: cannot access local variable 'timeout_seconds' where it is not associated with a value`.
Root cause: `train_streaming_fdm` initialized the torch distributed process group before calling the shared
`train_streaming_idm` trainer; `_distributed_runtime()` then skipped `init_process_group()` and returned a
`timeout_seconds` value that had never been assigned.

Committed failure evidence: `artifacts/fdm/g004_ddp_runtime_failure_snapshot.json`.
Reusable pod artifacts after the failure:

- `outputs/fdm_streaming_d2e_full_compact/fdm_streaming_split_summary.json` exists.
- `fdm_train_shards`: 16 files, `400,301,959,913` bytes.
- `fdm_target_shards`: 16 files, `347,089,780,692` bytes.
- No streaming stats, train cache, train history, checkpoint, summary, finalization, or passing audit yet.

The fix preserves `timeout_seconds` when the process group is already initialized and enables fail-closed
reuse of an existing materialized split summary/shards on restart. After pulling the fixed commit in the pod,
relaunch G004; it should skip the expensive materialization rewrite and proceed to stats/cache/DDP.
