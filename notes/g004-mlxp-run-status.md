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
