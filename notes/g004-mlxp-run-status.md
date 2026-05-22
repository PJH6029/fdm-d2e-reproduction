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
