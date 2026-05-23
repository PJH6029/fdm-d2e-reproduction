# G005 MLXP run status

- Reservation: `rsv-jeonghunpark-20260521-76e25a`
- Pod: `prod-rsv-jeonghunpark-20260521-76e25a`
- Namespace: `p-production`
- Repo path on pod: `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`

## 2026-05-23 launch readiness

After `G004-d2e-only-fdm-4xh200` was checkpointed complete in OMX, `scripts/plan_g005_launch.py` was rerun with:

```bash
uv run python scripts/plan_g005_launch.py \
  --source-evidence artifacts/aux/g005_aux_source_materialization_evidence.json \
  --eval-manifest-hashes artifacts/aux/d2e_eval_manifest_hashes.json \
  --require-eval-manifest-hashes \
  --require-namespace-ready
```

Result: `artifacts/aux/g005_launch_readiness.json` reported `status=ready`, `findings=0`.

## 2026-05-23 active candidate run

Current active G005 candidate:

- Pod checkout: `06b69c0`.
- Parent PID file: `outputs/cluster/g005_d2e_aux_best.pid`.
- Parent PID: `266056`.
- Watcher PID file: `outputs/cluster/g005_postrun_watcher.pid`.
- Watcher Python PID: `266067`.
- Command: `uv run python scripts/run_g005_aux_prior_candidate.py --config configs/model/g005_aux_prior_candidate.yaml --output artifacts/aux/g005_d2e_aux_train_run.json`.
- Watcher: `scripts/watch_g005_then_finalize.py` with `--source-evidence artifacts/aux/g005_aux_source_materialization_evidence.json`, `--eval-manifest-hashes artifacts/aux/d2e_eval_manifest_hashes.json`, and `--completion-ready`.

This first G005 candidate is a **CPU/IO-heavy source-specific auxiliary action-prior candidate**, not a neural 4×H200 auxiliary pretraining run. It consumes the full selected aux action-label corpus, writes D2E+aux predictions over the full D2E target eval stream, collects metrics and split statistics inline in the same pass, then the watcher runs the non-mutating G005 finalizer/audit. GPU utilization is expected to remain 0% for this candidate; future neural/visual aux candidates should use 4×H200 and must follow `notes/gpu-utilization-operating-rule.md`.

Latest observed progress at `2026-05-23T04:32:46Z`:

- `outputs/fdm_aux/d2e_aux_best/aux_action_prior_training.json` exists with `status=pass`, `total_rows_consumed=19,525,144`.
- The parent process was CPU-active at ~100% and writing `outputs/fdm_aux/d2e_aux_best/predictions.jsonl`.
- It was reading `outputs/fdm_streaming_d2e_full_compact/torch_model/predictions.jsonl` and target shard `outputs/fdm_streaming_d2e_full_compact/materialization_parts/target/part_00000.jsonl`.
- `artifacts/aux/g005_postrun_watcher_summary.json` reported `status=waiting_active_parent`.
- Terminal run summary, metrics, ablation summary, finalization summary, and passing completion audit were not present yet.

Claim boundary: do not checkpoint `G005-aux-data-best-model` complete until `artifacts/aux/g005_aux_finalization_summary.json` reports `status=pass` and `artifacts/aux/g005_aux_completion_audit.json` reports `status=pass`, `error_count=0`, then checkpoint OMX with a fresh active aggregate `get_goal` snapshot. Do not call `update_goal complete` for G005.

## 2026-05-23 13:37 KST CPU/IO prediction bottleneck and relaunch plan

Snapshot command evidence from the pod showed:

- Pod checkout: `06b69c0`.
- Parent `266056` and worker `266076` still active; worker CPU was ~100% and writing `outputs/fdm_aux/d2e_aux_best/predictions.jsonl`.
- `nvidia-smi` showed all four H200 GPUs at 0% utilization. This is expected for this CPU/IO-only action-prior candidate, but it is still a throughput bottleneck on reserved hardware.
- `outputs/fdm_aux/d2e_aux_best/predictions.jsonl` had `1,156,193` rows / `540,530,377` bytes, so the run was only a small fraction of the `16,698,646` D2E target rows.
- Terminal artifacts were still missing: `prediction_build_summary.json`, `metrics.json`, `statistical_comparison.json`, split-stat summary, ablation summary, run summary, finalization summary, and passing G005 audit.

A completion-blocking bug was identified in the pre-hardening code path: for multi-shard target evaluation, `outputs/fdm_aux/d2e_aux_best/d2e_target_records.jsonl` was linked to only the first target shard. `scripts/validate_g005_aux_completion.py` counts this file and would reject a full prediction file whose row count exceeds that first shard. Local hardening now links target records to the full monolithic D2E target JSONL and adds `d2e_only_prediction_paths` + `prediction_workers=16` so G005 prediction/metrics/split-stat collection processes G004 prediction recovery parts and target shards in parallel.

Next safe action after commit/push: capture a fresh pod snapshot, terminate the pre-hardening G005 parent/watcher only after backing up partial outputs, pull latest origin in the pod, relaunch `scripts/run_g005_aux_prior_candidate.py`, and restart `scripts/watch_g005_then_finalize.py`.

## 2026-05-23 14:03 KST fixed run completed; finalizer blocked by watcher/namespace hardening

The `3868187` relaunch completed the G005 candidate run itself:

- `artifacts/aux/g005_d2e_aux_train_run.json` reported `status=pass`, `exit_code=0`.
- `outputs/fdm_aux/d2e_aux_best/prediction_build_summary.json` reported `status=pass`, `rows=16,698,646`, `parallel_prediction=true`, `prediction_workers=16`.
- `artifacts/eval/g005_split_statistical_comparisons_summary.json` and `artifacts/aux/d2e_aux_ablation_summary.json` reported `status=pass`.
- `outputs/fdm_aux/d2e_aux_best/d2e_target_records.jsonl` was a symlink to the full monolithic D2E target JSONL, not the first target shard.

The post-run watcher did not finalize because the `uv` wrapper PID in `outputs/cluster/g005_d2e_aux_best.pid` remained as a zombie (`STAT=Z`), and the watcher treated `os.kill(pid, 0)` as alive. Local hardening now treats zombie PIDs as inactive. A second finalization blocker was also identified: namespace source evidence can contain absolute `.../outputs/aux/<dataset>` paths, while the G005 completion audit requires repo-relative `outputs/aux/<dataset>/...` namespaces. Local hardening now normalizes absolute namespace values back to repo-relative paths.

Next safe action after commit/push: pull latest origin in the pod, run `scripts/finalize_g005_aux_best_model.py --force-namespace --source-evidence artifacts/aux/g005_aux_source_materialization_evidence.json --eval-manifest-hashes artifacts/aux/d2e_eval_manifest_hashes.json --completion-ready`, then inspect `artifacts/aux/g005_aux_finalization_summary.json` and `artifacts/aux/g005_aux_completion_audit.json` before any OMX checkpoint.

## 2026-05-23 14:12 KST G005 finalization/audit pass

After pulling `7d3098f` in the pod, G005 finalization was run with:

```bash
uv run python scripts/finalize_g005_aux_best_model.py \
  --force-namespace \
  --source-evidence artifacts/aux/g005_aux_source_materialization_evidence.json \
  --eval-manifest-hashes artifacts/aux/d2e_eval_manifest_hashes.json \
  --completion-ready
```

Result:

- `artifacts/aux/g005_aux_finalization_summary.json`: `status=pass`, `g005_audit_status=pass`, `g005_audit_error_count=0`.
- `artifacts/aux/g005_aux_completion_audit.json`: `status=pass`, `error_count=0`.
- Audit counts: `target_records=16,698,646`, `predictions=16,698,646`.
- `artifacts/aux/g005_aux_namespace_manifest.json`: `completion_ready=true`, namespaces normalized to `outputs/aux/<source>/` for all selected aux sources.
- Small G005 evidence files were copied locally; raw `predictions.jsonl`, target records, and large PVC outputs remain on the MLXP pod/PVC.

G005 is ready for an OMX checkpoint with a fresh active aggregate `get_goal` snapshot. Do not call `update_goal complete`; only checkpoint `G005-aux-data-best-model` in OMX.

## 2026-05-23 14:14 KST G005 OMX checkpoint complete

`G005-aux-data-best-model` was checkpointed complete in OMX with fresh active aggregate `get_goal` snapshot. Evidence string referenced commit `2b97079`, passing finalization/audit artifacts, 16,698,646 prediction/target counts, 16-worker parallel prediction summary, split-stat summary, and ablation summary. The aggregate Codex goal remains active; do not call `update_goal complete` until G009/final quality gates complete.
