# G003 Full-Corpus Progress Monitoring

G003 is a long-running full D2E extraction + IDM training gate. The running MLXP
job must not be judged complete from partial shard logs. Use the non-mutating
progress monitor below to summarize extraction health while leaving the running
process untouched.

## Command

Inside the MLXP pod checkout:

```bash
uv run python scripts/monitor_g003_progress.py \
  --stale-seconds 7200 \
  --output artifacts/idm/g003_full_compact_parallel_progress.json
```

The command writes `artifacts/idm/g003_full_compact_parallel_progress.json` with:

- parent PID/running status from `outputs/cluster/g003_full_compact_parallel.pid`
- expected recording variants from `artifacts/sources/d2e_full_data_universe_manifest.json`
- decoded counts from shard logs and per-recording `decode_summary.json` files
- completed shard count
- no-progress/stale-log shard lists
- merged train/eval and IDM metrics existence checks

For a richer live supervision snapshot, also run:

```bash
uv run python scripts/audit_g003_live_health.py \
  --stale-seconds 7200 \
  --output artifacts/idm/g003_live_health_report.json
```

This second command is also non-mutating. It adds best-effort Linux `/proc`
topology evidence for the parent script, shard extractors, post-run watcher,
attached GPU monitor, merge/training/finalizer processes, inactive incomplete
shards, duplicate extractor processes, and low-active-extractor warnings. It is
handoff/recovery evidence only; it must not be treated as a completion or
quality-gate proof.

## Interpretation

- `status=running`: parent PID exists, progress is not stale, and G003 remains in
  progress.
- `status=review_stale_shards`: at least one shard log has not changed past the
  configured threshold; inspect before taking action because large original-video
  downloads can legitimately keep a shard quiet for a long time.
- `status=complete`: all shard summaries exist, merged train/eval files exist,
  and IDM metrics exist. This still needs the full G003 artifact checklist and
  OMX checkpoint before the story is complete.
- `status=not_running_partial`: decoded artifacts exist but the parent PID is no
  longer running; inspect logs and resume/relaunch safely.

For `g003_live_health_report.json`, key statuses are:

- `healthy_running`: the current stage has the expected live process topology
  for extraction, merge, or IDM training.
- `warn_live_health`: the parent is still alive but supervision evidence needs
  review, for example an incomplete shard has no active extractor or a monitor
  PID file is not live.
- `blocked_live_health`: stale-shard evidence requires operator review before
  resume/relaunch decisions.
- `complete_pending_audit`: progress artifacts look complete, but the full G003
  completion audit and OMX checkpoint are still required.

## Current G003 completion checklist

Do not checkpoint `G003-d2e-only-idm` complete until all are present and verified:

- `artifacts/sources/d2e_full_corpus_decode_summary.json`
- `outputs/data/d2e_full_corpus/train_core.jsonl`
- `outputs/data/d2e_full_corpus/target_all_eval.jsonl`
- `outputs/idm_streaming_d2e_full_compact/checkpoint.pt`
- `outputs/idm_streaming_d2e_full_compact/pseudolabels.jsonl`
- `outputs/idm_streaming_d2e_full_compact/predictions.jsonl`
- `outputs/idm_streaming_d2e_full_compact/metrics.json`
- `outputs/idm_streaming_d2e_full_compact/label_quality_report.json`
- `outputs/idm_streaming_d2e_full_compact/statistical_comparison.json`
- `artifacts/idm/idm_streaming_d2e_full_compact_summary.json`
- `artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json`

This monitor is progress evidence only. It is not a completion claim.

## Attached GPU-utilization evidence

The progress monitor above reports shard/PID health, but it does not sample GPU
utilization for the final 4×H200 evidence path. For an already-running integrated
parallel job, attach a separate sampler without restarting the job:

```bash
nohup uv run python scripts/attach_g003_gpu_monitor.py \
  --pid-file outputs/cluster/g003_full_compact_parallel.pid \
  --output artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv \
  --metadata-out artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json \
  --monitor-pid-file outputs/cluster/g003_attached_gpu_monitor.pid \
  --interval-seconds 30 \
  > artifacts/idm/g003_attached_gpu_monitor.log 2>&1 &
```

The attached monitor writes one CSV row per visible GPU per sample and exits
when the parent PID exits. A second invocation is fail-safe: it detects a live
monitor PID and records `existing_monitor_running` instead of duplicating
sampling. After the integrated run finishes, run:

```bash
uv run python scripts/build_g003_attached_train_run_summary.py
```

That summary remains non-passing until the integrated run evidence, checkpoint
metadata, metrics, and four-GPU monitor coverage are all present.


## Resume planning after interruption

If the parent process exits before all shard summaries exist, first generate a
read-only resume plan:

```bash
uv run python scripts/plan_g003_resume.py   --progress-report artifacts/idm/g003_full_compact_parallel_progress.json   --output artifacts/idm/g003_resume_plan.json
```

The plan lists incomplete shards and exact extraction commands. It deliberately
sets `runnable=false` while the original parent PID is still active. Only use the
shard commands after verifying the parent process is gone or after intentionally
passing `--allow-active-parent` for an operator-reviewed recovery.

The per-recording cache layout makes these shard reruns resumable: existing
`by_recording/.../all_records.jsonl` and `decode_summary.json` files are reused,
while each shard aggregate JSONL is rebuilt from per-recording records.
