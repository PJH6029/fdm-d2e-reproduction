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
