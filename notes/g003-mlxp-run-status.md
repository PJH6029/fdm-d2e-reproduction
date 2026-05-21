# G003 MLXP Run Status

Updated: 2026-05-21 KST

- Latest pushed commit pulled in pod: `7fe6b3f` (G004 convergence/scaling evidence support and G007 runtime adapter claim-boundary updates are present in the checkout; running G003 shard Python processes may have been launched before later pulls and continue with their loaded code).
- MLXP reservation: `rsv-jeonghunpark-20260521-76e25a`.
- Pod: `prod-rsv-jeonghunpark-20260521-76e25a` in namespace `p-production`.
- Reservation window: 2026-05-21 10:00+09:00 to 2026-05-24 09:00+09:00.
- GPU shape: 4×H200; `cluster_gpu_smoke.py --expected-gpus 4` passed.
- Pod bootstrap fixes applied manually: `python3 -m pip install --user uv`, `apt-get install -y ffmpeg`.
- Sequential extraction was stopped after timing evidence showed one 480p recording took ~9 min.
- Current command: `NUM_SHARDS=16 bash scripts/run_g003_d2e_full_idm_parallel.sh`.
- Current PID file in pod: `outputs/cluster/g003_full_compact_parallel.pid`.
- Main log in pod: `artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.log`.
- Shard logs in pod: `artifacts/sources/d2e_full_corpus_shard_*.log`.
- Monitor command:

```bash
kubectl -n p-production exec prod-rsv-jeonghunpark-20260521-76e25a -- bash -lc '
  cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
  echo HEAD=$(git rev-parse --short HEAD)
  ps -p $(cat outputs/cluster/g003_full_compact_parallel.pid) -o pid,stat,etime,cmd || true
  echo per_recording=$(find outputs/data/d2e_full_corpus_shards -path "*/by_recording/*/decode_summary.json" 2>/dev/null | wc -l)
  echo shard_summaries=$(find outputs/data/d2e_full_corpus_shards -maxdepth 2 -name decode_summary.json 2>/dev/null | wc -l)
  echo metrics=$(test -f outputs/idm_streaming_d2e_full_compact/metrics.json && echo yes || echo no)
  du -sh /root/work/data/d2e/cache outputs/data/d2e_full_corpus_shards outputs/data/d2e_full_corpus outputs/idm_streaming_d2e_full_compact 2>/dev/null || true
  for f in artifacts/sources/d2e_full_corpus_shard_*.log; do
    c=$(grep -c decoded "$f" || true)
    last=$(grep decoded "$f" | tail -1)
    echo $(basename "$f") count=$c "$last"
  done | tail -80
'
```

## 2026-05-21 11:44 KST progress snapshot

- Parallel extraction command was running from PID `9289`.
- Elapsed at snapshot: ~52 min.
- Decoded per-recording summaries: `23 / 918`; shard summaries: `0 / 16`; IDM metrics not yet produced.
- Cache size observed: ~26 GiB; shard output size observed earlier after latest pull: ~53 GiB.
- Pod repo was fast-forwarded while extraction was running; loaded shard Python processes are unaffected, and downstream merge/training will use the updated checkout.

## 2026-05-21 12:21 KST progress snapshot

- Parallel extraction command still running from PID `9289`.
- Pod checkout: `b3c8450`.
- Elapsed at snapshot: ~1h22m.
- Decoded per-recording summaries: `35 / 918`; shard summaries: `0 / 16`; IDM metrics not yet produced.
- Cache size observed: ~30 GiB; shard output size: ~90 GiB.
- Progress is still moving across most shards; no hard-stall evidence at this snapshot.


## 2026-05-21 12:35 KST progress snapshot

- Parallel extraction command still running from PID `9289`.
- Pod checkout: `118fb9c` at snapshot time.
- Elapsed at snapshot: ~1h35m.
- Decoded per-recording summaries: `40 / 918`; shard summaries: `0 / 16`; IDM metrics not yet produced.
- Progress is still moving; no hard-stall evidence at this snapshot.


## 2026-05-21 12:43 KST progress snapshot

- Parallel extraction command still running from PID `9289`.
- Pod checkout: `666d38e` at snapshot time.
- Elapsed at snapshot: ~1h42m.
- Decoded per-recording summaries: `43 / 918`; shard summaries: `0 / 16`; IDM metrics not yet produced.
- Cache size observed: ~31 GiB; shard output size: ~102 GiB.
- Progress is still moving; no hard-stall evidence at this snapshot.


## 2026-05-21 12:51 KST progress snapshot

- Parallel extraction command still running from PID `9289`.
- Pod checkout: `2aa1195` at snapshot time.
- Elapsed at snapshot: ~1h49m.
- Decoded per-recording summaries: `46 / 918`; shard summaries: `0 / 16`; IDM metrics not yet produced.
- Cache size observed: ~32 GiB; shard output size: ~114 GiB.
- Progress is still moving; no hard-stall evidence at this snapshot.


## 2026-05-21 12:56 KST progress snapshot

- Parallel extraction command still running from PID `9289`.
- Pod checkout: `7fa4d42` at snapshot time.
- Elapsed at snapshot: ~1h51m.
- Decoded per-recording summaries: `47 / 918`; shard summaries: `0 / 16`; IDM metrics not yet produced.
- Cache size observed: ~33 GiB; shard output size: ~116 GiB.
- Shards that were previously at 0 have started producing summaries; progress is still moving and no hard-stall evidence was observed.


## 2026-05-21 13:00 KST monitor snapshot

- Pod checkout synced to `b9600b0`.
- `scripts/monitor_g003_progress.py` ran in pod using `/root/.local/bin/uv` because non-login `kubectl exec` did not include `uv` on `PATH`.
- Monitor artifact copied back to `artifacts/idm/g003_full_compact_parallel_progress.json`.
- Monitor status: `running`; decoded recording variants: `50 / 918`; complete shards: `0 / 16`; stale shards: `[]`; no-progress shards: `[2]`; parent PID running: `true`.
- Merged train/eval and IDM metrics are still absent, so this is progress evidence only.
- Resume plan artifact `artifacts/idm/g003_resume_plan.json` currently reports `defer_active_parent` because the original parent PID is still active; do not run resume shard commands unless the parent exits or an operator intentionally reviews `--allow-active-parent`.


## 2026-05-21 13:07 KST progress and metadata patch snapshot

- Parallel extraction command still running from PID `9289`.
- Pod checkout at snapshot: `7f5f8b1` before the metadata patch commit below is synced.
- Elapsed at snapshot: ~2h05m.
- Decoded per-recording summaries: `59 / 918`; shard summaries: `0 / 16`; IDM metrics not yet produced.
- Streaming IDM metadata path was patched locally so the later training stage records config fingerprint/path, train/target paths, data-universe hash/fingerprint, split-contract hash/fingerprint/split id, source namespace, source ids/resolution tiers, and target eval tags.

Do **not** checkpoint `G003-d2e-only-idm` complete until all required artifacts exist:

- full decode summary covering all 918 D2E recording variants or audited exclusions with retry logs/reasons/impact,
- merged train/eval JSONLs,
- streaming IDM checkpoint,
- pseudolabels/predictions/metrics,
- label-quality/statistical comparison reports,
- run evidence JSON,
- validation evidence and committed artifact summaries.


## 2026-05-21 13:06 KST pod sync and monitor snapshot

- Pod checkout fast-forwarded to `7fe6b3f` (`Preserve G003 provenance in streaming IDM metadata`) while extraction was still running.
- Parallel extraction parent PID `9289` was still running; elapsed at sync was ~2h10m.
- `scripts/monitor_g003_progress.py` artifact was copied back to `artifacts/idm/g003_full_compact_parallel_progress.json`.
- Monitor status: `review_stale_shards`; decoded recording variants: `62 / 918`; complete shards: `0 / 16`; stale shards: `[8, 9, 13]`; no-progress shards: `[]`; parent PID running: `true`; IDM metrics absent.
- Follow-up process inspection showed the stale-shard child Python processes still alive, so this is a watch/review condition rather than a confirmed extraction failure. Continue monitoring before any resume/recovery action.


## 2026-05-21 13:21 KST process-aware monitor snapshot

- Pod checkout fast-forwarded to `057bd86` with process-aware G003 monitoring.
- Parallel extraction parent PID `9289` remained running; elapsed at monitor was ~2h19m.
- Monitor status: `running`; decoded recording variants: `67 / 918`; complete shards: `0 / 16`; stale shards: `[]`; long-running active shards: `[8, 9, 11, 13]`; no-progress shards: `[]`; active shard processes: `0..15`; IDM metrics absent.
- `artifacts/idm/g003_full_compact_parallel_progress.json` was refreshed from the pod. The previous stale-shard condition is now represented as long-running active shard evidence, not a recovery trigger.


## 2026-05-21 13:18 KST ETA monitor snapshot

- Pod checkout fast-forwarded to `d00990f` with throughput/ETA fields in the G003 monitor.
- Parallel extraction parent PID `9289` remained running; elapsed at monitor was ~2h23m.
- Monitor status: `running`; decoded recording variants: `69 / 918`; complete shards: `0 / 16`; long-running active shards: `[8, 9, 13]`; stale shards: `[]`; no-progress shards: `[]`; IDM metrics absent.
- Observed throughput: `37.74` recording variants/hour; ETA at current rate: `22.49` hours. This is telemetry only, not a G003 completion claim.


## G003 distributed IDM training guard

- Commit pending after the 2026-05-21 ETA snapshot updates `scripts/run_g003_d2e_full_idm_parallel.sh` to default `IDM_NPROC_PER_NODE=4` and run the post-merge IDM stage with `torchrun`.
- `configs/model/idm_streaming_d2e_full_compact.yaml` now records per-epoch validation checkpoints and convergence settings.
- Because the current parent bash process was launched before this script revision, verify the actual training command in `artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.log` after extraction completes. If it uses the old single-GPU command, rerun the IDM training stage with `torchrun --standalone --nproc-per-node=4` on the merged full-corpus JSONLs before checkpointing G003 complete.


## 2026-05-21 13:29 KST quality-gate/monitor snapshot

- Pod checkout remained at `ba4cbc7`; extraction parent PID `9289` still running.
- Monitor status: `running`; decoded recording variants: `71 / 918`; complete shards: `0 / 16`; long-running active shards: `[8, 9, 13]`; stale shards: `[]`; no-progress shards: `[]`; IDM metrics absent.
- Observed throughput: `35.94` recording variants/hour; ETA at current rate: `23.57` hours. This remains progress telemetry only.
- Final quality gate config now asserts G003/G004 provenance and 4×H200 run metadata; current audit fails as expected while artifacts are missing.
