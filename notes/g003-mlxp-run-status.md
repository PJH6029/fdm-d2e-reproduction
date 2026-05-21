# G003 MLXP Run Status

Updated: 2026-05-21 KST

- Latest pushed commit pulled in pod: `b3c8450` (G004 convergence/scaling evidence support and G007 runtime adapter claim-boundary updates are present in the checkout; running G003 shard Python processes may have been launched before later pulls and continue with their loaded code).
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

Do **not** checkpoint `G003-d2e-only-idm` complete until all required artifacts exist:

- full decode summary covering all 918 D2E recording variants or audited exclusions with retry logs/reasons/impact,
- merged train/eval JSONLs,
- streaming IDM checkpoint,
- pseudolabels/predictions/metrics,
- label-quality/statistical comparison reports,
- run evidence JSON,
- validation evidence and committed artifact summaries.
