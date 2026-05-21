# G003 MLXP Run Status

Updated: 2026-05-21 KST

- Latest pushed commit: `90f9c27`.
- MLXP reservation: `rsv-jeonghunpark-20260521-76e25a`.
- Pod: `prod-rsv-jeonghunpark-20260521-76e25a` in namespace `p-production`.
- Reservation window: 2026-05-21 10:00+09:00 to 2026-05-24 09:00+09:00.
- GPU shape: 4×H200; `cluster_gpu_smoke.py --expected-gpus 4` passed.
- Pod bootstrap fixes applied manually: `python3 -m pip install --user uv`, `apt-get install -y ffmpeg`.
- Sequential extraction was stopped after timing evidence showed one 480p recording took ~9 min.
- Current intended command: `NUM_SHARDS=16 bash scripts/run_g003_d2e_full_idm_parallel.sh`.
- Current PID file in pod: `outputs/cluster/g003_full_compact_parallel.pid`.
- Monitor command:

```bash
kubectl -n p-production exec prod-rsv-jeonghunpark-20260521-76e25a -- bash -lc '
  cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
  ps -p $(cat outputs/cluster/g003_full_compact_parallel.pid) -o pid,stat,etime,cmd || true
  find outputs/data/d2e_full_corpus_shards -path "*/by_recording/*/decode_summary.json" | wc -l
  for f in artifacts/sources/d2e_full_corpus_shard_*.log; do
    c=$(grep -c "decoded" "$f" || true)
    last=$(grep "decoded" "$f" | tail -1)
    echo "$(basename "$f") count=$c $last"
  done
  du -sh /root/work/data/d2e/cache outputs/data/d2e_full_corpus_shards 2>/dev/null || true
'
```

Do **not** checkpoint `G003-d2e-only-idm` complete until all required artifacts exist:

- full decode summary covering all 918 D2E recording variants,
- merged train/eval JSONLs,
- streaming IDM checkpoint,
- pseudolabels/predictions/metrics,
- run evidence JSON,
- validation evidence and committed artifact summaries.
