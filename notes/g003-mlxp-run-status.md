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


## 2026-05-21 13:34 KST G006 builder/monitor snapshot

- Pod checkout remained at `9fc1aa8`; extraction parent PID `9289` still running.
- Monitor status: `running`; decoded recording variants: `73 / 918`; complete shards: `0 / 16`; long-running active shards: `[8, 9, 13]`; stale shards: `[]`; no-progress shards: `[]`; IDM metrics absent.
- Observed throughput: `35.89` recording variants/hour; ETA at current rate: `23.55` hours. This remains progress telemetry only.
- Added a G006 final artifact builder in local worktree; it should only pass after G003/G004 split-aware comparison artifacts exist.


## 2026-05-21 13:40 KST split-stat builder/monitor snapshot

- Pod checkout remained at `8032fa8`; extraction parent PID `9289` still running.
- Monitor status: `running`; decoded recording variants: `74 / 918`; complete shards: `0 / 16`; long-running active shards: `[4, 8, 9, 13]`; stale shards: `[]`; no-progress shards: `[]`; IDM metrics absent.
- Observed throughput: `34.52` recording variants/hour; ETA at current rate: `24.45` hours. This remains progress telemetry only.
- Added a split-aware statistical comparison builder in local worktree; it should run after G003/G004 predictions exist and before the G006 final artifact builder.


## 2026-05-21 13:44 KST final-gate coverage snapshot

- Pod checkout remained at `a1db447`; extraction parent PID `9289` still running.
- Monitor status: `running`; decoded recording variants: `76 / 918`; complete shards: `0 / 16`; long-running active shards: `[8, 9, 13]`; stale shards: `[]`; no-progress shards: `[]`; IDM metrics absent.
- Observed throughput: `34.04` recording variants/hour; ETA at current rate: `24.74` hours. This remains progress telemetry only.
- Final quality gate config now requires G003 full decode coverage, split-stat summaries, and G006 build-summary pass artifacts; current audit fails as expected while artifacts are missing.


## 2026-05-21 13:51 KST G003 completion-audit snapshot

- Pod checkout remained at `a0fbd99`; extraction parent PID `9289` still running.
- Monitor status: `running`; decoded recording variants: `76 / 918`; complete shards: `0 / 16`; long-running active shards: `[8, 9, 13]`; stale shards: `[]`; no-progress shards: `[]`; IDM metrics absent.
- Observed throughput: `33.31` recording variants/hour; ETA at current rate: `25.28` hours. This remains progress telemetry only.
- Added `artifacts/idm/g003_full_idm_completion_audit.json`; current status is `fail` with `24` expected errors while G003 artifacts are missing/incomplete.

## 2026-05-21 13:56 KST post-G003-audit sync snapshot

- Pushed commit `8495d65` and fast-forwarded pod checkout to `8495d65`; `scripts/validate_g003_full_idm_completion.py` and config are present in pod.
- Parent PID `9289` still running; monitor status `running`; decoded recording variants `77 / 918`; complete shards `0 / 16`; long-running active shards `[8, 9, 13]`; stale/no-progress shards `[]`; IDM metrics absent.
- Pod-local G003 completion audit reports `fail` with `24` expected errors while full extraction/training artifacts are incomplete.

## 2026-05-21 14:02 KST post-G004-audit sync snapshot

- Pushed commit `e678dbf` and fast-forwarded pod checkout to `e678dbf`; `scripts/validate_g004_full_fdm_completion.py` and config are present in pod.
- Parent PID `9289` still running; decoded recording variants `77 / 918`; complete shards `0 / 16`; IDM metrics absent.
- Pod-local G004 completion audit reports `fail` with `26` expected errors because G003 is still in progress and G004 artifacts do not exist yet.

## 2026-05-21 14:08 KST post-G005-audit sync snapshot

- Pushed commit `02d4b14` and fast-forwarded pod checkout to `02d4b14`; `scripts/validate_g005_aux_completion.py` and config are present in pod.
- Parent PID `9289` still running; decoded recording variants `78 / 918`; complete shards `0 / 16`; IDM metrics absent.
- Pod-local G005 completion audit reports `fail` with `20` expected errors because G003/G004 are incomplete and aux artifacts do not exist yet.

## 2026-05-21 14:10 KST post-G008-audit sync snapshot

- Pushed commit `1cb6129` and fast-forwarded pod checkout to `1cb6129`; `scripts/validate_g008_live_suite_completion.py` and config are present in pod.
- Parent PID `9289` still running; decoded recording variants `81 / 918`; complete shards `0 / 16`; IDM metrics absent.
- Pod-local G008 completion audit reports `fail` with `11` expected errors because D2E-only training/live evidence prerequisites are incomplete.

## 2026-05-21 14:15 KST post-G006-audit sync snapshot

- Pushed commit `9f68cbf` and fast-forwarded pod checkout to `9f68cbf`; `scripts/validate_g006_completion.py` and config are present in pod.
- Parent PID `9289` still running; decoded recording variants `81 / 918`; complete shards `0 / 16`; IDM metrics absent.
- Pod-local G006 completion audit reports `fail` with `22` expected errors because G003/G004 and final evaluation artifacts are incomplete.

## 2026-05-21 14:23 KST post-G009-audit sync snapshot

- Pushed commit `1e57022` and fast-forwarded pod checkout to `1e57022`; `scripts/validate_g009_completion.py` and config are present in pod.
- Parent PID `9289` still running; decoded recording variants `82 / 918`; complete shards `0 / 16`; IDM metrics absent.
- Pod-local G009 completion audit reports `fail` with `9` expected errors because G003-G008 prerequisites and final regenerated package artifacts are incomplete.

## 2026-05-21 14:31 KST post-G007-audit sync snapshot

- Pushed commit `22717f6` and fast-forwarded pod checkout to `22717f6`; `scripts/validate_g007_completion.py` and config are present in pod.
- Parent PID `9289` still running; decoded recording variants `83 / 918`; complete shards `0 / 16`; IDM metrics absent.
- Local committed G007 completion audit reports `pass`. Pod-local G007 audit reports `fail` with `1` error only because `.omx/ultragoal/goals.json` is intentionally not present in the ignored pod checkout; use local worktree for OMX goal-state audits.

## 2026-05-21 14:38 KST post-G007-sync snapshot

- Pushed commit `d2e7304` and fast-forwarded pod checkout to `d2e7304`.
- Parent PID `9289` still running; decoded recording variants `87 / 918`; complete shards `0 / 16`; all 16 shard Python processes were still active; IDM metrics absent.
- The active integrated run may train with 4 ranks, but it was not launched through the dedicated standalone wrapper that writes the final G003 4×H200 monitor summary. Do not kill or restart it for this. Instead, after pulling the next commit, attach `scripts/attach_g003_gpu_monitor.py` to `outputs/cluster/g003_full_compact_parallel.pid`, then run `scripts/build_g003_attached_train_run_summary.py` after the integrated run exits.

## 2026-05-21 14:43 KST attached GPU monitor snapshot

- Pushed commit `f5ed43d` and fast-forwarded pod checkout to `f5ed43d`.
- Attached `scripts/attach_g003_gpu_monitor.py` to parent PID `9289` without restarting the integrated run. Attached monitor PID is `31950`.
- `artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv` exists in the pod with one header plus four GPU rows at first verification; GPU indices `0..3` were visible.
- Parent PID `9289` still running at elapsed `03:49:12`; decoded recording variants `87 / 918`; complete shards `0 / 16`; all 16 shard processes active; IDM metrics absent.
- Monitor status remained `running`; long-running active shards `[0, 2, 5, 6, 7, 8, 10, 11, 13, 14]`; stale/no-progress shards `[]`.
- Do not commit the live GPU-monitor CSV from local while the pod owns it as an untracked live output. Copy it back and package it after the integrated parent exits and the attached monitor writes its metadata JSON.

## 2026-05-21 14:57 KST G004 hardening sync snapshot

- Pushed commit `dadccb8` and fast-forwarded pod checkout to `dadccb8`; `scripts/predict_idm_streaming.py` and the G004 explicit train-core pseudo-label path are present in pod.
- Parent PID `9289` still running; decoded recording variants `90 / 918`; complete shards `0 / 16`; merged train/eval and IDM metrics absent.
- G003 monitor status remained `running`; stale/no-progress shards `[]`.
- G004 completion now requires prediction-only G003 IDM pseudo-labels on `train_core.jsonl` and FDM evaluation on untouched `target_all_eval.jsonl`; the older target-all-eval recording-tail FDM mode is rejected for completion evidence.

## 2026-05-21 15:04 KST causal G004 sync snapshot

- Pushed commit `bf16ac4` and fast-forwarded pod checkout to `bf16ac4`; G004 FDM config now requires `summary_causal_compact_grid8_time_prior_action`.
- Parent PID `9289` still running; decoded recording variants `92 / 918`; complete shards `0 / 16`; merged train/eval and IDM metrics absent.
- G003 monitor status remained `running`; stale/no-progress shards `[]`.
- G004 completion gate now rejects next-frame inverse-dynamics features for FDM and requires prior-action context provenance in the split summary.

## 2026-05-21 15:15 KST G005 namespace-gate sync snapshot

- Pushed commit `875f46f` and fast-forwarded pod checkout to `875f46f`; G005 completion now requires `artifacts/aux/g005_aux_namespace_manifest.json` with `completion_ready=true`, source-specific aux namespaces/action heads, no D2E heldout overlap, and byte-identical D2E eval-manifest hashes for D2E-only vs D2E+aux ablations.
- Parent PID `9289` still running; decoded recording variants `94 / 918`; complete shards `0 / 16`; merged train/eval and IDM metrics absent.
- Attached GPU monitor PID `31950` still running. The live pod-owned CSV `artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv` remains uncommitted; copy/package it only after the integrated parent exits and monitor metadata is written.
- G003 monitor status remained `running`; stale/no-progress shards `[]`.

## 2026-05-21 15:58 KST split-stat gating sync snapshot

- Pushed commit `6974f38` and fast-forwarded pod checkout to `6974f38`; future G003/G004 run wrappers now build preregistered split-specific statistical comparisons after successful training and record split-stat status in run summaries.
- Parent PID `9289` still running at elapsed `05:03:34`; attached GPU monitor PID `31950` still running at elapsed `01:14:40`.
- Monitor status remained `running`; decoded recording variants `109 / 918`; complete shards `0 / 16`; stale/no-progress shards `[]`; merged train/eval and IDM metrics absent.
- The active G003 parent was launched before commit `6974f38`, so do not assume it will run the newly added split-stat builder. If `artifacts/eval/g003_split_statistical_comparisons_summary.json` is absent after IDM predictions exist, run `uv run python scripts/build_split_statistical_comparisons.py --config configs/eval/g003_split_statistics.yaml` manually before `scripts/validate_g003_full_idm_completion.py`.
