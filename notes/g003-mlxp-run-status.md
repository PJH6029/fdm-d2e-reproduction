# G003 MLXP Run Status

Updated: 2026-05-22 KST

- Latest commit verified in the pod checkout before the current active G003 torchrun: `9a9f099`. Origin/local are intentionally ahead of the pod checkout (including tensor-cache implementation `8148678` and later note-only commits), but those commits are not pulled into the pod until the active G003 torchrun/finalization exits.
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

## 2026-05-21 16:01 KST manifest coverage sync snapshot

- Pushed commit `2c4c3fc` and fast-forwarded pod checkout to `2c4c3fc`; `artifacts/reproducibility/final_quality_gate_audit.json` is now included in the package manifest, removing the fixable G009 manifest-coverage blocker.
- Parent PID `9289` still running at elapsed `05:06:51`; attached GPU monitor PID `31950` still running at elapsed `01:17:57`.
- Monitor status remained `running`; decoded recording variants `111 / 918`; complete shards `0 / 16`; stale/no-progress shards `[]`; merged train/eval and IDM metrics absent.

## 2026-05-21 16:06 KST G006 manifest coverage sync snapshot

- Pushed commit `22c66c1` and fast-forwarded pod checkout to `22c66c1`; `artifacts/eval/g006_final_artifact_build_summary.json` is now included in the package manifest, removing the remaining configured-artifact manifest coverage finding from the final quality audit.
- Parent PID `9289` still running at elapsed `05:11:17`; attached GPU monitor PID `31950` still running at elapsed `01:22:23`.
- Monitor status remained `running`; decoded recording variants `111 / 918`; complete shards `0 / 16`; stale/no-progress shards `[]`; merged train/eval and IDM metrics absent.

## 2026-05-21 16:11 KST pre-checkpoint audit sync snapshot

- Pushed commit `58251dd` and fast-forwarded pod checkout to `58251dd`; story-level completion audits now support pre-checkpoint evidence gating via `require_goal_checkpoint_complete=false`, with final quality gates retaining the hard requirement that all stories are checkpointed complete.
- Parent PID `9289` still running at elapsed `05:16:24`; attached GPU monitor PID `31950` still running at elapsed `01:27:30`.
- Monitor status remained `running`; decoded recording variants `113 / 918`; complete shards `0 / 16`; stale/no-progress shards `[]`; merged train/eval and IDM metrics absent.

## 2026-05-21 16:18 KST G003 finalizer sync snapshot

- Pushed commit `e4a5524` and fast-forwarded pod checkout to `e4a5524`; `scripts/finalize_g003_integrated_run.py` is present and executable in the pod.
- Parent PID `9289` still running at elapsed `05:23:20`; attached GPU monitor PID `31950` still running at elapsed `01:34:26`.
- Monitor status remained `running`; decoded recording variants `115 / 918`; complete shards `0 / 16`; stale/no-progress shards `[]`; merged train/eval and IDM metrics absent.
- After PID `9289` exits, run `uv run python scripts/finalize_g003_integrated_run.py` before any G003 checkpoint. It builds missing split stats, synthesizes attached 4×H200 train evidence, and runs the G003 audit without mutating OMX state.

## 2026-05-21 16:23 KST G004 finalizer sync snapshot

- Pushed commit `65b5d24` and fast-forwarded pod checkout to `65b5d24`; `scripts/finalize_g004_d2e_full_fdm.py` is present and executable in the pod.
- Parent PID `9289` still running at elapsed `05:28:24`; attached GPU monitor PID `31950` still running at elapsed `01:39:30`.
- Monitor status remained `running`; decoded recording variants `116 / 918`; complete shards `0 / 16`; stale/no-progress shards `[]`; merged train/eval and IDM metrics absent.
- After the future G004 4×H200 run exits, run `uv run python scripts/finalize_g004_d2e_full_fdm.py` before any G004 checkpoint. It requires the run summary, builds missing split stats, and runs the G004 audit without mutating OMX state.

## 2026-05-21 17:21 KST live topology audit snapshot

- Pod checkout fast-forwarded to `639f9ba` with `scripts/audit_g003_live_health.py`.
- Parallel extraction parent PID `9289` remained running; post-run watcher and attached GPU monitor remained live.
- Progress monitor status: `running`; decoded recording variants: `131 / 918`; complete shards: `0 / 16`; stale shards: `[]`; no-progress shards: `[]`; IDM metrics absent.
- Live health audit status: `healthy_running`; stage: `extracting`; active extractor shards: `0..15`; inactive incomplete shards: `[]`; warnings: `[]`; errors: `[]`.
- Duplicate extractor processes were recorded as an observation only because uv wrapper and child Python processes can both expose the same shard commandline. This is handoff/recovery telemetry only, not G003 completion evidence.

## 2026-05-21 17:32 KST G008 gate sync snapshot

- Pod checkout fast-forwarded to `470c7c9` after the G008 live-control evidence gate hardening commit.
- G003 parent remained live; progress monitor status `running`, decoded recording variants `136 / 918`, complete shards `0 / 16`, stale/no-progress shards `[]`.
- Live health audit status `healthy_running`, stage `extracting`, active extractor shards `0..15`, warnings/errors `[]`.
- The G008 changes are validation/test hardening only and do not affect the already-loaded extraction workers; G003 remains in progress and must not be checkpointed.


## 2026-05-21 17:39 KST G004 watcher sync snapshot

- Pod checkout fast-forwarded to `7ce729c` after adding `scripts/watch_g004_then_finalize.py` and G004 wrapper PID-file support.
- G003 parent remained live; progress monitor status `running`, decoded recording variants `139 / 918`, complete shards `0 / 16`, stale/no-progress shards `[]`.
- Live health audit status `healthy_running`, stage `extracting`, active extractor shards `0..15`, warnings/errors `[]`.
- G004 watcher is future-run handoff tooling only; it does not affect current G003 extraction and does not checkpoint OMX/Codex state.


## 2026-05-21 17:48 KST G005 preflight/watcher sync snapshot

- Pod checkout fast-forwarded to `4f211ee` after adding `scripts/plan_g005_launch.py` and `scripts/watch_g005_then_finalize.py`.
- G003 parent remained live; progress monitor status `running`, decoded recording variants `143 / 918`, complete shards `0 / 16`, stale/no-progress shards `[]`.
- Live health audit status `healthy_running`, stage `extracting`, active extractor shards `0..15`, warnings/errors `[]`; long-running active shards `[0, 5, 6, 12, 13, 14, 15]` are active-process telemetry, not stale failures.
- G005 readiness is currently blocked as intended by incomplete G003/G004 D2E-only gates; G005 planner/watcher are future-run handoff tooling only and do not checkpoint OMX/Codex state.


## 2026-05-21 17:57 KST G006 readiness/watcher sync snapshot

- Pod checkout fast-forwarded to `49fb8c8` after adding `scripts/plan_g006_readiness.py` and `scripts/watch_g006_then_finalize.py`.
- G003 parent remained live; progress monitor status `running`, decoded recording variants `146 / 918`, complete shards `0 / 16`, stale/no-progress shards `[]`.
- Live health audit status `healthy_running`, stage `extracting`, active extractor shards `0..15`, warnings/errors `[]`; long-running active shards `[0, 3, 5, 6, 12, 13, 14, 15]` are active-process telemetry, not stale failures.
- G006 readiness is currently blocked as intended by incomplete G003/G004/G005 gates and missing split-stat artifacts; planner/watcher are future final-evaluation handoff tooling only and do not checkpoint OMX/Codex state.

## 2026-05-21 18:10 KST G008 readiness sync snapshot

- Pushed commit `5638ecc` and fast-forwarded pod checkout to `5638ecc`; `scripts/plan_g008_readiness.py`, its tests/docs, and `artifacts/harness/g008_readiness_plan.json` are present.
- Synced `.omx/ultragoal/{goals.json,ledger.jsonl,brief.md}` into the pod because `.omx/` is git-ignored and the pod checkout otherwise lacked current G007 completion state. This was a pod-local state sync only, not a Codex-goal mutation or story checkpoint.
- G008 readiness in the pod is blocked as intended: protocol `protocol_ready`; prerequisites report `G003-d2e-only-idm=in_progress`, `G004-d2e-only-fdm-4xh200=pending`, `G007-runtime-sdk-adapter=complete`; findings are incomplete G003/G004, missing trained checkpoint metadata, and missing live-game launch binaries; evidence validation is not collected yet.
- G003 parent remained live; progress monitor status `running`, decoded recording variants `149 / 918`, complete shards `0 / 16`, stale/no-progress shards `[]`, long-running active shards `[0, 3, 5, 12, 13, 14, 15]`.
- Live health audit status `healthy_running`, stage `extracting`, active extractor shards `0..15`, warnings/errors `[]`.
- No G003 or G008 checkpoint/completion claim was made.

## 2026-05-21 18:18 KST G009 readiness sync snapshot

- Pushed commit `fe1ca78` and fast-forwarded pod checkout to `fe1ca78`; `scripts/plan_g009_readiness.py`, its tests/docs, and `artifacts/reproducibility/g009_readiness_plan.json` are present.
- G009 readiness in the pod is blocked as intended by incomplete upstream prerequisites: `G003-d2e-only-idm=in_progress`, `G004-d2e-only-fdm-4xh200=pending`, `G005-aux-data-best-model=pending`, `G006-evaluation-failure-analysis=pending`, `G008-live-game-suite=pending`; completion projection remains `fail` with 6 errors.
- G003 parent remained live; progress monitor status `running`, decoded recording variants `152 / 918`, complete shards `0 / 16`, stale/no-progress shards `[]`, long-running active shards `[0, 5, 12, 13, 14, 15]`.
- Live health audit status `healthy_running`, stage `extracting`, active extractor shards `0..15`, warnings/errors `[]`.
- No story or aggregate checkpoint/completion claim was made.

## 2026-05-21 18:27 KST G003→G004 chain watcher sync snapshot

- Pushed commit `a269f3c` and fast-forwarded pod checkout to `a269f3c`; `scripts/watch_g003_then_launch_g004.py` is present and running in the pod.
- Chain watcher PID: `47480` (`outputs/cluster/g003_to_g004_chain_watcher.pid`); command includes `--launch --start-g004-watcher --poll-seconds 120` and writes `artifacts/fdm/g003_to_g004_chain_summary.json` plus `artifacts/fdm/g003_to_g004_chain.log`.
- Chain watcher status is `waiting_g003_parent`: it will not launch G004 until G003 parent PID exits, G003 finalization/audit pass, and `scripts/plan_g004_launch.py` reports ready. It never checkpoints OMX/Codex state.
- G003 parent remained live; progress monitor status `running`, decoded recording variants `161 / 918`, complete shards `0 / 16`, stale/no-progress shards `[]`, long-running active shards `[0, 5, 12, 13, 14, 15]`.
- Live health audit status `healthy_running`, stage `extracting`, active extractor shards `0..15`, warnings/errors `[]`.
- No story or aggregate checkpoint/completion claim was made.

## 2026-05-21 18:34 KST G004→G005 readiness-chain tooling sync snapshot

- Pushed commit `739e8eb` and fast-forwarded pod checkout to `739e8eb`; `scripts/watch_g004_then_plan_g005.py` is present for future post-G004 G005 readiness handoff.
- The G004→G005 chain was not started yet because G004 has not launched; start it after G004 parent exists and source-specific aux materialization/eval-hash evidence paths are known.
- Existing G003→G004 chain watcher remains `waiting_g003_parent`; it still has not launched G004.
- G003 parent remained live; progress monitor status `running`, decoded recording variants `162 / 918`, complete shards `0 / 16`, stale/no-progress shards `[]`, long-running active shards `[0, 5, 12, 13, 14, 15]`.
- Live health audit status `healthy_running`, stage `extracting`, active extractor shards `0..15`, warnings/errors `[]`.
- No story or aggregate checkpoint/completion claim was made.

## 2026-05-21 18:39 KST G005 eval-manifest hash evidence sync snapshot

- Pushed commit `dedfe53` and fast-forwarded pod checkout to `dedfe53`; `scripts/build_g005_eval_manifest_hashes.py` is present and ran in the pod.
- `artifacts/aux/d2e_eval_manifest_hashes.json` reports `status=pass`, `same_d2e_eval_manifests=true`, and `same_hash=true` for temporal, heldout-recording, and heldout-game split manifests. This is G005 namespace/readiness evidence only; it does not materialize aux sources or start G005 training.
- G003 parent remained live; progress monitor status `running`, decoded recording variants `164 / 918`, complete shards `0 / 16`, stale/no-progress shards `[]`, long-running active shards `[0, 5, 12, 13, 14]`.
- Live health audit status `healthy_running`, stage `extracting`, active extractor shards `0..15`, warnings/errors `[]`.
- Existing G003→G004 chain remains waiting for G003 parent/finalization; G004 has not launched.
- No story or aggregate checkpoint/completion claim was made.

## 2026-05-21 18:45 KST G005 aux-source materialization gate sync snapshot

- Pushed commit `1b71dc5` and fast-forwarded pod checkout to `1b71dc5`; `scripts/build_g005_aux_source_evidence.py` is present and ran in the pod.
- `artifacts/aux/g005_aux_source_materialization_evidence.json` currently reports `status=blocked`, `materialized_source_ids=[]`, and findings `aux_namespace_missing` for the three selected auxiliary sources. This is expected until selected aux files are materialized under `outputs/aux/<dataset_id>/train|val|test`.
- G005 namespace completion readiness now requires source-specific split hashes in addition to provenance/action-head/overlap fields.
- G003 parent remained live; progress monitor status `running`, decoded recording variants `164 / 918`, complete shards `0 / 16`, stale/no-progress shards `[]`, long-running active shards `[0, 5, 7, 12, 13, 14]`.
- Live health audit status `healthy_running`, stage `extracting`, active extractor shards `0..15`, warnings/errors `[]`.
- Existing G003→G004 chain remains waiting for G003 parent/finalization; G004 has not launched.
- No story or aggregate checkpoint/completion claim was made.

## 2026-05-22 00:22 KST accel64 path correction and retry hardening snapshot

- Local commit `af22ab0` was pushed to `origin/main`: central D2E downloads now retry bounded transient `URLError`/connection-reset/timeout-style failures and clean partial `.part` files. Local validation: `uv run pytest tests/test_d2e_real_contract.py -q` passed (`10` tests) and `uv run pytest -q` passed (`293` tests).
- Pod checkout was safely fast-forwarded from `4015ff7` to `e3adca6` after confirming the fast-forward did not reduce the 85 generated evidence/status lines; in-flight Python extractor processes keep their already-loaded code, but future repair relaunches will use the retry-hardened downloader. Pod validation: `uv run pytest tests/test_d2e_real_contract.py -q` passed (`10` tests).
- Canonical accel64 shard root is **`outputs/data/d2e_full_corpus_shards_accel64`**. Do not use the typo/order variant `outputs/data/d2e_full_corpus_accel64_shards` in health/resume/finalization commands.
- Correct accel64 health command refreshed `artifacts/idm/g003_accel64_live_health.json`: status `warn_live_health`, stage `extracting`, decoded `373 / 918`, complete shards `0 / 64`, active extractors `63 / 64`, warning only `inactive_incomplete_shards: [29]` because shard 29 is being repaired by an isolated process outside the original accel64 parent process tree.
- Isolated shard-29 repair was still running from PID `113351` with output dir `outputs/data/d2e_full_corpus_shards_accel64/shard_29`; elapsed ~21 min, log showed `4 / 14` selected recordings decoded, and no `decode_summary.json` yet.
- Canonical 16-shard G003 lane remained healthy/running in the previous probe: decoded `270 / 918`, active extractors `16 / 16`, parent PID `9289`; this remains the authoritative lane unless accel64 passes its own audit and is explicitly promoted.
- `G003-d2e-only-idm` must remain `in_progress`: `artifacts/idm/g003_full_idm_completion_audit.json` still fails because full decode, merged records, IDM checkpoint, predictions, label-quality metrics, and metadata are incomplete.

Recommended next command for accel64 health:

```bash
uv run python scripts/audit_g003_live_health.py \
  --shard-root outputs/data/d2e_full_corpus_shards_accel64 \
  --output artifacts/idm/g003_accel64_live_health.json \
  --pid-file outputs/cluster/g003_full_compact_accel64.pid \
  --watcher-pid-file outputs/cluster/g003_accel64_postrun_watcher.pid \
  --gpu-monitor-pid-file outputs/cluster/g003_accel64_attached_gpu_monitor.pid \
  --num-shards 64
```

## 2026-05-22 00:34 KST repair-aware monitor deployment snapshot

- Pushed commit `62b4104` and fast-forwarded the pod checkout from `f028633` to `62b4104` without reducing generated evidence/status lines (`85`).
- `scripts/monitor_g003_progress.py` and `scripts/audit_g003_live_health.py` now account for lane-scoped isolated repair PID files. For accel64, the default repair glob is `outputs/cluster/g003_accel64_shard_*_repair.pid`; for canonical G003 it remains lane-isolated as `outputs/cluster/g003_shard_*_repair.pid`.
- Pod validation after pull: `uv run pytest tests/test_g003_monitor.py -q` passed (`19` tests).
- Corrected accel64 live health now reports `healthy_running` instead of a false low-active warning: decoded `388 / 918`, active extractor shards `64 / 64`, inactive incomplete shards `[]`, repair pid evidence `outputs/cluster/g003_accel64_shard_29_repair.pid -> 113351` running for shard `29`.
- Canonical 16-shard lane remains healthy/running: decoded `274 / 918`, active extractor shards `16 / 16`, inactive incomplete shards `[]`.
- Shard 29 repair has not completed yet: `outputs/data/d2e_full_corpus_shards_accel64/shard_29/decode_summary.json` was still absent at this snapshot.
- `G003-d2e-only-idm` remains `in_progress`; local completion audit still fails with missing full decode/merge/IDM artifacts. Do not checkpoint G003 complete.

## 2026-05-22 00:44 KST accel64 log-lane correction and watcher restart

- Pushed and deployed commit `111a2be`, then replaced the accel64 postrun watcher with the current repair-aware watcher. The active extraction parents were not killed or restarted. New accel64 watcher Python PID at deployment: `122848`; it includes `--repair-pid-glob outputs/cluster/g003_accel64_shard_*_repair.pid`.
- Pushed and deployed commit `c9a9e57` so G003 progress/live-health/resume planning infer `artifacts/sources/g003_accel64` whenever accel64 shard or pid paths are used and `--log-dir` is omitted. This prevents canonical `artifacts/sources` logs from inflating accel64 progress telemetry.
- Important correction: earlier accel64 decoded counts such as `388/918`, `392/918`, and `397/918` were mixed-log telemetry from accel64 shard roots plus canonical log dir defaults. Treat those as invalid for accel64 progress. Lane-local accel64 evidence at this snapshot is `169 / 918` decoded, `0 / 64` complete shards, status `running`.
- Pod validation after `c9a9e57`: `uv run pytest tests/test_g003_monitor.py -q` passed (`21` tests).
- Accel64 live health with lane-local logs remains healthy: status `healthy_running`, active extractor shards `64 / 64`, inactive incomplete shards `[]`, repair pid evidence `outputs/cluster/g003_accel64_shard_29_repair.pid -> 113351` running.
- Accel64 resume plan now writes shard repair logs under `artifacts/sources/g003_accel64/...`; current plan status is `defer_active_parent`, as expected while the accel64 parent is running.
- Canonical G003 remains the D2E-only authoritative lane unless accel64 later passes its own audit and is explicitly promoted. `G003-d2e-only-idm` remains incomplete; do not checkpoint until the completion audit passes.

## 2026-05-22 00:49 KST watcher summary lane-evidence rollout

- Pushed and deployed commit `018336d`. G003 postrun watcher/finalizer summaries now include resolved `progress.log_dir` and `progress.repair_pid_glob` so lane-local evidence can be audited directly from watcher/finalizer artifacts.
- The accel64 postrun watcher was restarted again without touching active extraction parents. New watcher Python PID at deployment: `123879`.
- Pod validation after pull: `uv run pytest tests/test_g003_postrun_watcher.py tests/test_g003_integrated_finalization.py -q` passed (`6` tests).
- New accel64 watcher summary confirms lane-local evidence fields: status `waiting_active_parent`, decoded `172 / 918`, `progress.log_dir=/mnt/ddn/prod-runs/jeonghunpark/code/continuous-gui-poc/fdm-d2e-reproduction/artifacts/sources/g003_accel64`, `progress.repair_pid_glob=/mnt/ddn/prod-runs/jeonghunpark/code/continuous-gui-poc/fdm-d2e-reproduction/outputs/cluster/g003_accel64_shard_*_repair.pid`, `pid_running=true`.
- G003 remains incomplete and must not be checkpointed complete until the completion audit passes after full decode, merge, IDM training/eval, label-quality, and split-stat artifacts exist.

## 2026-05-22 00:52 KST canonical watcher lane-evidence rollout

- Restarted only the canonical G003 postrun watcher on current checkout `2c1d034`; active extraction parents were not touched.
- Pod validation before restart: `uv run pytest tests/test_g003_postrun_watcher.py tests/test_g003_integrated_finalization.py -q` passed (`6` tests).
- Old canonical watcher Python PID `41388` was replaced by new Python PID `124896`.
- New canonical watcher summary confirms lane evidence: status `waiting_active_parent`, decoded `277 / 918`, `progress.log_dir=/mnt/ddn/prod-runs/jeonghunpark/code/continuous-gui-poc/fdm-d2e-reproduction/artifacts/sources`, `progress.repair_pid_glob=/mnt/ddn/prod-runs/jeonghunpark/code/continuous-gui-poc/fdm-d2e-reproduction/outputs/cluster/g003_shard_*_repair.pid`, `pid_running=true`.
- Fresh lane-local probe at this continuation: canonical `277 / 918`, active `16 / 16`; accel64 `174 / 918`, active `64 / 64`, shard 29 repair PID `113351` still running and `shard_29/decode_summary.json` still absent.
- G003 remains incomplete; completion audit still fails with missing full decode/merge/IDM artifacts. Do not checkpoint complete.

## 2026-05-22 00:57 KST extraction activity audit artifacts

- Pushed and deployed commit `67d2d03`, adding `scripts/audit_g003_extraction_activity.py` and tests. The tool writes non-mutating filesystem activity evidence for long-running extraction shards; it is liveness context only, not completion evidence.
- Pod validation after pull: `uv run pytest tests/test_g003_extraction_activity.py -q` passed (`1` test).
- Generated and copied back current activity artifacts:
  - `artifacts/idm/g003_extraction_activity.json`: canonical shard root `outputs/data/d2e_full_corpus_shards`, log dir `artifacts/sources`, `16` shards with activity, `277` per-recording summaries, `0 / 16` complete shard summaries.
  - `artifacts/idm/g003_accel64_extraction_activity.json`: accel64 shard root `outputs/data/d2e_full_corpus_shards_accel64`, log dir `artifacts/sources/g003_accel64`, `64` shards with activity, `177` per-recording summaries, `0 / 64` complete shard summaries.
- Fresh pod probe in this continuation showed canonical `277 / 918` and accel64 `175 / 918` in progress before the activity audit; the copied activity artifact reflects a slightly newer accel64 per-recording-summary count (`177`) from filesystem summaries.
- Shard 29 repair PID `113351` was still running, and `outputs/data/d2e_full_corpus_shards_accel64/shard_29/decode_summary.json` remained absent.
- G003 remains incomplete; no checkpoint/update_goal was made.

## 2026-05-22 01:11 KST O(N²) action-binning fix and patched restart

- Diagnosed the accel64 shard-29 stall as CPU-bound Python after video/frame extraction, not a live network/download stall: the child had high CPU, no socket/file descriptors, cached Medieval Dynasty media was already present, and the in-progress recording directory had no active frame files. The likely hot path was `build_window_records` rescanning every action event for every 50 ms bin on long recordings.
- Pushed commit `3b0ca6d` (`Remove quadratic D2E action binning bottleneck`): `build_window_records` now buckets action events by bin once, G003 shard launch defaults BLAS/OpenMP-style thread fanout to `1`, and `scripts/extract_d2e_full_corpus.py` emits per-recording stage timing logs (`download_*`, `decode_mcap_*`, `extract_frames_*`, `build_records_*`, cached resumes) that completion monitors ignore unless a row has `decoded`.
- Local validation before push: `uv run pytest -q tests/test_d2e_real_contract.py` passed (`11` tests) and `uv run pytest -q` passed (`300` tests). Pod validation after pull: `uv run pytest -q tests/test_d2e_real_contract.py tests/test_full_corpus_extraction_contract.py` passed (`13` tests).
- Pod checkout fast-forwarded to `3b0ca6d`. Old canonical and accel64 shard logs were archived under pod-local `artifacts/sources/g003_restart_archive_20260521T161028Z/` and `artifacts/sources/g003_accel64/restart_archive_20260521T161028Z/`; restart metadata was written to pod-local `artifacts/idm/g003_restart_20260521T161028Z.json`.
- Canonical and accel64 extraction lanes were intentionally restarted on patched code so already-completed per-recording summaries can resume while in-flight old-code long recordings are retried with linear action binning. New live PIDs at the first post-restart probe:
  - canonical parent `126930`, postrun watcher `126959`, attached GPU monitor `126954`, G003→G004 chain watcher `126960`;
  - accel64 parent `126937`, postrun watcher `126963`, attached GPU monitor `126961`.
- First post-restart health probe (`2026-05-22 01:11 KST`): canonical status `running`, decoded `277 / 918`, active Python extractors `16`; accel64 status `running`, decoded `181 / 918`, active Python extractors `64`; both live-health audits reported `healthy_running` and watcher summaries reported `waiting_active_parent`. Accel64 increased from the pre-restart lane-local `178 / 918` to `181 / 918` within roughly one minute after restart.
- New logs now show `recording_resume_cached` stage rows for previously completed recordings. These stage rows are liveness/diagnostic evidence only; G003 completion still requires full decode, merge, IDM training/eval, label-quality, split statistics, and `artifacts/idm/g003_full_idm_completion_audit.json` status `pass`.
- No G003/aggregate checkpoint and no Codex `update_goal` call was made; `G003-d2e-only-idm` remains `in_progress`.

## 2026-05-22 01:25 KST streaming frame-feature deployment and restart

- Pushed commit `1d99123` (`Stream D2E frame features through ffmpeg`): when `keep_frames=false` the D2E video feature extractor now streams `rgb24` raw frames from ffmpeg instead of writing/re-reading transient PPM files. The PPM path remains available for debug runs that keep frame files.
- Local validation before push: `uv run pytest -q tests/test_d2e_real_contract.py` passed (`12` tests) and `uv run pytest -q` passed (`301` tests). Pod validation after pull: `uv run pytest -q tests/test_d2e_real_contract.py tests/test_full_corpus_extraction_contract.py` passed (`14` tests).
- Pod checkout fast-forwarded to `1d99123`. Canonical and accel64 G003 extraction lanes were intentionally restarted again so live workers use the streaming frame path. Old post-`3b0ca6d` logs were archived under pod-local `artifacts/sources/g003_restart_archive_20260521T161637Z/` and `artifacts/sources/g003_accel64/restart_archive_20260521T161637Z/`; restart metadata was written to pod-local `artifacts/idm/g003_restart_20260521T161637Z.json`.
- New live PIDs after streaming restart: canonical parent `135989`, canonical postrun watcher `136019`, G003→G004 chain watcher `136017`; accel64 parent `135996`, accel64 postrun watcher `136023`. Both lanes reported `healthy_running` at follow-up probes.
- Follow-up probe at `2026-05-22 01:22 KST`: canonical progress still `277 / 918`, accel64 progress still `181 / 918` by per-recording summary count while many cached rows replayed; active extractor coverage remained canonical `16 / 16` and accel64 `64 / 64`.
- Streaming path evidence appeared in live accel64 logs: e.g. shard `41` logged `extract_frames_done` with `73,851` frame features in `216.817s`, then `annotate_records_done`; shard `20` logged `extract_frames_done` with `25,150` frame features in `126.021s`; shard `44` logged `extract_frames_done` with `64,756` frame features in `264.498s`. These are throughput/liveness observations, not G003 completion evidence.
- Per-recording summary counts had not yet increased beyond canonical `277` / accel64 `181` at the probe because current new long recordings were between extract/build/JSON write/summary stages. Continue monitoring lane-local logs and summary counts; do not checkpoint G003 until the completion audit passes.

### 2026-05-22 01:26 KST streaming follow-up

- Follow-up monitor after the streaming restart showed accel64 progress increasing again: monitor decoded `186 / 918`, filesystem per-recording summaries `187`; canonical stayed `277 / 918`. Both lanes remained `running` with no stale/long-running shard warnings.
- Grep over current canonical and accel64 shard logs found no `Traceback`, `CalledProcessError`, `RuntimeError`, `No space left`, or killed-process fatal lines at this probe.
- G003 remains non-terminal; continue monitoring until full decode, merge, IDM training/eval, and completion audit pass.

## 2026-05-22 02:32 KST accel64-focused execution snapshot

- After accel64 had become the clearly faster full-corpus lane, canonical 16-shard extraction was stopped to reduce CPU contention and to leave the primary parent inactive for a future audited accel64 promotion. Canonical per-recording summaries were preserved (`287` summaries at stop time) and can still be resumed if accel64 fails.
- Pod-local stop evidence was written under `artifacts/idm/g003_canonical_stop_20260521T171444Z.json`. Killed process roots included canonical parent `135989`, canonical postrun watcher `136019`, canonical attached GPU monitor, and G003→G004 chain watcher `136017`. Accel64 parent `135996` and accel64 watcher `136023` stayed alive.
- Accel64 remained healthy immediately after the canonical stop: monitor `439 / 918`, live health `healthy_running`, active extractors `64 / 64`.
- Follow-up monitor at `2026-05-22 02:32 KST` showed accel64 continuing to progress: monitor `482 / 918`, live-health progress `484 / 918`, filesystem per-recording summaries `484`, status `running`, no stale/no-progress shard lists, live health `healthy_running`.
- Stage diagnosis showed the remaining workload dominated by `extract_frames_start` on original/high-resolution recordings and some `download_video_start`; active extractor processes were consuming CPU rather than idle. No fatal lines were found in current accel64 logs. A broad `du -sh` over cache was terminated after it ran too long; `df -h` still showed the mounted PVC at `140T` total, `120T` used, `21T` available (`86%`).
- G003 remains non-terminal: accel64 has not yet produced merged full-corpus JSONLs, IDM metrics/checkpoints, split-statistics, or a passing accel64/canonical completion audit. Do not checkpoint G003 until accel64 finishes, passes its own audit, is promoted to canonical paths, and canonical `artifacts/idm/g003_full_idm_completion_audit.json` reports `pass`.

### 2026-05-22 02:49 KST accel64-only monitor follow-up

- After stopping canonical, accel64 continued to advance under the same parent `135996`: `484 -> 494 -> 503 -> 515 / 918` over the 02:33-02:49 KST monitor window. Status remained `running` with no stale/no-progress shards; merged JSONLs and IDM metrics had not appeared yet, so extraction was still active.
- Live process sampling still showed extraction workers only (no merge/train/torchrun yet). Continue waiting on accel64 extraction; promotion/G004 handoff remains pending.

### 2026-05-22 02:56 KST local evidence-path hardening while accel64 keeps running

- Fresh pod probe on checkout `b568f0b` showed accel64 still healthy and extracting: monitor `528 / 918`, live health `healthy_running`, active extractors `64 / 64`, parent PID `135996`, watcher status `waiting_active_parent`. No merged full-corpus JSONLs, IDM metrics/checkpoint, finalization summary, or completion audit existed yet.
- Local commit `ea8ed9d` hardens future G003 evidence generation by exporting lane-specific `IDM_SUMMARY` defaults, including the accel64 summary path `artifacts/idm/idm_streaming_d2e_full_compact_accel64_summary.json`. Local validation passed: `uv run pytest tests/test_training_run_scripts.py tests/test_g003_accel64_isolation.py` and full `uv run pytest` (`301` tests).
- Important deployment note: `ea8ed9d` was pushed to origin but intentionally **not pulled into the active pod checkout** because the pod is running the accel64 parent shell from `b568f0b`. The current active run can still pass because the postrun finalizer/watcher already uses explicit accel64 paths for completion auditing; deploy `ea8ed9d` only after the active accel64 parent exits, or if a future restart is needed.
- G003 remains non-terminal. Do not checkpoint complete until accel64 finishes extraction, merge, IDM training/eval, split stats, finalization audit, promotion to canonical paths, and canonical `g003_full_idm_completion_audit.json` reports `pass`.

### 2026-05-22 03:17 KST accel64 extraction still healthy

- Pod checkout remained at `b568f0b` (new local notes/evidence-path commits are intentionally not deployed while the active parent shell runs).
- Accel64 monitor advanced to `578 / 918` decoded recording variants, status `running`, parent alive, watcher `waiting_active_parent`, filesystem summary count `578`.
- Live health remained `healthy_running` with `64 / 64` active extractors. The monitor flagged long-running shards `[13, 56]`, but the recommendation was `continue_monitor_long_recordings`; no stale/no-progress shards were reported.
- Full-corpus merged JSONLs, IDM metrics/checkpoint, integrated finalization summary, and accel64 completion audit were still absent, so G003 remains non-terminal and must not be checkpointed complete.

### 2026-05-22 03:51 KST accel64 crosses 600 variants with completed shards

- Pod checkout still remained at `b568f0b`; local/origin remained ahead for notes/evidence-path hardening and is intentionally not deployed while the active accel64 parent runs.
- Accel64 progressed over the monitoring window: `582 / 918` at 03:20 KST, `603 / 918` at 03:30 KST, `623 / 918` at 03:40 KST, and `647 / 918` at 03:51 KST.
- At 03:51 KST, `complete_shards=2 / 64`, status `running`, parent alive, watcher `waiting_active_parent`, live health `healthy_running`, active extractors `62 / 62`, stale/no-progress shards `[]`, and long-running shards `[19, 33, 38, 41]` with continue-monitor guidance.
- Terminal G003 artifacts were still absent: accel64 full decode summary, merged train/eval JSONLs, IDM metrics/checkpoint, run evidence, finalization summary, split-stat summary, and accel64 audit. G003 remains non-terminal and no checkpoint/update_goal was made.

### 2026-05-22 05:03 KST accel64 776/918 healthy extraction

- Pod checkout still remained at `b568f0b`; newer local/origin commits were not deployed while the active accel64 parent was running.
- Accel64 monitoring progressed through the window: `650 / 918` at 03:51 KST, `669 / 918` at 04:02, `689 / 918` at 04:12, `708 / 918` at 04:22, `723 / 918` at 04:32, `741 / 918` at 04:43, `758 / 918` at 04:53, and `776 / 918` at 05:03.
- At 05:03 KST, `complete_shards=15 / 64`, parent alive, watcher `waiting_active_parent`, live health `healthy_running`, active extractors `49 / 49`, stale/no-progress shards `[]`, long-running shards `[11, 12]` with continue-monitor guidance.
- No terminal G003 artifacts existed yet: accel64 full decode summary, merged train/eval JSONLs, IDM metrics/checkpoint, run evidence, finalization summary, split-stat summary, and accel64 audit remained absent. G003 remains non-terminal.

### 2026-05-22 06:15 KST accel64 tail extraction at 864/918

- Pod checkout still remained at `b568f0b`; newer local/origin commits were intentionally not deployed while the active accel64 parent was running.
- Accel64 progressed over this monitoring window: `776 / 918` at 05:04 KST, `791 / 918` at 05:14, `805 / 918` at 05:24, `820 / 918` at 05:34, `833 / 918` at 05:45, `844 / 918` at 05:55, `852 / 918` filesystem summaries at 06:05, and `864 / 918` at 06:15.
- At 06:15 KST, `complete_shards=36 / 64`, parent alive, watcher `waiting_active_parent`, live health `healthy_running`, active extractors `28 / 28`, stale/no-progress shards `[]`, long-running shards `[]`, recommendation `continue_waiting`.
- Terminal G003 artifacts remained absent: no accel64 full decode summary, merged train/eval JSONLs, IDM metrics/checkpoint, run evidence, finalization summary, split-stat summary, or accel64 audit. G003 remains non-terminal and no checkpoint/update_goal was made.

### 2026-05-22 09:25 KST full extraction/merge succeeded; IDM training resumed after distributed timeout fix

- Accel64 full extraction reached `918 / 918` per-recording summaries and `64 / 64` shard summaries. The merge stage completed successfully with run-log evidence: `shards=64`, `variants=918`, `records=35,909,652`, `train_core=19,211,006`, `eval=16,698,646`, `failures=0`.
- Merged artifacts exist on the pod: `artifacts/sources/d2e_full_corpus_decode_summary_accel64.json` (`~1.2M`), `outputs/data/d2e_full_corpus_accel64/train_core.jsonl` (`367G`), and `outputs/data/d2e_full_corpus_accel64/target_all_eval.jsonl` (`319G`).
- The first integrated torchrun failed before producing IDM metrics/checkpoints because non-rank0 workers timed out at the first NCCL barrier after 600 seconds while rank0 was scanning full-corpus streaming stats. Root-cause evidence was recorded in pod-local `artifacts/idm/g003_accel64_failed_training_timeout.json` and the failed run log.
- Pushed/deployed commit `3d3d9c5`: adds configurable `distributed_timeout_seconds=21600`, precomputes streaming IDM stats before torchrun, and adds `scripts/run_g003_accel64_training_resume.sh` to resume from merged accel64 artifacts without rerunning extraction. Pod targeted tests passed (`9 passed`).
- Relaunched the accel64 training-only resume at 09:14 KST with parent PID `250901`, fresh GPU monitor, and fresh postrun watcher. At 09:25 KST, `scripts/precompute_streaming_idm_stats.py` was CPU-active and no IDM metrics/checkpoints existed yet. The live-health warning `parent_running_no_known_worker` is expected during this precompute stage because the health classifier does not yet label the new precompute helper as an IDM worker.
- G003 remains non-terminal: do not checkpoint complete until resumed IDM training, split stats, finalization, accel64 audit, promotion, and canonical G003 audit pass.

## 2026-05-22 09:45 KST accel64 training-timeout recovery patch

- Accel64 extraction/merge is complete on the pod (`918 / 918` variants, `64 / 64` shards), but the resumed training lane was still in single-process `precompute_streaming_idm_stats.py` after ~30 min and had read only ~17.4 GiB of the ~393 GiB merged `train_core.jsonl`; projected wall time was too high for the G003 gate.
- Local patch adds scalable recovery mechanics before redeploying:
  - streaming IDM stats can scan multiple shard JSONLs and merge Welford moments/category counts;
  - accel64 config uses `train_records_glob` / `target_records_glob`, `precompute_num_workers=32`, and bounded convergence validation (`convergence_eval_max_examples=262144`);
  - distributed IDM destroys the process group after synchronized training so non-rank0 workers do not wait during rank0 full-target prediction;
  - split-stat builder can stream ordered prediction/ground-truth rows using precomputed train stats instead of loading full train/target/prediction JSONLs into memory;
  - G003 live-health classifies stats precompute as `idm_stats_precompute` rather than warning `parent_running_no_known_worker`.
- Validation before commit/deploy: `uv run pytest -q` -> `305 passed`.
- Next operational step: commit/push, pull in pod, stop the slow precompute parent, and restart `scripts/run_g003_accel64_training_resume.sh` so G003 continues from already-merged accel64 data with parallel stats and sharded distributed training.

## 2026-05-22 10:26 KST accel64 stats complete / 4xH200 training active

- Pod pulled `626ba3d` and restarted the accel64 resume lane from existing full merge artifacts.
- Shard-parallel stats precompute completed successfully and wrote `outputs/idm_streaming_d2e_full_compact_accel64/streaming_stats.json` with `19,211,006` train examples and input dim `620`.
- `torchrun --standalone --nproc-per-node=4 scripts/train_idm_streaming.py --config configs/model/idm_streaming_d2e_full_compact_accel64.yaml --require-torch` is active under parent PID file `outputs/cluster/g003_full_compact_accel64.pid`.
- Live health reports `healthy_running` / `idm_training`; 4 GPUs have allocated training memory. `train_history.json`, checkpoint, final metrics, split stats, and G003 audit are still pending.

### 2026-05-22 11:15 KST G004 shard-aware prep deployed while G003 trains

- G003 accel64 remains non-terminal but healthy: full extraction/merge is complete (`918 / 918` variants, `64 / 64` shards, `19,211,006` train-core rows, `16,698,646` target-eval rows). Shard-parallel stats precompute is complete (`streaming_stats.json`, input dim `620`).
- Active 4×H200 torchrun is still in `idm_training` under parent PID `251593`; rank workers `252006`-`252009` were CPU-active. `train_history.json`, checkpoint, metrics, split-stat summary, finalization summary, and completion audit were still absent at this snapshot.
- Local/pushed commit `14f4f60` prepares G004 before the hard gate opens: FDM materialization now keeps audit-required monolithic evidence files while also writing train/target shard JSONLs, passes shard paths into DDP training, raises G004 distributed timeout to 24h, bounds convergence validation, uses parallel stats precompute, and makes G004 split statistics stream over the target shard glob with precomputed train stats.
- Validation: `uv run pytest -q` passed (`307` tests). Pod checkout was fast-forwarded to `14f4f60` without stopping the active G003 Python/torchrun processes, so future G004 launch sees the shard-aware patch.
- G003 remains non-terminal. Do not checkpoint complete until accel64 training/eval/finalization pass, promotion to canonical paths succeeds, and canonical `artifacts/idm/g003_full_idm_completion_audit.json` reports `pass`.

### 2026-05-22 11:38 KST kubeconfig monitor handoff

- The previously used production kubeconfig at `/home/top321902/.kube/mlxp/jeonghunpark/production-kubeconfig.yaml` began returning Kubernetes `Unauthorized` while the MLXP reservation API still reported the production pod as running.
- The debug kubeconfig at `/home/top321902/.kube/mlxp/jeonghunpark/debug-kubeconfig.yaml` authenticated successfully for the same `p-production` namespace and pod, so monitoring was switched to that kubeconfig without touching the training process.
- Local old watch session using the stale production kubeconfig was killed; a new watch loop is running via `/tmp/g003_train_watch_debugkcfg.sh`.


### 2026-05-22 12:26 KST G003 still healthy; local G004 trainer optimization pushed

- G003 accel64 remains non-terminal. Full extraction/merge and stats precompute are complete, and 4×H200 IDM torchrun is still active under parent PID file `outputs/cluster/g003_full_compact_accel64.pid`.
- Fresh probe at 12:22 KST reported live health `healthy_running`, stage `idm_training`, decoded `918 / 918`, complete shards `64 / 64`, and watcher status `waiting_active_parent`. No `train_history.json`, checkpoint, metrics, split-stat summary, finalization summary, or completion audit existed yet.
- Rank file-progress probe showed rank workers still CPU-active in epoch streaming; the slowest inferred rank was at ~0.67 of its assigned first-epoch shard bytes while one rank had already exhausted its current file handle and was likely waiting in the DDP join path. Continue waiting rather than restarting.
- Local commits `60a25da` and `8148678` were pushed to origin to reduce repeated per-batch tensor/lookup setup and add opt-in chunked tensor training caches for future runs. Validation passed: `uv run pytest -q tests/test_streaming_idm_contract.py tests/test_streaming_fdm_contract.py` and full `uv run pytest -q` (`307` tests).
- Do not pull `60a25da`/`8148678` into the pod while the active G003 torchrun is still running. Pull it after G003 training/finalization exits and before G004 launch/promotion-related follow-up, so the running source tree is not mutated under active Python workers.
- No G003 OMX checkpoint and no aggregate `update_goal` call was made.


### 2026-05-22 13:36 KST G003 first full-corpus IDM epoch completed

- G003 accel64 4×H200 torchrun completed epoch 1 over the full train-core corpus: `19,211,006` examples, `4,692` aggregate batches, loss `1.5158540560842337`.
- Rank0 convergence evaluation wrote `outputs/idm_streaming_d2e_full_compact_accel64/train_history.json` and `convergence_report.json` with `262,144` target examples. Composite validation score after epoch 1: `0.1058949944649953` (`keyboard_accuracy=0.006023485509156915`, `mouse_button_f1=0.002607845267847441`, `mouse_move_pearson=0.30905365261798157`).
- The same torchrun continued into epoch 2 without restart; a 13:36 KST rank-progress probe showed all four ranks reading their first assigned epoch-2 shard paths.
- Final G003 artifacts are still pending: epochs 2-3, final checkpoint, full target prediction/pseudolabels, metrics, split statistics, integrated finalization summary, accel64 completion audit, promotion to canonical paths, and canonical completion audit. Do not checkpoint G003 yet.


### 2026-05-22 13:50 KST prediction-resume hardening prepared locally

- Local/origin code now includes `resume_predictions=true` support for `predict_streaming_idm_checkpoint`: if matching `pseudolabels.jsonl` and `predictions.jsonl` prefixes exist, the predictor verifies sequence order, recomputes metrics over the prefix, and appends inference for the remaining records.
- The G004 train-core pseudo-label config enables this resume path. Commit `314e55b` also enables `resume_predictions` in the full G003 IDM and G004 FDM training configs, so interrupted final target prediction can resume after a future rerun. This is future-run hardening only; it was validated locally and must not be pulled into the active pod checkout until the current G003 torchrun/finalization exits.


### 2026-05-22 13:58 KST checkpoint-output recovery helper prepared locally

- Local/origin code now includes `scripts/recover_idm_streaming_outputs.py` and `recover_streaming_idm_outputs_from_checkpoint`. If a full-corpus IDM run has saved `checkpoint.pt` but exits before prediction/metrics/metadata/summary finish, this helper reruns or resumes checkpoint prediction and reconstructs `checkpoint_metadata.json` plus the train-summary artifact without retraining.
- Validation passed locally with targeted recovery tests and full pytest. This remains future-run hardening only; do not pull into the active pod checkout until the current G003 torchrun/finalization exits.
