# AGENTS.md — Full-Corpus D2E/FDM Research Handoff

This repository is a serious research reproduction of the **publicly inferable recipe shape** of FDM-1 for game-control data using D2E. It is not a smoke demo and must never be presented as closed-source FDM-1 parity.

These instructions apply to this repository and all child paths unless a deeper `AGENTS.md` overrides them.

## Non-negotiable research contract

- **No FDM-1 parity claim.** We can reproduce/approximate method ideas, not claim equivalence to the closed-source system.
- **No non-game, robotics, or car-transfer claims.** Keep scope to game/desktop control.
- **No weak smoke-only success.** Final success requires real D2E training/evaluation, not tiny subsets or dry-run paths.
- **Full D2E gate:** consume D2E 480p plus available original/FHD/QHD sources where required by the active plan. Audited exclusions are allowed only with retry logs, reasons, and impact analysis.
- **D2E-only gates before broader claims:** D2E-only IDM/FDM results, metrics, ablations, and failure analysis must be separately reported before D2E+aux or runtime success claims.
- **D2E+aux may be final best/primary** only after D2E-only hard gates are complete and D2E-only vs D2E+aux ablations are reported.
- **Live evidence target:** open-source graphical games/tasks with closed-loop control evidence, latency/failure logs, replay/video evidence, and statistical improvement. Do not claim live commercial-game control from current artifacts.
- **Reusable artifacts target:** trained checkpoints plus inference SDK/game-ready adapter demo suitable for later plug-and-play integration.

## Operating rules for coding agents

- Prefer `uv` for dependency sync, Python execution, tests, and training/cluster commands.
- Commit regularly after coherent, verified milestones. Do not accumulate one huge commit.
- Treat sustained 4×H200 GPU idle time as a blocker/risk for this research ultragoal. During MLXP runs, actively monitor utilization, separate expected CPU/IO materialization from real GPU training, and prefer sharded/parallel/cache/prediction implementations that minimize idle GPU wall-clock while preserving completion audits and reproducibility artifacts.
- Use the Lore commit protocol for every commit: intent-first subject plus meaningful trailers (`Constraint:`, `Rejected:`, `Confidence:`, `Scope-risk:`, `Directive:`, `Tested:`, `Not-tested:`). Include `Co-authored-by: OmX <omx@oh-my-codex.dev>` when appropriate.
- Preserve configs, manifests, hashes, dataset fingerprints, split contracts, checkpoints, predictions, metrics, reports, and monitor artifacts. Future agents should resume from committed files, not chat history.
- Keep claims evidence-bound. If a metric/harness claim lacks committed evidence, phrase it as pending/future work.
- Do not commit secrets, tokens, kubeconfigs, private reservation payloads, or unredacted sensitive MLXP data.

## Current ultragoal state (authoritative as of 2026-05-23 KST)

Active aggregate Codex objective:

> Complete approved full-corpus FDM-D2E ultragoal stories G001-G009 in `.omx/ultragoal/goals.json`, preserving D2E-only hard gates before D2E+aux/runtime claims.

Current `.omx/ultragoal/goals.json` status:

- `G001-data-universe-audit` — complete.
- `G002-split-leakage-contract` — complete.
- `G003-d2e-only-idm` — complete and checkpointed in OMX.
- `G004-d2e-only-fdm-4xh200` — pending in OMX; latest MLXP run failed after reusable split materialization and must be relaunched from the fixed commit.
- `G005-aux-data-best-model` — pending.
- `G006-evaluation-failure-analysis` — pending.
- `G007-runtime-sdk-adapter` — complete for the adapter-contract slice only.
- `G008-live-game-suite` — pending.
- `G009-report-repro-package` — pending.

Do **not** mark the Codex goal or aggregate ultragoal complete until G001-G009 are all complete and final quality gates pass.

## Current G004 MLXP run/restart state

Latest known live run snapshot: 2026-05-23 09:48 KST.

- Reservation: `rsv-jeonghunpark-20260521-76e25a`.
- Pod: `prod-rsv-jeonghunpark-20260521-76e25a`, namespace `p-production`.
- Pod repo path: `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`.
- The last pod checkout was `d38a3b1`; it failed after completing split materialization.
- No active `run_g004`/`torchrun`/`watch_g004` processes were observed at 2026-05-23T00:48:00Z.
- Failure evidence: `artifacts/fdm/g004_ddp_runtime_failure_snapshot.json`; root cause was `timeout_seconds` unbound in `_distributed_runtime()` when `train_streaming_fdm` had already initialized the process group.
- Reusable materialization artifacts remain on the pod: split summary exists; `fdm_train_shards` has 16 files / `400,301,959,913` bytes; `fdm_target_shards` has 16 files / `347,089,780,692` bytes.
- Fixed code must be pulled into the pod before relaunch. The fixed trainer preserves `timeout_seconds` for already-initialized process groups and reuses the existing split summary/shards by default (`reuse_materialized_split_summary=true`), so relaunch should skip the expensive materialization rewrite and move to stats/cache/DDP.
- Claim boundary: this is failure/restart evidence only. G004 remains incomplete until `artifacts/fdm/g004_d2e_full_fdm_finalization_summary.json` reports pass and `artifacts/fdm/g004_full_fdm_completion_audit.json` reports `status=pass`, `error_count=0`, followed by OMX checkpointing with a fresh `get_goal` snapshot. Do not call `update_goal complete` for G004 in aggregate mode.

Useful G004 monitor command:

```bash
KUBECONFIG=/home/top321902/.kube/mlxp/jeonghunpark/debug-kubeconfig.yaml \
  kubectl --request-timeout=600s -n p-production exec -i prod-rsv-jeonghunpark-20260521-76e25a -- bash -s <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
date -Iseconds
cat outputs/cluster/g004_d2e_full_fdm_4xh200.pid 2>/dev/null || true
pgrep -af 'run_g004|torchrun|train_fdm_streaming|watch_g004|multiprocessing.spawn' || true
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits || true
REMOTE
```

## Historical G003 MLXP run (complete; do not treat as active)

The bullets in this section are preserved historical context from before G003 completion. Do not follow their active-process/pull/checkpoint instructions as current state; G003 is complete in OMX and the live pod work is G004.

Latest known live run snapshot: 2026-05-22 16:45 KST.

- Reservation: `rsv-jeonghunpark-20260521-76e25a`.
- Pod: `prod-rsv-jeonghunpark-20260521-76e25a`, namespace `p-production`.
- Pod repo path: `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`.
- Active pod checkout for the currently running G003 torchrun is `9a9f099`.
- Local/origin are intentionally ahead of the active pod checkout, with latest known local/origin `d5fdafc` adding tensor-cache, prediction-resume validation, and IDM/FDM recovery helpers. **Do not pull latest origin into the pod while the current G003 Python workers are still running.** Pull only after the active G003 torchrun/finalization exits, before recovery/promotion/G004 launch.
- Accel64 extraction and merge are complete: `918 / 918` recording variants, `64 / 64` shards, `35,909,652` total records, `19,211,006` train-core rows, `16,698,646` target-eval rows, `failures=0`.
- Active command: `bash scripts/run_g003_accel64_training_resume.sh`, parent PID file `outputs/cluster/g003_full_compact_accel64.pid`, observed parent PID `251593`.
- Active training command: `uv run torchrun --standalone --nproc-per-node=4 scripts/train_idm_streaming.py --config configs/model/idm_streaming_d2e_full_compact_accel64.yaml --require-torch`. Rank worker PIDs observed at the snapshot: `252006`, `252007`, `252008`, `252009`.
- Shard-parallel stats precompute is complete: `outputs/idm_streaming_d2e_full_compact_accel64/streaming_stats.json` exists with `19,211,006` examples and input dim `620`.
- Live health reports `healthy_running` / `idm_training`; the post-run watcher reports `waiting_active_parent`. GPU monitor CSV path: `artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64.csv`.
- Epochs 1 and 2 completed and wrote `outputs/idm_streaming_d2e_full_compact_accel64/train_history.json` plus `convergence_report.json`. Epoch 1: loss `1.5158540560842337`, validation score `0.1058949944649953`, keyboard accuracy `0.006023485509156915`, mouse-button F1 `0.002607845267847441`, mouse-move Pearson `0.30905365261798157`. Epoch 2: loss `1.448147530049278`, validation score `0.11558532553625012`, keyboard accuracy `0.006652199484860162`, mouse-button F1 `0.023052525721358397`, mouse-move Pearson `0.3170512514025318`. Epoch 3 started by 2026-05-22 16:45 KST.
- Terminal artifacts were still absent at the snapshot: `metrics.json`, `checkpoint_metadata.json`, `checkpoint.pt`, `pseudolabels.jsonl`, `predictions.jsonl`, accel64 IDM summary, split-stat summary, integrated finalization summary, and completion audit. G003 remains non-terminal.
- A stale `outputs/cluster/g003_to_g004_chain_watcher.pid` and `artifacts/fdm/g003_to_g004_chain_summary.json` may exist in the pod, but no `watch_g003_then_launch_g004.py` process was observed at 2026-05-22 14:09 KST. Do not rely on that stale watcher to launch G004; start a fresh current watcher only after G003 audit promotion and OMX checkpointing.

Useful pod monitor command:

```bash
KUBECONFIG=/home/top321902/.kube/mlxp/jeonghunpark/debug-kubeconfig.yaml \
  kubectl --request-timeout=600s -n p-production exec -i prod-rsv-jeonghunpark-20260521-76e25a -- bash -s < /tmp/g003_resume_probe.sh
```

`kubectl exec` is flaky with `connect: cannot assign requested address`; retry with a short sleep. If `production-kubeconfig.yaml` returns `Unauthorized`, use `/home/top321902/.kube/mlxp/jeonghunpark/debug-kubeconfig.yaml` with `-n p-production` (verified on 2026-05-22 11:38 KST). Non-login pod shells may not include `uv` on `PATH`, so `export PATH="$HOME/.local/bin:$PATH"` or call `/root/.local/bin/uv`.

After the active G003 accel64 torchrun/finalization exits:

1. Pull latest origin in the pod to pick up recovery/promotion hardening.
2. If `checkpoint.pt` exists but metadata/summary/predictions are missing, recover without retraining:

   ```bash
   uv run python scripts/recover_idm_streaming_outputs.py \
     --config configs/model/idm_streaming_d2e_full_compact_accel64.yaml
   ```

3. Ensure the accel64 finalization/audit reports pass.
4. Promote accel64 artifacts to canonical paths with:

```bash
uv run python scripts/promote_g003_accel64_to_canonical.py
```

Then verify canonical `artifacts/idm/g003_full_idm_completion_audit.json` reports `status=pass`, checkpoint `G003-d2e-only-idm` complete through OMX only after calling Codex `get_goal` and passing the snapshot JSON to `omx ultragoal checkpoint --goal-id G003-d2e-only-idm --status complete ...`. Do not mark the aggregate Codex goal complete. After the checkpoint is recorded, launch G004 with a fresh current watcher requiring the G003 checkpoint:

```bash
nohup uv run python scripts/watch_g003_then_launch_g004.py \
  --replace-existing-watcher \
  --launch \
  --start-g004-watcher \
  --require-g003-goal-checkpoint \
  --output artifacts/fdm/g003_to_g004_chain_summary.json \
  --watcher-pid-file outputs/cluster/g003_to_g004_chain_watcher.pid \
  --poll-seconds 60 \
  > artifacts/fdm/g003_to_g004_chain.log 2>&1 &
```

## Current G005 auxiliary materialization

Latest known live run snapshot: 2026-05-22 14:12 KST.

- G005 source materializer PID `49075` has exited; no active G005 materialization/watcher/training processes were observed.
- `artifacts/aux/g005_aux_materialization_progress.json` reports `status=pass`, `error_count=0`, `pid_running=false`.
- `artifacts/aux/g005_aux_runtime_env.json` reports `status=pass`, `error_count=0`, and the pod environment had been restored with `uv sync --extra d2e --extra train --extra test` after adding `array-record`. Do not rerun a narrower `uv sync --extra d2e` alone in the pod because it removes train/test packages.
- `artifacts/aux/g005_aux_namespace_manifest.json` reports `completion_ready=true`; local committed small artifacts already include namespace/source/materialization evidence. Do not commit raw `outputs/aux/**/raw` archives.
- `artifacts/aux/g005_aux_materialization_watcher_summary.json` reports `status=g005_launch_not_ready`; `artifacts/aux/g005_aux_completion_audit.json` still reports `status=fail` with `error_count=51`, as expected while G003/G004 D2E-only prerequisites and G005 training/ablation evidence are incomplete.
- The materialization watcher never starts G005 training or checkpoints OMX/Codex state. Do not launch G005 training until G003 and G004 are complete and checkpointed.

Useful pod monitor command:

```bash
kubectl -n p-production exec prod-rsv-jeonghunpark-20260521-76e25a -- bash -lc '
  cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
  export PATH="$HOME/.local/bin:$PATH"
  echo HEAD=$(git rev-parse --short HEAD)
  ps -p $(cat outputs/cluster/g003_full_compact_parallel.pid) -o pid,stat,etime,cmd || true
  /root/.local/bin/uv run python scripts/monitor_g003_progress.py --output /tmp/g003_progress.json
  cat /tmp/g003_progress.json
'
```

`kubectl exec` is flaky with `connect: cannot assign requested address`; retry with a short sleep. Non-login pod shells may not include `uv` on `PATH`, so use `/root/.local/bin/uv` or export `$HOME/.local/bin`.

## G003 completion gate

Do **not** checkpoint `G003-d2e-only-idm` complete until all of these exist and are validated:

- full decode summary covering all expected D2E recording variants, or audited exclusions with retry logs/reasons/impact,
- merged `outputs/data/d2e_full_corpus/train_core.jsonl` and `target_all_eval.jsonl`,
- streaming IDM checkpoint and metadata under `outputs/idm_streaming_d2e_full_compact/`,
- `checkpoint_metadata.json` and `resolved_config.json` proving config/data-universe/split/source provenance,
- pseudolabels, predictions, metrics, label-quality report, statistical comparison report, train history, convergence report,
- committed run evidence and monitor/evaluation summaries.

The local G003 completion audit is `configs/eval/g003_full_idm_completion.yaml` + `scripts/validate_g003_full_idm_completion.py` and writes `artifacts/idm/g003_full_idm_completion_audit.json`. It is expected to fail while extraction/training artifacts are missing, but must report `status=pass` before G003 checkpointing.

The current streaming IDM config must carry:

- `data_universe: artifacts/sources/d2e_full_data_universe_manifest.json`,
- `split_contract: artifacts/sources/d2e_full_split_contract.json`,
- `source_namespace: d2e_full_corpus`.

### Current-run 4×H200 monitor evidence

The active authoritative G003 lane is now accel64, not the older canonical
16-shard lane. Do not restart training just to change monitor evidence. The
active post-run watcher is expected to use accel64 evidence paths:

- PID file: `outputs/cluster/g003_full_compact_accel64.pid`
- GPU monitor CSV: `artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64.csv`
- attached-monitor metadata: `artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64_attached.json`
- train-run summary: `artifacts/idm/g003_d2e_full_idm_4xh200_train_run_accel64.json`
- integrated finalization summary: `artifacts/idm/g003_accel64_integrated_finalization_summary.json`
- completion audit: `artifacts/idm/g003_full_idm_completion_accel64_audit.json`

When accel64 audit passes, `scripts/promote_g003_accel64_to_canonical.py`
symlinks/promotes the accel64 outputs into canonical G003 paths and rebuilds the
canonical train-run summary/audit. Use that promotion path instead of trying to
manually copy monitor evidence.

## Dataset and split artifacts

Primary source/split artifacts:

- `artifacts/sources/d2e_full_data_universe_manifest.json`
- `artifacts/sources/d2e_full_split_contract.json`
- `docs/d2e_full_data_universe.md`
- `docs/d2e_full_split_contract.md`
- `configs/data/d2e_full_corpus.yaml`

Generalization split requirements include within-recording temporal heldout, heldout-recording, and heldout-game reporting. Avoid mixing auxiliary data into D2E heldout namespaces.

## Auxiliary-data policy

Auxiliary game/action datasets may be used within the 5TiB storage envelope and may become the final best/primary model **only after** D2E-only hard gates and ablations are complete.

Current planning artifacts:

- `docs/auxiliary_data_plan.md`
- `artifacts/sources/aux_game_action_dataset_candidates.json`
- terminal G005 must additionally populate `artifacts/aux/g005_aux_namespace_manifest.json`.

Required separation:

- source-specific namespaces under `outputs/aux/<dataset_id>/...`,
- clear license/provenance/storage accounting, including `source_url`, `license_id`, `provenance_sha256`, and source split hashes per selected aux source,
- source-specific action heads/token namespaces for every non-D2E action space,
- D2E-only vs D2E+aux metrics on the same D2E eval split,
- no auxiliary leakage into D2E heldout recordings/games.

## G004 completion gate

Do **not** checkpoint `G004-d2e-only-fdm-4xh200` complete until `scripts/validate_g004_full_fdm_completion.py` reports `status=pass` in `artifacts/fdm/g004_full_fdm_completion_audit.json`. This audit requires G003 complete, D2E-only FDM-from-IDM-pseudolabel provenance, split count consistency, prediction coverage, convergence-report evidence, split-stat summaries, 4×H200 run metadata, and GPU-monitor coverage for all expected GPU indices.

Current G004 hardening: FDM training must use G003 IDM prediction-only
pseudo-labels over `outputs/data/d2e_full_corpus/train_core.jsonl`, written to
`outputs/idm_streaming_d2e_full_compact/fdm_train_core_pseudolabels/pseudolabels.jsonl`,
then evaluate against untouched `outputs/data/d2e_full_corpus/target_all_eval.jsonl`.
The old recording-tail split over `target_all_eval` is local-debug only and is
rejected by the G004 completion audit via `counts.mode == explicit_target`.
G004 FDM features must use
`summary_causal_compact_grid8_time_prior_action` so the FDM input contains
current-frame/temporal/prior-action context, not next-frame inverse-dynamics
features.
`scripts/run_g004_d2e_full_fdm_4xh200.sh` auto-generates the train-core
pseudo-labels with `scripts/predict_idm_streaming.py` after G003 checkpoint
artifacts exist.

Current G004 throughput hardening: commit `14f4f60` keeps the monolithic FDM
train/target JSONLs required by the completion audit, but also writes
`outputs/fdm_streaming_d2e_full_compact/fdm_train_shards/shard_*.jsonl` and
`outputs/fdm_streaming_d2e_full_compact/fdm_target_shards/shard_*.jsonl`.
`train_streaming_fdm` passes those shard paths into the shared streaming trainer
so 4×H200 DDP ranks parse disjoint shard files instead of each rank re-reading
the full monolith. `configs/eval/g004_split_statistics.yaml` is streaming and
must compare predictions against the target shard glob, not the monolithic
target JSONL, because prediction order follows the target shard path order.
Latest local/origin trainer hardening adds optional `training_cache_dir`
chunked tensor caches for IDM/FDM training epochs. It also adds
`resume_predictions=true` for checkpoint inference so interrupted G003/G004
prediction passes can append after already-written `pseudolabels.jsonl` /
`predictions.jsonl` rows while recomputing metrics over the prefix. If a
full-corpus IDM run saves `checkpoint.pt` but exits before metadata/summary,
run `uv run python scripts/recover_idm_streaming_outputs.py --config <train-config>`
after pulling latest origin to rebuild prediction, metrics, checkpoint metadata,
and train summary without retraining. For G004 FDM wrapper recovery after
`outputs/fdm_streaming_d2e_full_compact/torch_model/checkpoint.pt` exists, run
`uv run python scripts/recover_fdm_streaming_outputs.py --config configs/model/fdm_streaming_d2e_full_compact.yaml`
to rebuild torch-model outputs plus FDM checkpoint metadata/summary without
rematerializing or retraining. This is intended for future G003 restarts and
G004 runs after the active G003 torchrun exits; do not pull new trainer commits
into the pod while the current G003 Python workers are still running.

Latest G004 launch preflight: run
`uv run python scripts/plan_g004_launch.py --check-gpus` in the pod before
starting the G004 4×H200 command. It requires G003 audit pass plus G003 OMX
checkpoint complete by default, verifies FDM input artifacts, records whether
train-core pseudo-labels already exist or will be generated, writes
`artifacts/fdm/g004_launch_readiness.json`, and does not launch training or
mutate OMX state.

Latest script hardening: commit `6974f38` makes future G003/G004 run wrappers
build split-specific statistical comparisons automatically after successful
training. The already-running G003 parent PID `9289` was launched before this
commit, so do not assume it will emit
`artifacts/eval/g003_split_statistical_comparisons_summary.json`; after the
parent exits and IDM predictions exist, run
`uv run python scripts/build_split_statistical_comparisons.py --config configs/eval/g003_split_statistics.yaml`
manually if the split-stat summary is absent before auditing/checkpointing G003.

Latest audit hardening: commit `58251dd` configures story-level completion
audits as pre-checkpoint evidence gates with
`require_goal_checkpoint_complete=false`. They still report current OMX story
status, but final quality gates are responsible for requiring all stories to be
checkpointed `complete`. This prevents a circular blocker when preparing
evidence before checkpointing G003/G004/etc.

Latest G003 finalization helper: `scripts/finalize_g003_integrated_run.py`
builds any missing G003 split stats, synthesizes attached 4×H200 train-run
evidence, and runs the G003 completion audit. For the active accel64 lane, the
post-run watcher already supplies accel64-specific arguments and should finalize
after parent PID `251593` exits. It refuses to proceed while the parent is still
running unless explicitly overridden and does not mutate OMX state.

Latest G003 post-run watcher: for active accel64, the watcher writes
`artifacts/idm/g003_accel64_postrun_watcher_summary.json`, waits for the G003
parent to exit, then invokes the non-mutating G003 finalizer. It never
checkpoints G003.

Latest G004 finalization helper: commit `65b5d24` adds
`scripts/finalize_g004_d2e_full_fdm.py`. After a G004 4×H200 run exits, run
`uv run python scripts/finalize_g004_d2e_full_fdm.py` to require the G004 run
summary, build any missing split stats, and run the G004 completion audit
without mutating OMX state.

## G005 completion gate

Do **not** checkpoint `G005-aux-data-best-model` complete until `scripts/validate_g005_aux_completion.py` reports `status=pass` in `artifacts/aux/g005_aux_completion_audit.json`. This audit requires G003/G004 complete, selected aux provenance/storage policy, `artifacts/aux/g005_aux_runtime_env.json` with `status=pass`, `artifacts/aux/g005_aux_namespace_manifest.json` with `completion_ready=true`, separated `outputs/aux/<dataset_id>/...` namespaces, source-specific action heads, byte-identical D2E eval-manifest hashes for D2E-only vs D2E+aux, ablation across all required splits, no aux leakage into D2E heldouts, target split tags, prediction coverage, and run evidence. Build that namespace manifest with `scripts/build_g005_aux_namespace_manifest.py` from explicit per-source materialization evidence and D2E eval hash evidence rather than editing it by hand.

Latest G005 finalization helper: use
`uv run python scripts/finalize_g005_aux_best_model.py --source-evidence <...> --eval-manifest-hashes <...> --completion-ready`
after D2E-only G003/G004 gates and the D2E+aux training/ablation run finish. It
builds/reuses the namespace manifest, requires G005 run summary evidence, runs
the G005 completion audit, and does not mutate OMX state.

## G006 completion gate

Do **not** checkpoint `G006-evaluation-failure-analysis` complete until `scripts/validate_g006_completion.py` reports `status=pass` in `artifacts/eval/g006_completion_audit.json`. This audit requires G003/G004/G005 complete, endpoint statistics, failure analysis, claim taxonomy, readiness audit, final artifact-build summary, required splits/endpoints, required failure axes, documented non-rejections/examples, required claim states with evidence paths for claimable/documented claims, and forbidden claim boundaries.

Latest G006 finalization helper: after G003/G004/G005 are complete and split-stat
artifacts exist, run `uv run python scripts/finalize_g006_evaluation.py`. It
rebuilds final endpoint statistics, failure analysis, and claim taxonomy, runs
readiness plus completion audits, writes a finalization summary, and does not
mutate OMX state.

## G007 completion gate

Do **not** modify or rely on completed `G007-runtime-sdk-adapter` evidence unless `scripts/validate_g007_completion.py` reports `status=pass` in `artifacts/runtime/g007_completion_audit.json`. This audit preserves G007 as a safe deterministic adapter-contract story only, not G008 live game control or any commercial-game claim.

## G008 completion gate

Do **not** checkpoint `G008-live-game-suite` complete until `scripts/validate_g008_live_suite_completion.py` reports `status=pass` in `artifacts/harness/g008_live_suite_completion_audit.json`. This audit requires completed D2E-only training prerequisites, G007 runtime-adapter evidence, trained checkpoint metadata, live evidence validation `quality_gate.status=pass`, statistical comparison evidence, and hashed video/replay/latency/failure artifacts. Protocol readiness alone never satisfies G008.

Latest G008 finalization helper: after collecting explicit live evidence, run
`uv run python scripts/finalize_g008_live_suite.py --evidence artifacts/harness/<run>/live_suite_evidence.json`.
It writes the protocol report, validates evidence to
`artifacts/harness/g008_live_open_game_suite_evidence_validation.json`, runs the
G008 completion audit, writes a finalization summary, and does not mutate OMX
state. Do not use it without `--evidence` except for non-terminal diagnostics.

## Runtime and harness boundary

Current adapter-contract evidence is not live commercial-game control. Future live evidence must use open-source graphical games/tasks and include:

- target game/version/config/map/task protocol,
- explicit `evidence_mode` of `live_desktop_control` or `live_graphical_game_control`,
- screen capture + frame → inference → action → next-frame closed loop,
- OS/input adapter focus guard, kill switch, and rate limits,
- latency/FPS/dropped-frame/input logs,
- video/replay/trace evidence,
- multiple episodes/seeds and baseline comparison.

Commercial-game plug-and-play should be treated as an artifact/API compatibility target, not a current empirical claim.

## G009 completion gate

Do **not** checkpoint `G009-report-repro-package` complete until `scripts/validate_g009_completion.py` reports `status=pass` in `artifacts/reproducibility/g009_completion_audit.json`. This audit requires G001-G008 complete, final report/evidence/runbook docs, claim-boundary audit pass, final quality audit `status=pass`, and package-manifest coverage with matching hashes for the final quality audit and other required report artifacts.

Latest G009 finalization helper: run
`uv run python scripts/finalize_g009_report_package.py` to refresh the
claim-boundary audit, final-quality audit, package manifest, and G009 completion
audit in one fail-closed sequence. It writes
`artifacts/reproducibility/g009_finalization_summary.json` and does not mutate
OMX state or checkpoint G009.

## Verification commands

Run targeted checks first for the files you changed, then broader gates before committing claims.

Common checks:

```bash
uv run pytest -q
uv run python scripts/audit_claim_boundaries.py --output artifacts/reproducibility/claim_boundary_audit.json
uv run python scripts/build_repro_package_manifest.py --output artifacts/reproducibility/package_manifest.json
uv run python scripts/validate_g003_full_idm_completion.py --allow-fail
uv run python scripts/validate_g004_full_fdm_completion.py --allow-fail
uv run python scripts/validate_g005_aux_completion.py --allow-fail
uv run python scripts/validate_g006_completion.py --allow-fail
uv run python scripts/validate_g007_completion.py
uv run python scripts/validate_g008_live_suite_completion.py --allow-fail
uv run python scripts/validate_g009_completion.py --allow-fail
uv run python scripts/validate_final_quality_gates.py --allow-fail
```

Expected while G003-G009 remain incomplete: `validate_final_quality_gates.py --allow-fail` reports missing-artifact errors. This is not a blocker unless a completed goal is missing required evidence.

For streaming IDM edits:

```bash
uv run python -m py_compile src/fdm_d2e/training/streaming_idm.py scripts/train_idm_streaming.py
uv run pytest tests/test_streaming_idm_contract.py -q
```

## Documentation map

Read/update these when changing claims or handoff state:

- `notes/ultragoal-operating-notes.md` — persistent constraints, decisions, and active-run reminders.
- `notes/g003-mlxp-run-status.md` — MLXP pod/run snapshots.
- `notes/fdm-d2e-full-reproduction-deep-interview.md` — clarified requirements.
- `notes/fdm-d2e-full-reproduction-ralplan.md` — approved plan.
- `docs/d2e_full_idm_pipeline.md` — G003 pipeline and checkpoint metadata contract.
- `docs/final_quality_gates.md` and `configs/eval/final_quality_gates.yaml` — completion gate definitions.
- `.omx/ultragoal/goals.json` and `.omx/ultragoal/ledger.jsonl` — workflow state and checkpoint ledger.

Regenerate `artifacts/reproducibility/package_manifest.json` after changing durable docs/artifacts so hashes stay current.
