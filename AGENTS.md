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
- Use the Lore commit protocol for every commit: intent-first subject plus meaningful trailers (`Constraint:`, `Rejected:`, `Confidence:`, `Scope-risk:`, `Directive:`, `Tested:`, `Not-tested:`). Include `Co-authored-by: OmX <omx@oh-my-codex.dev>` when appropriate.
- Preserve configs, manifests, hashes, dataset fingerprints, split contracts, checkpoints, predictions, metrics, reports, and monitor artifacts. Future agents should resume from committed files, not chat history.
- Keep claims evidence-bound. If a metric/harness claim lacks committed evidence, phrase it as pending/future work.
- Do not commit secrets, tokens, kubeconfigs, private reservation payloads, or unredacted sensitive MLXP data.

## Current ultragoal state (2026-05-21 KST)

Active aggregate Codex objective:

> Complete approved full-corpus FDM-D2E ultragoal stories G001-G009 in `.omx/ultragoal/goals.json`, preserving D2E-only hard gates before D2E+aux/runtime claims.

Current `.omx/ultragoal/goals.json` status:

- `G001-data-universe-audit` — complete.
- `G002-split-leakage-contract` — complete.
- `G003-d2e-only-idm` — **in_progress** and active.
- `G004-d2e-only-fdm-4xh200` — pending.
- `G005-aux-data-best-model` — pending.
- `G006-evaluation-failure-analysis` — pending.
- `G007-runtime-sdk-adapter` — complete for the adapter-contract slice only.
- `G008-live-game-suite` — pending.
- `G009-report-repro-package` — pending.

Do **not** mark the Codex goal or aggregate ultragoal complete until G001-G009 are all complete and final quality gates pass.

## Current G003 MLXP run

Latest known live run snapshot: 2026-05-21 16:49 KST.

- Reservation: `rsv-jeonghunpark-20260521-76e25a`.
- Pod: `prod-rsv-jeonghunpark-20260521-76e25a`, namespace `p-production`.
- Pod repo path: `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`.
- Current run command: `NUM_SHARDS=16 bash scripts/run_g003_d2e_full_idm_parallel.sh`.
- Parent PID file: `outputs/cluster/g003_full_compact_parallel.pid`; last observed PID `9289` running.
- Pod checkout contains finalizer hardening commits through `7311f33` (`Make G006 evaluation finalization repeatable`); G003/G004/G005/G006/G008 finalizers, split-stat generation, completion audits, and package-manifest updates are present in pod.
- Latest monitor artifact: `artifacts/idm/g003_full_compact_parallel_progress.json`.
- Last decoded count: `123 / 918` recording variants; shard summaries `0 / 16`; IDM metrics absent.
- Monitor status was `running`; stale/no-progress shard lists empty. Treat as progress telemetry only until parent exits or shard logs/processes stop progressing.
- Parent PID `9289` was still running at elapsed `05:51:10`; attached GPU monitor PID `31950` was still running at elapsed `02:02:16` and writing `artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv` in the pod. Do not commit/push that live CSV from local until the run exits, because the pod currently owns it as an untracked live output.

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

### Current-run 4×H200 monitor recovery

The active integrated run predates the dedicated standalone train wrapper, so do
not restart it just to create GPU-monitor evidence. The attached monitor path is
now installed in the pod and currently running as PID `31950`:

```bash
kubectl -n p-production exec prod-rsv-jeonghunpark-20260521-76e25a -- bash -lc '
  cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
  git fetch origin main && git pull --ff-only origin main
  export PATH="$HOME/.local/bin:$PATH"
  nohup /root/.local/bin/uv run python scripts/attach_g003_gpu_monitor.py \
    --pid-file outputs/cluster/g003_full_compact_parallel.pid \
    --output artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv \
    --metadata-out artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json \
    --monitor-pid-file outputs/cluster/g003_attached_gpu_monitor.pid \
    --interval-seconds 30 \
    > artifacts/idm/g003_attached_gpu_monitor.log 2>&1 &
  echo monitor_launcher_pid=$!
'
```

When the integrated run exits, synthesize
`artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json` with:

```bash
uv run python scripts/build_g003_attached_train_run_summary.py
```

The summary builder is fail-closed and requires integrated run evidence,
checkpoint metadata/metrics, attached-monitor metadata, and CSV coverage for all
four GPU indices before it reports `exit_code=0`.

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

Latest G003 finalization helper: commit `e4a5524` adds
`scripts/finalize_g003_integrated_run.py`. After parent PID `9289` exits, run
`uv run python scripts/finalize_g003_integrated_run.py` to build any missing
G003 split stats, synthesize attached 4×H200 train-run evidence, and run the
G003 completion audit. It refuses to proceed while the parent is still running
unless explicitly overridden and does not mutate OMX state.

Latest G004 finalization helper: commit `65b5d24` adds
`scripts/finalize_g004_d2e_full_fdm.py`. After a G004 4×H200 run exits, run
`uv run python scripts/finalize_g004_d2e_full_fdm.py` to require the G004 run
summary, build any missing split stats, and run the G004 completion audit
without mutating OMX state.

## G005 completion gate

Do **not** checkpoint `G005-aux-data-best-model` complete until `scripts/validate_g005_aux_completion.py` reports `status=pass` in `artifacts/aux/g005_aux_completion_audit.json`. This audit requires G003/G004 complete, selected aux provenance/storage policy, `artifacts/aux/g005_aux_namespace_manifest.json` with `completion_ready=true`, separated `outputs/aux/<dataset_id>/...` namespaces, source-specific action heads, byte-identical D2E eval-manifest hashes for D2E-only vs D2E+aux, ablation across all required splits, no aux leakage into D2E heldouts, target split tags, prediction coverage, and run evidence. Build that namespace manifest with `scripts/build_g005_aux_namespace_manifest.py` from explicit per-source materialization evidence and D2E eval hash evidence rather than editing it by hand.

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
