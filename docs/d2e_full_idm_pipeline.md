# D2E Full-Corpus IDM Pipeline

This is the G003 execution path for the full-corpus D2E-only IDM story. It is
designed for MLXP/PVC execution, not local smoke-only evidence.

## Inputs

- Data universe: `artifacts/sources/d2e_full_data_universe_manifest.json`
- Leakage-safe split contract: `artifacts/sources/d2e_full_split_contract.json`
- Extraction config: `configs/data/d2e_full_corpus.yaml`
- Streaming IDM config: `configs/model/idm_streaming_d2e_full_compact.yaml`

The extraction config includes both `d2e_480p` and `d2e_original`. Do not add a
per-recording/bin cap for final G003 evidence; caps are only for local debugging.

## Cluster command

From the MLXP PVC checkout:

```bash
cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
git pull --ff-only origin main
uv sync --frozen --extra d2e --extra test --extra train
NUM_SHARDS=16 bash scripts/run_g003_d2e_full_idm_parallel.sh
```

The parallel script launches disjoint recording-variant extraction shards,
merges split-aware JSONL files, trains a streaming IDM without loading all D2E
windows into GPU memory, builds the preregistered split-specific statistical
comparisons, and writes a run evidence JSON under `artifacts/idm/`.

Use the sequential `scripts/run_g003_d2e_full_idm.sh` only for debugging. The
uncapped full 480p+original corpus should use parallel shards; `NUM_SHARDS=16`
is the current MLXP setting for a 128-core H200 production pod.

## Expected outputs

- `outputs/data/d2e_full_corpus/all_records.jsonl`
- `outputs/data/d2e_full_corpus/train_core.jsonl`
- `outputs/data/d2e_full_corpus/target_temporal.jsonl`
- `outputs/data/d2e_full_corpus/target_heldout_recording.jsonl`
- `outputs/data/d2e_full_corpus/target_heldout_game.jsonl`
- `outputs/data/d2e_full_corpus/target_all_eval.jsonl`
- `artifacts/sources/d2e_full_corpus_decode_summary.json`
- `outputs/idm_streaming_d2e_full_compact/checkpoint.pt`
- `outputs/idm_streaming_d2e_full_compact/pseudolabels.jsonl`
- `outputs/idm_streaming_d2e_full_compact/predictions.jsonl`
- `outputs/idm_streaming_d2e_full_compact/metrics.json`
- `outputs/idm_streaming_d2e_full_compact/split_temporal_statistical_comparison.json`
- `outputs/idm_streaming_d2e_full_compact/split_heldout_recording_statistical_comparison.json`
- `outputs/idm_streaming_d2e_full_compact/split_heldout_game_statistical_comparison.json`
- `artifacts/eval/g003_split_statistical_comparisons_summary.json`
- `artifacts/idm/idm_streaming_d2e_full_compact_summary.json`
- `artifacts/idm/g003_d2e_full_idm_run_full_compact.json`
- `artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json` for the
  parallel run path.
- Downstream G004 FDM path after these artifacts exist:
  `docs/d2e_full_fdm_pipeline.md`.

The G003 completion audit is intentionally stricter than artifact existence. It
must prove the D2E-only universe contains and the decode summary actually covers
both required D2E sources/resolution tiers: `d2e_480p` / `480p` and
`d2e_original` / `original_fhd_qhd`, with 459 recording variants from each
source. A full count of 918 without those source/tier counts is not sufficient
for checkpointing `G003-d2e-only-idm`.

## Operational notes from the first MLXP attempt

- Production base image may not include `uv`; install with
  `python3 -m pip install --user uv` and export `PATH="$HOME/.local/bin:$PATH"`.
- Production base image may not include `ffmpeg`; install with
  `apt-get update -y && apt-get install -y ffmpeg`.
- Keep the D2E cache source-namespaced (`.../cache/d2e_480p`,
  `.../cache/d2e_original`) because the two Hugging Face repos share
  game/recording filenames.
- The first sequential uncapped attempt decoded one 480p recording in about
  nine minutes; do not rely on the sequential script for the full corpus.

## Claim boundary

G003 is complete only after the uncapped run consumes all included D2E source
variants from the data universe and produces checkpoint/metrics/pseudolabel
artifacts. The compact-feature trainer is a D2E-only IDM baseline/labeler; it is
not an FDM-1 parity claim and it is not live-game evidence.


## Progress monitor

For long MLXP extraction/training runs, use the non-mutating monitor documented
in `docs/g003_progress_monitoring.md`:

```bash
uv run python scripts/monitor_g003_progress.py \
  --stale-seconds 7200 \
  --output artifacts/idm/g003_full_compact_parallel_progress.json
```

The monitor summarizes decoded/expected recording variants, completed shards,
stale/no-progress shards, parent PID state, and whether merged train/eval plus
IDM metrics exist. It is progress evidence only and does not complete G003.

## Attached 4×H200 GPU monitor for an already-running integrated run

If the integrated parallel script was launched before the standalone
`run_g003_idm_training_4xh200.sh` wrapper is used, do **not** restart or kill the
long-running extraction/training process just to add monitor evidence. Attach a
non-mutating GPU sampler to the parent PID instead:

```bash
nohup uv run python scripts/attach_g003_gpu_monitor.py \
  --pid-file outputs/cluster/g003_full_compact_parallel.pid \
  --output artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv \
  --metadata-out artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_attached.json \
  --monitor-pid-file outputs/cluster/g003_attached_gpu_monitor.pid \
  --interval-seconds 30 \
  > artifacts/idm/g003_attached_gpu_monitor.log 2>&1 &
```

The monitor is idempotent: if `outputs/cluster/g003_attached_gpu_monitor.pid`
points at a live process, it records `existing_monitor_running` and does not
start a duplicate sampler unless `--force` is explicitly used. This evidence is
only GPU-utilization telemetry; it does not prove G003 completion.

After the integrated run finishes and writes
`artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json`, synthesize the
required G003 train-run summary:

```bash
uv run python scripts/build_g003_attached_train_run_summary.py
```

The builder only passes when the integrated run evidence, IDM summary,
checkpoint metadata, metrics, attached monitor metadata, and a GPU monitor CSV
covering all four GPU indices exist. Until then it writes
`artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json` with `exit_code=2` and
explicit findings so the final gate remains fail-closed.

For the active integrated run that predates automatic split-stat generation, use
the post-run finalizer once the parent PID has exited:

```bash
uv run python scripts/finalize_g003_integrated_run.py
```

The finalizer is safe by default: if the parent PID is still running, it writes
`artifacts/idm/g003_integrated_finalization_summary.json` with
`status=blocked_active_parent` and does not build downstream artifacts. After
the run exits, it builds missing G003 split-stat comparisons, synthesizes the
attached 4×H200 train-run summary, and runs the G003 completion audit. It does
not checkpoint OMX state; checkpoint `G003-d2e-only-idm` only after the
finalizer and `artifacts/idm/g003_full_idm_completion_audit.json` report pass.

To avoid missing the parent exit in long MLXP sessions, start the non-mutating
post-run watcher:

```bash
nohup uv run python scripts/watch_g003_then_finalize.py \
  > artifacts/idm/g003_postrun_watcher.log 2>&1 &
```

The watcher writes `outputs/cluster/g003_postrun_watcher.pid` while active and
periodically updates `artifacts/idm/g003_postrun_watcher_summary.json`. Once the
parent exits, it runs `scripts/finalize_g003_integrated_run.py` with the same
fail-closed gates. It never checkpoints G003 or mutates OMX/Codex state.

## Distributed IDM training

`scripts/run_g003_d2e_full_idm_parallel.sh` defaults to
`IDM_NPROC_PER_NODE=4` and launches the streaming IDM stage with `torchrun`
after shard merge. The active config records one validation checkpoint per epoch
and a preregistered convergence report:

- `eval_interval_epochs=1`,
- `convergence_score=composite_primary`,
- `plateau_patience=3`,
- `plateau_min_relative_improvement=0.01`.

The checkpoint metadata must report `distributed.enabled=true` and
`distributed.world_size=4` for the primary G003 multi-GPU run. If the currently
running extraction was launched before this script revision, verify the actual
training command in `artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.log`;
if it fell back to single-GPU training, rerun only the IDM training stage with
`torchrun` on the merged full-corpus JSONLs before checkpointing G003 complete.

The standalone rerun/recovery entry point is:

```bash
NPROC_PER_NODE=4 EXPECTED_GPUS=4 bash scripts/run_g003_idm_training_4xh200.sh
```

It requires merged full-corpus JSONLs, runs GPU smoke, launches `torchrun`, and
writes `artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json` plus
`artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv`. These artifacts are
part of the final G003 gate so the IDM claim includes explicit multi-GPU
training evidence. By default it also runs
`scripts/build_split_statistical_comparisons.py --config configs/eval/g003_split_statistics.yaml`
after successful training and records the split-stat summary status in the run
summary. Set `BUILD_SPLIT_STATS=0` only for local recovery/debug runs; terminal
G003 evidence still requires the summary and all three split comparison files.


## G003 checkpoint metadata contract

The streaming IDM training step writes `outputs/idm_streaming_d2e_full_compact/checkpoint_metadata.json`
and `resolved_config.json`. These artifacts must carry explicit provenance for
G003 checkpointing:

- config fingerprint and originating config path,
- train/target JSONL paths,
- D2E data-universe path/hash/fingerprint,
- split-contract path/hash/fingerprint and split id,
- source namespace (`d2e_full_corpus`), source ids, resolution tiers, and target
  eval split tags,
- pseudolabel/prediction/metrics/label-quality/statistical-comparison paths.

This metadata is required so later FDM and report stages can prove they used the
D2E-only full-corpus split rather than historical bounded or auxiliary data.

## G003 completion audit

Before checkpointing `G003-d2e-only-idm` complete, run:

```bash
uv run python scripts/validate_g003_full_idm_completion.py
```

During active extraction/training this may be run with `--allow-fail`, but a
terminal G003 checkpoint requires `artifacts/idm/g003_full_idm_completion_audit.json`
to report `status == pass`. The repo config runs this audit as a pre-checkpoint
evidence gate (`require_goal_checkpoint_complete=false`): it reports the current
OMX goal status but does not fail solely because G003 has not yet been
checkpointed. The final quality gate separately verifies that G003 is complete.
The audit checks full decode coverage, merged JSONL counts,
pseudolabel/prediction counts, checkpoint metadata provenance, required target
split tags, split-stat summary status, and 4×H200 run evidence. The 4×H200
evidence includes `nproc_per_node == 4`, `expected_gpus == 4`, embedded
`run_summary.gpu_monitor_status.covers_expected_gpus == true`, and a GPU
monitor CSV with rows covering all four GPU indices. A run summary or CSV that
merely exists is not enough for terminal G003 completion.
