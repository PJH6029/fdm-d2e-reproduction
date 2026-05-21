# D2E Full-Corpus FDM Pipeline

This is the G004 execution path prepared while the G003 full-corpus IDM run is
still producing IDM pseudo-labels. It is a D2E-only FDM lane and does not make
an FDM-1 parity claim.

## Inputs

- D2E FDM-train records: `outputs/data/d2e_full_corpus/train_core.jsonl`
- D2E heldout target records: `outputs/data/d2e_full_corpus/target_all_eval.jsonl`
- IDM pseudo-labels for FDM training:
  `outputs/idm_streaming_d2e_full_compact/fdm_train_core_pseudolabels/pseudolabels.jsonl`
- FDM config: `configs/model/fdm_streaming_d2e_full_compact.yaml`
- Endpoints: `configs/eval/primary_endpoints.yaml`
- G003 IDM metadata: `outputs/idm_streaming_d2e_full_compact/checkpoint_metadata.json`
- D2E universe/split provenance:
  `artifacts/sources/d2e_full_data_universe_manifest.json` and
  `artifacts/sources/d2e_full_split_contract.json`

The pseudo-label and record JSONLs are order-joined by `sequence_id`. G004 now
uses a prediction-only G003 IDM pass over `train_core.jsonl` to create the FDM
training labels, then evaluates on the untouched `target_all_eval.jsonl`. This
preserves heldout-recording and heldout-game target namespaces instead of
training the FDM on earlier windows from those heldout recordings/games. A
sequence mismatch fails the run instead of silently mixing artifacts.

## FDM-1 recipe alignment boundary

The public FDM-1 recipe describes the FDM as next-action prediction conditioned
on prior frames and actions, after an IDM labels videos with keyboard/mouse
tokens. G004 therefore uses a causal compact feature mode for the FDM stage:

- `summary_causal_compact_grid8_time_prior_action`
- current frame summary + current compact grid/luma + temporal basis
- previous action-token sketch from `prior_action_tokens`
- no next-frame or frame-delta features in the FDM input

For offline evaluation this is teacher-forced on previous actions only:
training rows use the previous IDM pseudo-label, and target rows use the
previous D2E ground-truth action within the same target recording. This remains
an offline next-action evaluation, not closed-loop action execution; G008 is the
separate closed-loop/live-suite gate.

## Training/evaluation split

`src/fdm_d2e/training/streaming_fdm.py` materializes:

- `fdm_train_pseudolabeled_records.jsonl`: D2E records whose
  `ground_truth_tokens` are replaced by IDM-generated tokens and whose
  `prior_action_tokens` contain the previous IDM-generated action.
- `fdm_target_ground_truth_records.jsonl`: recording-tail heldout records that
  retain real D2E ground-truth tokens for evaluation and carry previous-action
  teacher-forcing context only.
- `fdm_streaming_split_summary.json`: counts, input hashes, and split
  fingerprint.
- `resolved_config.json` and `checkpoint_metadata.json`: config fingerprint,
  source IDM metadata/hash, D2E universe/split-contract metadata, source
  namespace, source ids, resolution tiers, split names, and target eval tags.

Default full-corpus split is explicit train/target:

- train: `train_core.jsonl` with G003 IDM pseudo-labels;
- target: `target_all_eval.jsonl` with real D2E labels and temporal,
  heldout-recording, and heldout-game tags.

The older per-recording temporal tail mode remains available for local
debugging if no `target_records_path` is supplied, but the G004 completion audit
requires `counts.mode == explicit_target`.

The local-debug tail split defaults are:

- `fdm_train_fraction=0.75`
- `min_target_per_recording=1`

This preserves the no-oracle-control boundary for FDM training while still
evaluating against real D2E labels.

## 4×H200 command

After G003 has completed and the pod checkout has the latest code:

```bash
cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
git pull --ff-only origin main
uv sync --frozen --extra d2e --extra test --extra train
NPROC_PER_NODE=4 EXPECTED_GPUS=4 bash scripts/run_g004_d2e_full_fdm_4xh200.sh
```

The script runs a GPU smoke check, launches the streaming action trainer through
`torchrun`, builds split-specific statistical comparisons for the preregistered
temporal/heldout-recording/heldout-game splits, and writes:

- `outputs/idm_streaming_d2e_full_compact/fdm_train_core_pseudolabels/pseudolabels.jsonl`
- `artifacts/idm/idm_streaming_d2e_full_compact_fdm_train_core_pseudolabels_summary.json`
- `outputs/fdm_streaming_d2e_full_compact/checkpoint_metadata.json`
- `outputs/fdm_streaming_d2e_full_compact/resolved_config.json`
- `outputs/fdm_streaming_d2e_full_compact/summary.json`
- `outputs/fdm_streaming_d2e_full_compact/fdm_streaming_split_summary.json`
- `outputs/fdm_streaming_d2e_full_compact/torch_train_summary.json`
- `outputs/fdm_streaming_d2e_full_compact/torch_model/checkpoint.pt`
- `outputs/fdm_streaming_d2e_full_compact/torch_model/convergence_report.json`
- `outputs/fdm_streaming_d2e_full_compact/torch_model/predictions.jsonl`
- `outputs/fdm_streaming_d2e_full_compact/torch_model/metrics.json`
- `outputs/fdm_streaming_d2e_full_compact/torch_model/statistical_comparison.json`
- `outputs/fdm_streaming_d2e_full_compact/split_temporal_statistical_comparison.json`
- `outputs/fdm_streaming_d2e_full_compact/split_heldout_recording_statistical_comparison.json`
- `outputs/fdm_streaming_d2e_full_compact/split_heldout_game_statistical_comparison.json`
- `artifacts/eval/g004_split_statistical_comparisons_summary.json`
- `artifacts/fdm/fdm_streaming_d2e_full_compact_summary.json`
- `artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json`
- `artifacts/fdm/g004_d2e_full_fdm_4xh200_gpu_monitor.csv`

The run summary records `gpu_monitor_status.covers_expected_gpus`. Terminal
G004 evidence must show monitor rows for all expected GPU indices (`0..3` for
the 4×H200 run); a CSV that merely exists is not enough.
The run summary also records `split_stats_summary_exists` and
`split_stats_status`. `BUILD_SPLIT_STATS=0` is reserved for local debug/recovery
only; terminal G004 evidence still requires passing split-stat artifacts.

`configs/model/fdm_streaming_d2e_full_compact.yaml` enables per-epoch
validation checkpoints and a preregistered plateau rule:

- score: `composite_primary` over keyboard accuracy, mouse-button F1/accuracy,
  and mouse-move Pearson when present;
- plateau: `<1%` relative score improvement over `3` consecutive validation
  checkpoints;
- final full metrics/statistics are still computed on the full target stream.


## G004 completion audit

Before checkpointing `G004-d2e-only-fdm-4xh200` complete, run:

```bash
uv run python scripts/validate_g004_full_fdm_completion.py
```

During upstream G003/G004 execution this may be run with `--allow-fail`, but a
terminal G004 checkpoint requires `artifacts/fdm/g004_full_fdm_completion_audit.json`
to report `status == pass`. The audit checks G003/G004 goal state, D2E-only
FDM-from-IDM-pseudolabel provenance, split materialization counts, prediction
coverage, explicit train-core → target-all-eval split mode, causal FDM feature
mode, prior-action context provenance, target split tags, convergence-report
presence, split statistics, and 4×H200 run evidence.
The 4×H200 evidence includes both `nproc_per_node == 4` and GPU-monitor coverage
for all expected GPU indices.

## Claim boundary

G004 is not complete merely because this path runs. Completion still requires:

- full G003 IDM pseudo-label inputs;
- 4×H200 run logs, GPU-monitor coverage for all expected GPU indices, and checkpoint metadata;
- convergence/saturation evidence or a documented stricter preregistered rule;
- baseline/statistical comparisons with Holm-adjusted endpoint claims;
- failure analysis for non-rejected endpoints and unstable variants.
