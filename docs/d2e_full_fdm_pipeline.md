# D2E Full-Corpus FDM Pipeline

This is the G004 execution path prepared while the G003 full-corpus IDM run is
still producing IDM pseudo-labels. It is a D2E-only FDM lane and does not make
an FDM-1 parity claim.

## Inputs

- D2E records: `outputs/data/d2e_full_corpus/target_all_eval.jsonl`
- IDM pseudo-labels: `outputs/idm_streaming_d2e_full_compact/pseudolabels.jsonl`
- FDM config: `configs/model/fdm_streaming_d2e_full_compact.yaml`
- Endpoints: `configs/eval/primary_endpoints.yaml`

The pseudo-label and record JSONLs are order-joined by `sequence_id`. This is
intentional: the G003 IDM predictor writes pseudo-labels while streaming the
same target record file, so G004 can build FDM train/eval files with O(1)
memory. A sequence mismatch fails the run instead of silently mixing artifacts.

## Training/evaluation split

`src/fdm_d2e/training/streaming_fdm.py` materializes:

- `fdm_train_pseudolabeled_records.jsonl`: D2E records whose
  `ground_truth_tokens` are replaced by IDM-generated tokens.
- `fdm_target_ground_truth_records.jsonl`: recording-tail heldout records that
  retain real D2E ground-truth tokens for evaluation.
- `fdm_streaming_split_summary.json`: counts, input hashes, and split
  fingerprint.

Default split is per-recording temporal tail:

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
`torchrun`, and writes:

- `outputs/fdm_streaming_d2e_full_compact/checkpoint_metadata.json`
- `outputs/fdm_streaming_d2e_full_compact/summary.json`
- `outputs/fdm_streaming_d2e_full_compact/torch_model/checkpoint.pt`
- `outputs/fdm_streaming_d2e_full_compact/torch_model/predictions.jsonl`
- `outputs/fdm_streaming_d2e_full_compact/torch_model/metrics.json`
- `outputs/fdm_streaming_d2e_full_compact/torch_model/statistical_comparison.json`
- `artifacts/fdm/fdm_streaming_d2e_full_compact_summary.json`
- `artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json`

## Claim boundary

G004 is not complete merely because this path runs. Completion still requires:

- full G003 IDM pseudo-label inputs;
- 4×H200 run logs and checkpoint metadata;
- convergence/saturation evidence or a documented stricter preregistered rule;
- baseline/statistical comparisons with Holm-adjusted endpoint claims;
- failure analysis for non-rejected endpoints and unstable variants.
