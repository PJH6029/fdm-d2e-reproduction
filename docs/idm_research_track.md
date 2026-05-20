# IDM Research Track

G4 now has an executable neural-IDM path over decoded real D2E records.

## Current implementation

- `src/fdm_d2e/training/neural_idm.py` implements deterministic tiny neural IDM variants for mouse-delta prediction from decoded video-frame features.
- `scripts/train_idm_real.py --config configs/model/idm_real_sample.yaml` trains the configured variants on the decoded D2E sample split.
- Outputs under `outputs/idm_real/` include checkpoint JSON, checkpoint metadata, pseudo-label JSONL, filtered pseudo-label JSONL, and metrics per variant.
- `artifacts/idm/idm_real_sample_summary.json` is a source-control-safe summary with metrics and predeclared endpoint comparisons.

## Calibration/filtering

Each pseudo-label receives a confidence derived from the trained model residual scale. The config threshold writes both full and filtered pseudo-label artifacts. Later full-scale runs should tune this threshold on a validation split and record label-retention/quality curves.

## Multi-recording preflight

`configs/data/d2e_real_multi_apex8.yaml` and `configs/model/idm_real_multi_apex8.yaml` run an eight-recording Apex Legends preflight (512 decoded windows; 384 train / 128 heldout; 8 heldout recording clusters). The summary artifacts are:

- `artifacts/sources/d2e_multi_decode_apex8_summary.json`
- `artifacts/eval/baseline_stat_eval_multi_apex8.json`
- `artifacts/idm/idm_real_multi_apex8_summary.json`

The best current neural IDM variants improve mean mouse endpoints over the `last_seen_train` movement baseline, but Holm-corrected significance is not yet clean. Treat this as the escalation signal for larger MLXP-scale extraction/training, not as final G4 completion.

## Torch/MLXP trainer path

`src/fdm_d2e/training/torch_idm.py` and `scripts/train_idm_torch.py` provide the H200-ready IDM trainer path. It consumes the same decoded JSONL contracts, trains a Torch MLP IDM, and writes `idm_checkpoint_metadata.v1`, pseudo-label JSONL, filtered pseudo-label JSONL, checkpoint, metrics, and a summary. Local runs without the train extra exit cleanly; MLXP runs should execute after `uv sync --frozen --extra d2e --extra test --extra train`:

```bash
uv run python scripts/train_idm_torch.py --config configs/model/idm_torch_apex8.yaml --require-torch
```

## Completion caveat

This milestone proves the real-D2E neural IDM training/evaluation path on the decoded sample and already beats the `global_majority` movement baseline on sample mouse endpoints. It is not the final G4 completion proof: G4 still needs full selected-D2E training on MLXP storage, enough recording clusters for the strong statistical bar, and durable trained checkpoints before the ultragoal story should be checkpointed complete.
