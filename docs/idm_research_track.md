# IDM Research Track

G4 now has an executable neural-IDM path over decoded real D2E records.

## Current implementation

- `src/fdm_d2e/training/neural_idm.py` implements deterministic tiny neural IDM variants for mouse-delta prediction from decoded video-frame features.
- `scripts/train_idm_real.py --config configs/model/idm_real_sample.yaml` trains the configured variants on the decoded D2E sample split.
- Outputs under `outputs/idm_real/` include checkpoint JSON, checkpoint metadata, pseudo-label JSONL, filtered pseudo-label JSONL, and metrics per variant.
- `artifacts/idm/idm_real_sample_summary.json` is a source-control-safe summary with metrics and predeclared endpoint comparisons.

## Calibration/filtering

Each pseudo-label receives a confidence derived from the trained model residual scale. The config threshold writes both full and filtered pseudo-label artifacts. Later full-scale runs should tune this threshold on a validation split and record label-retention/quality curves.

## Completion caveat

This milestone proves the real-D2E neural IDM training/evaluation path on the decoded sample and already beats the `global_majority` movement baseline on sample mouse endpoints. It is not the final G4 completion proof: G4 still needs full selected-D2E training on MLXP storage, enough recording clusters for the strong statistical bar, and durable trained checkpoints before the ultragoal story should be checkpointed complete.
