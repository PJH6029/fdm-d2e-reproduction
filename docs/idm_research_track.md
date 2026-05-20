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

## H200 evidence snapshot

The current G4 evidence is real-D2E/H200-backed but still incomplete.  The
strongest useful artifacts are:

| Run | Artifact | Data | Trained IDM evidence | Interpretation |
| --- | --- | --- | --- | --- |
| Apex8 rich-motion | `artifacts/idm/g4_h200_idm_run_h200_richmotion.json` | 8 Apex recordings; 384 train / 128 heldout; H200 GPU | `mouse_move_pearson` delta `0.3041`, Holm-adjusted p `0.0495`; `mouse_move_scale_ratio_distance` delta `0.6275`, Holm-adjusted p `0.015`; keyboard/button not significant | Frame-pair grid/shift features recover meaningful inverse mouse dynamics on the smaller real split. |
| Apex16 rich-motion | `artifacts/idm/g4_h200_idm_run_h200_richmotion16.json` | 16 Apex recordings; 768 train / 256 heldout; H200 GPU | `keyboard_accuracy` delta `0.1008`, Holm-adjusted p `0.005`; mouse/button not significant | Scaling adds keyboard evidence but exposes a mouse generalization regression versus `last_seen_train`. |
| Apex16 categorical sweep | `artifacts/idm/idm_torch_apex16_sweep_h200.json` | Same Apex16 split | 30 categorical-weight/threshold variants; best rows preserve keyboard significance only | Categorical loss/threshold tuning alone does not recover mouse endpoints. |
| Apex16 capacity sweep | `artifacts/idm/idm_torch_apex16_capacity_sweep_h200.json` | Same Apex16 split | 64 depth/width variants; zero-categorical-loss linear heads show apparent mouse-button wins | Treat zero-loss categorical wins as invalid failure-analysis clues because the categorical head is untrained; capacity reduction does not solve mouse motion. |
| Apex36 rich-motion | `artifacts/idm/g4_h200_idm_run_h200_richmotion36b.json` | 36 Apex recordings; 1728 train / 576 heldout; H200 GPU | `keyboard_accuracy` delta `0.0704`, Holm-adjusted p `0.0045`; mouse Pearson raw p `0.0635` / Holm p `0.4445`; mouse button `0.0` accuracy | More clusters make the keyboard result robust and strengthen failure evidence: shared-head rich-motion MLP still does not clear mouse/click endpoints at scale. |
| Apex36 residual mouse head | `artifacts/idm/g4_h200_idm_run_apex36_residual.json` | Same Apex36 split | `keyboard_accuracy` delta `0.1108`, Holm p `0.0`; `mouse_move_scale_ratio_distance` delta `0.1720`, Holm p `0.0`; mouse Pearson raw p `0.0655` / Holm p `0.4585`; mouse button not significant | Predicting residual motion over the last-seen baseline is a real improvement for scale calibration, but it still does not solve direction/correlation or click recovery. |
| Shooter32 rich-motion | `artifacts/idm/g4_h200_idm_run_h200_shooter32.json` | 32 shooter/action recordings; 2304 train / 768 heldout; H200 GPU | `keyboard_accuracy` Holm p `0.0`; mouse button raw p did not survive Holm; mouse motion endpoints failed | Domain mixing/click-richer data helps keyboard robustness but does not by itself satisfy G4 mouse/click criteria. |
| Shooter32 residual mouse head | `artifacts/idm/g4_h200_idm_run_shooter32_residual.json` | Same Shooter32 split | `keyboard_accuracy` delta `0.1047`, Holm p `0.0`; mouse button Holm p `1.0`; mouse Pearson Holm p `1.0`; scale-ratio failed | Residual motion does not transfer the Apex36 scale-ratio improvement to mixed shooter/action data. |
| Shooter32 categorical sweep | `artifacts/idm/idm_torch_shooter32_sweep_h200.json` | Same Shooter32 split | 72 loss-weight / positive-cap / threshold variants; best mouse-button row reached `0.125` accuracy with raw p `0.0235` but Holm p `0.2115`; best mouse Pearson Holm p `1.0` | A trained categorical sweep improves the raw click signal but remains below the strong statistical bar; this rules out global threshold/pos-weight tuning as a sufficient G4 fix. |
| Shooter32 grid8-time group-calibrated sweep | `artifacts/idm/idm_torch_shooter32_grid8_time_group_calibrated_sweep_h200.json` | Same Shooter32 split | 36 grid8 + temporal-basis + focal-loss + group-exact calibration variants; best keyboard row `0.1030` accuracy with Holm p `0.0`; best mouse-button row `0.075` accuracy with Holm p `0.8685`; best mouse Pearson Holm p `1.0` | Higher spatial resolution, train-only exact group calibration, and temporal bin bases preserve keyboard significance but regress click evidence versus the simpler categorical sweep. |

Current conclusion: G4 has meaningful non-smoke IDM progress across real D2E
splits, including H200 checkpoint metadata and pseudo-label artifacts, but it
should remain `in_progress`.  Apex36 rules out "just add more Apex recordings"
as a sufficient fix for the mouse/click endpoints, while Shooter32 rules out a
simple "mix in more click-heavy shooter recordings plus tune a global
categorical threshold" fix. The grid8-time group-calibrated sweep further rules
out naive spatial-detail and periodic-bin features for click recovery. The next
credible completion attempt should move beyond independent frame-pair MLP heads:
for example a sequential IDM that conditions on recent action/state history and
reports button false-positive rates, or a predeclared trained portfolio that
combines a motion-specialized head with a click-specialized head and evaluates
all endpoints without untrained categorical logits or post-hoc heldout
thresholding.

## Completion caveat

This milestone proves the real-D2E neural IDM training/evaluation path on the decoded sample and already beats the `global_majority` movement baseline on sample mouse endpoints. It is not the final G4 completion proof: G4 still needs full selected-D2E training on MLXP storage, enough recording clusters for the strong statistical bar, and durable trained checkpoints before the ultragoal story should be checkpointed complete.
