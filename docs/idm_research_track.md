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
| Shooter32 sequential-history sweep | `artifacts/idm/idm_torch_shooter32_seq_history_sweep_h200.json` | Same Shooter32 split | 36 autoregressive action-history variants; best keyboard row `0.1130` accuracy with Holm p `0.0`; best mouse-button row `0.1000` accuracy with raw p `0.006` but Holm p `0.054`; mouse Pearson/scale endpoints failed | Causal action history nearly clears the mouse-button correction gate without heldout leakage, but autoregressive feedback hurts mouse-motion correlation and still misses the predeclared strong bar. |
| Shooter32 sequential-history focused FP audit | `artifacts/idm/idm_torch_shooter32_seq_history_focused_fp_h200.json` | Best seq-history click variant under expanded button metrics | Mouse-button positive accuracy `0.1000`, precision `0.0074`, F1 `0.0138`, `540` predicted button examples, no-button false-positive rate `0.6964` | The near-significant click result is not harness-safe: it mostly comes from button spam. Future click work must optimize precision/false-positive rate, not only positive-example accuracy. |
| Shooter32 sequential-history F-beta focused audit | `artifacts/idm/idm_torch_shooter32_seq_history_fbeta_focused_h200.json` | Same focused variant with train-only group F-beta calibration (`beta=0.5`) | Mouse-button positive accuracy `0.0`, precision `0.0`, `21` predicted button examples, no-button false-positive rate `0.0247`; keyboard still non-significant in this focused run | Precision-aware calibration suppresses click spam but becomes too conservative, confirming that a better click model/objective is needed rather than threshold-only repair. |
| Shooter32 softmax button-head sweep | `artifacts/idm/idm_torch_shooter32_seq_button_softmax_sweep_h200.json` | Same Shooter32 split; exact-set mouse-button softmax head with explicit no-button class and train-tail F-beta threshold calibration | 24 H200 variants; 4 variants reject both `keyboard_accuracy` and `mouse_button_accuracy` after Holm correction. Best precision row: mouse-button accuracy `0.125`, precision `0.2273`, F1 `0.1613`, no-button false-positive rate `0.0192`, mouse-button Holm p `0.0405`; mouse motion still fails (`mouse_move_pearson` Holm p `1.0`, scale-ratio Holm p `1.0`) | Learning no-button as a class fixes the click-spam failure mode and clears the click correction gate for several variants, but it does not solve mouse motion. G4 remains incomplete until a single predeclared IDM/prediction artifact also clears or explicitly composes motion evidence without heldout leakage. |
| Shooter32 residual softmax button-head sweep | `artifacts/idm/idm_torch_shooter32_seq_button_softmax_residual_sweep_h200.json` | Same Shooter32 split; softmax button head plus residual mouse targets and causal autoregressive heldout residual baselines | 24 H200 variants; no button/motion endpoint rejects. Best Pearson row: `mouse_move_pearson` `0.2432`, raw p `0.134` / Holm p `1.0`; mouse-button accuracy `0.05`, precision `0.0741`, no-button false-positive rate `0.0330`; keyboard remains significant | Causal residual feedback improves raw mouse correlation versus the absolute softmax sweep but does not clear the strong statistical bar and regresses click recovery. This is useful failure evidence for a future predeclared portfolio or stronger motion head, not a completion artifact. |
| Shooter32 softmax-click + residual-motion portfolio diagnostic | `artifacts/idm/idm_shooter32_softmax_click_residual_motion_portfolio_h200.json` | One composed heldout artifact from predeclared group sources: keyboard/click tokens from the best softmax click-head row and mouse movement from the best residual Pearson row | Keyboard accuracy `0.1096`; mouse-button accuracy `0.125`, precision `0.2273`, no-button false-positive rate `0.0192`; mouse Pearson `0.2432`, scale ratio `1.2573`; Holm rejects only `keyboard_accuracy` and `mouse_button_accuracy` | The composer proves specialist token streams can be evaluated as a single reproducible artifact with source fingerprints. The selected residual motion source still fails motion significance, so the portfolio diagnostic remains non-terminal G005 evidence. |
| Shooter32 motion-only residual sweep | `artifacts/idm/idm_torch_shooter32_motion_only_residual_sweep_h200.json` | Same residual/autoregressive setup with categorical and button losses set to zero; width/depth motion-specialist grid | 9 H200 variants; no motion endpoint rejects. Best Pearson row reaches `0.1790` with scale ratio `1.1224`, below the residual+softmax raw Pearson (`0.2432`) and non-significant | Removing categorical/button loss does not improve Shooter32 motion; shared supervision was not the main motion bottleneck. |

Current conclusion: G4 has meaningful non-smoke IDM progress across real D2E
splits, including H200 checkpoint metadata and pseudo-label artifacts, but it
should remain `in_progress`.  Apex36 rules out "just add more Apex recordings"
as a sufficient fix for the mouse/click endpoints, while Shooter32 rules out a
simple "mix in more click-heavy shooter recordings plus tune a global
categorical threshold" fix. The grid8-time group-calibrated sweep further rules
out naive spatial-detail and periodic-bin features for click recovery. The
sequential-history sweep is the strongest non-leaky positive-click attempt so
far under independent button logits (Holm p `0.054`) but still fails the
predeclared correction gate, regresses mouse motion, and has an unacceptable
no-button false-positive rate (`0.6964`) on the focused audit. Precision-aware
F-beta calibration reduces the false-positive rate to `0.0247` but loses all
true click positives. The exact-set softmax button head is the first trained
click objective to clear Holm correction while keeping no-button false positives
low. The residual+softmax sweep improves raw mouse Pearson but still does not
reject motion endpoints and weakens click recovery, so the next credible
completion attempt should design a stronger motion head before recomposing the
portfolio. The portfolio composer now preserves source fingerprints and
evaluates a single heldout artifact, but the first softmax-click/residual-motion
diagnostic still rejects only keyboard and mouse-button endpoints. A
motion-only residual loss sweep also failed, suggesting the next motion attempt
needs better temporal/visual representation or more targeted target modeling,
not merely reweighting away categorical losses. In every case, the evaluation
must use one predeclared heldout prediction artifact without untrained
categorical logits or post-hoc heldout thresholding.

## Completion caveat

This milestone proves the real-D2E neural IDM training/evaluation path on the decoded sample and already beats the `global_majority` movement baseline on sample mouse endpoints. It is not the final G4 completion proof: G4 still needs full selected-D2E training on MLXP storage, enough recording clusters for the strong statistical bar, and durable trained checkpoints before the ultragoal story should be checkpointed complete.
