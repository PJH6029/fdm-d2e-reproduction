# FDM Research Track

G5 now has a real-D2E, H200-backed FDM path that trains from IDM pseudo-labels, evaluates on heldout D2E windows, and clears all four predeclared FDM primary endpoints after Holm correction for the selected calibrated branch.

## Data and training signal

- Data split: `outputs/data/real_multi_shooter64` from 64 shooter/action D2E recordings (`4,608` train windows, `1,536` heldout windows).
- IDM teacher: selected Shooter64 surface-motion IDM handoff from G4, plus the recall-oriented `button_softmax_threshold_override=0.5` pseudo-label generation path.
- FDM training: `configs/model/fdm_shooter64_surface_motion_fulltrain_bth05.yaml` trains the Torch FDM on IDM-generated train-split pseudo-labels and evaluates against real D2E heldout labels.
- Scale calibration branch: `configs/model/fdm_bth05_d2e_train_scale_calibrated.yaml` preserves the trained bth05 FDM predictions, then rescales mouse-motion tokens using **train-split D2E ground-truth motion scale only**. It does not consume heldout/target labels (`calibration_uses_target_ground_truth=false`) and keeps endpoint reference baselines tied to the pseudo-labeled FDM training stream.

## Selected G5 artifact

| Field | Value |
| --- | --- |
| Model | `fdm_bth05_d2e_train_scale_calibrated` |
| Local artifact | `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/summary.json` |
| Remote output | `outputs/fdm_bth05_d2e_train_scale_calibrated/summary.json` |
| Source predictions | `outputs/fdm_shooter64_surface_motion_fulltrain_bth05/torch_model/predictions.jsonl` |
| Calibration records | `outputs/data/real_multi_shooter64/train.jsonl` |
| Baseline train records | `outputs/fdm_shooter64_surface_motion_fulltrain_bth05/fdm_train_pseudolabeled_records.jsonl` |
| Heldout target | `outputs/fdm_shooter64_surface_motion_fulltrain_bth05/fdm_target_ground_truth_records.jsonl` |
| Checkpoint metadata SHA-256 | `38fa66dd65276ad3d818a6a6de38f86bcd3db440a8bd9a034bbc11f478df65fd` |
| Predictions SHA-256 | `012844afe3c4c8eebe50c8f809247815f7056c8f0b763c5148257229e37d587a` |
| Summary SHA-256 | `0d6ebc68a2fe8fdb98fd95bf159b1eeb166015be41fd73e7dc1db228099dbd1b` |

## Primary endpoint results

Heldout metrics for the selected branch:

- Keyboard accuracy: `0.110119` over `672` keyboard-positive examples.
- Mouse-button accuracy: `0.220588` over `68` button-positive examples; precision `0.348837`; no-button false-positive rate `0.018392`.
- Mouse-move Pearson: `0.175390` over `2,810` axis values.
- Mouse scale ratio: `1.089469`.

Holm-corrected statistical comparison (`configs/eval/primary_endpoints.yaml`):

| Endpoint | Reference | Delta | Holm p | Reject 0.05 |
| --- | --- | ---: | ---: | --- |
| `keyboard_accuracy` | `noop` | `0.085627` | `0.0` | yes |
| `mouse_button_accuracy` | `noop` | `0.133030` | `0.025` | yes |
| `mouse_move_pearson` | `last_seen_train` | `0.235562` | `0.0` | yes |
| `mouse_move_scale_ratio_distance` | `last_seen_train` | `0.324230` | `0.0` | yes |

## Ablations and failure analysis feeding the selected branch

- Neural FDM button/recall sweeps (`artifacts/fdm/fdm_shooter64_fulltrain_button_sweep_h200.json`, `artifacts/fdm/fdm_shooter64_recall_beta_sweep_h200.json`) consistently improved keyboard and mouse-direction signal, and several variants improved click precision, but the raw neural FDM scale endpoint did not survive Holm correction.
- KNN/retrieval FDM sweep (`artifacts/fdm/fdm_knn_shooter64_surface_sweep_h200.json`) showed that nonparametric visual retrieval can clear keyboard+button for some settings, but it has weak motion correlation and does not solve scale.
- Pure IDM-pseudo-label scale calibration (`configs/model/fdm_bth05_recording_scale_calibrated.yaml`) improved direction but still missed scale significance (`mouse_move_scale_ratio_distance` Holm p `1.0`).
- D2E train-split scale calibration is therefore the smallest non-heldout intervention that fixes the identified scale failure while leaving keyboard/click/direction predictions from the trained FDM branch intact.

## Reproduction commands

On the MLXP PVC repo path after `git pull` and `uv sync --frozen --extra d2e --extra test --extra train`:

```bash
uv run python scripts/train_fdm_real.py --config configs/model/fdm_shooter64_surface_motion_fulltrain_bth05.yaml
uv run python scripts/calibrate_fdm_predictions.py --config configs/model/fdm_bth05_d2e_train_scale_calibrated.yaml
```

Local verification for the code/config contract:

```bash
uv run python -m py_compile src/fdm_d2e/training/calibrated_fdm.py scripts/calibrate_fdm_predictions.py
uv run pytest -q
```

## Caveats

This is not an FDM-1 parity claim. The selected branch uses real D2E train-split labels for post-hoc mouse-scale calibration, so report it as a train-labeled calibration variant rather than a pure IDM-pseudo-label FDM. Heldout labels are not used in training or calibration. G6 should treat pure-pseudo scale calibration, KNN retrieval, and train-labeled calibration as separate ablation axes.
