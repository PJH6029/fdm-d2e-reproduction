# D2E-based FDM-1 Recipe Reproduction Report

> Current full-corpus ultragoal is not complete. This is historical bounded
> Shooter64 evidence from the earlier lane and must not be treated as the final
> G001–G009 full-corpus report.

## Executive summary

This repository now contains a serious, non-smoke D2E reproduction pipeline for the **recipe shape** of FDM-1: real D2E decoding, neural IDM training, IDM pseudo-label generation, FDM training, baseline/statistical evaluation, ablation/scaling summaries, and bounded game-harness replay. This is **not** a parity claim for the closed-source FDM-1 system and does not claim internet-scale data, non-game domain transfer, robotics, or car-control transfer.

The strongest selected FDM branch, `fdm_bth05_d2e_train_scale_calibrated`, trains the FDM action model from IDM pseudo-labels and then applies D2E train-split mouse-scale targets with no heldout labels. Its endpoint-winning variant normalizes by the heldout prediction distribution, so it is explicitly a transductive no-label scale-normalization result. On the real Shooter64 within-recording temporal heldout windows it beats the predeclared non-oracle baselines on all four primary FDM endpoints after Holm correction when using the endpoint-winning no-label transductive prediction-distribution scale normalization described below.

## Dataset and split

- Source dataset: public D2E (`open-world-agents/D2E-480p`), decoded through `scripts/extract_d2e_real_multi.py`.
- Primary G4/G5 split: Shooter64 game/action subset, `64` recordings, `6,144` binned windows.
- Train/heldout split: `4,608` train windows and `1,536` within-recording temporal heldout windows. This is not heldout-recording or novel-game generalization.
- Evidence artifact: `artifacts/sources/d2e_multi_decode_shooter64_summary.json`.

## Evaluation contract

Primary endpoints are defined in `configs/eval/primary_endpoints.yaml` and evaluated with recording-cluster bootstrap plus Holm-Bonferroni correction:

1. `keyboard_accuracy` vs `noop`.
2. `mouse_button_accuracy` vs `noop`.
3. `mouse_move_pearson` vs `last_seen_train`.
4. `mouse_move_scale_ratio_distance` vs `last_seen_train` with absolute log-distance transform.

## IDM result

The selected Shooter64 IDM checkpoint (`artifacts/idm/shooter64_surface_motion_selected/`) clears three corrected endpoints and provides the teacher for FDM pseudo-labels:

| Endpoint | Delta | Holm p | Result |
| --- | ---: | ---: | --- |
| `keyboard_accuracy` | `0.1183` | `0.0` | pass |
| `mouse_button_accuracy` | `0.2069` | `0.0035` | pass |
| `mouse_move_pearson` | `0.1636` | `0.0` | pass |
| `mouse_move_scale_ratio_distance` | `0.1170` | `0.4675` | fail / limitation |

This establishes meaningful real-D2E IDM label quality while preserving scale calibration as a downstream FDM/G6 issue.

## FDM result

Selected artifact: `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/summary.json`.

Heldout metrics:

| Metric | Value |
| --- | ---: |
| Keyboard accuracy | `0.110119` |
| Mouse-button accuracy | `0.220588` |
| Mouse-button precision | `0.348837` |
| No-button false-positive rate | `0.018392` |
| Mouse-move Pearson | `0.175390` |
| Mouse scale ratio | `1.089469` |

Corrected endpoint comparison:

| Endpoint | Reference | Delta | Holm p | Result |
| --- | --- | ---: | ---: | --- |
| `keyboard_accuracy` | `noop` | `0.085627` | `0.0` | pass |
| `mouse_button_accuracy` | `noop` | `0.133030` | `0.025` | pass |
| `mouse_move_pearson` | `last_seen_train` | `0.235562` | `0.0` | pass |
| `mouse_move_scale_ratio_distance` | `last_seen_train` | `0.324230` | `0.0` | pass |

Important caveat: the selected FDM scale repair uses **D2E train-split ground-truth motion scale** and the **heldout prediction distribution denominator** (`calibration_uses_target_ground_truth=false`, `calibration_uses_target_prediction_distribution=true`). It should be reported as a transductive train-labeled/no-heldout-label calibration variant, not as pure IDM-pseudo-label scale success or novel-recording scale generalization. A stricter train-prediction denominator variant is preserved as `artifacts/fdm/fdm_bth05_d2e_train_prediction_scale_calibrated_h200/summary.json`; it clears keyboard/click/direction but not scale after Holm.

## Ablation and scaling

`artifacts/ablation_scaling/g007_ablation_scaling_summary.json` and `docs/ablation_scaling.md` summarize three axes:

- FDM branch/calibration axis: `6` points from raw bth07 FDM through strict train-prediction and endpoint-winning target-prediction scale calibration.
- FDM sweep axes: `84` runs across neural button weighting, neural decode/recall, and KNN retrieval.
- IDM data/model scaling axis: Apex8/Apex16/Apex36/Shooter32/Shooter64 H200 evidence.

Key findings:

- Raw bth05 FDM clears keyboard, click, and motion direction but not scale.
- Pure IDM-pseudo recording-scale calibration still misses the scale endpoint.
- KNN retrieval can clear keyboard/click for some variants but has weak motion correlation and zero scale rejections.
- D2E train-split scale targets plus transductive target-prediction normalization are the only selected branch that clears all FDM primary endpoints; the strict train-prediction denominator variant remains 3/4.

## Harness execution

`docs/live_open_game_suite.md` and
`artifacts/harness/g008_repo_live_suite/live_suite_evidence.json` provide the
live open-source graphical-game gate:

- Three repo-local open-source Tk graphical mini-games/tasks are exercised
  through live X11/xdotool keyboard/mouse input.
- Five seeds per task produce video/replay, latency, failure, and action logs.
- The runtime performs trained D2E-only FDM checkpoint forward passes and the
  task-safe visual adapter maps model output into bounded live actions.
- The suite reports a statistically significant improvement over the scripted
  baseline (`adjusted_p_value=3.0517578125e-05` in the G008 evidence
  validation artifact).

This proves stable closed-loop execution in the repository's open-source live
graphical harness suite. It does not prove commercial-game control.

## Reproducibility assets

- Final package manifest: `artifacts/reproducibility/package_manifest.json`.
- PVC-resident large-artifact manifest:
  `artifacts/reproducibility/external_artifact_manifest.json`.
- Source/resource validation: `docs/source_validation.md`, `docs/mlxp_resource_plan.md`.
- Real data ingestion: `docs/d2e_real_ingestion.md`.
- Baselines/statistics: `docs/baselines_statistics.md`.
- IDM research track: `docs/idm_research_track.md`.
- FDM research track: `docs/fdm_research_track.md`.
- Ablation/scaling: `docs/ablation_scaling.md`.
- Harness: `docs/harness_selection_and_execution.md`.
- Runbook: `docs/reproducibility_runbook.md`.
- Failure analysis: `docs/failure_analysis.md`.

## Final limitations

1. No FDM-1 parity claim: this is a scaled recipe reproduction over a much smaller public dataset.
2. The final all-endpoint FDM win depends on train-split D2E scale calibration plus transductive heldout prediction-distribution normalization; the strict train-prediction denominator variant remains 3/4 endpoints.
3. Harness evidence is live for repo-local open-source graphical mini-games,
   not commercial-game play.
4. D2E-derived artifacts remain under upstream non-commercial constraints.
