# Experiment Log

## Smoke run

The smoke run should record:

- command sequence
- config files
- data manifest path/hash
- pseudo-label artifact path/hash
- FDM checkpoint metadata path/hash
- metrics JSON path/hash
- rollout smoke path/hash
- known gaps and skipped categories

Initial implementation uses deterministic synthetic D2E-shaped fixtures for local reproducibility; real D2E sample paths can replace the fixture after source-contract validation.

## G5 FDM Shooter64 H200 track

- Trained neural FDM variants from selected Shooter64 IDM pseudo-labels on the real D2E train split (`4,608` pseudo-labeled windows) and evaluated on `1,536` heldout D2E windows.
- Button/recall sweeps and KNN retrieval sweeps are preserved under `artifacts/fdm/` as failure/ablation evidence.
- Selected artifact: `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/summary.json`.
- Command: `uv run python scripts/calibrate_fdm_predictions.py --config configs/model/fdm_bth05_d2e_train_scale_calibrated.yaml` after the bth05 FDM training run.
- Result: all four predeclared primary endpoints reject after Holm correction: keyboard p `0.0`, mouse-button p `0.025`, mouse Pearson p `0.0`, mouse scale-ratio-distance p `0.0`.
- Caveat: scale is fixed by train-split D2E ground-truth calibration (`calibration_uses_target_ground_truth=false`), not by pure IDM pseudo-labels alone.
