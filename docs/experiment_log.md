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

## G6 ablation/scaling summary

- Generated `artifacts/ablation_scaling/g007_ablation_scaling_summary.json` and `docs/ablation_scaling.md` from source-controlled H200 artifacts.
- Axes summarized: FDM branch/calibration (`5` points), FDM sweeps (`84` total runs across neural button-weight, neural decode/recall, and KNN retrieval axes), and IDM data/model scaling (`5` points from Apex8 through Shooter64).
- Quality gate: pass; selected FDM branch rejects all four primary endpoints, at least two ablation/scaling axes are present, and total sweep coverage is non-smoke.
- Verification: `uv run python -m py_compile scripts/summarize_ablation_scaling.py`; JSON contract assertions for status/axes/run count; `uv run pytest -q` (`55 passed`).

## G7 harness selection/execution

- Implemented deterministic repo-local game/game-adjacent harnesses in `src/fdm_d2e/rollout/game_harness.py` and CLI `scripts/run_game_harness_eval.py`.
- Candidate catalog contains five game-like environments; all five passed install/control probes.
- Replayed the trained `fdm_bth05_d2e_train_scale_calibrated` prediction stream from `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/predictions.jsonl`.
- Quality gate passed: `3/3` tasks, `3/2` required environments, `5/3` install/control probes.
- Verification: `uv run python -m py_compile src/fdm_d2e/rollout/game_harness.py scripts/run_game_harness_eval.py`; `uv run pytest -q` (`58 passed`).
