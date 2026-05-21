# Reproducibility Runbook

> Current full-corpus ultragoal is not complete. The Shooter64 commands below
> are historical bounded evidence; use `docs/d2e_full_idm_pipeline.md` and
> `docs/d2e_full_fdm_pipeline.md` for the active full-corpus path.

## Local setup

Prefer `uv`.

```bash
uv sync --frozen --extra d2e --extra test --extra train
uv run pytest -q
```

## Real D2E Shooter64 data

```bash
uv run python scripts/extract_d2e_real_multi.py \
  --config configs/data/d2e_real_multi_shooter64.yaml \
  --summary-copy artifacts/sources/d2e_multi_decode_shooter64_summary.json
```

Expected primary split: `outputs/data/real_multi_shooter64` with `4,608` train and `1,536` within-recording temporal heldout windows. `scripts/prepare_d2e_real.py` only writes manifest-level files; the multi-recording extraction script above is required for the train/heldout JSONL used by IDM/FDM configs.

## IDM training and pseudo-labeling

```bash
uv run python scripts/train_idm_torch.py --config configs/model/idm_torch_shooter64_surface_motion.yaml --require-torch
uv run python scripts/predict_idm_torch.py --config configs/model/idm_predict_shooter64_train_button_recall.yaml
```

Selected checkpoint evidence is under `artifacts/idm/shooter64_surface_motion_selected/`.

## FDM training and calibration

```bash
uv run python scripts/train_fdm_real.py --config configs/model/fdm_shooter64_surface_motion_fulltrain_bth05.yaml
uv run python scripts/predict_idm_torch.py --config configs/model/fdm_bth05_predict_train_for_scale.yaml
uv run python scripts/calibrate_fdm_predictions.py --config configs/model/fdm_bth05_d2e_train_prediction_scale_calibrated.yaml  # strict 3/4 endpoint variant
uv run python scripts/calibrate_fdm_predictions.py --config configs/model/fdm_bth05_d2e_train_scale_calibrated.yaml  # endpoint-winning transductive no-label variant
```

Selected evidence is under `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/`.

## Ablation/scaling summary

```bash
uv run python scripts/summarize_ablation_scaling.py \
  --output-json artifacts/ablation_scaling/g007_ablation_scaling_summary.json \
  --output-md docs/ablation_scaling.md
```

## Harness gate

```bash
uv run python scripts/run_game_harness_eval.py --config configs/harness/g008_game_harness.yaml
```


## G009 completion audit

Before checkpointing `G009-report-repro-package` complete, run:

```bash
uv run python scripts/audit_claim_boundaries.py --output artifacts/reproducibility/claim_boundary_audit.json
uv run python scripts/build_repro_package_manifest.py --output artifacts/reproducibility/package_manifest.json
uv run python scripts/validate_g009_completion.py
```

Prefer the fail-closed finalizer for terminal and handoff runs:

```bash
uv run python scripts/finalize_g009_report_package.py
```

It refreshes the claim-boundary audit, final-quality audit, package manifest,
and G009 completion audit, then writes
`artifacts/reproducibility/g009_finalization_summary.json`. It does not mutate
OMX state or checkpoint G009.

During upstream G003-G008 execution the last command may be run with
`--allow-fail`, but a terminal G009 checkpoint requires
`artifacts/reproducibility/g009_completion_audit.json` to report `status == pass`.
The audit checks prerequisite goal state, final report/evidence/runbook docs,
claim-boundary audit status, package-manifest coverage and hashes, and final
quality audit presence.

## Package manifest

```bash
uv run python scripts/build_repro_package_manifest.py --output artifacts/reproducibility/package_manifest.json
```

## MLXP/PVC path

Cluster execution uses the PVC repo path:

```bash
cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
git pull --ff-only origin main
uv sync --frozen --extra d2e --extra test --extra train
```

The Docker/cluster launcher path is documented in `docs/cluster_runbook.md`.
