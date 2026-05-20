# Shooter64 Surface-Motion IDM Selected Checkpoint

This directory is the selected G005 IDM handoff artifact from the H200 Shooter64 sweep.

- Source dataset summary: `artifacts/sources/d2e_multi_decode_shooter64_summary.json`
- Sweep artifact: `artifacts/idm/idm_torch_shooter64_surface_motion_sweep_h200.json`
- Selected variant: `lw0.5_pc20_th0.35_h128_d2_lossfocal_fg2_calgroup_exact_calibrated_cb1_cf0.2_bhsoftmax_bth0.5_btmfbeta_calibrated_blw0.25_bcw4_bnw1.5_bpw1_mhaxis_softmax_malw0.5_mrlw0.25_macw4`
- Training config family: `configs/model/idm_torch_shooter64_surface_motion.yaml`
- Data config: `configs/data/d2e_real_multi_shooter64.yaml`

Included files are copied from the MLXP PVC path for reproducibility:

- `checkpoint.pt` — trained Torch checkpoint.
- `checkpoint_metadata.json` — dataset fingerprint, model/config metadata, calibration diagnostics, and source paths.
- `predictions.jsonl` — heldout prediction tokens used for evaluation.
- `pseudolabels.jsonl` and `pseudolabels.filtered.jsonl` — IDM pseudo-label outputs for downstream FDM training.
- `metrics.json` and `statistical_comparison.json` — primary metric and Holm-corrected baseline comparisons.
- `summary.json` — train summary tying the artifacts together.

Selected corrected endpoints versus baselines:

- `keyboard_accuracy`: delta `0.1183`, Holm p `0.0`.
- `mouse_button_accuracy`: delta `0.2069`, Holm p `0.0035`; precision `0.2679`; no-button FPR `0.0266`.
- `mouse_move_pearson`: delta `0.1636`, Holm p `0.0`; Pearson `0.3040`.
- Known limitation: `mouse_move_scale_ratio_distance` delta `0.1170`, Holm p `0.4675` (not significant).

This is a G005 IDM research-track handoff, not an FDM-1 parity claim.
