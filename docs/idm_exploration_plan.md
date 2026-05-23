# IDM Exploration Plan

G004 converts the failure audit into a bounded IDM exploration lane for G005.

## Selected Full-Corpus Candidates

1. `configs/model/idm_streaming_d2e_full_surface_calibrated.yaml`
   - Surface/grid compact features with a larger MLP.
   - Streaming group F-beta threshold calibration for keyboard and mouse-button logits.
   - Train-set mouse absolute-ratio gain calibration for movement scale.
   - Greedy row-balanced tensor-cache sharding for 4xH200 DDP.

2. `configs/model/idm_streaming_d2e_full_luma_temporal_calibrated.yaml`
   - Luma temporal convolution over `summary_luma16_stack5_time`.
   - Same calibrated categorical and mouse-scale decisions.
   - Used as the backup if the surface/grid MLP plateaus.

## Promotion Gate

G005 may promote a candidate only after full-corpus training/evaluation beats the G003 paper target contract and reports split-level statistics. G004 small/medium H200 evidence is directional only; it is not a full-D2E success claim.

## GPU Policy

No GPU reservation is held for G004 after local code/artifact verification. G005 should reserve 4xH200 only when ready to run one of the selected full-corpus configs.
