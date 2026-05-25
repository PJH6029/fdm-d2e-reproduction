# G005 IDM Alignment Diagnostics

Date: 2026-05-25 KST.

Purpose: determine whether failed full-corpus IDM candidates are mostly caused by row/timestamp alignment drift before launching another 4xH200 training run.

## Evidence

- Diagnostic code: `scripts/diagnose_idm_alignment_shifts.py`.
- Unit tests: `tests/test_idm_alignment_shifts.py`.
- Video-stack 1M-row probe:
  - Artifact: `artifacts/idm/g005_idm_video_stack_luma96_offsets012_keysoftmax_alignment_shift_probe_1m.json`.
  - W&B: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/15l3sid5`.
  - Rows: `1,000,000`; sequence mismatches: `0`; recording fragments: `59`.
  - Best shift for keyboard, mouse-button, Pearson X, and Pearson Y is `0`.
  - Shift-0 paper-default metrics: keyboard `0.012195`, mouse-button `0.027992`, Pearson X `0.027102`, Pearson Y `0.022137`.
- State-token 1M-row probe:
  - Artifact: `artifacts/idm/g005_idm_state_luma_pair_alignment_shift_probe_1m.json`.
  - W&B: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/2g53neln`.
  - Rows: `1,000,000`; sequence mismatches: `0`; recording fragments: `59`.
  - Best shift for mouse-button, Pearson X, and Pearson Y is `0`.
  - Shift-0 paper-default metrics: keyboard `0.0`, mouse-button `0.018863`, Pearson X `0.413933`, Pearson Y `0.050528`.
  - Strict no-button FPR remains high at about `0.2545` across shifts.

## Interpretation

The failed candidates are not primarily explained by a simple row offset. For both video-stack and state-token branches, shift `0` is the best or tied-best setting, and non-zero shifts do not produce a hidden paper-target win.

The state-token branch has a separate action-namespace failure: sampled predictions emit key-state tokens such as `KEY_DOWN_87`, while D2E paper-default target rows use event tokens such as `KEY_PRESS_87` and `KEY_RELEASE_87`. This explains why paper-default keyboard accuracy is `0.0` in the alignment probe even though an `empty_bins_as_correct=true` report can look less bad.

## Next Recipe Constraint

Do not spend another full 4xH200 run on state-token labels unless the model has a dedicated D2E event-token head for keyboard press/release and mouse button events. For video/frame recipes, prioritize stronger visual representation or teacher labeling over row-shift correction.
