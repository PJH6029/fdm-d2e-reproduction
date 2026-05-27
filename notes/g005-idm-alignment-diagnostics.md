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

## State Eventification Probe

Follow-up diagnostic: convert the trained state-token checkpoint predictions from held-state tokens (`KEY_DOWN_*`, `MOUSE_*_DOWN`) into D2E event tokens by recording-local differencing.

- Code: `scripts/convert_state_predictions_to_events.py`.
- Debounce 1 row:
  - Summary: `artifacts/idm/g005_idm_state_luma_pair_eventified_d1_prefix1m_summary.json`.
  - Metrics: `artifacts/idm/g005_idm_state_luma_pair_eventified_d1_prefix1m_paper_metrics.json`.
  - W&B: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/myvluxqq`.
  - 1M prefix paper-default metrics: keyboard `0.009490`, mouse-button `0.015058`, Pearson X `0.413933`, Pearson Y `0.050528`.
  - Strict mouse-button F1 `0.026942`, no-button FPR `0.061712`.
- Debounce 2 rows:
  - Summary: `artifacts/idm/g005_idm_state_luma_pair_eventified_d2_prefix1m_summary.json`.
  - Metrics: `artifacts/idm/g005_idm_state_luma_pair_eventified_d2_prefix1m_paper_metrics.json`.
  - 1M prefix paper-default metrics: keyboard `0.006343`, mouse-button `0.007142`, Pearson X `0.413933`, Pearson Y `0.050528`.
  - Strict mouse-button F1 `0.013184`, no-button FPR `0.023228`.

Conclusion: eventifying the held-state checkpoint fixes the token namespace enough to produce valid event tokens and good no-button FPR, but it does not recover keyboard or mouse-button paper metrics. The trained state checkpoint is therefore useful as mouse-X/FPR diagnostic evidence, not as the G005 paper-target IDM.

## Raw112 Offset-2 Long-Train Prefix Probe

Follow-up after the 2-epoch offset-2 raw-video prefix rejection: a 12-epoch 2GPU probe reused the already materialized two train-shard cache and `shard_30` target cache to distinguish undertraining from architecture failure.

- Evidence summary: `artifacts/idm/g005_idm_video_pair_raw112_offset2_keysoftmax_long12_prefix320000_train2shards_probe_chain_summary.json`.
- Rejection record: `artifacts/idm/g005_idm_video_pair_raw112_offset2_keysoftmax_long12_prefix320000_train2shards_rejection.json`.
- Rows: `227,580` train rows, `84,512` target rows.
- Train loss improved through epoch 12, but paper-compatible target metrics remained unusable: keyboard `0.0`, mouse-button `0.001680`, Pearson X `-0.006546`, Pearson Y `0.013523`, scale ratios X/Y `3.658/3.227`.
- Strict local no-button FPR stayed bounded (`0.047153`), but button F1 remained `0.003121`.

Conclusion: longer prefix training does not rescue the non-leaky raw112 offset-2 CNN candidate. Do not promote this candidate to a full 4xH200 G005 run; pivot to a stronger pretrained visual-action representation, exact released G-IDM diagnostics, or a new architecture before spending another full GPU reservation.
