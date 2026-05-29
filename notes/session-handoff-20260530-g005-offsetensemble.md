# Self handoff — G005 FDM-1-shaped IDM paper-target work

Timestamp: 2026-05-30 04:05 KST.

## Active objective / hard boundary

Continue renewed ultragoal execution with active story `G005-g014-idm-full-paper-target` in `.omx/ultragoal/goals.json`. **Do not checkpoint G005 complete**: no candidate currently beats D2E paper-reported Generalist IDM metrics. Do not call `update_goal` or `omx ultragoal checkpoint --status complete` for G005 unless a future full/paper-target evidence gate actually passes with Codex `get_goal` reconciliation.

Global mission remains: reproduce the public FDM-1 IDM/FDM recipe shape on D2E, not arbitrary methods. Use video/screen tokens + noncausal masked action-token diffusion / iterative unmasking for IDM; later FDM should use transformer-family interleaved frame/action data. No FDM-1 parity claim, no target/eval-label calibration.

## What happened in this session

1. Confirmed the quota increase is available from prior evidence (`production_effective_gpu_quota=8`, grant `grant-req-1-gpu-quota-increase-20260529-94abef`).
2. Finished and rejected the `candidate_reranker_predict5k` probe:
   - Evidence commit: `509a22f`.
   - Metrics: keyboard `0.01775568181818182`, mouse-button `0.004032258064516129`, mouse-button F1 `0.0`, mouse Pearson X/Y `null/null`, no-button FPR `0.017912291537986413`.
   - Reservation `rsv-jeonghunpark-20260530-0396ce` cancelled.
3. Implemented temporal source-offset candidate ensembling to test D2E NEP/timing mismatch while staying inside the FDM-1 masked action-token recipe:
   - Code commit: `ab62cc2`.
   - Probe launcher namespace fix/reservation evidence: `3bcf35d`.
4. Wider `offsetensemble_predict5k` probe on `rsv-jeonghunpark-20260530-39a28b` was aborted for throughput: >25 minutes, zero prediction rows. Abort marker copied locally:
   - `artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_offsetensemble_predict5k_throughput_abort.json`.
5. Prepared and ran the narrowed `offsetensemble_fast1k` probe:
   - Code commit: `91018f7`.
   - Pod: `prod-rsv-jeonghunpark-20260530-39a28b`, node4 GPU1, commit `91018f7`.
   - It completed before handoff; artifacts were copied locally.
   - Metrics: keyboard `0.016377171215880892`, mouse-button `0.0`, mouse-button F1 `0.0`, mouse Pearson X/Y `0.026625633715173874/null`, no-button FPR `0.011133603238866396`, rows `1000`.
   - It is negative; reservation `rsv-jeonghunpark-20260530-39a28b` was cancelled and GPU is idle.

## Local files to commit from this handoff turn

- `artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_offsetensemble_fast1k_summary.json`
- `artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_offsetensemble_fast1k_compact_summary.json`
- `artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_offsetensemble_fast1k_wandb_status.json`
- `artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_offsetensemble_fast1k.log`
- `artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_offsetensemble_fast1k_rejection.json`
- `artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_offsetensemble_predict5k_throughput_abort.json`
- `artifacts/cluster/g005_offsetensemble_fast1k_cancel_20260530.json`
- Forced-add ignored small output files:
  - `outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_train320k_offsetensemble_fast1k/paper_metrics.json`
  - `outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_train320k_offsetensemble_fast1k/resolved_config.json`
- This handoff note.

## Safe termination state

- No `offsetensemble_fast1k`, `predict_idm_temporal_masked_diffusion.py`, or `uv run` process was running at the last pod poll.
- `nvidia-smi` showed `0 MiB` GPU memory and 0% utilization after completion.
- Reservation `rsv-jeonghunpark-20260530-39a28b` cancel API response status is `cancelled`; local redacted evidence is `artifacts/cluster/g005_offsetensemble_fast1k_cancel_20260530.json`.
- If a later check shows the pod still exists, treat it as cancelled/terminating and do not launch more work there unless a new reservation is created.

## Next recommended G005 pivot

Reject the current train320k raw96 checkpoint family, candidate-score reranking, and temporal source-offset ensembling. They all keep no-button FPR low but fail keyboard/button/mouse metrics by huge margins. Next work should be a more fundamental recipe-shaped change, not another threshold sweep:

1. Revisit action/event target construction against D2E official semantics (especially paper-compatible keyboard/button denominators) and verify whether event-state full-corpus baseline’s `keyboard=0.2026`, `button=0.1777`, `mouse_x=0.7502` can be converted into a recipe-faithful masked-token/teacher path without leaking target labels.
2. Consider separated released G-IDM teacher infrastructure only after keeping it clearly labeled as teacher/baseline, not our G005 paper-target success.
3. If reserving GPUs again, start with a small bounded decision probe and cancel immediately if GPU is idle or terminal metrics remain negative.
