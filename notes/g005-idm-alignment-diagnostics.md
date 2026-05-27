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

## Released G-IDM Chunked Exact-Split Infrastructure

Date: 2026-05-27 KST.

After rejecting the raw112 offset-2 long-train prefix candidate, the next non-leaky path is to make released Generalist-IDM exact-split inference usable as a baseline/teacher diagnostic before spending another full 4xH200 run.

Implemented local chunked inference support for the generated D2E `inference_desktop_minimal.py` wrapper:

- Adds `--start-time` and `--timestamp-offset` patching so each video chunk is cut with ffmpeg but stamped back into the original recording timeline.
- Adds chunk planning from manifest `bin_index_min/max` and `timestamp_min_ns` so temporal heldout rows no longer require replaying the entire recording prefix.
- Adds chunked manifest rows with `prediction_mcap_paths` and `prediction_timestamps_aligned_to_ground_truth=true`.
- Extends conversion/target extraction to accept multiple MCAP chunks per recording and skip first-screen auto-shift when chunk timestamps are already ground-truth aligned.

Validation evidence:

- `uv run pytest -q tests/test_gidm_adapter.py tests/test_gidm_baseline_contract.py tests/test_g005_idm_paper_target.py` => 22 passed.
- `python3 -m py_compile` over the modified G-IDM runner/adapter/pipeline scripts passed.
- Dry-run artifact: `artifacts/eval/g006_gidm_chunked_dry_run_summary.json` plans one heldout recording as 59 five-second chunks and verifies chunk-manifest generation without committing the large generated manifest.

Claim boundary: this is G-IDM baseline/teacher infrastructure, not G005 paper-target success and not our-IDM metric evidence.

### Chunked G-IDM bounded pilot limiter

Added `--max-chunks` to the released G-IDM manifest runner and exact-split pipeline so GPU pilots can validate one or a few timestamp-aligned chunks without scheduling every chunk from a heldout recording. Dry-run evidence: `artifacts/eval/g006_gidm_chunked_maxchunks_dry_run_summary.json` selected one recording and exactly three chunks with `--max-recordings 2 --max-chunks 3`.

This is a utilization guardrail: use it for smoke/pilot validation only, not for G005 completion or exact-split metric claims.

## Chunked released G-IDM 2GPU pilot

Date: 2026-05-27 KST.

Reservation `rsv-jeonghunpark-20260527-a89102` on production node 4 GPUs `[1,2]` ran a bounded released Generalist-IDM chunked pilot and was cancelled after evidence collection to avoid idle GPU time.

Evidence:

- First live attempt failed before model execution because the production Codex image lacked system `ffmpeg`: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_max2chunks_wrapper_summary.json`.
- First fallback retry still failed because the generated command left literal `ffmpeg` as argv[0]: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_retry_ffmpegfallback_wrapper_summary.json`.
- Executable-fallback retry succeeded on two 5-second chunks: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_retry2_execfallback_wrapper_summary.json` with `completed_chunks=2`, `failed_chunks=0`.
- W&B sidecar run: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_retry2_execfallback_wandb_sidecar_status.json`.
- Pilot conversion/metrics over the scoped recording target rows: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_retry2_paper_metrics.json` (`rows_seen=5855`). Metrics are intentionally not a success claim because only the first two chunks have predictions; most bins are empty.
- Hash manifest: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_retry2_artifact_summary.json`.

Claim boundary: this proves the chunked released G-IDM execution path can produce timestamp-aligned MCAP chunks and local JSONL conversion. It does not satisfy G005 paper-target win, full exact-split G-IDM baseline, or our-IDM training evidence.

### Target-timed chunk scheduling fix

The successful two-chunk G-IDM pilot used decode-summary count manifest rows with missing `bin_index_min/max`, so chunks were scheduled from video time `0s` while the Apex temporal target rows begin near bin `23420` / video time `1171s`. This made the pilot execution valid infrastructure evidence but its metrics mostly empty-bin diagnostics.

Added target timing enrichment for G-IDM manifests from either by-recording JSONL roots or an extracted target-record JSONL. Local pilot artifact `artifacts/eval/g006_gidm_chunked_pilot_2gpu_retry2_timing_enrichment_summary.json` enriches `d2e_480p:Apex_Legends/0805_01` to `bin_index_min=23420`, `timestamp_min_ns=1173417098600`; dry-run `artifacts/eval/g006_gidm_chunked_pilot_2gpu_retry2_timed_chunk_dry_run_summary.json` now schedules the first chunk at `start_time_seconds=1171.0` and `timestamp_offset_seconds=1173.4170986`.

Next live pilot should rerun chunked G-IDM on this timing-enriched manifest; previous two MCAP chunks remain useful only as executable-path evidence.

### Timing-correct chunked G-IDM pilot and covered-window metrics

Date: 2026-05-28 KST.

After the local target-timing enrichment, a fresh 2GPU production pilot used reservation `rsv-jeonghunpark-20260527-ec840b` on node 4 GPUs `[1,2]` and was cancelled after evidence collection. The pod checkout was reset to `fedd123` and used the timing-enriched Apex temporal target row (`bin_index_min=23420`, `timestamp_min_ns=1173417098600`).

Code/evidence hardening before the run:

- `write_chunked_gidm_manifest` now records explicit `prediction_mcap_chunks` with `start_time_seconds`, `timestamp_offset_seconds`, and `[timestamp_start_ns, timestamp_end_ns_exclusive)`.
- Target extraction and MCAP conversion now support `filter_to_prediction_windows` / `filter_targets_to_prediction_windows` so bounded pilots evaluate only rows covered by predicted chunks instead of silently treating unpredicted target rows as empty predictions.
- Dry-run evidence: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_timed_window_dry_run_summary.json` and `artifacts/eval/g006_gidm_chunked_pilot_2gpu_timed_window_manifest.json` plan chunks at `start000001171000` and `start000001176000` with timestamp offsets `1173.4170986` and `1178.4170986`.

Live timed pilot evidence:

- Wrapper: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_timed_retry_wrapper_summary.json` — `status=pass`, `completed_chunks=2`, `failed_chunks=0`.
- Inference summary: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_timed_retry_inference_summary.json` — both chunks succeeded; elapsed seconds were about 60.4s and 64.6s.
- Chunk outputs: `outputs/gidm_exact_split/predicted_mcap/d2e_480p_Apex_Legends_0805_01_chunks/d2e_480p_Apex_Legends_0805_01/chunk_0000_start000001171000_dur000005000.mcap` and `chunk_0001_start000001176000_dur000005000.mcap`.
- Target-window extraction/conversion: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_timed_retry_target_extraction_summary.json` and `artifacts/eval/g006_gidm_chunked_pilot_2gpu_timed_retry_conversion_summary.json` — 200 covered target rows and 200 prediction rows.
- Metrics: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_timed_retry_paper_metrics.json` — `rows_seen=200`, status pass; metrics remain poor (`keyboard=0`, `button=0`, mouse Pearson X/Y `-0.0515/0.0886`) and are diagnostic only.
- W&B sidecar: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_timed_retry_wandb_sidecar_status.json` (`0mt0e772`). The sidecar counted older pilot MCAP files in the same predicted directory, so use the scoped artifact summary rather than sidecar final_mcap_count for chunk-completion claims.
- Scoped hash summary: `artifacts/eval/g006_gidm_chunked_pilot_2gpu_timed_retry_artifact_summary.json` — `status=pass`, `completed_chunk_count=2`, `planned_chunk_count=2`, `target_rows=200`, `prediction_rows=200`, `metric_rows=200`.

Claim boundary: this proves timing-correct released G-IDM chunk scheduling, execution, target-window filtering, conversion, and W&B logging. It is not an exact-split baseline result, does not show our IDM beats paper metrics, and does not complete G005/G006.

Follow-up hardening in the same session patched the W&B sidecar to count only manifest-planned chunk paths for future runs, because this run shared a predicted directory with older pilot MCAPs. Next step: proceed to a longer exact-split G-IDM shard once a 4GPU quota/reservation is available; in parallel, keep G005 IDM architecture work focused on beating paper targets rather than promoting these poor released-GIDM pilot metrics.

### Current G005 candidate revalidation after timed G-IDM pilot

Date: 2026-05-28 KST.

Re-ran current G005 paper-target validators and summarized existing candidate metrics in `artifacts/idm/g005_idm_candidate_revalidation_summary.json`. Status remains `fail_no_candidate_meets_paper_targets`.

Top current full-corpus diagnostic candidates:

- `event_state_duration_context_full`: strongest overall paper metrics so far (keyboard `0.2026`, mouse-button `0.1777`, Pearson X/Y `0.7502/0.6970`, strict button F1 `0.2936`, no-button FPR `0.0480`) but still misses D2E paper targets and is rejected because event-state context lacks closed-loop predicted-context evidence.
- `transition_hazard_full`: strict F1/FPR remain useful (`0.2726`, `0.0526`) but paper keyboard/mouse targets are far below target and it has the same closed-loop context audit issue.
- `video_stack_luma96_full` and raw112 variants remain non-leaky visual baselines with bounded FPR but near-zero paper keyboard/button and poor mouse motion; do not promote alone.

Recommended branch: repair the event-state-duration context candidate into a non-leaky closed-loop predicted-context model, then combine it with endpoint-specialist heads/calibration for keyboard and mouse-button. Use prefix/small-shard gates before spending another full 4xH200 run. The released G-IDM timed pilot is now infrastructure for future exact-split baseline/teacher diagnostics, not a G005 success path by itself.

### Planned closed-loop/dropout repair branch

Implemented a non-GPU prep branch for the next G005 prefix gate:

- `scripts/materialize_state_context_dropout_train.py` creates deterministic train-core shards where a configurable fraction of prior state/action context is replaced by `NOOP`/empty duration fields. This targets the train/eval distribution shift that made the event-state-duration context checkpoint collapse under closed-loop prediction.
- `configs/model/idm_streaming_d2e_full_event_state_duration_context_dropout035_closed_loop_prefix320k.yaml` trains a 320k-row prefix candidate on 35% context-dropout train shards and evaluates target rows with `closed_loop_state_context=true`, `seed_from_train=false`, `seed_from_target_prior=true`, and `prediction_workers=1`.
- `scripts/run_g005_idm_event_state_duration_context_dropout035_closed_loop_prefix.sh` materializes bounded dropout shards, trains the prefix candidate, and builds paper-compatible prefix metrics.

Verification: `uv run pytest -q tests/test_state_context_dropout_materializer.py tests/test_streaming_idm_closed_loop_state_context.py tests/test_training_run_scripts.py` => 22 passed.

This branch is a prefix gate only. Promote to a full 4xH200 run only if it materially improves closed-loop keyboard/button/mouse metrics over `g005_idm_event_state_duration_context_closed_loop_prefix320k` while preserving no-button FPR.

### Context-dropout closed-loop prefix rejection

Date: 2026-05-28 KST.

Ran the planned 2GPU prefix gate for deterministic 35% train-context dropout plus closed-loop target inference on reservation `rsv-jeonghunpark-20260528-92815d` (production node 4 GPUs `[1,2]`, cancelled immediately after evidence collection). W&B sidecar run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/1aphyal5`.

Evidence:

- `artifacts/idm/g005_idm_event_state_duration_context_dropout035_materialization_summary.json` — 64 train shards, `1,280,000` rows, dropout fraction `0.350139`.
- `artifacts/idm/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_artifact_summary.json` — status `pass`, 320k predictions/pseudolabels, 2GPU monitor rows with max GPU utilization 100%.
- `artifacts/idm/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_paper_metrics.json` — status `pass`, alignment rows `320,000`, but metrics are a rejection.
- `artifacts/idm/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_rejection.json` — explicit negative decision.

Result: reject this branch. Paper-compatible metrics remained far below targets: keyboard `0.006398`, mouse-button `0.008884`, mouse Pearson X/Y `0.0214/0.0089`, scale ratios X/Y `18.0/44.7`. Strict mouse-button F1 was only `0.01549`, and no-button FPR exploded to `0.7219` overall (`~1.0` on heldout_game and heldout_recording). This does not improve over the previous closed-loop prefix and must not be promoted to a full G005 run.

Next hypothesis: closed-loop evaluation may be corrupting the state tracker by consuming shard-glob target rows out of per-recording chronological order. Before spending another training run, retest the stronger full event-state-duration checkpoint on the same prefix row set sorted by `(recording_id, timestamp_ns, sequence_id)` with `closed_loop_state_context=true`. If that still fails, pivot to endpoint-specialist/mixture heads with conservative calibration rather than more global context dropout.


### Chronological closed-loop prefix rejection

Date: 2026-05-28 KST.

Ran the chronological closed-loop prefix retest on production reservation `rsv-jeonghunpark-20260528-927122` (node 4 GPU `[1]`; follow-on extension `rsv-jeonghunpark-20260528-47a5e5` was cancelled unused after evidence collection). The pod checkout was `15588db`; local reproducibility fixes were later committed in `8619a53` and the follow-up config/logging fix commit. W&B artifact run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/gifrgefk`.

Evidence:

- `artifacts/idm/g005_idm_event_state_duration_context_chrono_prefix320k_materialization_summary.json` — materialized 320k target rows with zero per-recording timestamp violations.
- `artifacts/idm/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_artifact_summary.json` — status `pass`; predictions and pseudolabels both have 320k rows with hashes recorded on DDN.
- `artifacts/idm/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_paper_metrics.json` — status `pass`, alignment rows `320,000`.
- `artifacts/idm/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_rejection.json` — explicit negative decision.

Result: reject this branch. Chronological materialization reproduced the same closed-loop prefix metrics, so target row ordering is not the main failure mode. No-button FPR is now bounded (`0.0165` overall), but only because the model is underactive: paper keyboard `0.000606`, mouse-button `0.00419`, Pearson X/Y `0.0145/-0.00144`, strict button F1 `0.00728`. This misses every paper target and must not be promoted to a full G005 run.

Next branch: endpoint-specialist/mixture heads plus conservative heldout calibration. Prefix gates must require no-button FPR `<=0.10` and meaningful improvements in keyboard/button/mouse metrics before any 4xH200 full run.
