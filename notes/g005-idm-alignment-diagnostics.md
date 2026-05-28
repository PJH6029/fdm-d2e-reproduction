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


### Endpoint-mixture prefix matrix rejection

Date: 2026-05-28 KST.

Used `production-storage-shell-4` (no new GPU reservation) to run a CPU-only endpoint-mixture prefix matrix from existing full-corpus prediction JSONLs. The committed runner/config at `fea9099` combines `event_state_duration_context` mouse/keyboard with several state-luma/event-context mouse-button policies over the first 320k target rows.

Evidence:

- `artifacts/idm/g005_idm_endpoint_mixture_matrix_summary.json` — `status=rejected_no_policy_meets_paper_targets`; best button policy and best FPR-gated policy are both `event_all`.
- `artifacts/idm/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k_paper_metrics.json` — the proposed state-luma button gated by event-context detector keeps FPR low (`0.00860`) but button accuracy collapses to `0.00890`.
- `artifacts/idm/g005_idm_endpoint_mixture_matrix_event_all_paper_metrics.json` — best prefix policy among the matrix: keyboard `0.1990`, mouse-button `0.1726`, Pearson X/Y `0.6048/0.5986`, strict button F1 `0.2809`, no-button FPR `0.0375`.
- `artifacts/idm/g005_idm_endpoint_mixture_matrix_rejection.json` — explicit negative decision.

Result: reject post-hoc endpoint recombination. State-luma button predictions either increase no-button FPR (`0.1447` when used directly) or lose button accuracy when gated/intersected. Event-context alone remains best under the FPR gate and still misses all paper targets by large margins.

Next branch must be a learned endpoint-specialist architecture rather than token-level recombination: NEP/future-offset visual context for keyboard/buttons, exact-count or exact-set heads, and heldout-calibrated thresholds. Prefix gates must beat the `event_all` prefix baseline while keeping no-button FPR `<=0.10`.

### 2026-05-28 KST — NEP-style future luma-window prefix rejected

Ran `g005-luma-window-nep100-prefix320k` on production reservation
`rsv-jeonghunpark-20260528-081f41` (node 4, 1×H200, cancelled after evidence
collection). Code checkout: `9eb3094`. W&B sidecar run:
`https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/b1gnvqbr`.

This branch materialized a nonleaky 320k/320k prefix with compact luma offsets
`[0, 2, 4, 6, 8]` on top of event-state-duration context and trained a
`luma_temporal_conv` IDM. The materializer was intentionally split-independent
(train rows cannot borrow target/eval frames from the same recording).

Evidence:

- `artifacts/idm/g005_idm_event_state_duration_luma_window_nep100_prefix320k_materialization_summary.json` — 320k train + 320k target rows.
- `outputs/idm_streaming_d2e_full_event_state_duration_luma_window_nep100_prefix320k/{checkpoint_metadata.json,metrics.json,train_history.json,convergence_report.json}` — prefix training metadata copied locally; raw checkpoint/predictions remain on DDN.
- `artifacts/idm/g005_idm_event_state_duration_luma_window_nep100_prefix320k_paper_metrics.json` — paper-compatible prefix metrics with 320k aligned rows and no sequence mismatches.
- `artifacts/idm/g005_idm_event_state_duration_luma_window_nep100_prefix320k_rejection.json` — explicit rejection decision.

Metrics: keyboard `0.0190`, mouse-button `0.0307`, Pearson X/Y
`0.5804/0.5546`, scale ratio X/Y `1.5370/1.2588`, strict button F1 `0.0492`,
overall no-button FPR `0.0764`. Heldout-game no-button FPR is `0.1073`, above
the <=0.10 hard gate. This is much worse than the event-context prefix matrix on
key/button endpoints, so naive NEP-style compact luma windows are not promoted.

Operational finding: the JSON-expanded luma-window prefix produced 14–15GB JSONL
files and averaged only `0.184%` GPU utilization during the 1GPU run. Future
visual/NEP probes should not reserve GPUs for JSON materialization/stats; use
CPU/storage-shell precompute, tensor caches, or the existing video-cache path.

Next: target a learned endpoint-specialist/calibration branch over the stronger
event-context/event_all signal instead of adding global visual context. Prefix
gates must beat `event_all` simultaneously on keyboard, button, and mouse Pearson
while keeping no-button FPR <=0.10 before any full-corpus run.

### 2026-05-28 KST — event-context global threshold sweep rejected

Ran `g005_event_context_threshold_sweep_prefix320k` on production reservation
`rsv-jeonghunpark-20260528-236ade` (node 4, 1×H200, cancelled after evidence
collection). Code checkout: `eca4a5d`. The storage-shell attempt failed first
because `production-storage-shell-4` has a 2Gi memory limit and was OOMKilled;
the successful run used a memory-rich production pod. No W&B run was created for
this prediction-only sweep.

The sweep re-decoded the existing full event-state-duration checkpoint over the
first 320k target rows with global key/button thresholds `{0.05, 0.10}`. It was
a calibration-only test intended to falsify whether the strong event-context
checkpoint is merely under-decoded.

Evidence:

- `artifacts/idm/g005_idm_event_context_threshold_sweep_prefix320k_summary.json` — 4 completed threshold combos.
- `artifacts/idm/g005_idm_event_context_threshold_sweep_prefix320k/*_paper_metrics.json` — paper-compatible metrics for every combo.
- `artifacts/idm/g005_idm_event_context_threshold_sweep_prefix320k_rejection.json` — explicit rejection decision.

Result: reject threshold-only decoding. The best combo was
`key0p05_button0p1`, with keyboard `0.00494`, mouse-button `0.01285`, Pearson
X/Y `0.8002/0.6427`, strict button F1 `0.00044`, and no-button FPR `0.8884`.
This is far worse than the calibrated `event_all` prefix and misses the FPR gate
by a large margin. Lowering button thresholds recovers neither exact buttons nor
safe FPR; it mostly overfires no-button rows.

Operational finding: repeated JSON prediction averaged only `0.194%` GPU
utilization. Future decode sweeps should export logits once or run in a
memory-rich CPU context; do not spend H200 time on repeated JSON re-prediction.

Next: threshold-only and post-hoc recombination paths are exhausted. The next
G005 candidate must change the learned endpoint heads/training objective over
event-context features, e.g. exact-set/hierarchical transition heads with
heldout-calibrated priors, and must beat `event_all` on a prefix before any full
run.


### 2026-05-28 KST — hierarchical exact-set prefix rejected

Ran `g005-hierarchical-exactset-prefix320k` on production reservation
`rsv-jeonghunpark-20260528-0002ff` (node 4, 1×H200, cancelled after evidence
collection). Code checkout: `deb1ec0`. W&B sidecar run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/v4gxczt0`.

This branch replaced independent multilabel decoding for keyboard/mouse-button
with learned event/no-event + positive exact-set heads over the event-state-duration
features. The intended hypothesis was to preserve the low no-button false-positive
rate while improving exact key/button sets.

Evidence:

- `artifacts/idm/g005_idm_event_state_duration_hierarchical_prefix320k_train_materialization_summary.json` and
  `artifacts/idm/g005_idm_event_state_duration_hierarchical_prefix320k_target_materialization_summary.json` — 320k/320k
  chronological prefix materialization; target rows cover temporal,
  heldout-recording, and heldout-game tags.
- `outputs/idm_streaming_d2e_full_event_state_duration_hierarchical_prefix320k/{checkpoint_metadata.json,metrics.json,train_history.json,convergence_report.json}`
  — local reproducibility metadata; raw checkpoint/predictions remain on DDN/PVC.
- `artifacts/idm/g005_idm_event_state_duration_hierarchical_prefix320k_paper_metrics.json` — paper-compatible metrics over
  320k aligned target rows with zero missing/mismatched sequence IDs.
- `artifacts/idm/g005_idm_event_state_duration_hierarchical_prefix320k_rejection.json` — explicit negative decision.

Result: reject this branch. It keeps no-button FPR under the 0.10 gate on every
heldout tag (overall `0.0444`; heldout-game
`0.0794`; heldout-recording
`0.0522`; temporal
`0.0230`), but paper-compatible
keyboard `0.0825`, mouse-button
`0.1476`, Pearson X/Y
`0.0912/0.0402`, and strict button F1
`0.2499` all fail the paper-target gate. It is
also worse than the `event_all` prefix baseline on keyboard, button, and motion
(Pearson X/Y deltas `-0.5136` /
`-0.5584`).

Operational finding: the 1×H200 run completed in under one reservation hour, but
GPU sampling shows bursty/low utilization (mean `1.00%`,
max `91.0%`) because JSON cache/prediction phases
dominate this 320k prefix. Do not scale this exact architecture to a full G005 run.

Next: abandon pure exact-set event heads as the main improvement route. The next
candidate should preserve the strong event-context motion signal and add a
stronger high-recall categorical/action prior or sequence model for key/button
sets, with prefix acceptance requiring improvement over `event_all` on all major
endpoints before any 4×H200 full run.

### 2026-05-28 KST — state-delta / repeat-prior diagnostic

Ran CPU-only prefix heuristics on `production-storage-shell-4` using the same
320k chronological target prefix. These are diagnostic upper-bound/recipe probes,
not valid completion candidates, because `next_state_delta_*` uses the next row's
held-state metadata to infer the current key/button transition.

Evidence:

- `artifacts/idm/g005_idm_prefix_context_heuristic_matrix.json` — previous-event
  and prior-state baseline policies.
- `artifacts/idm/g005_idm_state_delta_oracle_prefix320k_metrics.json` — noncausal
  next-state delta oracle.
- `artifacts/idm/g005_idm_key_repeat_prior_prefix320k_metrics.json` — train-prefix
  held-key repeat priors layered on the next-state delta and previous-motion signal.

Findings:

- Previous motion alone gives Pearson X/Y `0.7687/0.7425` with near-perfect scale
  ratio (`1.0002/1.0000`), so the motion gap is now primarily a short autoregressive
  motion-continuation problem, not a visual-feature problem on this prefix.
- Next-state delta gives mouse-button `0.9754`, strict button F1 `0.9933`, and
  no-button FPR `0.0`, beating the paper mouse-button target as an oracle. This
  means button failures are largely state-transition modeling/decoding failures.
- The same next-state delta reaches only keyboard `0.4590`; adding a train-prefix
  held-key repeat prior improves best keyboard to `0.5426`, still below the paper
  keyboard target `0.73`. Keyboard repeat/timing remains the hard blocker.

Next branch: train or decode a keyboard-repeat specialist over held-state duration,
previous key repeats, and frame/next-frame context, while reusing previous-motion
continuation and state-transition button decoding as separate heads. Do not treat
state-delta oracle metrics as claimable because they use future held-state metadata.


### 2026-05-28 KST — causal keyboard repeat/release prior rejected

Ran a CPU-only causal held-key duration policy matrix on `production-storage-shell-4`
over the same 320k train/target prefix. Unlike the state-delta upper bound, this
matrix predicts only from causal metadata available before the current action bin:
held key code, held duration, time since last key transition, and previous key
release tokens.

Evidence:

- `artifacts/idm/g005_idm_causal_keyboard_repeat_policy_matrix.json` — full policy
  matrix over global/code-specific hold-duration contexts and thresholds.
- `src/fdm_d2e/eval/state_transition_diagnostics.py` and
  `scripts/build_g005_state_transition_diagnostics.py` — reproducible builder now
  includes this causal matrix.
- `tests/test_state_transition_diagnostics.py` — regression coverage for the
  causal matrix plus state-delta/repeat diagnostics.

Result: reject simple causal duration priors. The best policy is `code_hold_mod_pressrelease_th0.3` with
keyboard `0.1978` over `88535` paper-compatible key samples. That is
below the `event_all` prefix keyboard (`0.1990`) and far below the paper target
`0.73`, while the noncausal next-state/repeat upper bound reached only `0.5426`.
This rules out a purely tabular held-duration repeat/release fix.

Next: keyboard likely requires a learned sequence/action-state model that predicts
future key state or current event repeats from richer temporal context, not just
held duration. GPU should still be gated by a prefix run that beats `event_all` on
keyboard, button, and motion simultaneously.


### Event-state-duration sequence-prior prefix rejection

Date: 2026-05-28 KST.

Ran a 1GPU prefix gate on reservation `rsv-jeonghunpark-20260528-873c77` (production node 4 GPU `[1]`, cancelled immediately after evidence collection). The pod checkout was `8b6402e`; W&B sidecar status is `artifacts/idm/g005_idm_event_state_duration_sequence_prior_prefix320k_wandb_sidecar_status.json`.

Evidence:

- `artifacts/idm/g005_idm_event_state_duration_sequence_prior_prefix320k_run_summary.json` — status `pass`, 320k target predictions, reservation expiration recorded.
- `artifacts/idm/g005_idm_event_state_duration_sequence_prior_prefix320k_paper_metrics.json` — status `pass`, alignment rows `320,000`, zero sequence mismatches.
- `artifacts/idm/g005_idm_event_state_duration_sequence_prior_prefix320k_rejection.json` — explicit negative decision.

Result: reject this branch. Paper-compatible all-row metrics were keyboard `0.009123`, mouse-button `0.041381`, Pearson X/Y `0.6426/0.5972`, scale ratios X/Y `1.124/1.127`. Strict button F1 was `0.06498`. No-button FPR was acceptable overall (`0.0668`) and across heldout_game/heldout_recording/temporal (`0.0897/0.0660/0.0491`), but the model badly underperforms the current endpoint-mixture/event-context baseline and misses every paper target. Do not promote to a full 4xH200 run.

Operational note: the run spent roughly 25 minutes in CPU stats/cache construction before GPU training. Future action-history candidates should precompute stats/cache in a CPU/storage shell before reserving H200s.

Next branch: change the supervision/modeling problem rather than adding another global temporal wrapper. Prioritize teacher-assisted event decoding or causal per-recording latent state estimation with prefix gates, and keep exact-split released G-IDM infrastructure separate until the paper-target objective is met.

### 2026-05-28 KST — NEP/offset target-autocorrelation diagnostic rejects simple label shift

Ran a CPU/storage-shell diagnostic on `production-storage-shell-4` over the same
320k `d2e_event_state_duration_hierarchical_prefix320k` target rows using a clean
`de4eab3` worktree. This checks ground-truth target autocorrelation at row shifts
`-4..4` (50ms bins) to test whether D2E's NEP-τ=100ms semantics could be rescued
by a simple target/label offset before spending more H200 time.

Evidence:

- `artifacts/idm/g005_idm_nep_offset_target_autocorr_prefix320k.json` — compact
  summary with paper-target comparison and baseline comparison.
- `artifacts/idm/g005_idm_nep_offset_target_autocorr_prefix320k_raw.json` — raw
  alignment-shift diagnostics.
- `scripts/build_g005_nep_offset_diagnostics.py` and
  `src/fdm_d2e/reporting/g005_nep_offset_diagnostics.py` — reproducible builder.
- `tests/test_g005_nep_offset_diagnostics.py` — regression coverage for expected
  shift extraction and claim boundary.

Result: reject a simple NEP/label-offset GPU branch. The expected +2 row shift
(+100ms) has keyboard `0.17448`, mouse-button `0.02147`, Pearson X/Y
`0.41486/0.38618`, and strict button F1 `0.0300`; it misses every paper target
except scale ratio. No nonzero shift meets all paper targets. The best nonzero
motion shift is `-1` with Pearson X/Y `0.7687/0.7424`, still below the D2E paper
`0.796/0.783` targets, while the best key/button shifts remain very low
(keyboard `0.19085`, mouse-button `0.05744`).

Interpretation: D2E NEP-style timing is not just a row-index mismatch in our
prefix target rows. Continue with a branch that changes the supervision/modeling
problem (teacher-assisted event decoding, causal per-recording latent state, or a
learned future-key-state/event decoder) and prefix-gate it before any full 4xH200
promotion. Do not claim target-autocorrelation metrics as trained-model evidence.

### 2026-05-28 KST — prepared frozen frame-embedding prefix branch

Implemented a new prefix-gated representation branch after rejecting simple
label-offset/action-history variants. This branch materializes JSONL rows with
`__streaming_idm_features` built from frozen per-frame embeddings plus the
existing compact state-duration/event-context features, then reuses the streaming
IDM trainer in MLP mode. It is intended to test whether stronger pretrained
visual state representation helps keyboard/button decoding before any full-corpus
4×H200 promotion.

New artifacts/scripts:

- `src/fdm_d2e/data/frame_embedding_materializer.py`
- `scripts/materialize_frame_embedding_features.py`
- `configs/model/idm_streaming_d2e_full_frozen_frame_embedding_prefix320k.yaml`
- `configs/eval/g005_idm_frozen_frame_embedding_prefix320k_paper_metrics.yaml`
- `scripts/run_g005_idm_frozen_frame_embedding_prefix.sh`

Default real backend is `hf-vision` with `facebook/dinov2-small`, offsets `0,2`,
CLS pooling, normalized embeddings, embedding deltas, and state-duration summary
features. A `dummy-stat` backend provides deterministic dependency-light tests
and verifies that streaming stats honor the generated
`__streaming_idm_features` override.

Claim boundary: this is infrastructure/preparation only. It is not G005 success
evidence unless a downstream prefix IDM run beats the paper-target metrics under
`empty_bins_as_correct=false`.

Storage/CPU sanity gate:

- `artifacts/idm/g005_idm_frozen_frame_embedding_dummy_compact_luma_target_sample1k_materialization_summary.json`
  reports `status=pass` on 1,000 real D2E target-prefix rows from
  `production-storage-shell-4`.
- The gate used `dummy-stat`, `frame_source=compact-luma`, offsets `0,2`,
  `feature_dim=1004`, and `missing_frames=0`; wall clock was about 9 seconds.
- This validates row/provenance preservation plus streaming feature override
  materialization on real D2E rows without requiring an H200 reservation or
  `ffmpeg` on the storage shell. It is not trained-model evidence.
- `artifacts/idm/g005_idm_frozen_frame_embedding_hf_tiny_compact_luma_target_sample16_materialization_summary.json`
  additionally validates the `hf-vision` backend on 16 real D2E rows with
  `hf-internal-testing/tiny-random-vit`, `hf_preprocess=manual-imagenet`,
  `frame_source=compact-luma`, `feature_dim=1064`, and `missing_frames=0`.
  This only proves the HF/manual-preprocess code path; it is not a candidate
  model result and does not replace a real DINO/ViT prefix gate.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_small_compact_luma_target_sample16_materialization_summary.json`
  validates the intended default frozen model `facebook/dinov2-small` on 16 real
  D2E rows with `embedding_dim_per_frame=384`, `feature_dim=2120`, and
  `missing_frames=0`. CPU wall clock was about 281 seconds for only 16 rows, so
  any prefix-scale DINO materialization should run on a reserved GPU with
  utilization monitoring rather than on the storage shell.

### 2026-05-28 KST — debug GPU DINO materialization gate

Used the active debug reservation `rsv-jeonghunpark-20260527-1ea75c`
(`p-debug`, pod `debug-rsv-jeonghunpark-20260527-1ea75c`, expires
2026-05-28 11:00 KST) to validate the new `dinov2-torchhub` backend on a
1,024-row real D2E target sample copied from the production storage shell. No
production reservation was created.

Evidence:

- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_opt_compact_luma_target_sample1024_materialization_summary.json`
  — `status=pass`, `rows_written=1024`, `feature_dim=2120`, elapsed `45.75s`.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_opt_compact_luma_target_sample1024_gpu_monitor.csv`
  — debug H200 monitor evidence for the optimized batch-128 run.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_opt_compact_luma_target_sample1024_b512_materialization_summary.json`
  — batch-512 rerun, `status=pass`, elapsed `46.30s`.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_opt_compact_luma_target_sample1024_b512_gpu_monitor.csv`
  — debug H200 monitor evidence for batch 512.

Finding: torchhub DINO now runs on the GPU pod without transformers/torchvision,
but the current JSONL materializer is still too low-throughput and too low-util
for prefix/full scale. Batch 512 did not improve wall-clock versus batch 128.
Do not launch full 320k train+target materialization with this monolithic path.
Next implementation should shard/parallelize extraction and/or write tensor-cache
features directly before any larger H200 reservation.

### 2026-05-28 KST — sharded debug GPU DINO materialization

Added contiguous `--skip-rows` support and ran two 512-row `dinov2-torchhub`
compact-luma shards concurrently over the same 1,024-row real D2E sample on the
active debug reservation `rsv-jeonghunpark-20260527-1ea75c` (`p-debug`, pod
`debug-rsv-jeonghunpark-20260527-1ea75c`).

Evidence:

- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_sharded1024_final_run_summary.json`
  — `status=pass`, `rows_written=1024`, `shard_count=2`, shard elapsed seconds
  `26.28` and `26.70`.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_sharded1024_shard0_summary.json`
  — `rows_written=512`, `source_rows_scanned=512`, `source_rows_skipped=0`.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_sharded1024_shard1_summary.json`
  — `rows_written=512`, `source_rows_scanned=1024`, `source_rows_skipped=512`.
- Per-shard progress/log artifacts and
  `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_sharded1024_late_gpu_monitor.csv`
  were copied locally for auditability.

Finding: contiguous sharding works and improves 1,024-row wall-clock throughput
versus the prior monolithic ~45.7s run, but the monitor evidence is late/weak
because the ad-hoc shell launcher used command substitution for background PIDs
and did not wait/monitor correctly from before launch. Treat this as sharded
materialization correctness/throughput evidence, not as strong H200 utilization
proof.

Next: add a proper shard launcher that starts `nvidia-smi` before launching
shards, stores child PIDs without command substitution, waits fail-closed, and
summarizes shard plus monitor evidence before any larger prefix/full extraction.
Claim boundary: materialization throughput diagnostic only; no trained IDM metric
evidence and no G005 success claim.

### 2026-05-28 KST — fail-closed frame-embedding shard launcher

Implemented a reusable shard launcher for the frozen frame-embedding branch so
future DINO prefix extraction does not depend on ad-hoc shell backgrounding.

Implementation:

- `scripts/run_frame_embedding_shards.py` builds contiguous row shard plans,
  starts `nvidia-smi` before materializer subprocesses, assigns optional
  `CUDA_VISIBLE_DEVICES` per shard, waits fail-closed on every child, records
  per-shard logs/progress/summaries, optionally concatenates shard JSONLs into
  the monolithic path expected by the current streaming-IDM config, and writes a
  combined summary with parsed GPU-utilization statistics.
- `scripts/run_g005_idm_frozen_frame_embedding_prefix.sh` now supports
  `EMBED_SHARD_COUNT>1`, `EMBED_SHARD_DEVICES`, and shard-local monitor settings;
  `EMBED_BACKEND=dinov2-torchhub` uses the train extra so torch is available in
  normal `uv` environments.
- `tests/test_frame_embedding_shard_launcher.py` covers shard planning,
  dummy-stat subprocess materialization/concat, and nvidia-smi CSV parsing.

Verification: `python3 -m py_compile scripts/run_frame_embedding_shards.py
scripts/materialize_frame_embedding_features.py
src/fdm_d2e/data/frame_embedding_materializer.py`; `bash -n
scripts/run_g005_idm_frozen_frame_embedding_prefix.sh`; and `uv run pytest -q
tests/test_frame_embedding_materializer.py tests/test_frame_embedding_shard_launcher.py
tests/test_training_run_scripts.py` => 30 passed.

Claim boundary: launcher hardening only. The next GPU step must rerun a small
sharded DINO materialization with this launcher and valid pre-launch monitoring
before scaling to 320k train/target extraction or training.

### 2026-05-28 KST — shard-launcher debug GPU gate

Reran the 1,024-row DINO materialization gate with the new fail-closed launcher
on the still-active debug reservation `rsv-jeonghunpark-20260527-1ea75c`
(before the 11:00 KST expiration). The pod worktree was reset to `4c84904`.

Evidence:

- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_launcher1024_run_summary.json`
  — `status=pass`, `rows_written=1024`, `combined_rows=1024`, `shard_count=2`,
  elapsed `26.18s`, `monitor_started_before_shards=true`.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_launcher1024_shard0_summary.json`
  and `..._shard1_summary.json` — both `status=pass`, `rows_written=512`;
  shard 1 records `source_rows_skipped=512`, proving the contiguous skip range.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_launcher1024_gpu_monitor.csv`
  — pre-launch monitor with 52 samples over GPUs 0 and 1.
- Per-shard logs/progress files record the exact materializer commands and
  torchhub DINO cache path.

Finding: the launcher fixes the previous evidence bug: it waits on child PIDs,
concatenates the shard outputs, and captures a monitor from before launch. The
short 1,024-row gate still shows low mean utilization (GPU0 mean `2.58%`, GPU1
mean `0.12%`, max `64%`/`3%`) even though both GPUs allocate DINO memory, so do
not scale straight to a full 320k train+target extraction expecting high H200
utilization. Next gate should either use larger per-shard row counts to amortize
model load/JSON overhead or write/read tensor caches directly, and it must keep
the same launcher summary/monitor evidence.

Claim boundary: materialization launcher validation only; no trained IDM metric
evidence and no G005 success claim.

### 2026-05-28 KST — 4,096-row launcher amortization gate

Used the same debug reservation before expiration to run a larger 4,096-row
`dinov2-torchhub` gate with the validated shard launcher, after copying a fresh
4,096-row real D2E target sample from `production-storage-shell-4`.

Evidence:

- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_launcher4096_run_summary.json`
  — `status=pass`, `rows_written=4096`, `combined_rows=4096`, two shards,
  elapsed `93.64s`.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_launcher4096_gpu_monitor.csv`
  — pre-launch monitor with 186 samples. GPU0/GPU1 max utilization reached
  `100%`/`89%`, but mean utilization remained low at `4.13%`/`1.41%`.
- Per-shard summary/progress/log artifacts record 2,048 rows per shard and exact
  skip/scanned row counts.

Finding: increasing from 1,024 to 4,096 rows amortizes startup enough to observe
real H200 spikes on both debug GPUs, but the mean utilization is still too low
for a full 4×H200 extraction to be considered healthy. The bottleneck is likely
JSONL row IO/serialization and CPU-side preprocessing around the DINO forward
rather than model compute alone.

Decision: do not run a full frozen-frame extraction with the JSON-expanded path
as-is. The next implementation should add a more compact tensor-cache/materialized
feature path or otherwise batch CPU preprocessing/serialization so H200 wall-clock
is dominated by model forward work before spending a production reservation.
Claim boundary: materialization throughput evidence only; no trained IDM metric
evidence and no G005 success claim.

### 2026-05-28 KST — vectorized compact-luma preprocessing gate

After commit `03efbb7` vectorized compact-luma byte-frame batching before the
DINO forward pass, reran the same 4,096-row launcher gate on the debug pod.

Evidence:

- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_launcher4096_vec_run_summary.json`
  — `status=pass`, `rows_written=4096`, `combined_rows=4096`, elapsed `91.67s`.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_launcher4096_vec_gpu_monitor.csv`
  — valid pre-launch monitor, GPU0/GPU1 max utilization `10%`/`59%`, means
  `0.24%`/`1.14%`.
- Per-shard logs no longer show the previous non-writable `torch.frombuffer`
  warning.

Finding: vectorized preprocessing slightly improves wall-clock versus the prior
4,096-row gate (`93.64s` → `91.67s`) and cleans the warning, but it does not fix
mean GPU utilization. This falsifies the hypothesis that the per-frame tensor
construction loop was the dominant bottleneck. Full DINO extraction still needs a
larger change (binary/tensor cache writer, more aggressive batching around model
forward, or a different representation branch) before production-scale GPU use.

Claim boundary: preprocessing/throughput diagnostic only; no trained IDM metric
evidence and no G005 success claim.
