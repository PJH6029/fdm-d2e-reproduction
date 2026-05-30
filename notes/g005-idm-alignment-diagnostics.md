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

### 2026-05-28 KST — external feature-cache contract for frozen-frame branch

Implemented the next non-GPU scaling fix after the DINO JSONL path remained
low-utilization: materialized rows can now store only
`__streaming_idm_feature_cache` references while the actual feature vectors live
in a torch `.pt` tensor cache. The same refs are consumed by streaming IDM stats,
training-cache construction, training, and prediction.

Implementation:

- `scripts/materialize_frame_embedding_features.py` accepts `--feature-cache-out`
  and `--thin-output`.
- `scripts/run_frame_embedding_shards.py` accepts `--feature-cache-dir` and
  passes per-shard cache paths to the materializer.
- `scripts/run_g005_idm_frozen_frame_embedding_prefix.sh` supports
  `EMBED_FEATURE_CACHE=1`, `EMBED_THIN_OUTPUT=1`, and
  `EMBED_FEATURE_CACHE_ROOT=...` so prefix extraction can avoid JSON-expanded
  feature arrays.
- `src/fdm_d2e/training/streaming_idm.py` can read external feature-cache refs
  through an LRU cache and can train/predict through the existing training-cache
  path from those refs.

Evidence:

- `artifacts/idm/g005_idm_frozen_frame_embedding_feature_cache_contract_summary.json`
  — real D2E 128-row compact-luma contract probe with `status=pass`.
- Inline JSON for the 128-row probe was `2,607,781` bytes. Thin JSON plus feature
  cache was `170,055` bytes, a `15.33x` total byte reduction while preserving
  `input_dim=36` through `scan_streaming_idm_stats`.
- `artifacts/idm/g005_idm_frozen_frame_embedding_feature_cache_contract_inline_summary.json`
  and `..._thin_summary.json` preserve materializer provenance/hashes.
- Unit/integration coverage now includes materializer cache refs, shard-launcher
  cache refs, and a tiny end-to-end streaming IDM train/predict run from external
  feature-cache rows.

Verification: `python3 -m py_compile` for the materializer, streaming IDM, and
launcher scripts; `bash -n scripts/run_g005_idm_frozen_frame_embedding_prefix.sh`;
`uv run pytest -q tests/test_frame_embedding_materializer.py
tests/test_frame_embedding_shard_launcher.py tests/test_streaming_idm_contract.py
tests/test_training_run_scripts.py` => 67 passed.

Claim boundary: feature-cache/scaling infrastructure only. It does not provide a
trained IDM metric win. The next GPU branch should rerun a monitored DINO gate
with `EMBED_FEATURE_CACHE=1 EMBED_THIN_OUTPUT=1`; only if utilization/throughput
is materially better should it scale to the 320k prefix and train a candidate.

### 2026-05-28 KST — cache-backed DINO launcher gate

Reserved a short debug slot `rsv-jeonghunpark-20260528-af91c8` on group B
(2×H200, 10:00–11:00 KST) to rerun the 4,096-row DINO gate with the new external
feature-cache path from commit `5ed61a9`.

Evidence:

- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_cache_launcher4096_run_summary.json`
  — `status=pass`, `rows_written=4096`, `combined_rows=4096`, elapsed `90.26s`,
  `thin_output=true`, per-shard feature caches enabled.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_cache_launcher4096_gate_summary.json`
  — compares the cache-backed output against the previous JSON-expanded 4,096-row
  DINO output.
- The previous JSON-expanded combined output was `158,722,812` bytes. Thin JSON
  plus two feature-cache tensors is `40,249,279` bytes (`3.94x` total reduction;
  the JSON component alone is `28.8x` smaller).
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_cache_launcher4096_gpu_monitor.csv`
  — valid pre-launch monitor. GPU0/GPU1 both reached `100%` max, but means were
  still low (`2.31%`/`1.11%`).

Decision: cache-backed rows solve the JSON size/serialization artifact problem
and are required for any future frozen-frame prefix run, but they do not fix the
DINO extraction utilization bottleneck by themselves. Do not promote directly to
full 320k prefix extraction on production H200s from this evidence alone. The
next frozen-frame option would need larger shard amortization, more aggressive
GPU batching, or direct tensor-cache extraction that reduces CPU row handling;
otherwise pivot to a stronger teacher/representation branch.

Claim boundary: materialization and storage-efficiency evidence only; no trained
IDM metric evidence and no G005 success claim.

### 2026-05-28 KST — tensor-cache direct path and native-luma DINO gate

After the cache-backed DINO gate still averaged low H200 utilization, two follow-up
materialization changes were tested on short debug reservations and both were
kept evidence-bound as materialization gates only.

First, commit `99bfa9b` made external feature-cache mode keep DINO embedding
outputs as tensors instead of converting every embedding vector to Python lists
and then rebuilding a tensor for `torch.save`. Debug reservation
`rsv-jeonghunpark-20260528-28ad03` ran the 4,096-row DINO gate twice:

- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_cache_direct4096_run_summary.json`
  — batch 256, `status=pass`, elapsed `89.34s`, only `1.01x` faster than the
  prior cache-backed 4,096-row gate.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_cache_direct4096_b1024_run_summary.json`
  — batch 1024, `status=pass`, elapsed `88.49s`, still not a material speedup;
  mean nvidia-smi utilization was lower because the remaining bottleneck was
  outside the DINO forward.

This rejected the hypothesis that Pythonizing DINO output vectors was the main
H200-idle cause.

Second, commit `e0d0e86` removed the actual dominant CPU loop: compact-luma rows
were being expanded from 16×16 to 224×224 with Python nested loops before the
batched tensor preprocessor. The materializer now keeps native 16×16 luma bytes
and lets `torch.nn.functional.interpolate` resize the whole batch.

Debug reservation `rsv-jeonghunpark-20260528-d40653` validated the fix:

- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_native_luma4096_b1024_run_summary.json`
  — 4,096 real D2E rows, batch 1024, elapsed `13.12s`, about `6.88x` faster
  than the prior cache-backed 4,096-row gate; both GPUs reached `100%` max.
- `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_native_luma8192_b2048_run_summary.json`
  — 8,192 real D2E rows, batch 2048, elapsed `20.17s`, `406` rows/s on 2×H200;
  both GPUs reached `100%` max with valid pre-launch monitor evidence.
- Summary artifact:
  `artifacts/idm/g005_idm_frozen_frame_embedding_dinov2_torchhub_gpu_native_luma_gate_summary.json`.

Decision: native-luma tensor-cache mode is now good enough for the next bounded
frozen-DINO prefix materialization/training gate. It is still not trained IDM
metric evidence and does not satisfy G005. Next step is a 320k prefix DINO IDM
train/eval gate, then promote only if paper-compatible keyboard, mouse-button,
and mouse-motion metrics materially beat the current event-context prefix/full
baselines while preserving no-button FPR.

### 2026-05-28 KST — 320k frozen-DINO compact-luma prefix rejected

- Launched production 2×H200 run `rsv-jeonghunpark-20260528-ed08d8` on Node 6 GPUs 0–1 for `G005` frozen-DINO prefix; a 4×H200 reservation was rejected by quota (`requested 4 GPUs exceeds quota 2`). A back-to-back extension `rsv-jeonghunpark-20260528-719318` was scheduled and then cancelled after evidence collection; an idle debug reservation was also cancelled. Current reservation list was empty after cleanup.
- The first launch failed from an env quoting bug, the second from concurrent `torch.hub` DINO cache extraction, and the third from `ffmpeg` absence in the production base image. The terminal run used manual torchhub prewarm plus `EMBED_FRAME_SOURCE=compact-luma`, matching the native-luma tensor path. Follow-up hardening `895ae53` adds launcher-side torchhub cache prewarm for future shard launches.
- Evidence copied locally: `artifacts/idm/g005_idm_frozen_frame_embedding_prefix320k_run_summary.json`, `artifacts/idm/g005_idm_frozen_frame_embedding_prefix320k_paper_metrics.json`, `artifacts/idm/g005_idm_frozen_frame_embedding_prefix320k_gpu_monitor.csv`, source/materialization summaries, and small output metadata under `outputs/idm_streaming_d2e_full_frozen_frame_embedding_prefix320k/`.
- Result: materialization and training completed on 320k train / 320k target rows, but metric screen failed badly: paper-compatible keyboard `0.0206`, mouse-button `0.1046`, Pearson X/Y `0.0199/0.0102`, scale ratios X/Y `3.465/2.076`; strict button F1 `0.1814`; heldout-game no-button FPR `0.1096` exceeds the 0.10 split gate.
- Decision: reject frozen-DINO compact-luma embeddings as a G005 full-corpus promotion candidate. Treat native-luma DINO as useful infrastructure only; next branch must change supervision/decoding (teacher-assisted event decoding, causal per-recording latent state/state-machine estimation, or released G-IDM distillation if feasible), with a prefix metric gate before expensive full-corpus 4×H200 work.

### 2026-05-28 KST — key-event taxonomy exposes the remaining keyboard bottleneck

Ran a CPU-only taxonomy diagnostic on `production-storage-shell-4` from clean
worktree `fdm-d2e-reproduction-key-taxonomy-5bb75e4`, using the same 320k
chronological target prefix as the rejected DINO branch.

Evidence:

- `scripts/build_g005_key_event_taxonomy.py`
- `src/fdm_d2e/eval/key_event_taxonomy.py`
- `tests/test_key_event_taxonomy.py`
- `artifacts/idm/g005_idm_key_event_taxonomy_prefix320k.json`

Findings:

- Rows: `320,000`; key tokens: `80,548`; key rows: `60,313`.
- Only `36.48%` of key tokens are visible from prior→next held-state differencing.
- `63.49%` are hidden/repeat tokens; the dominant class is `held_repeat_press`
  (`50,169` tokens), not rare same-bin taps (`34` press + `34` release).
- A diagnostic oracle that adds current-row hidden/repeat key tokens to state-delta
  buttons and previous-motion reaches keyboard `0.74175` and mouse-button
  `0.97544` on the prefix, crossing the paper keyboard/button targets. This is
  not claimable because it inspects current-row ground truth, but it proves the
  paper-target gap is specifically a key-repeat/tap decoder problem rather than
  impossible state-transition labeling.
- Motion remains below the paper target under previous-motion continuation alone
  (Pearson X/Y `0.7687/0.7425`), so a complete G005 candidate still needs a
  stronger motion source in addition to the repeat-key specialist.

Decision: do not reserve H200s for more frozen visual/global temporal wrappers
until a non-leaky key-repeat/tap specialist beats the `event_all` prefix keyboard
baseline and approaches the oracle target. The next branch should explicitly
model held-repeat keypresses (likely with richer phase/history features or a
specialized per-key head), pair state-delta-style mouse-button decoding with safe
FPR, and separately improve motion beyond previous-motion continuation.

### 2026-05-28 KST — aligned repeat-key table specialist rejected

Implemented and ran a CPU-only key-repeat specialist matrix before reserving any
H200s for a neural repeat-key branch. The runner trains press/release lookup
tables from train-prefix causal fields and composes them with the existing
`event_state_duration_context` base prediction stream. A first chronological-run
attempt was discarded as diagnostic only because the base prediction stream was
shard-ordered; the committed runner now records sequence-id alignment errors.

Evidence:

- `src/fdm_d2e/eval/key_repeat_specialist.py`
- `scripts/build_g005_key_repeat_specialist_matrix.py`
- `tests/test_key_repeat_specialist.py`
- `artifacts/idm/g005_idm_key_repeat_specialist_matrix_aligned_prefix50k.json`

Result: reject simple table-based repeat-key decoding. On a shard-order aligned
50k prefix, alignment reports zero sequence-id mismatches. The best policy is
still `base_all` with keyboard `0.1550`, mouse-button `0.1614`, Pearson X/Y
`0.7598/0.6794`, strict button F1 `0.2675`, and no-button FPR `0.0390`.
Every repeat-table replacement/union policy reduces keyboard; the best
replacement reaches only `0.1216` keyboard. This confirms that the key-event
oracle gap cannot be closed by simple train-prefix hold-age lookup tables.

Decision: do not promote this repeat-key table path to a GPU run. The next G005
candidate must use a learned sequence/teacher signal for key repeats/taps (or a
released-GIDM distillation target) while retaining strict alignment checks for
any base-stream composition.

### 2026-05-28 KST — phase-aware repeat-key heuristic rejected

Tested whether keyboard-repeat failures are recoverable from causal bin phase
features before launching a learned repeat-key GPU run. The diagnostic trains
press probabilities from train-prefix held-key code, hold duration, and
`sequence_id`/timestamp phase, then adds predicted held-key press tokens to the
aligned `event_state_duration_context` base stream.

Evidence:

- `src/fdm_d2e/eval/key_phase_repeat_diagnostic.py`
- `scripts/build_g005_key_phase_repeat_diagnostic.py`
- `tests/test_key_phase_repeat_diagnostic.py`
- `artifacts/idm/g005_idm_key_phase_repeat_diagnostic_prefix50k.json`

Result: reject handcrafted phase features. On a 50k aligned prefix, alignment has
zero sequence mismatches. Base keyboard is `0.155047`; the best phase policy
(`code_holdbucket_phase_period6_threshold0.65`) reaches only `0.155303`, a
+`0.000256` absolute improvement and nowhere near the paper target. This is too
small to justify an H200 run.

Decision: learned/teacher-assisted key-repeat supervision remains the next
viable direction; handcrafted hold-age/phase tables are exhausted.

### 2026-05-28 KST — learned hashed key-repeat diagnostic improves but remains far below target

After rejecting handcrafted hold/phase tables, implemented a lightweight learned
hashed logistic key specialist that predicts held-key press/release events from
causal event-state-duration features and composes with the aligned
`event_state_duration_context` base stream.

Evidence:

- `src/fdm_d2e/eval/key_hash_sequence_diagnostic.py`
- `scripts/build_g005_key_hash_sequence_diagnostic.py`
- `tests/test_key_hash_sequence_diagnostic.py`
- `artifacts/idm/g005_idm_key_hash_sequence_diagnostic_prefix50k.json`
- `artifacts/idm/g005_idm_key_hash_sequence_diagnostic_prefix50k_e2_lr01.json`
- `artifacts/idm/g005_idm_key_hash_sequence_diagnostic_prefix320k_e2_lr01.json`

Result: promising diagnostic, not success. On the aligned 320k prefix, the best
policy `press_only_union_base_keys_press0.65` improves keyboard from base
`0.1990` to `0.2337` with zero sequence-id mismatches, while preserving base
mouse-button `0.1726`, Pearson X/Y `0.8002/0.6427`, strict button F1 `0.2809`,
and no-button FPR `0.0375`. This is the first repeat-key branch to materially
beat the aligned base keyboard metric.

However, it remains far below the paper keyboard target `0.73`, does not improve
mouse-button, and leaves Pearson Y below the paper target. Do not checkpoint G005
or promote this exact CPU model as a final candidate. The next branch should use
this as evidence for a stronger learned sequence/teacher-assisted key-repeat
model (e.g. neural sequence specialist, released-GIDM teacher targets, or richer
per-key temporal state) and must prefix-gate against the 320k aligned metrics.

### 2026-05-28 KST — compact visual hash does not improve learned repeat-key gate

Extended the learned hashed key-repeat diagnostic with optional compact luma
transition hash features, then ran a CPU/storage-shell aligned prefix gate before
spending GPU time.

Evidence:

- `src/fdm_d2e/eval/key_hash_sequence_diagnostic.py`
- `scripts/build_g005_key_hash_sequence_diagnostic.py`
- `tests/test_key_hash_sequence_diagnostic.py`
- `artifacts/idm/g005_idm_key_hash_sequence_visual_diagnostic_prefix50k_e2_lr01.json`

Result: reject this visual-hash branch. On the aligned 50k prefix, the run has
zero sequence-id mismatches and best policy
`press_only_union_base_keys_press0.5`, but keyboard reaches only `0.19496`.
That is above the aligned base `0.15505`, but below the non-visual learned hash
50k gate (`0.19928`) and far below the paper keyboard target `0.73`. Button and
motion metrics are inherited from the base stream (button `0.16138`, Pearson
X/Y `0.75981/0.67937`) and also miss paper targets.

Decision: do not reserve H200s for compact visual-transition hash features in
this repeat-key specialist. The next G005 branch should move beyond additive
hashed features toward a stronger learned sequence/teacher-assisted model or
released G-IDM exact-split distillation, while preserving aligned-prefix gates.

### 2026-05-28 KST — top-key vocabulary binary gate rejected

Broadened the hashed key specialist beyond currently held keys by adding a
train-prefix top-key vocabulary option, then ran a CPU/storage-shell aligned
50k prefix gate with top 16 key codes.

Evidence:

- `src/fdm_d2e/eval/key_hash_sequence_diagnostic.py`
- `scripts/build_g005_key_hash_sequence_diagnostic.py`
- `tests/test_key_hash_sequence_diagnostic.py`
- `artifacts/idm/g005_idm_key_hash_sequence_topvocab16_diagnostic_prefix50k_e2_lr01.json`

Result: reject this top-vocabulary binary specialist. Alignment has zero
sequence-id mismatches and the trained model saw `10,250,608` candidate examples
across two epochs, but `base_all` remains the best policy at keyboard `0.15505`.
The best specialist policy (`press_only_union_base_keys_press0.9`) reaches only
keyboard `0.11229`, below both base and the held-only learned hash 50k gate
(`0.19928`). Adding non-held key candidates this way overfires/overconfuses key
tokens rather than closing the hidden-repeat/new-key gap.

Decision: do not reserve GPUs for the top-key vocabulary hash branch. Continue
G005 with a qualitatively stronger sequence/teacher-assisted approach, not wider
additive tabular/hash key candidates.

### 2026-05-28 KST — held-only hash scaling to 2M train rows regresses

Ran the best nonterminal held-key hash specialist with a larger CPU/storage-shell
train prefix before spending H200 time on the branch.

Evidence:

- `artifacts/idm/g005_idm_key_hash_sequence_diagnostic_train2m_prefix320k_e1_lr01.json`
- comparison: `artifacts/idm/g005_idm_key_hash_sequence_diagnostic_prefix320k_e2_lr01.json`

Result: reject more-row scaling for this hash model. The 2M-row / 1-epoch run
keeps zero sequence-id mismatches on the aligned 320k eval prefix and trains on
`1,848,111` held-key examples, but best policy
`press_only_union_base_keys_press0.95` reaches keyboard `0.22216`. This is above
base `0.19900`, but below the prior 320k-train / 2-epoch held-hash result
`0.23365` and still far below the paper keyboard target `0.73`. Button and
motion remain inherited from the base stream (button `0.17257`, Pearson X/Y
`0.80024/0.64272`).

Decision: do not reserve GPUs or expand full-corpus training for this hash
specialist. The evidence now rejects additive hash/table variants, visual hash,
top-vocabulary binary broadening, and simple train-row scaling; the next G005
attempt must introduce a new sequence-state/teacher-assisted mechanism.

### 2026-05-28 KST — mouse-button hash specialist rejected

Added a CPU prefix diagnostic for mouse-button down/up events using prior button
held-state, previous events, prior action tokens, and hashed online features.
A first broad 320k sweep was intentionally terminated after ~31 minutes because
73 policies over 320k rows were too slow for a single CPU gate; the bounded
50k/narrow-threshold gate below is the retained evidence.

Evidence:

- `src/fdm_d2e/eval/button_hash_sequence_diagnostic.py`
- `scripts/build_g005_button_hash_sequence_diagnostic.py`
- `tests/test_button_hash_sequence_diagnostic.py`
- `artifacts/idm/g005_idm_button_hash_sequence_diagnostic_prefix50k_e2_lr01.json`

Result: reject this button hash branch. Alignment has zero sequence-id
mismatches and the model trains on `1,920,000` button-code examples across two
epochs, but `base_all` remains best at mouse-button accuracy `0.16138`, strict
button F1 `0.26748`, and no-button FPR `0.03899`. The best specialist policy
(`replace_base_buttons_down0.95_up0.65`) drops button accuracy to `0.08046` and
raises no-button FPR to `0.10274`, exceeding the desired <=`0.10` gate.

Decision: do not reserve GPUs for this mouse-button hash specialist. The button
endpoint needs a different learned head, calibrated logits, or teacher signal;
prior-button-state hashing is not sufficient.

### 2026-05-28 KST — count-aware double-press hash rejected

Found a structural issue in earlier key-hash diagnostics: they deduped key tokens
before paper-metric evaluation, while D2E target bins can contain repeated key
presses. On the aligned 320k target prefix, `16,532` key-token cases have count
`2` and `16,511` rows contain at least one repeated key token. Added a
double-press hash head and count-preserving max-count union policy, then ran a
CPU/storage-shell aligned 50k gate.

Evidence:

- `src/fdm_d2e/eval/key_hash_sequence_diagnostic.py`
- `scripts/build_g005_key_hash_sequence_diagnostic.py`
- `tests/test_key_hash_sequence_diagnostic.py`
- `artifacts/idm/g005_idm_key_hash_sequence_countaware_diagnostic_prefix50k_e2_lr01.json`

Result: reject this count-aware hash branch. It trains on `555,438` held-key
examples with `31,850` double-press positives and has zero sequence-id
mismatches, but best overall policy reaches keyboard `0.19446`, below the
non-count held-hash 50k result `0.19928`. The best explicit count-aware policy
(`press_count_union_base_keys_press0.35_double0.95`) reaches only `0.19353`.

Decision: count multiplicity is a real bottleneck, but a separate hashed
double-press head does not solve it. Next G005 branch should use a stronger
sequence/teacher model that jointly predicts key identity, repeat count, release,
and closed-loop state, rather than additive hash heads.

### 2026-05-28 KST — joint key-state table diagnostic rejected

Implemented a bounded joint sequence-state diagnostic that predicts the full key-event multiset from causal held-key/state contexts instead of independent per-key hash heads. The storage-shell gate used a clean worktree at `67532a3`, existing aligned `event_state_duration_context` base predictions, 100k train rows, and the first 50k target rows with selected lookup contexts (`held_codes_only`, `held_bucket_only`, `held_mod_since_phase`, `chain:specific_to_global`).

Evidence:

- `src/fdm_d2e/eval/joint_key_state_diagnostic.py`
- `scripts/build_g005_joint_key_state_diagnostic.py`
- `tests/test_joint_key_state_diagnostic.py`
- `artifacts/idm/g005_idm_joint_key_state_diagnostic_prefix50k_train100k_selected.json`
- `artifacts/idm/g005_idm_joint_key_state_diagnostic_prefix50k_train100k_selected_storage.log`

Result: reject this joint table branch. Alignment has zero sequence-id mismatches and the best policy (`joint_union_held_bucket_only_top_th0.05_s1`) improves the 50k base keyboard from `0.15505` to only `0.16187`, while preserving base button `0.16138`, Pearson X/Y `0.75981/0.67937`, strict button F1 `0.26748`, and no-button FPR `0.03899`. This is below the previous held-key hash 50k gate (`0.19928`) and far below the paper keyboard target `0.73`.

Decision: do not reserve GPUs for this table/memorization specialist. Independent hash/table and joint tabular sequence-state variants are exhausted; the next viable G005 branch must move to a stronger learned sequence/teacher-assisted mechanism rather than another CPU table over held-key metadata.

### 2026-05-28 KST — visual/state action-memory retrieval rejected

Implemented a CPU prefix diagnostic that retrieves train-token multisets from quantized visual/state contexts (`frame.features`, `next_frame_features`, 4×4 pooled grid transitions, held-key state, previous key context) and composes the retrieved action memory with the aligned `event_state_duration_context` base stream.

Evidence:

- `src/fdm_d2e/eval/visual_action_retrieval_diagnostic.py`
- `scripts/build_g005_visual_action_retrieval_diagnostic.py`
- `tests/test_visual_action_retrieval_diagnostic.py`
- `artifacts/idm/g005_idm_visual_action_retrieval_diagnostic_prefix50k_train100k.json`
- `artifacts/idm/g005_idm_visual_action_retrieval_diagnostic_prefix50k_train100k_storage.log`

Result: reject visual/state action-memory retrieval. The storage-shell gate used clean worktree `c75f86c`, 100k train rows, 50k target rows, and zero sequence-id mismatches. Best policy `retrieval_union_categorical_base_motion_state_only_th0.2_s1` reached keyboard `0.15816` versus base `0.15505`, while button dropped to `0.15585`; Pearson X/Y stayed at base `0.75981/0.67937`; no-button FPR was `0.04110`. This underperforms both the held-key hash 50k gate (`0.19928`) and the joint key-state table (`0.16187`), and is far below paper target `0.73`.

Decision: do not promote approximate visual/action-memory retrieval. The immediate remaining viable route is not another quantized CPU memory/table branch; use released-GIDM/teacher debugging or a genuinely stronger neural sequence model with explicit repeat-key supervision and calibrated motion/button heads.

### Warmup-trimmed released-GIDM teacher pilot rejection

Date: 2026-05-28 KST.

Ran a real 2GPU warmup-trimmed released `open-world-agents/Generalist-IDM-1B` pilot on MLXP reservation `rsv-jeonghunpark-20260528-82c473` (production node 4 GPUs `[1,2]`, cancelled after evidence collection). The pod used a clean detached worktree at commit `df25e79`, copied `.env` for W&B/HF access without committing secrets, and logged W&B run `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/zp6wpkpb`.

Evidence:

- `artifacts/eval/g006_gidm_warmup_trim_pilot_pipeline_summary.json` — pipeline `status=pass`, one completed 15s chunk, 100 eval rows, zero missing predictions.
- `artifacts/eval/g006_gidm_warmup_trim_pilot_paper_metrics.json` — paper-compatible metric pass over aligned rows but metric values reject the branch.
- `artifacts/idm/g005_gidm_warmup_trim_pilot_rejection.json` — explicit negative decision and sanitized reservation summary.
- `outputs/gidm_warmup_trim_pilot/` — small local target/prediction/MCAP pilot evidence copied from the pod.

Result: reject warmup trimming as a released-GIDM teacher/timing rescue for G005. On the 100-row temporal eval window, keyboard accuracy was `0.0`, mouse Pearson X/Y were `-0.0304/0.1131`, scale ratios were too high (`3.04/1.87`), and the window had no mouse-button positives, so button accuracy was not measured. No-button FPR was `0.0`, but only because no button predictions occurred. This does not approach the D2E paper targets and must not be promoted to a G005 paper-target path or G006 exact-split completion.

Next branch: pivot back to a learned non-leaky sequence/state IDM architecture or a different teacher signal; do not spend a larger G-IDM warmup-trim scaling run unless a separate timing/alignment explanation appears.

Follow-up row-shift sweep over the copied 100-row warmup pilot (`artifacts/eval/g006_gidm_warmup_trim_pilot_shift_sweep.json`) found that timing rescue is insufficient: the best keyboard shift was +47 rows / +2.35s with keyboard `0.1786` on only 53 overlapping rows; the best mouse-Pearson shift was +48 rows with Pearson X/Y `0.6477/0.6805` but keyboard only `0.0645`. This confirms that simple timestamp/row shifting does not turn the released-GIDM warmup pilot into a viable G005 paper-target path.

### Per-key repeat-clock prefix rejection

Date: 2026-05-28 KST.

Implemented and ran a CPU/storage-shell per-key repeat-clock diagnostic (`scripts/build_g005_key_repeat_clock_diagnostic.py`, commit `1f04203`) to test whether key-repeat failures were caused by missing per-key last-press/release timing. The gate used 20k train rows, 10k target rows, top-8 key candidates, predicted-clock and teacher-forced-clock variants, and the current event-state-duration context predictions as the non-key/base stream.

Evidence:

- `artifacts/idm/g005_idm_key_repeat_clock_diagnostic_prefix10k_train20k.json` — diagnostic payload with zero sequence mismatches.
- `artifacts/idm/g005_idm_key_repeat_clock_prefix10k_train20k_rejection.json` — explicit negative decision.
- `outputs/idm_diagnostics/g005_key_repeat_clock_prefix10k_train20k_predictions.jsonl` — small prediction sample copied from storage shell.

Result: reject this branch. `base_all` remained best on the 10k prefix (`keyboard=0.11678`, mouse-button accuracy `0.22667`, Pearson X/Y `0.76516/0.74335`, no-button FPR `0.01866`). The best predicted-clock repeat policy slightly underperformed base (`keyboard=0.11671`) and no teacher-forced clock policy justified promotion. Do not reserve GPUs or run a larger table-clock sweep for this exact approach.

### 2026-05-28 KST — base-offset released-GIDM timing diagnostic retained, still rejected

After the local reboot/resume, revalidated the warmup-trimmed released-GIDM
pilot's suspicious `~2.4s` timing offset without reserving GPUs. Added a
configurable chunk timestamp mode to the released-GIDM runner so future bounded
pilots can explicitly test `ground_truth_aligned`, `video_relative`, and
`ground_truth_plus_base` stamping instead of hand-editing generated scripts.
Also added an auditable base-offset row-shift diagnostic over the copied warmup
pilot JSONLs.

Evidence:

- `src/fdm_d2e/eval/gidm_runner.py` now exposes `chunk_timestamp_mode` for
  chunked released-GIDM pilots; default remains `ground_truth_aligned` for
  existing evidence paths.
- `scripts/run_gidm_manifest_inference.py --chunk-timestamp-mode ...` and
  `src/fdm_d2e/eval/gidm_exact_pipeline.py` pass the mode through to chunk
  planning/manifest generation.
- `src/fdm_d2e/eval/gidm_timestamp_diagnostic.py` and
  `scripts/build_g006_gidm_base_offset_shift_diagnostic.py` generate the retained
  base-offset diagnostic.
- `artifacts/eval/g006_gidm_warmup_base_offset_shift_diagnostic.json` — status
  `pass`, decision `rejected_no_base_offset_shift_meets_paper_targets`.

Result: the manifest base offset is `2.4170986s` (`48.34` 50ms rows), matching
the earlier sweep's best-shift neighborhood. However, base-offset correction
still does not approach paper targets. Best keyboard is shift `+47` / `2.35s`
with keyboard `0.1786`; best mouse-Pearson is shift `+48` / `2.40s` with
Pearson X/Y `0.6477/0.6805`, keyboard only `0.0645`, no button positives, and
all `paper_targets_pass=false`.

Decision: keep timestamp-mode configurability for future exact-split G-IDM
baseline diagnostics, but do not spend a larger G005 teacher run on this pilot
unless a new alignment hypothesis is backed by stronger covered-window evidence.
The immediate G005 path remains a stronger learned non-leaky sequence/teacher
IDM, not promoting released-GIDM warmup timing rescue.

### 2026-05-28 KST — released-GIDM ffmpeg seek-mode diagnostic infrastructure

Added a second bounded released-GIDM alignment knob for future teacher pilots: `chunk_seek_mode`. The historical/default path remains `input_fast`, which emits `ffmpeg -ss <start> -i <video> ...`. The new diagnostic path `output_accurate` emits `ffmpeg -i <video> -ss <start> ...` so a future tiny GPU pilot can test whether keyframe/input-seek drift explains the poor warmup-trimmed released-GIDM alignment.

Evidence:

- `src/fdm_d2e/eval/gidm_runner.py` now carries `chunk_seek_mode` through chunk plans, runner rows, and chunked manifests; default is `input_fast` to preserve existing evidence.
- `scripts/run_gidm_manifest_inference.py --chunk-seek-mode {input_fast,output_accurate}` and `src/fdm_d2e/eval/gidm_exact_pipeline.py` pass the mode through.
- `artifacts/eval/g006_gidm_warmup_seekmode_dry_run_summary.json` and `artifacts/eval/g006_gidm_warmup_seekmode_dry_run_chunked_manifest.json` prove the output-accurate plan/manifest without reserving GPUs.
- Unit tests cover generated upstream script patching, runner CLI propagation, exact-pipeline default propagation, and manifest seek-mode recording.

Decision: this is infrastructure only, not a metric win. Do not treat it as G005 completion evidence. A future GPU pilot is only justified if it is tightly bounded (for example one 15s warmup chunk) and followed immediately by conversion/paper metrics plus reservation cancellation.

### 2026-05-28 KST — output-accurate released-GIDM seek pilot rejected

Ran the bounded real-GPU follow-up for the `chunk_seek_mode=output_accurate` hypothesis on MLXP reservation `rsv-jeonghunpark-20260528-a71ffb` (production node 4, one H200 GPU, cancelled after artifact copy). The pod used detached worktree commit `d3f9298`, copied `.env` for W&B/HF access without committing secrets, and logged W&B run `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/nmnwjjbc` for the terminal finalize pass.

Evidence:

- `configs/eval/g006_gidm_warmup_seekmode_pilot.yaml` — exact bounded pilot config with `chunk_seek_mode=output_accurate`, `resume=false`, one 15s chunk, and absolute PVC by-recording roots.
- `artifacts/eval/g006_gidm_warmup_seekmode_pilot_inference_summary.json` — one chunk completed, `elapsed_seconds=178.736`, `seek_mode=output_accurate`, output SHA retained.
- `artifacts/eval/g006_gidm_warmup_seekmode_pilot_paper_metrics.json` — 100 aligned rows, paper-metric artifact `status=pass` but metric values reject the branch.
- `artifacts/eval/g006_gidm_warmup_seekmode_pilot_gpu_monitor.csv` — 39 monitor rows, max GPU utilization 48%, 16 nonzero-GPU samples.
- `artifacts/idm/g005_gidm_warmup_seekmode_pilot_rejection.json` — explicit negative decision and sanitized reservation summary.
- `outputs/gidm_warmup_seekmode_pilot/` — small target/prediction/MCAP evidence copied locally.

Result: reject output-accurate ffmpeg seek as a released-GIDM teacher rescue. Metrics on the same 100-row warmup window stayed at keyboard `0.0`, mouse Pearson X/Y `-0.0304/0.1131`, scale ratios `3.04/1.87`, and no mouse-button positives. This matches the previous warmup-trim failure pattern and is nowhere near paper targets. Do not spend a larger G-IDM run on seek placement.

### 2026-05-28 KST — expanded/count-aware repeat priors rejected

Ran two CPU/storage-shell diagnostics after the state-delta oracle showed that
hidden held-key repeats dominate the remaining keyboard gap.

Evidence:

- `artifacts/idm/g005_state_transition_expanded_repeat_context_diagnostics_summary.json`
- `artifacts/idm/g005_idm_expanded_repeat_context_key_repeat_prior_prefix320k_metrics.json`
- `artifacts/idm/g005_idm_expanded_repeat_context_causal_keyboard_repeat_policy_matrix.json`
- `artifacts/idm/g005_idm_key_press_multiplicity_prefix320k.json`
- `artifacts/idm/g005_idm_key_repeat_count_prior_prefix320k_metrics.json`
- `artifacts/idm/g005_idm_repeat_context_count_prior_rejection.json`

Findings:

- Adding hold-modulo context to the binary repeat prior only improves the
  noncausal keyboard upper-bound from the previous `0.5426` to `0.5430`.
- The target prefix has many duplicate binned key repeats: `16,530` key-press
  token occurrences have count `2` in the first `320,000` target rows.
- A count-aware repeat prior that can emit two `KEY_PRESS_*` tokens still reaches
  only keyboard `0.5463` (`global_hold_mod_ge12_th0.5`), far below the `0.73`
  paper target. It preserves the noncausal state-delta mouse-button result
  (`0.9754`) and previous-motion Pearson (`0.7687/0.7425`), but those are still
  not valid G005 completion evidence and motion remains below the paper target.

Decision: reject tabular held-duration/modulo/count priors. The next branch must
learn a causal latent key-state/repeat-count decoder or use teacher-assisted
sequence supervision; simple context tables are exhausted. Preserve
state-transition button and autoregressive motion as specialist-head ideas only
after replacing their noncausal state/motion sources with trainable causal
predictors.

### 2026-05-28 KST — count-preserving hierarchical exact-set prefix rejected

Implemented an opt-in `keyboard_exact_set_preserve_counts` path so the
streaming IDM exact-set keyboard head can train and decode duplicate key tokens
inside a D2E bin instead of collapsing them to a set. Then ran a bounded 1×H200
prefix gate on reservation `rsv-jeonghunpark-20260528-7524c1` using commit
`bbae584`, 320k train rows, 320k target rows, W&B sidecar logging, and cancelled
the reservation after copying artifacts.

Evidence:

- `src/fdm_d2e/training/streaming_idm.py` — count-preserving stats,
  cache identity, exact-set target generation, and decode-compatible class
  handling via `keyboard_exact_set_preserve_counts`.
- `tests/test_streaming_idm_contract.py` — duplicate-key stats/cache/decode
  regression test.
- `configs/model/idm_streaming_d2e_full_event_state_duration_counted_hierarchical_prefix320k.yaml`
  and `scripts/run_g005_idm_event_state_duration_counted_hierarchical_prefix.sh`.
- `artifacts/idm/g005_idm_event_state_duration_counted_hierarchical_prefix320k_paper_metrics.json`
  — paper-compatible metrics, `status=pass`, 320k aligned rows, zero sequence
  mismatches.
- `outputs/idm_streaming_d2e_full_event_state_duration_counted_hierarchical_prefix320k/metrics.json`
  — strict/local metric artifact.
- `artifacts/idm/g005_idm_event_state_duration_counted_hierarchical_prefix320k_rejection.json`
  — explicit negative decision with split metrics and duplicate-class stats.

Findings:

- Count preservation is real in the train prefix: 499 counted keyboard classes
  vs 413 collapsed classes, 91 duplicate-counted classes, and 15,711 duplicate
  counted training rows; top duplicate is two `KEY_PRESS_87` tokens (3,963
  rows).
- Metrics remain far below paper targets: all-row paper-compatible keyboard
  `0.0898`, mouse-button `0.1507`, Pearson X/Y `0.1206/0.0277`. Heldout splits
  are similarly poor: temporal keyboard `0.0925`, heldout-recording `0.0979`,
  heldout-game `0.0779`.
- It only slightly improves collapsed hierarchical prefix keyboard/button
  (`0.0825/0.1476 -> 0.0898/0.1507`) and badly underperforms the sequence-prior
  motion branch (`0.6426/0.5972` Pearson X/Y), so it is not a promotion path.

Decision: reject count-preserving hierarchical exact-set as a standalone G005
paper-target route. Keep the implementation because it fixes a real label
fidelity bug, but the next branch should combine count-preserving key labels
with a stronger causal visual/motion/sequence teacher or specialist ensemble;
do not launch a full-corpus counted-only 4×H200 run.

### 2026-05-29 KST — video-head masked-diffusion probe rejected; real-video DINO pivot prepared

After quota approval, reserved production 4×H200 reservation
`rsv-jeonghunpark-20260529-579a85` (node 4 GPUs `[1,2,3,4]`, 22:00–01:00 KST)
and ran `g005_idm_temporal_masked_diffusion_luma2_videohead_prefix80k_epoch3`
from commit `7c650aa` with W&B sidecar logging. The run completed successfully
and was then cancelled to avoid idle H200 time.

Evidence:

- `artifacts/cluster/g005_quota_request_status_20260529_after_user_approval.json`
  — approved production quota grant observed (`effective_gpu_quota=8`, 400
  approved GPU-hours).
- `artifacts/cluster/g005_videohead_prefix80k_auto_launch_20260529.json` — pod
  launch summary for `prod-rsv-jeonghunpark-20260529-579a85`.
- `artifacts/idm/g005_idm_temporal_masked_diffusion_luma2_videohead_prefix80k_epoch3_h200_run.json`
  and compact summary — terminal run evidence, `exit_code=0`, GPU monitor pass.
- `outputs/idm_temporal_masked_diffusion_d2e_luma2_videohead_prefix80k_epoch3/paper_metrics.json`
  — paper-compatible metric artifact.
- `artifacts/cluster/g005_videohead_prefix80k_cancel_20260529.json` — reservation
  cancelled after the negative run.

Result: reject this candidate. Paper-compatible metrics were keyboard
`0.010206`, mouse-button `0.005871`, mouse Pearson X/Y `null/null`; strict
no-button FPR was `0.024505`, so only the FPR constraint passed. Candidate
family diagnostics still show the video-token confidence heads collapse toward a
small set of high-frequency tokens instead of ranking exact sparse key/button
identity.

Decision: do not add more calibration-only candidate scorers to this temporal
masked-diffusion branch. The next representation pivot is actual D2E video-frame
conditioning rather than compact `luma16` proxies: `video_idm._VideoFrameStream`
now falls back to OpenCV `VideoCapture` when the production image lacks an
`ffmpeg` binary, and a bounded real-video DINO prefix wrapper has been prepared:

- `configs/model/idm_streaming_d2e_full_frozen_frame_embedding_realvideo_prefix16k.yaml`
- `configs/eval/g005_idm_frozen_frame_embedding_realvideo_prefix16k_paper_metrics.yaml`
- `scripts/run_g005_idm_frozen_frame_embedding_realvideo_prefix16k.sh`

This remains FDM-1-recipe-aligned only as a representation probe (screen-video
encoder + action-token IDM metrics); it is not completion evidence until a
real-video prefix gate beats the D2E paper targets and then scales under the
full hard gates.

## 2026-05-29T23:20 KST — Real-video frozen-DINO prefix16k gate negative

After quota increase approval, a 4×H200 bounded real-video gate ran on `rsv-jeonghunpark-20260529-c93357` / `prod-rsv-jeonghunpark-20260529-c93357`. The initial launch exposed a GPU-idle operational bug: cv2 fallback decoded real videos by seeking every frame. Commit `5391069` fixed this with sequential nearby cv2 decoding plus coarse seeks, and the same reservation was relaunched successfully. Commit `248f176` fixes the shared wrapper run-summary path.

Evidence copied locally:

- `artifacts/idm/g005_idm_frozen_frame_embedding_realvideo_prefix16k_compact_summary.json`
- `artifacts/idm/g005_idm_frozen_frame_embedding_realvideo_prefix16k_paper_metrics.json`
- `artifacts/idm/g005_idm_frozen_frame_embedding_realvideo_prefix16k_run_summary.json`
- `artifacts/idm/g005_idm_frozen_frame_embedding_realvideo_*materialization_summary*.json`
- `artifacts/cluster/g005_realvideo_prefix16k_relaunch_optimized_20260529.json`
- `artifacts/cluster/g005_realvideo_prefix16k_cancel_20260529.json`

Materialization is now operational (`16k` train `26.17s`, `16k` target `25.28s`, real `video` frame source, `dinov2-torchhub`). Metric result is negative: all-target-prefix paper-compatible keyboard key accuracy `0.000368`, mouse-button accuracy `0.0`, mouse Pearson X/Y `0.00946/0.00371`; strict no-button FPR `0.0` from emitting no buttons. This does not beat paper targets, does not cover heldout-recording/game rows in the bounded prefix, and must not checkpoint `G005-g014-idm-full-paper-target`.

### 2026-05-29 KST — balanced real-video raw96 prefix32k launch failed before training

- Reservation/pod: `rsv-jeonghunpark-20260529-b8d1e3` / `prod-rsv-jeonghunpark-20260529-b8d1e3` on 4×H200 node4 GPUs.
- Launch commit: `68a2236`; source materialization succeeded (`32,000` balanced train rows, `24,000` target rows split evenly across temporal / heldout-recording / heldout-game).
- Training failed before first optimizer step with `ValueError: only one element tensors can be converted to Python scalars` in `_maybe_tensorize_features` because `raw_video_feature_storage=tensor` returned a list of frame tensors and the shared tensorization path called `torch.tensor(list_of_tensors)` instead of stacking.
- GPU monitor evidence intentionally records zero utilization for this failed pre-train gate; G005 remains incomplete and no paper-target claim is made.
- Fix direction: stack list-of-tensor features in `_maybe_tensorize_features`, keep the raw-frame FDM-1-shaped config unchanged, then relaunch on the same reservation while quota is available.

### 2026-05-29 KST — balanced real-video raw96 prefix32k terminal negative; train320k follow-up prepared

- Relaunched `g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_prefix32k` on the same `rsv-jeonghunpark-20260529-b8d1e3` reservation after commit `75c25b8` fixed tensorized raw-frame features.
- The run completed with `exit_code=0` and 4×H200 training evidence, but metric status is non-terminal negative: all-row paper-compatible keyboard `0.009311`, mouse-button `0.006849`, mouse Pearson X/Y `0.000154/-0.000441`; strict mouse-button F1 `0.013917`; no-button FPR `0.018676` passes only the FPR gate.
- Evidence copied locally: `artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_prefix32k_h200_{run,compact_summary,gpu_monitor}.json/csv`, `artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_prefix32k_summary.json`, source balanced summaries, and small output metadata under `outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_prefix32k/`.
- Do not checkpoint `G005-g014-idm-full-paper-target` from this run. It proves the actual-D2E-video/FDM-1-shaped trainer is operational but not useful yet.
- Follow-up prepared: a larger real-video `train320k/target24k` branch with a 512-dim/6-layer patch-token masked-diffusion IDM and a distributed raw-frame feature cache so future 4×H200 launches do not duplicate raw-frame decode on every rank.

### 2026-05-30 KST — balanced real-video raw96 train320k terminal negative

Ran `g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_target24k` on the active 4×H200 MLXP pod from commit `d642668`. The run completed with `exit_code=0`, 320k balanced D2E train rows, 24k balanced target rows (temporal/heldout-recording/heldout-game), W&B sidecar logging, and distributed raw-frame feature-cache evidence. It is **not** G005 completion evidence.

Observed all-split paper-compatible metrics: keyboard key accuracy `0.0086006`, mouse-button accuracy `0.0`, strict mouse-button F1 `0.0`, mouse Pearson X/Y `0.0055316/null`, and no-button FPR `0.012024`. Only the no-button FPR gate passed; all paper-target quality gates failed. The model mostly emitted common key press/release tokens plus `MOUSE_DY_Z0`, indicating a candidate ranking/budget failure rather than an operational training failure.

Operational diagnosis: distributed feature-cache hardening worked and reduced duplicated raw-frame decode across ranks. Training used all four H200s, but post-training probability/prediction remained rank0-only, leaving GPUs 1–3 idle during final prediction. Future H200 branches should vectorize/distribute post-checkpoint calibration/prediction before another large run.

### 2026-05-30 KST — stratified calibration prediction sweep remains negative

After the train320k failure, added train-only `temporal_calibration_strategy=stratified_action` so budget calibration no longer samples a near-noop train tail. A prediction-only sweep from the train320k checkpoint over a 5k temporal target prefix used 2k stratified calibration rows, disabled retrieval to avoid recomputing full fit features, and reused raw-video cache support where possible.

Result remains negative: keyboard key accuracy improved only to `0.01486`, mouse-button accuracy/F1 stayed `0.0/0.0`, mouse Pearson X/Y `0.02470/-0.00157`, and no-button FPR `0.0`. The calibration now sees dense positives, but candidate scoring still ranks keyboard/button/mouse movement too poorly for paper targets. Do not promote this sweep to full 24k/full-corpus evaluation.

### 2026-05-30 KST — quota grant remains active; idle continuation cancelled

After the user confirmed the quota increase approval, MLXP API inspection showed
`production_effective_gpu_quota=8` with active grant
`grant-req-1-gpu-quota-increase-20260529-94abef` and no current GPU workload on
the continuation pod. The train320k and stratified prediction sweeps were already
terminal negative, so the 4×H200 continuation reservation was cancelled to avoid
burning approved GPU-hours while the next candidate-scoring redesign is still
local/research work.

Evidence:

- `artifacts/cluster/g005_realvideo_raw96_train320k_continuation_cancel_20260530.json`
  — `rsv-jeonghunpark-20260530-d58f2b` cancellation audit.
- Post-cancel API check: current production reservations `0`; quota remains
  effective at 8 GPUs with approved hours remaining.

G005 remains incomplete. Do not checkpoint `G005-g014-idm-full-paper-target`; the
next GPU reservation should wait for a materially different FDM-1-shaped IDM
candidate and a distributed/vectorized prediction path.

### 2026-05-30 KST — next local pivot: train-heldout candidate-score reranker

Implemented a split-safe confidence-reranker path for the failed train320k raw96
masked-diffusion checkpoint. This keeps the public FDM-1 recipe boundary: the
IDM still produces masked action-token candidates and iterative unmasking, while
a small family-specific logistic confidence scorer is fit only on held-out D2E
train candidate rows to approximate the unpublished FDM-1 confidence ranking
stage. Target labels are never used for fitting or threshold calibration.

New artifacts/code paths:

- `_fit_candidate_score_reranker` and application helpers in
  `src/fdm_d2e/training/temporal_masked_diffusion_idm_trainer.py`.
- Prediction script integration in `scripts/predict_idm_temporal_masked_diffusion.py`.
- Decision-probe config:
  `configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_train320k_candidate_reranker_predict5k.yaml`.
- Wrapper:
  `scripts/run_g005_idm_temporal_raw96_train320k_candidate_reranker_predict5k.sh`.

Validation: `python3 -m py_compile` for the trainer/prediction script,
`bash -n` for the wrapper, and `uv run pytest -q tests/test_masked_diffusion_idm_trainer.py tests/test_training_run_scripts.py tests/test_fdm1_recipe_alignment.py`
passed (`76 passed`). Do not reserve 4×H200 for this yet; first run the 5k
prediction-only probe from the existing train320k checkpoint/cache when a small
reservation or reusable pod is available. If the reranker cannot move keyboard,
button, and mouse metrics materially on 5k, reject it without a full 24k/4×H200
promotion.

### 2026-05-30 KST — reranker probe throughput hardening

The first live 1×H200 reranker prediction process spent its initial minutes in
CPU/raw-video precompute with the GPU allocated but idle, because stratified
calibration rows cause random MKV frame access and the target 5k prefix did not
reuse the existing 24k raw-video feature cache. Local hardening now adds opt-in
prefix feature-cache reuse for prediction diagnostics and reduces the reranker
5k decision probe to 1,024 stratified calibration rows plus a 500-row target
candidate diagnostic cap. This preserves the split-safe train-heldout-only
calibration boundary and should be used before any rerun if the old process does
not reach inference promptly.

### 2026-05-30 KST — candidate reranker rejected; temporal source-offset ensemble prepared

The 1×H200 prediction-only reranker probe from the `train320k` raw96 masked-diffusion checkpoint completed on reservation `rsv-jeonghunpark-20260530-0396ce` and was copied locally before cancellation. It is negative evidence, not G005 completion evidence: 5k target-prefix paper-compatible metrics were keyboard `0.01775568181818182`, mouse-button `0.004032258064516129`, mouse Pearson X/Y `null/null`, strict button F1 `0.0`, and no-button FPR `0.017912291537986413`. The reservation was cancelled at 2026-05-30 03:04 KST to preserve approved GPU-hours.

Evidence:

- `artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_candidate_reranker_predict5k_rejection.json`
- `artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_candidate_reranker_predict5k_compact_summary.json`
- `outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_train320k_candidate_reranker_predict5k/paper_metrics.json`
- `artifacts/cluster/g005_candidate_reranker_predict5k_cancel_20260530.json`

The next local branch stays within the public FDM-1 masked action-token recipe but targets the alignment failure noted by earlier diagnostics: D2E uses temporal-offset next-event prediction, while the exact FDM-1 IDM action-token timing convention is unpublished. The new temporal source-offset ensemble merges candidate tokens predicted at neighboring masked action-token offsets (`[-2,-1,0,1,2]`) for center-bin emission, then uses train-heldout family-budget calibration only. This is not target-label calibration and not an FDM-1 parity claim.

New code/config paths:

- `_temporal_final_probabilities`, `_temporal_candidate_rows_from_probabilities`, and offset-weight helpers in `src/fdm_d2e/training/temporal_masked_diffusion_idm_trainer.py`.
- `configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_train320k_offsetensemble_predict5k.yaml`.
- `scripts/run_g005_idm_temporal_raw96_train320k_offsetensemble_predict5k.sh`.

Validation: `python3 -m py_compile src/fdm_d2e/training/temporal_masked_diffusion_idm_trainer.py scripts/train_idm_temporal_masked_diffusion.py scripts/predict_idm_temporal_masked_diffusion.py`, `uv run pytest -q tests/test_masked_diffusion_idm_trainer.py tests/test_training_run_scripts.py tests/test_fdm1_recipe_alignment.py` (`77 passed`), and `uv run python scripts/validate_fdm1_recipe_alignment.py` (`status=pass`). Do not checkpoint `G005-g014-idm-full-paper-target`; run the offset-ensemble as a bounded prediction probe only, and reject it without a full 4×H200 promotion unless it materially improves all paper-target endpoints.

### 2026-05-30 KST — offsetensemble 5k aborted for throughput; fast1k probe prepared

The first offsetensemble launch (`predict5k`, five source offsets) was aborted after more than 25 minutes because it stayed CPU-bound and wrote zero prediction rows while holding GPU memory. This was a throughput failure, not metric evidence. The live reservation is still useful for a smaller decision probe, so the next branch narrows to three source offsets `[-1,0,1]`, 256 stratified train-heldout calibration rows, 1k target rows, and smaller candidate diagnostics.

New fast probe paths:

- `configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_train320k_offsetensemble_fast1k.yaml`
- `scripts/run_g005_idm_temporal_raw96_train320k_offsetensemble_fast1k.sh`

Validation remains passing (`77` targeted tests and FDM-1 recipe-alignment audit). If fast1k is also negative or too slow, reject temporal offset ensembling and pivot to a more fundamental training/representation change rather than widening the same candidate path.

### 2026-05-31 KST — family-count auxiliary branch prepared

After rejecting the train320k raw96 checkpoint family, candidate reranking, and temporal source-offset ensembling, the next G005 branch targets the observed failure mode directly: exact key/button/mouse candidates often exist but are ranked too low or emitted with a global per-family budget that cannot decide row-local sparse action counts.

Implemented a recipe-faithful family-count auxiliary for the temporal masked-diffusion IDM. The model still denoises masked action-token slots from noncausal video/action-token context, but now learns train-only capped per-family action-token counts (`keyboard`, `mouse_button`, `mouse_move`) and can use those probabilities to gate candidate confidence and row-local family budgets during iterative unmasking. This is an approximation of the unpublished FDM-1 confidence/unmask-count machinery, not target-label calibration and not FDM-1 parity.

Prepared bounded next-probe paths:

- `configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_counthead_prefix32k.yaml`
- `scripts/run_g005_idm_temporal_raw96_patch_axisclass_realvideo_counthead_prefix32k.sh`

Validation before any GPU reservation: `python3 -m py_compile` for the temporal trainer/train/predict scripts, `bash -n` for the wrapper, targeted pytest (`79 passed`), and `uv run python scripts/validate_fdm1_recipe_alignment.py` all pass. Do not checkpoint `G005-g014-idm-full-paper-target` from this implementation alone; run it only as a bounded prefix gate first and promote only if keyboard/button/mouse metrics materially move while no-button FPR remains controlled.
