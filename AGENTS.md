# AGENTS.md — D2E/FDM Reproduction Handoff Guide

This repository is a serious research reproduction of the **recipe shape** of FDM-1 on public D2E data. It is not a smoke demo and must not be presented as closed-source FDM-1 parity.

These instructions apply to this repository and all child paths unless a deeper `AGENTS.md` overrides them.

## Operating principles

- Prefer `uv` for dependency sync, Python execution, tests, training scripts, and cluster launch wrappers.
- Commit regularly after coherent, verified milestones. Do not batch substantial work into one huge commit.
- Use the Lore commit protocol for commits: intent-first subject plus useful trailers such as `Tested:`, `Not-tested:`, `Confidence:`, and `Scope-risk:`.
- Preserve artifacts, configs, hashes, dataset fingerprints, prediction files, and reports. Future agents should be able to resume from files, not chat context.
- Do not introduce new dependencies unless the research/verification need is clear and recorded.
- Keep claims evidence-bound. If a metric or harness claim is not supported by a committed artifact, phrase it as future work.

## Non-negotiable claim boundaries

Do **not** claim any of the following unless new evidence explicitly supports it:

- FDM-1 parity or equivalence to the closed-source FDM-1 system.
- Full D2E-scale training. Current primary training used a bounded Shooter64 subset, not the full corpus.
- Non-game-domain, robotics, or car-control transfer.
- Live commercial-game control. Current harness evidence is deterministic repo-local game-adjacent replay.
- Pure target-free/non-transductive all-endpoint FDM scale success. The selected all-endpoint branch has an important calibration caveat below.

## Current completed state

The OMX ultragoal is complete as of commit `322758a`:

- `G001` source/resource prerequisite validation — complete.
- `G002` real D2E ingestion/contracts — complete.
- `G003` training platform / Docker / cluster path — complete.
- `G004` baselines/statistical evaluation — complete.
- `G005` IDM research track — complete.
- `G006` FDM research track — complete.
- `G007` ablation/scaling experiments — complete.
- `G008` harness selection/execution — complete.
- `G009` report/reproducibility package — complete.

Durable state/evidence:

- `.omx/ultragoal/goals.json` and `.omx/ultragoal/ledger.jsonl`
- `docs/final_research_report.md`
- `docs/failure_analysis.md`
- `docs/reproducibility_runbook.md`
- `docs/evidence_index.md`
- `artifacts/reproducibility/package_manifest.json`

## Dataset and training scale actually used

Current primary evidence does **not** consume all of D2E.

- Public D2E-480p inventory: `459` paired recordings, `29` games, about `267h` according to upstream source validation.
- Primary selected subset: Shooter64 / `real_multi_shooter64`.
- Primary subset scale: `64` recordings, `7` games, `6,144` binned windows.
- Split: `4,608` train windows and `1,536` within-recording temporal heldout windows.
- Per-recording cap: `96` windows, 50ms bins, 64×64 feature path.
- This is a bounded real-D2E training/evaluation subset, not full-corpus training and not heldout-recording or heldout-game generalization.

Key source artifacts:

- `configs/data/d2e_real_multi_shooter64.yaml`
- `artifacts/sources/d2e_multi_decode_shooter64_summary.json`
- `outputs/data/real_multi_shooter64/train.jsonl` on the MLXP/PVC training path when regenerated
- `outputs/data/real_multi_shooter64/heldout.jsonl` on the MLXP/PVC training path when regenerated

## Selected IDM evidence

Selected IDM artifact:

- `artifacts/idm/shooter64_surface_motion_selected/`

Important facts:

- Train records: `4,608`; target/eval records: `1,536`.
- Model family: torch MLP IDM with surface-motion features, focal categorical loss, exact-set/softmax button calibration, axis-softmax mouse head.
- It clears Holm-corrected keyboard, mouse-button, and mouse-move Pearson endpoints.
- Mouse scale-ratio distance remains non-significant and is a recorded limitation.
- Mouse-button claims must report precision/F1 and no-button false-positive rate, not just positive-class accuracy.

## Selected FDM evidence and caveat

Selected all-endpoint FDM artifact:

- `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/summary.json`
- `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/checkpoint_metadata.json`
- `configs/model/fdm_bth05_d2e_train_scale_calibrated.yaml`

Important facts:

- FDM label source: IDM pseudo-labels.
- Training examples: `4,608`; target/eval examples: `1,536`.
- It beats predeclared baselines on keyboard, mouse-button, mouse-move Pearson, and mouse scale-ratio distance after Holm correction.
- The endpoint-winning branch uses D2E train-split ground-truth motion-scale calibration and a heldout target-prediction distribution denominator.
- Metadata flags for the selected branch:
  - `calibration_uses_target_ground_truth=false`
  - `calibration_uses_target_prediction_distribution=true`
- Therefore phrase it as a **transductive no-heldout-label target-prediction normalization result**, not as pure IDM-pseudo-label scale success or novel-recording scale generalization.

Strict comparison branch:

- `artifacts/fdm/fdm_bth05_d2e_train_prediction_scale_calibrated_h200/summary.json`
- `configs/model/fdm_bth05_d2e_train_prediction_scale_calibrated.yaml`
- This avoids target prediction-distribution normalization and remains a documented `3/4` endpoint result.

## Evaluation/statistics contract

Primary endpoint config:

- `configs/eval/primary_endpoints.yaml`

Use the existing repo-native metrics/statistics pipeline:

- `src/fdm_d2e/eval/baselines.py`
- `src/fdm_d2e/eval/statistics.py`
- recording-cluster bootstrap
- Holm-Bonferroni correction

Primary endpoints:

1. `keyboard_accuracy` vs `noop`.
2. `mouse_button_accuracy` vs `noop`.
3. `mouse_move_pearson` vs `last_seen_train`.
4. `mouse_move_scale_ratio_distance` vs `last_seen_train`.

For new model claims, include:

- raw metric values,
- baseline/reference,
- delta,
- Holm-adjusted p-value,
- reject/non-reject status,
- recording/game split details,
- exact artifact paths and hashes.

## Ablation/scaling evidence

Ablation/scaling artifacts:

- `artifacts/ablation_scaling/g007_ablation_scaling_summary.json`
- `docs/ablation_scaling.md`
- `scripts/summarize_ablation_scaling.py`

Current summary includes:

- FDM branch/calibration axis: 6 points.
- FDM sweep axes: 84 runs across neural button weighting, neural decode/recall, and KNN retrieval.
- IDM scaling axis: Apex8/Apex16/Apex36/Shooter32/Shooter64.

Future scaling work should prioritize:

- full D2E or much larger multi-game data consumption,
- heldout-recording and heldout-game splits,
- stronger pure-pseudo/self-supervised mouse-scale estimation,
- multi-GPU training that improves throughput rather than only launching dry-run-compatible commands.

## Harness evidence and live-game boundary

Current harness artifact:

- `artifacts/harness/g008_game_harness_eval.json`
- `docs/harness_selection_and_execution.md`
- `configs/harness/g008_game_harness.yaml`
- `src/fdm_d2e/rollout/game_harness.py`

Current result:

- Five deterministic game-adjacent candidates.
- Five install/control probes passing.
- Trained FDM prediction replay passes five tasks across five repo-local environments.

This proves bounded action-sequence stability for D2E-shaped keyboard/mouse token streams. It does **not** prove live commercial-game play.

To upgrade to live commercial-game evidence, future agents need at minimum:

- an explicit target game/version/config/map/task protocol,
- OS-level keyboard/mouse input adapter with focus guard, kill switch, and rate limits,
- live screen capture and closed-loop frame → inference → action → next-frame control,
- latency/FPS/dropped-frame/input logs,
- video plus MCAP-like trace capture,
- baseline comparisons across multiple episodes/seeds,
- offline/single-player or otherwise permissioned environments only; avoid public multiplayer/anti-cheat contexts.

Suggested future files:

- `src/fdm_d2e/rollout/screen_capture.py`
- `src/fdm_d2e/rollout/os_input.py`
- `src/fdm_d2e/rollout/live_game_harness.py`
- `configs/harness/live_<game>.yaml`
- `scripts/run_live_game_harness.py`

## Reproducibility commands

Prefer these commands before claiming completion after edits:

```bash
uv run python -m py_compile scripts/build_repro_package_manifest.py scripts/summarize_ablation_scaling.py scripts/run_game_harness_eval.py src/fdm_d2e/rollout/game_harness.py src/fdm_d2e/training/calibrated_fdm.py
uv run pytest -q
uv run python scripts/summarize_ablation_scaling.py --output-json artifacts/ablation_scaling/g007_ablation_scaling_summary.json --output-md docs/ablation_scaling.md
uv run python scripts/run_game_harness_eval.py --config configs/harness/g008_game_harness.yaml
uv run python scripts/build_repro_package_manifest.py --output artifacts/reproducibility/package_manifest.json
```

Use narrower targeted commands first for small edits, then run the broader suite before committing research claims.

## Cluster / MLXP handoff

- Use the `mlxp-reservation-api` skill for SNUPI MLXP reservation workflows.
- GPU scheduling may be planned autonomously with up to 4× H200 when useful.
- Cluster workflow: edit locally, push, then pull inside the pod PVC path:
  - `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`
- Docker registry username: `pjh6029`; auth has been configured previously.
- Do not commit secrets, tokens, kubeconfigs, or live reservation payloads containing sensitive data.
- Current/sensitive MLXP snapshots should stay ignored; only safe redacted summaries belong in artifacts.

## Documentation map

Read these before changing claims:

- `docs/final_research_report.md` — top-level results and limitations.
- `docs/failure_analysis.md` — failed approaches and why selected caveats remain.
- `docs/reproducibility_runbook.md` — rerun commands and cluster path.
- `docs/evidence_index.md` — hashes and package artifacts.
- `notes/ultragoal-operating-notes.md` — persistent user preferences.

When extending the project, update the relevant docs and regenerate `artifacts/reproducibility/package_manifest.json` so the evidence package remains reproducible.
