# Ultragoal Operating Notes

Persistent user preferences and non-negotiable research constraints for the D2E/FDM reproduction ultragoal.

## Workflow preferences

- Use the requested progression: `$deep-interview` → `$ralplan` → `$ultragoal`, with sub-goals implemented by `$ralph` or `$team` when useful.
- Prefer `uv` for Python dependency resolution, script execution, testing, training launch wrappers, and environment workflows whenever practical.
- Commit regularly after coherent, verified milestones; do not batch the ultragoal into one huge final commit.
- Follow the Lore commit protocol for commit messages.
- Keep notes/evidence in repo-local artifacts and OMX state so future sessions can resume without relying on chat context.

## Research bar

- This is serious FDM-1-style reproduction research on the D2E dataset, not a smoke-path proof of concept.
- Do **not** claim FDM-1 parity, non-game-domain transfer, robotics/car transfer, commercial-game live control, or success from weak smoke-only evidence.
- Train/evaluate on real D2E data and preserve reusable checkpoints, configs, dataset fingerprints, logs, predictions, and reproducibility artifacts.
- Success requires trained IDM/FDM models to meaningfully beat smoke/baseline methods under predeclared metrics with a strong statistical bar.
- Trained policies must eventually execute stable action sequences in desktop/game harnesses, not only offline metrics.
- Final package must include method, baseline comparison, ablation/scaling curves, failure analysis, reproducible training pipeline, and trained checkpoints.

## Current ultragoal gates

- Active aggregate Codex/OMX ultragoal covers G001–G009 in `.omx/ultragoal/goals.json`; do not complete the aggregate Codex goal until all stories are complete.
- Current completed gates:
  - `G001-data-universe-audit`: full D2E-480p + D2E-Original universe manifest and storage/license report.
  - `G002-split-leakage-contract`: temporal, heldout-recording, and heldout-game split/leakage contract.
  - `G007-runtime-sdk-adapter`: reusable SDK/action decoder/safety adapter/latency logger/deterministic replay contract only; this is not G008 live-game success.
- Current active gate:
  - `G003-d2e-only-idm`: full-corpus D2E-only IDM extraction/training/evaluation.
- Pending gates:
  - `G004-d2e-only-fdm-4xh200`, `G005-aux-data-best-model`, `G006-evaluation-failure-analysis`, `G007-runtime-sdk-adapter`, `G008-live-game-suite`, `G009-report-repro-package`.
- D2E-only gates must finish before D2E+aux or runtime success claims. D2E+aux may become the primary/best final model, but D2E-only results/ablations remain mandatory and separately reported.
- Strong FDM/IDM evidence should include keyboard, mouse movement, and mouse-button endpoints; mouse-button claims must report precision/F1 and no-button false-positive rate, not only positive-class accuracy.

## Dataset and auxiliary-data policy

- Final success requires full usable D2E consumption across 480p and available original/FHD/QHD sources; audited exclusions are allowed only with retry/failure logs and impact counts.
- The project is not limited to D2E if public gameplay/action-event datasets fit within the 5TiB storage cap and have valid provenance/license/action-event utility.
- D2E+aux may be the primary/best model for final harness and report claims (`d2e_aux_may_be_primary`), provided D2E-only hard gates and comparisons are preserved.
- Latest deep-interview decision: `d2e_aux_may_be_primary`; D2E+aux may be the best/primary final model only after D2E-only hard gates, with separate D2E-only vs D2E+aux ablation evidence.

## Resource / cluster policy

- Use the `mlxp-reservation-api` skill for SNUPI MLXP GPU reservation workflows.
- User authorized cluster GPU reservations/scheduling without further confirmation during this ultragoal; use up to 4×H200 and design multi-GPU-capable training paths.
- Cluster workflow: edit locally, push, then pull in the pod PVC path `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction` before running GPU jobs.
- Docker registry username: `pjh6029`; auth is already configured.

## Current cluster handoff

- Current reservation/pod: `rsv-jeonghunpark-20260521-76e25a` / `prod-rsv-jeonghunpark-20260521-76e25a` in namespace `p-production`.
- Pod repo path: `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`.
- Current G003 full-corpus IDM run path: `NUM_SHARDS=16 bash scripts/run_g003_d2e_full_idm_parallel.sh`.
- Runtime adapter contract evidence: `artifacts/runtime/g007_runtime_replay_adapter_contract.json`; commits `34cddb7` and `e858114`; OMX checkpointed `G007-runtime-sdk-adapter` complete locally.
- Do not checkpoint `G003-d2e-only-idm` complete until all 918 source variants are decoded/merged, streaming IDM training finishes, reports/metrics are validated, and evidence summaries are committed.
