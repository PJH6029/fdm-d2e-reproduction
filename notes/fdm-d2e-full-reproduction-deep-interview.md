# Handoff Note: Full D2E FDM-Style Reproduction Deep Interview

Date: 2026-05-21

This tracked note mirrors the local `.omx/specs/deep-interview-fdm-d2e-full-reproduction.md` artifact because `.omx/` is git-ignored in this repository. Treat the local `.omx` spec as the immediate workflow input when present, and this note as the durable cross-agent/cluster handoff source.

## Workflow

User-requested flow:

```text
deep-interview -> ralplan -> ultragoal -> ralph/team subgoals
```

Recommended next command in an OMX runtime:

```bash
$plan --consensus --direct .omx/specs/deep-interview-fdm-d2e-full-reproduction.md
```

If the local `.omx` spec is absent after a fresh clone, reconstruct it from this note before running ralplan.

## Core objective

Build a serious research reproduction of the open/observable parts of FDM-1-style methodology in the game domain, using D2E plus allowed public auxiliary game-action datasets. This is not a smoke path and must not claim FDM-1 parity.

Reference anchors:
- D2E project page: https://worv-ai.github.io/d2e/
- FDM-1 public blog: https://si.inc/posts/fdm1/

## Clarified requirements

### Full-corpus training bar

- Success requires convergence-level full training: all usable D2E data must be trained to validation saturation/convergence.
- A convenience early stop is not acceptable for final success.
- Multiple MLXP reservations and long walltime are acceptable.
- 4×H200 scaling run is required.

### Dataset/source bar

- D2E 480p alone is not enough for this renewed goal.
- Original/FHD/QHD D2E sources must be included when available.
- Every recording must be classified as usable or excluded.
- Exclusions are allowed only with retry logs, concrete failure reason, and impact summary.

### Auxiliary data policy

- The work is not limited to D2E.
- Public gameplay video/action-event datasets may be used within the 5TiB storage budget if they support inverse-dynamics supervision or action/event labels.
- Final best model and harness success may be D2E+aux.
- D2E-only run/ablation/report remains mandatory so claims are separated.

### Generalization/evaluation bar

Required splits:
- within-recording temporal heldout
- heldout-recording
- heldout-game

Required evidence:
- IDM label-quality analysis
- IDM/FDM vs smoke baselines with strong statistical bar
- ablation/scaling curves
- failure analysis report

### Artifact/runtime bar

Checkpoint plus docs is not sufficient. The deliverable must include:
- checkpoints/configs/tokenizer/action schema/training logs
- reusable real-time inference SDK
- preprocessing and action-decoder interface
- latency logging
- game-ready adapter demo

### Live harness bar

Live open-source graphical-game evidence must be suite-level:
- multiple games/tasks/seeds
- live closed-loop control, not just offline deterministic replay
- statistical improvement vs baseline
- latency/failure logs
- replay/video evidence

### Non-goals and claim boundaries

- No FDM-1 parity claim.
- No non-game-domain scope.
- No robotics/car-transfer scope.
- No weak smoke-only result.
- No commercial-game live-control claim unless later evidence actually supports it.
- No 480p-only full-success claim.

## Decision boundaries for future agents

Agents may decide without further confirmation:
- IDM/FDM architecture variants and training stack details
- dataset mix within 5TiB
- baseline/ablation/scaling matrix
- open-source graphical game targets, subject to feasibility/licensing/safety
- GPU reservation schedule up to 4×H200, per user authorization
- whether each subgoal should use ralph or team

Agents must not silently decide:
- to complete the ultragoal without full usable-D2E convergence evidence
- to exclude recordings without audited retry/failure records
- to claim FDM-1 parity
- to claim commercial-game plug-and-play/live-control beyond the produced SDK/adapter evidence

## Local OMX artifacts from this interview

These are ignored by git but exist in the current worktree:

- `.omx/context/fdm-d2e-full-reproduction-20260520T235132Z.md`
- `.omx/interviews/fdm-d2e-full-reproduction-20260521T000638Z.md`
- `.omx/specs/deep-interview-fdm-d2e-full-reproduction.md`

Final ambiguity: 18% / threshold 20%.
Readiness gates: non-goals explicit, decision boundaries explicit, pressure pass complete, closure audit pass.
