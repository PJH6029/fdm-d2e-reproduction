# FDM-1-style D2E reproduction research contract

**Status:** active contract for the reset ultragoal started on 2026-06-01 KST.  
**Canonical roadmap/spec:** `ROADMAP.md`.
**Durable plan:** `.omx/ultragoal/goals.json` (ignored by Git but authoritative inside this runtime).  
**Progress reports:** `artifacts/reports/`.  
**Gate config:** `configs/eval/fdm1_d2e_research_gates.json`.

`ROADMAP.md` is the canonical roadmap for goal execution. This document is an
operational summary and governance layer; if it omits detail from `ROADMAP.md`,
the roadmap still applies. If it appears to conflict with `ROADMAP.md`, preserve
the user-provided roadmap and update this contract instead of weakening the goal.

## Source anchors

- D2E project page: <https://worv-ai.github.io/d2e/>. The page describes D2E as a desktop/game interaction corpus with OWA Toolkit, a Generalist-IDM, public 480p and FHD/QHD dataset links, and a 50ms/temporal-offset style evaluation context. G002 must pin the actual Hugging Face dataset revision/commit before any training claim.
- FDM-1 public post: <https://si.inc/posts/fdm1/>. The public recipe has three high-level stages: train an IDM on labeled screen recordings, use that IDM to label much larger video data, then train an autoregressive FDM for next-action prediction over keyboard and mouse movement tokens. The post also describes a masked-diffusion, non-causal IDM with iterative confidence unmasking; video compression via a masked/self-supervised objective; interleaved frame/action FDM training; 49-bin mouse movement tokenization; and next-click-position prediction.

These public pages guide recipe shape only. They do **not** disclose enough detail for parity, and this project must never claim closed-source FDM-1 parity.

## Reset decision

Earlier repo artifacts and OMX histories were oriented toward prior D2E-paper-metric or partial-pipeline goals. They remain useful for source-code reuse, provenance, and regression comparisons, but they are **not** completion evidence for this reset unless a new gate explicitly re-audits them against the FDM-1-style contract below.

The active objective is now:

> Reproduce the publicly inferable FDM-1 training recipe shape on full D2E-480p game data, with all-game video-token, IDM, pseudo-label, and FDM stages; baseline and ablation comparisons; full-corpus 4xH200 scaling evidence; failure analysis; stable checkpoint-backed desktop/game action-sequence harness evidence; and a reproducible report package.

## Non-negotiable claim boundaries

- No FDM-1 parity claim.
- No non-game, robotics, car-transfer, or commercial-game-control claim.
- No subset-only or smoke-only success claim.
- D2E-480p full-corpus gates precede broader best-model or harness claims.
- D2E+aux may be explored only after D2E-only evidence is preserved and ablated.
- Harness evidence must be framed as action-sequence execution in a controlled desktop/open-game environment, not proof of general autonomous game solving unless separately measured.

## Active goal map

| Goal | Short name | Required outcome |
| --- | --- | --- |
| G001 | Research reset/governance | This contract, gate config, AGENTS reset notice, and first progress report exist. |
| G002 | D2E universe/splits | Pinned D2E-480p revision, all-game inventory, categories, action stats, and leakage-safe split/scale manifests. |
| G003 | 50ms action-token pipeline | D2E/OWAMcap reader, 20Hz sampling, 50ms bins, K-slot tokenizer/detokenizer, overflow/accounting, alignment validation. |
| G004 | Baselines/metrics | No-op, zero-mouse, previous-action, action-prior/action-only, metric harness, per-game/category reports. |
| G005 | V-JEPA2 video encoders | VE-0/1/2/3 candidates, feature caches, compressor/adaptation configs, throughput and promotion report. |
| G006 | IDM candidates | Non-causal masked-diffusion IDM, causal/prior baselines, future-offset/context/K/mouse/VE ablations, calibration. |
| G007 | Pseudo-label datasets | D_GT, D_PSEUDO_ALL, D_PSEUDO_FILTERED, D_MIX with confidence/entropy/provenance manifests. |
| G008 | FDM candidates | GT/pseudo/filtered/mix FDM, action-only/video-only/game-ID ablations, teacher-forced and logged free-run evaluation. |
| G009 | Full-corpus 4xH200 scaling | Ready-only MLXP H200 reservations, multi-GPU full-corpus runs, utilization evidence, scale/context/model curves. |
| G010 | Evaluation/failure analysis | RQ1-RQ5 synthesis with stats, pseudo/GT ratios, held-out-game generalization, failures, and claim taxonomy. |
| G011 | Harness execution | Checkpoint-backed desktop/open-game action-sequence harness with focus/kill/rate guards and latency/replay/video logs. |
| G012 | Final report/repro package | Final report, runbooks, configs, manifests, checkpoint metadata, package hashes, final audits and review evidence. |

## Reporting cadence

- Write a markdown progress report under `artifacts/reports/` at every coherent milestone and before/after MLXP reservations that materially affect GPU usage.
- Every report must list: active goal, changed files/artifacts, validation evidence, known risks, next action, and claim boundary.
- Keep large/raw data, private kubeconfigs, API tokens, and unredacted reservation payloads out of Git.

## MLXP resource policy

- Use `mlxp.md` only as local operational input; do not copy secrets/tokens into committed files.
- Use debug/production board inspection before reservation when feasible.
- Production H200 reservations are allowed without additional user confirmation by the user directive, but the MLXP skill's live-action guard requires exact-payload confirmation before live production creation. If this conflict appears during execution, prefer a non-production debug reservation or stop before production creation with a clearly recorded blocker rather than violating the skill guard.
- Reserve 4xH200 only when the workload is ready to train/evaluate; cancel when blocked or idle.
- Each GPU run must emit utilization monitor evidence and distinguish CPU/IO preparation from expected GPU compute.

## Verification policy

- For docs/governance changes: `git diff --check`, JSON validation for gate files, and `omx ultragoal status` reconciliation.
- For code changes: targeted tests first, then relevant integration/audit scripts before claims.
- For ML experiments: preserve configs, manifests, hashes, checkpoints, predictions, metrics, monitor CSVs, launch/cancel/detail evidence, and failure logs.
- For final completion: run final verification, anti-slop cleanup on changed files, rerun verification, independent code review, and final quality-gate checkpoint.

## Stop condition

The aggregate Codex goal remains active until all `.omx/ultragoal/goals.json` stories are complete, G012 final quality gates pass, and the final checkpoint is recorded. Intermediate story completion must not call `update_goal`.
