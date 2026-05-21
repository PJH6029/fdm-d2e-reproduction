# Tracked Mirror: Full D2E FDM-Style Ralplan

The canonical planning artifacts are under `.omx/plans/`, which is ignored by git. This tracked mirror records the revised consensus plan state.

## Consensus status

- Planner draft: complete.
- Architect review: `ITERATE`, incorporated.
- First Critic review: `ITERATE`, incorporated.
- Second Critic review: `APPROVE`.
- Execution: approved for `$ultragoal` handoff; implementation should occur only inside the approved ultragoal/subgoal workflow.

## Decision

The plan is **Gated Option B**:

- D2E-only full-corpus lane is the protected scientific lane.
- Aux/runtime lanes may run in parallel only as isolated contract/prototype lanes until D2E-only ingestion, split, convergence, and evaluation gates pass.
- D2E+aux may be the best model only after D2E-only metrics are independently reported.

## Local ignored artifacts created

- `.omx/plans/ralplan-fdm-d2e-full-reproduction.md`
- `.omx/plans/prd-fdm-d2e-full-reproduction.md`
- `.omx/plans/test-spec-fdm-d2e-full-reproduction.md`

## Hard gates

1. Data Universe ADR: exact D2E 480p + original/FHD/QHD sources, revisions, hashes, license metadata, status enum, exclusion logs.
2. Split/Leakage Charter: temporal, heldout-recording, heldout-game, no cross-resolution duplicate leakage.
3. D2E-only gates before aux claims: inventory, train run, convergence curves, endpoint stats, report section.
4. Calibration registry: train-only, transductive no-label, oracle/disallowed.
5. Model fidelity taxonomy: repo-native action FDM vs FDM-style autoregressive/interleaved implementation; no FDM-1 parity.
6. Resource gates: <=5TiB unless approved request; measured 4×H200 throughput/utilization/wall-clock/checkpoint cadence.
7. Runtime safety: open-source/offline games only, no anti-cheat/public multiplayer, focus guard, kill switch, rate limits.
8. Claim taxonomy: D2E-only, D2E+aux, transductive, heldout-recording, heldout-game, runtime replay, live closed-loop.

## Numeric gates

- Inventory: 100% status coverage.
- Convergence: <1% relative validation improvement over 3 consecutive eval checkpoints, or stricter preregistered rule.
- Statistics: claimed endpoint win requires Holm-adjusted p <0.05 where multiple endpoints are tested.
- Live suite: >=3 open-source/offline graphical games, >=3 tasks/scenarios total, >=5 seeds/episodes per task unless stricter protocol applies.
- Runtime latency: record p50/p95; p95 >150ms requires failure analysis.

## Ledger-ready goal draft

G1 data audit -> G2 split/leakage -> G3 D2E-only IDM -> G4 D2E-only FDM + 4×H200 -> G5 aux/best model -> G6 eval/failure -> G7 SDK/adapter -> G8 live game suite -> G9 report/package.

Each goal has dependencies, outputs, done criteria, block criteria, and verification evidence in `.omx/plans/ralplan-fdm-d2e-full-reproduction.md`.

## Approved launch hint

```bash
$ultragoal .omx/plans/ralplan-fdm-d2e-full-reproduction.md
```

## Ultragoal ledger created

A fresh local `.omx/ultragoal` aggregate ledger was created from the approved plan after archiving the older bounded Shooter64 ledger and a transient bad 103-goal auto-split.

Current G1-G9 IDs:

1. `G001-data-universe-audit` — data universe audit.
2. `G002-split-leakage-contract` — split/leakage contract.
3. `G003-d2e-only-idm` — D2E-only IDM full-corpus training.
4. `G004-d2e-only-fdm-4xh200` — D2E-only FDM + 4×H200 scaling.
5. `G005-aux-data-best-model` — auxiliary data integration and best model.
6. `G006-evaluation-failure-analysis` — expanded evaluation/statistics/failure analysis.
7. `G007-runtime-sdk-adapter` — reusable inference SDK and game adapter.
8. `G008-live-game-suite` — live open-source graphical game suite.
9. `G009-report-repro-package` — final report and reproducibility package.

The aggregate Codex objective is: complete approved full-corpus FDM-D2E ultragoal stories G001-G009 in `.omx/ultragoal/goals.json`, preserving D2E-only hard gates before D2E+aux/runtime claims.
