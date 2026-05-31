# G001 progress report — research reset and governance

**Date:** 2026-06-01 KST  
**Branch:** `research/fdm1-d2e-ultragoal`  
**Active OMX story:** `G001-research-reset-and-governance`  
**Aggregate Codex goal:** durable ultragoal plan in `.omx/ultragoal/goals.json`.

## Outcome

The repo now has a clean reset contract for the new FDM-1-style D2E objective. Previous D2E-paper-metric-oriented artifacts are explicitly treated as provenance/source-code references only, not terminal success evidence for the reset.

## Evidence created

- `.omx/ultragoal/brief.md`, `.omx/ultragoal/goals.json`, `.omx/ultragoal/ledger.jsonl` — runtime ultragoal plan with 12 stories.
- `docs/fdm1_d2e_research_contract.md` — active human-readable research contract.
- `configs/eval/fdm1_d2e_research_gates.json` — machine-readable gate map.
- `AGENTS.md` — reset notice for future agents.
- `artifacts/reports/2026-06-01_g001_research_reset.md` — this report.

## Source anchors checked

- D2E project page: <https://worv-ai.github.io/d2e/>.
- FDM-1 public post: <https://si.inc/posts/fdm1/>.

Source interpretation is deliberately conservative: the public pages define recipe shape and dataset targets only; they do not support parity claims.

## Claim boundary

No model/dataset/training success is claimed by G001. This is governance only. G002 must pin the live D2E-480p dataset revision and produce all-game manifests before data-pipeline claims.

## Next action

Start G002: inspect D2E dataset/code paths, pin dataset revision, define all-game inventory and leakage-safe split manifests. No H200 production reservation is needed until data preparation and GPU-ready training commands exist.
