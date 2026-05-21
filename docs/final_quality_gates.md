# Final Quality Gate Audit

The final G001–G009 ultragoal must not be completed from partial evidence,
historical bounded Shooter64 artifacts, deterministic replay smoke, or green unit
tests alone. The authoritative completion audit is:

- Config: `configs/eval/final_quality_gates.yaml`
- Validator: `scripts/validate_final_quality_gates.py`
- Module: `src/fdm_d2e/reporting/quality_gates.py`
- Current audit artifact: `artifacts/reproducibility/final_quality_gate_audit.json`

Run it with:

```bash
uv run python scripts/validate_final_quality_gates.py
```

During active work, record the current incomplete state without failing the shell:

```bash
uv run python scripts/validate_final_quality_gates.py --allow-fail
```

## What it enforces

The audit checks, without mutating OMX/Codex goal state:

1. `.omx/ultragoal/goals.json` exists and every configured story is `complete`.
2. G001/G002 source and split manifests exist.
3. G003 full-corpus IDM artifacts exist: decode summary, merged train/eval JSONL,
   checkpoint, pseudolabels, predictions, metrics, label-quality report, and
   statistical comparison. The IDM checkpoint/run metadata must also prove
   D2E-only provenance (`source_namespace=d2e_full_corpus`, data-universe and
   split-contract artifacts present), 4×H200 distributed training evidence
   (`distributed.enabled=true`, `world_size=4`, run `exit_code=0`), full
   decode coverage (`selected_recording_variants=918`, `num_shards=16`, no
   failures), per-split statistical summaries for temporal, heldout-recording,
   and heldout-game splits, and `g003_full_idm_completion_audit.status=pass`.
4. G004 D2E-only FDM artifacts exist: checkpoint, predictions, metrics,
   statistical comparison, convergence report, split-stat summaries,
   `g004_full_fdm_completion_audit.status=pass`, and 4×H200 run evidence. The
   FDM metadata must prove IDM-pseudolabel training, no oracle ground-truth
   control, D2E-only provenance, source-IDM metadata linkage, distributed
   world-size/run-exit evidence, GPU-monitor coverage for all expected H200
   indices, and per-split statistical summaries for temporal,
   heldout-recording, and heldout-game splits.
5. G005 aux artifacts exist and remain separated from D2E-only namespaces; `g005_aux_namespace_manifest.completion_ready=true`, byte-identical D2E eval-manifest hashes, source-specific action-head namespaces, `g005_aux_completion_audit.status=pass`, and `d2e_aux_ablation_summary.status=pass` are required before D2E+aux can be primary/best.
6. G006 final endpoint statistics, failure-analysis, claim-taxonomy, readiness,
   final artifact-build summary, and `g006_completion_audit` artifacts exist and report `status == pass`;
   G003/G004/G005 must be complete, D2E+aux comparison must be `claimable`, live-suite claims must remain
   `not_claimed_until_g008`, and claimable/documented claims must carry evidence paths.
7. G007 runtime SDK adapter evidence remains present and `g007_completion_audit.status=pass`.
8. G008 live open-source graphical-game evidence validation exists and has
   `quality_gate.status == pass`, and `g008_live_suite_completion_audit.status=pass`; protocol readiness does not count.
9. G009 final report, evidence index, reproducibility runbook, package manifest,
   claim-boundary audit, final-quality audit with `status == pass`, and
   `g009_completion_audit.status=pass` exist.
10. Required configured artifacts that already exist are represented in the
    reproducibility package manifest.
11. The claim-boundary audit reports `status == pass`.

## Current expected state

While G003 is still extracting/training, the audit is expected to fail. That is
useful: it enumerates exactly which gates and artifacts remain unproven and
prevents accidental aggregate goal completion.

The aggregate Codex goal should only be completed after this audit passes on
fresh artifacts and OMX checkpoints for all G001–G009 are complete.

Story-level completion audits are configured as pre-checkpoint evidence gates
(`require_goal_checkpoint_complete=false`) so they can pass on real artifacts
before the corresponding OMX checkpoint is written. They still report the
current story status. The final quality gate is the layer that requires every
story status to be `complete`.
