# G003 fail-closed run-path hardening — 2026-06-01 KST

## Target

Move `G003-50ms-action-token-dataset-pipeline` closer to a safe full D2E-480p MLXP/PVC run by tightening fail-closed checks before expensive merge/finalization and preserving the small evidence needed after a pod run.

## Changes

- `scripts/preflight_g003_fdm1_action_dataset_pod.py`
  - Now validates ROADMAP-canonical action-tokenization invariants before launch:
    - 50ms bins / 20fps video stream.
    - default K=8 fixed event slots.
    - required special tokens including `MASK_ACTION`, `NO_ACTION`, `PAD_ACTION`, and `EVENT_OVERFLOW`.
    - compound 49-axis mouse movement bins with 24 increasing positive boundaries.
    - next-click auxiliary target defaults of 1.0s and 32x18 grid.
- `scripts/run_g003_fdm1_action_dataset_sharded_pipeline.sh`
  - Now exits before shard merge when any shard subprocess returns non-zero, instead of allowing downstream merge/finalization work to start after a failed extraction batch.
- `scripts/build_fdm1_g003_pod_evidence_copyback.py`
  - Copyback plan now includes the shard summary, shard logs, and shard PID directory alongside the completion audit/evidence bundle so failed/retried full-corpus runs remain auditable.
- `artifacts/cluster/fdm1_g003_pod_evidence_copyback_plan.json` and `.sh`
  - Regenerated with the expanded small-evidence copyback path list.

## Verification

```bash
bash -n scripts/run_g003_fdm1_action_dataset_sharded_pipeline.sh artifacts/cluster/fdm1_g003_pod_evidence_copyback.sh
uv run python -m py_compile scripts/preflight_g003_fdm1_action_dataset_pod.py scripts/build_fdm1_g003_pod_evidence_copyback.py scripts/merge_d2e_full_corpus_shards.py scripts/launch_g003_fdm1_action_dataset_pod.py scripts/build_fdm1_g003_checkpoint_handoff.py scripts/monitor_g003_fdm1_action_dataset_pod.py scripts/build_fdm1_g003_evidence_bundle.py
uv run pytest tests/test_preflight_g003_fdm1_action_dataset_pod.py tests/test_fdm1_g003_pod_evidence_copyback.py tests/test_run_g003_fdm1_sharded_pipeline_script.py tests/test_merge_d2e_full_corpus_shards.py tests/test_launch_g003_fdm1_action_dataset_pod.py tests/test_fdm1_g003_checkpoint_handoff.py tests/test_monitor_g003_fdm1_action_dataset_pod.py tests/test_fdm1_g003_evidence_bundle.py -q
uv run python -m json.tool artifacts/cluster/fdm1_g003_pod_evidence_copyback_plan.json
```

Observed evidence: `30 passed` locally; copyback plan JSON parses.

## Claim boundary

This is launch/copyback safety hardening only. It does not run full D2E-480p, does not generate G003 completion artifacts, and must not be used to checkpoint G003 complete until the real pod run produces passing completion-audit and evidence-bundle artifacts.

## Checkpoint status after hardening

A guarded checkpoint handoff was rebuilt with the fresh active Codex goal snapshot at `.omx/tmp/fdm1_g003_get_goal_snapshot.json` and remains correctly blocked because the real full-corpus pod evidence is absent:

```bash
uv run python scripts/build_fdm1_g003_checkpoint_handoff.py --codex-goal-json .omx/tmp/fdm1_g003_get_goal_snapshot.json --allow-blocked
```

Observed blocker codes: `monitor_missing`, `completion_audit_missing`, `evidence_bundle_missing`, `monitor_not_pass`, `completion_audit_not_pass`, `evidence_bundle_not_pass`.

## Pod launch self-PID preflight fix

The first live pod launch attempt on reservation `rsv-jeonghunpark-20260601-5c5df9` reached the latest branch, but the sharded pipeline stopped before extraction because the launch wrapper writes its background PID before the pipeline's own preflight. The preflight then interpreted that same PID as an already-active competing G003 run.

Fix:

- `run_g003_fdm1_action_dataset_sharded_pipeline.sh` now detects when `outputs/cluster/fdm1_g003_action_dataset_pipeline.pid` contains its own `$$` and passes `--allow-active-pid` only for that self-PID case.
- This preserves duplicate-run protection in the outer launch wrapper while allowing the wrapped pipeline to pass its internal preflight.

Verification:

```bash
bash -n scripts/run_g003_fdm1_action_dataset_sharded_pipeline.sh
uv run pytest tests/test_run_g003_fdm1_sharded_pipeline_script.py tests/test_preflight_g003_fdm1_action_dataset_pod.py -q
```

Observed evidence: `7 passed` locally.
