# G003 checkpoint handoff tooling — 2026-06-01 KST

## Target

Prepare the final post-run step for `G003-50ms-action-token-dataset-pipeline` without mutating OMX state prematurely.  The helper creates a checkpoint-ready handoff only after the pod monitor, completion audit, evidence bundle, and `.omx/ultragoal/goals.json` all prove that G003 is ready to checkpoint.

## Added artifacts

- `scripts/build_fdm1_g003_checkpoint_handoff.py`
  - Reads:
    - `artifacts/cluster/fdm1_g003_action_dataset_pod_monitor.json`
    - `artifacts/sources/fdm1_g003_action_dataset_completion_audit.json`
    - `artifacts/sources/fdm1_g003_evidence_bundle_manifest.json`
    - `.omx/ultragoal/goals.json`
  - Requires monitor/audit/bundle `status=pass` and G003 still `in_progress`.
  - Optionally validates a fresh Codex `get_goal` JSON snapshot is still `active` and matches the aggregate ultragoal objective.
  - Emits `artifacts/cluster/fdm1_g003_checkpoint_handoff.json` with the exact `omx ultragoal checkpoint ...` command, but never runs it.
- `tests/test_fdm1_g003_checkpoint_handoff.py`
  - Covers ready handoff, monitor gate failure, Codex objective mismatch, and blocked CLI output.
- `scripts/monitor_g003_fdm1_action_dataset_pod.py`
  - Now invokes the handoff builder after monitor status reaches `pass`.
- `scripts/run_g003_fdm1_action_dataset_pipeline.sh`
  - Now emits the handoff JSON at the end of the pipeline in `--allow-blocked` mode so missing pass evidence remains explicit instead of silent.
- `artifacts/cluster/fdm1_g003_action_dataset_pod_launch_plan.json`
  - Post-launch checks include handoff generation.

## Intended final checkpoint flow after pod run

1. Verify monitor/audit/bundle are pass:

```bash
uv run python scripts/monitor_g003_fdm1_action_dataset_pod.py --refresh-audit --build-bundle-if-pass
```

2. Save a fresh Codex `get_goal` snapshot from the agent/tool context to:

```text
.omx/tmp/fdm1_g003_get_goal_snapshot.json
```

3. Build the handoff with snapshot validation:

```bash
uv run python scripts/build_fdm1_g003_checkpoint_handoff.py \
  --codex-goal-json .omx/tmp/fdm1_g003_get_goal_snapshot.json
```

4. Only if the handoff reports `ready_to_checkpoint`, run the emitted checkpoint command.

## Verification

```bash
uv run python -m py_compile scripts/build_fdm1_g003_checkpoint_handoff.py scripts/monitor_g003_fdm1_action_dataset_pod.py scripts/launch_g003_fdm1_action_dataset_pod.py
uv run pytest tests/test_fdm1_g003_checkpoint_handoff.py tests/test_monitor_g003_fdm1_action_dataset_pod.py tests/test_launch_g003_fdm1_action_dataset_pod.py -q
uv run python scripts/build_fdm1_g003_checkpoint_handoff.py --output /tmp/fdm1_g003_handoff_local.json --allow-blocked
uv run python -m json.tool /tmp/fdm1_g003_handoff_local.json
```

Observed evidence:

- Checkpoint handoff tests: `4 passed`.
- Local handoff currently reports `blocked` because full pod monitor, completion audit, and evidence bundle are absent before the MLXP full-corpus materialization run.

## Claim boundary

This tool is a checkpoint handoff generator only. It does not mutate OMX, does not call Codex goal tools, and does not make G003 complete. G003 completion still requires a passing full D2E-480p action-token materialization run and a fresh active Codex goal snapshot.
