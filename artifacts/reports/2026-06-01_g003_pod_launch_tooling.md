# G003 pod launch tooling progress — 2026-06-01 KST

## Target

Move G003 closer to full-corpus action-token materialization without starting an unconfirmed live MLXP reservation.  The new helper prepares the exact in-pod launch path for the ROADMAP-canonical 50ms action-token dataset pipeline after a reservation exists.

## Added artifacts

- `scripts/launch_g003_fdm1_action_dataset_pod.py`
  - Writes an audited JSON launch plan and executable shell launcher.
  - Defaults to dry-run/write-only mode.
  - Refuses `--execute` unless the process is inside an MLXP Kubernetes pod workspace (`KUBERNETES_SERVICE_HOST` plus the pod repo path).
  - Refuses duplicate pipeline launches when the existing pid file is still active unless `--replace-existing` is supplied.
  - Records a pod launch context with branch, git head, pid path, log path, and claim boundary.
- `artifacts/cluster/fdm1_g003_action_dataset_pod_launch_plan.json`
  - Generated launch plan for `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction` on branch `research/fdm1-d2e-ultragoal`.
- `artifacts/cluster/fdm1_g003_action_dataset_pod_launch.sh`
  - Generated shell script for the reserved pod.
- `tests/test_launch_g003_fdm1_action_dataset_pod.py`
  - Unit/CLI coverage for command generation, env validation, default write-only behavior, and execute refusal outside the pod.

## Intended pod command after an approved/reserved MLXP pod exists

```bash
cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
git fetch origin research/fdm1-d2e-ultragoal
git checkout research/fdm1-d2e-ultragoal
git pull --ff-only
export PATH="$HOME/.local/bin:$PATH"
uv run python scripts/launch_g003_fdm1_action_dataset_pod.py --execute
```

The generated launch shell then runs:

```bash
bash scripts/run_g003_fdm1_action_dataset_pipeline.sh \
  > artifacts/logs/fdm1_g003_action_dataset_pipeline.log 2>&1 &
```

and writes the pid to:

```text
outputs/cluster/fdm1_g003_action_dataset_pipeline.pid
```

## Verification

```bash
uv run python -m py_compile scripts/launch_g003_fdm1_action_dataset_pod.py
uv run pytest tests/test_launch_g003_fdm1_action_dataset_pod.py -q
uv run python scripts/launch_g003_fdm1_action_dataset_pod.py
bash artifacts/cluster/fdm1_g003_action_dataset_pod_launch.sh  # expected local refusal with rc=2
```

Observed evidence:

- `tests/test_launch_g003_fdm1_action_dataset_pod.py`: `6 passed`.
- Local generated shell refused outside Kubernetes with: `refusing: this launch script must run inside the MLXP Kubernetes pod`.

## Claim boundary

This is launch readiness only. G003 remains incomplete until the full D2E-480p materialization run completes, `validate_fdm1_g003_action_dataset_completion.py` reports pass, the evidence bundle is built, and the OMX ultragoal ledger is checkpointed with a fresh Codex goal snapshot.

## Monitor/final-status hardening

Added after the initial pod launch helper:

- `scripts/monitor_g003_fdm1_action_dataset_pod.py`
  - Reads the launch pid/log paths and reports `running`, `incomplete`, `failed_or_interrupted`, `audit_pass_bundle_missing`, or `pass`.
  - Tails the launch log and surfaces fatal patterns such as `Traceback`, `RuntimeError`, `No space left`, `Killed`, and CUDA OOM.
  - Reads the G003 completion audit and evidence-bundle manifest; it only reports full `pass` when both are pass.
  - Emits artifact existence/size/hash evidence while avoiding expensive hashes for large JSONL payloads.
  - Supports `--refresh-audit` and `--build-bundle-if-pass` for post-run collection.
- `scripts/run_g003_fdm1_action_dataset_pipeline.sh`
  - Now runs the monitor at the end of a successful pipeline after audit and evidence bundle creation.
- `artifacts/cluster/fdm1_g003_action_dataset_pod_launch_plan.json`
  - Post-launch checks now include the monitor command.

Additional verification:

```bash
uv run python -m py_compile scripts/monitor_g003_fdm1_action_dataset_pod.py scripts/launch_g003_fdm1_action_dataset_pod.py
uv run pytest tests/test_monitor_g003_fdm1_action_dataset_pod.py tests/test_launch_g003_fdm1_action_dataset_pod.py -q
uv run python scripts/monitor_g003_fdm1_action_dataset_pod.py --output /tmp/fdm1_g003_monitor_local.json
uv run python -m json.tool /tmp/fdm1_g003_monitor_local.json
```

Observed evidence:

- Monitor/launcher tests: `10 passed`.
- Local monitor status is currently `incomplete`, as expected before the MLXP full-corpus materialization run.
