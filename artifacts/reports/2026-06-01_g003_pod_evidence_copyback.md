# G003 pod evidence copyback tooling — 2026-06-01 KST

## Target

After the MLXP full-corpus G003 run, copy only small evidence artifacts back into the local repo while leaving large action-slot JSONL packs on the PVC. This keeps the checkpoint evidence auditable without accidentally committing raw full-corpus outputs.

## Added artifacts

- `scripts/build_fdm1_g003_pod_evidence_copyback.py`
  - Reads `configs/eval/fdm1_g003_action_dataset_completion.yaml`.
  - Excludes artifact keys listed in `omit_sha256_artifact_keys` such as `action_slots` and split JSONL packs.
  - Includes small completion evidence, pod preflight/monitor/handoff JSON, finalization summaries, evidence bundle, alignment report, and launch log.
  - Emits both JSON plan and executable shell script.
- `artifacts/cluster/fdm1_g003_pod_evidence_copyback_plan.json`
  - Template plan using pod placeholder `REPLACE_WITH_POD_NAME`.
- `artifacts/cluster/fdm1_g003_pod_evidence_copyback.sh`
  - Template `kubectl exec ... tar` copyback command preserving repo-relative paths.
- `tests/test_fdm1_g003_pod_evidence_copyback.py`
  - Verifies large JSONL packs are excluded, output-hash roles are retained, and shell generation uses kubectl/tar.

## Intended use after pod run

Replace the pod placeholder with the actual reservation pod name and optionally add `--kubeconfig`:

```bash
uv run python scripts/build_fdm1_g003_pod_evidence_copyback.py \
  --pod <actual-pod-name> \
  --kubeconfig /path/to/kubeconfig.yaml

bash artifacts/cluster/fdm1_g003_pod_evidence_copyback.sh
```

Then run:

```bash
uv run python scripts/monitor_g003_fdm1_action_dataset_pod.py \
  --output artifacts/cluster/fdm1_g003_action_dataset_pod_monitor.local.json
uv run python scripts/build_fdm1_g003_checkpoint_handoff.py --allow-blocked
```

## Verification

```bash
uv run python -m py_compile scripts/build_fdm1_g003_pod_evidence_copyback.py
uv run pytest tests/test_fdm1_g003_pod_evidence_copyback.py -q
uv run python scripts/build_fdm1_g003_pod_evidence_copyback.py \
  --pod REPLACE_WITH_POD_NAME \
  --output artifacts/cluster/fdm1_g003_pod_evidence_copyback_plan.json \
  --shell-out artifacts/cluster/fdm1_g003_pod_evidence_copyback.sh
uv run python -m json.tool artifacts/cluster/fdm1_g003_pod_evidence_copyback_plan.json
```

Observed evidence:

- Copyback tests: `3 passed`.
- Generated template copies 18 small evidence paths and excludes 9 large action-slot JSONL artifacts.

## Claim boundary

This is a copyback plan only. It does not reserve GPUs, does not run the pod workload, does not prove G003 completion, and does not mutate OMX state.
