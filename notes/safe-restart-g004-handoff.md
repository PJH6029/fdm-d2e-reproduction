# Safe restart handoff — G004 D2E-only FDM 4×H200
This file is the restart handoff for the next Codex agent/thread. It is non-terminal evidence only. Do **not** checkpoint G004 complete from this file.
## Why restart
- The current Codex `get_goal` tool is failing with `no such table: thread_goals`.
- OMX ultragoal state is still durable under `.omx/ultragoal/`, so restart the Codex thread rather than mutating goal state here.

## Current ultragoal state
- OMX aggregate plan: `complete=4`, `pending=5`, `failed=0`, `aggregateComplete=false`.
- Complete: `G001-data-universe-audit`, `G002-split-leakage-contract`, `G003-d2e-only-idm`, `G007-runtime-sdk-adapter`.
- Current live work: `G004-d2e-only-fdm-4xh200` is still ledger `pending`, but the MLXP pod run is active.
- Pending after G004: `G005`, `G006`, `G008`, `G009`.

## Live pod/process state at snapshot
- Snapshot artifact: `artifacts/fdm/g004_safe_restart_live_snapshot.json`
- Captured UTC: `2026-05-22T23:19:42Z`
- Reservation: `rsv-jeonghunpark-20260521-76e25a`
- Pod: `prod-rsv-jeonghunpark-20260521-76e25a` in namespace `p-production`
- Pod repo path: `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`
- Pod git head at launch/run: `c090d6a81f28239096aad649d9f3809b26d86da0`
- Parent PID: `260000` (`outputs/cluster/g004_d2e_full_fdm_4xh200.pid`)
- Watcher status: `waiting_active_parent`; elapsed seconds at snapshot: `28622.795221567154`

Relevant process roles observed:
```text
 260000       1 S     0.0  0.0    07:57:48 bash scripts/run_g004_d2e_full_fdm_4xh200.sh
```
```text
 260010  260000 S     0.0  0.0    07:57:48 bash scripts/run_g004_d2e_full_fdm_4xh200.sh
```
```text
 260083       1 Sl    0.0  0.0    07:57:47 uv run python scripts/watch_g004_then_finalize.py --replace-existing-watcher --poll-seconds 60 --pid-file outputs/cluster/g004_d2e_full_fdm_4xh200.pid --watcher-pid-file outputs/cluster/g004_postrun_watcher.pid
```
```text
 260088  260083 S     0.0  0.0    07:57:47 /mnt/ddn/prod-runs/jeonghunpark/code/continuous-gui-poc/fdm-d2e-reproduction/.venv/bin/python3 scripts/watch_g004_then_finalize.py --replace-existing-watcher --poll-seconds 60 --pid-file outputs/cluster/g004_d2e_full_fdm_4xh200.pid --watcher-pid-file outputs/cluster/g004_postrun_watcher.pid
```
```text
 261469  260010 Sl    0.0  0.0    06:43:21 uv run torchrun --standalone --nproc-per-node=4 scripts/train_fdm_streaming.py --config configs/model/fdm_streaming_d2e_full_compact.yaml
```
```text
 261472  261469 Sl    0.3  0.0    06:43:21 /mnt/ddn/prod-runs/jeonghunpark/code/continuous-gui-poc/fdm-d2e-reproduction/.venv/bin/python3 /mnt/ddn/prod-runs/jeonghunpark/code/continuous-gui-poc/fdm-d2e-reproduction/.venv/bin/torchrun --standalone --nproc-per-node=4 scripts/train_fdm_streaming.py --config configs/model/fdm_streaming_d2e_full_compact.yaml
```
```text
 261541  261472 Rsl  87.2  0.0    06:43:09 /mnt/ddn/prod-runs/jeonghunpark/code/continuous-gui-poc/fdm-d2e-reproduction/.venv/bin/python3 -u scripts/train_fdm_streaming.py --config configs/model/fdm_streaming_d2e_full_compact.yaml
```
```text
 261542  261472 Ssl   0.0  0.0    06:43:09 /mnt/ddn/prod-runs/jeonghunpark/code/continuous-gui-poc/fdm-d2e-reproduction/.venv/bin/python3 -u scripts/train_fdm_streaming.py --config configs/model/fdm_streaming_d2e_full_compact.yaml
```
```text
 261543  261472 Ssl   0.0  0.0    06:43:09 /mnt/ddn/prod-runs/jeonghunpark/code/continuous-gui-poc/fdm-d2e-reproduction/.venv/bin/python3 -u scripts/train_fdm_streaming.py --config configs/model/fdm_streaming_d2e_full_compact.yaml
```
```text
 261544  261472 Ssl   0.0  0.0    06:43:09 /mnt/ddn/prod-runs/jeonghunpark/code/continuous-gui-poc/fdm-d2e-reproduction/.venv/bin/python3 -u scripts/train_fdm_streaming.py --config configs/model/fdm_streaming_d2e_full_compact.yaml
```
```text
 262031  262017 S     0.0  0.0       00:00 grep -E PID|run_g004|torchrun|train_fdm_streaming|watch_g004
```

## Stage interpretation
- Full train-core IDM pseudo-label generation for G004 is complete: `19,211,006` rows.
- The pod is inside `torchrun`, but actual GPU model training has **not** started yet at snapshot.
- Rank 0 is CPU/IO-bound, materializing FDM train/target JSONL + shards. Ranks 1–3 are mostly barrier-waiting. GPU util can be 0% in this stage.
- `fdm_streaming_split_summary.json`, `training_cache_manifest.json`, `train_history.json`, `summary.json`, and `g004_d2e_full_fdm_4xh200_run.json` were still missing at snapshot.

Key artifacts at snapshot:
- `outputs/idm_streaming_d2e_full_compact/fdm_train_core_pseudolabels/pseudolabels.jsonl`: exists=True bytes=11638487567 mtime=2026-05-22T16:36:17Z
- `outputs/idm_streaming_d2e_full_compact/fdm_train_core_pseudolabels/predictions.jsonl`: exists=True bytes=7613175080 mtime=2026-05-22T16:36:17Z
- `artifacts/idm/idm_streaming_d2e_full_compact_fdm_train_core_pseudolabels_summary.json`: exists=True bytes=19866 mtime=2026-05-22T16:36:22Z
- `outputs/fdm_streaming_d2e_full_compact/fdm_train_pseudolabeled_records.jsonl`: exists=True bytes=400298678216 mtime=2026-05-22T23:04:21Z
- `outputs/fdm_streaming_d2e_full_compact/fdm_target_ground_truth_records.jsonl`: exists=True bytes=21978497976 mtime=2026-05-22T23:19:43Z
- `outputs/fdm_streaming_d2e_full_compact/fdm_streaming_split_summary.json`: exists=False bytes=0 mtime=
- `outputs/fdm_streaming_d2e_full_compact/torch_model/train_history.json`: exists=False bytes=0 mtime=
- `outputs/fdm_streaming_d2e_full_compact/summary.json`: exists=False bytes=0 mtime=
- `artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json`: exists=False bytes=0 mtime=

## Safe restart procedure for next agent
1. Open a fresh Codex thread in this same repo/worktree.
2. Read `AGENTS.md`, `.omx/ultragoal/goals.json`, `.omx/ultragoal/ledger.jsonl`, `notes/g004-mlxp-run-status.md`, and this file.
3. Run `omx ultragoal status --json`. Expect G004 still pending unless another agent checkpointed it.
4. Call `get_goal` in the new thread. If no active goal exists, create the aggregate objective from `.omx/ultragoal/goals.json` (`codexObjective`). Do not call `update_goal complete` for G004; aggregate completion is only for final G009.
5. Inspect the pod before doing anything else. Do **not** launch a duplicate G004 if parent PID is still alive.

Suggested pod inspection command:
```bash
KCFG=/home/top321902/.kube/mlxp/jeonghunpark/debug-kubeconfig.yaml
NS=p-production
POD=prod-rsv-jeonghunpark-20260521-76e25a
KUBECONFIG="$KCFG" kubectl -n "$NS" exec -i "$POD" -- bash -s <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
date -Iseconds
cat outputs/cluster/g004_d2e_full_fdm_4xh200.pid 2>/dev/null || true
pgrep -af 'run_g004|torchrun|train_fdm_streaming|watch_g004' || true
python - <<'PY'
import json, pathlib
p=pathlib.Path("artifacts/fdm/g004_postrun_watcher_summary.json")
print(json.dumps(json.load(p.open()), indent=2)[:4000] if p.exists() else "watcher missing")
PY
for p in \
 outputs/fdm_streaming_d2e_full_compact/fdm_streaming_split_summary.json \
 outputs/fdm_streaming_d2e_full_compact/torch_model/train_history.json \
 outputs/fdm_streaming_d2e_full_compact/summary.json \
 artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json \
 artifacts/fdm/g004_d2e_full_fdm_finalization_summary.json \
 artifacts/fdm/g004_full_fdm_completion_audit.json; do
  [ -e "$p" ] && stat -c "%n bytes=%s mtime=%y" "$p" || echo "missing $p"
done
REMOTE
```

## If parent is still alive
- Let it continue unless you intentionally decide to optimize/restart the G004 run.
- Monitor for these stage transitions: split summary appears -> training cache manifest appears -> `train_history.json` grows -> `summary.json` + run summary appear -> watcher finalizes.
- Do not checkpoint G004 until `artifacts/fdm/g004_d2e_full_fdm_finalization_summary.json` is pass and `artifacts/fdm/g004_full_fdm_completion_audit.json` has `status=pass`, `error_count=0`.

## If parent has exited
- Read `artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json`. If `exit_code != 0`, diagnose and do not checkpoint complete.
- If finalization/audit pass, pull small artifacts back locally, rebuild `artifacts/reproducibility/package_manifest.json`, commit with Lore protocol, then checkpoint G004 using a fresh `get_goal` snapshot in the new thread.

## If deciding to restart the pod run itself
- This is separate from safe Codex-thread restart and should be deliberate. Preserve/backup current outputs first.
- Kill only the G004 parent tree on pod if needed; do not kill unrelated processes. Current parent PID is in `outputs/cluster/g004_d2e_full_fdm_4xh200.pid`.
- Prefer implementing materialization/cache/prediction parallelization before relaunch, because current GPU util is 0% during CPU/IO preprocessing.
