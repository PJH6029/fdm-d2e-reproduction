#!/usr/bin/env bash
set -euo pipefail

# Durable recovery wrapper for the G005 4xH200 prefix320k promotion launch.
# Reads the MLXP API token from mlxp.md/.env-style variables without printing it.
# This script is intentionally reservation-specific by default but overridable.

REPO_LOCAL="${REPO_LOCAL:-$(pwd)}"
RES_ID="${RES_ID:-rsv-jeonghunpark-20260529-4f61cb}"
PROJECT_ID="${PROJECT_ID:-production}"
NAMESPACE="${NAMESPACE:-p-production}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-/home/top321902/.kube/mlxp/jeonghunpark/debug-kubeconfig.yaml}"
BASE_URL="${BASE_URL:-http://147.46.219.248:8000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_POLLS="${MAX_POLLS:-90}"
MIN_REMAINING_SECONDS="${MIN_REMAINING_SECONDS:-1800}"
REMOTE_BASE="${REMOTE_BASE:-/root/work/code/continuous-gui-poc}"
REMOTE_REPO_PRIMARY="${REMOTE_REPO_PRIMARY:-fdm-d2e-reproduction-g005-lumalong-13023f0}"
REMOTE_REPO_FALLBACK="${REMOTE_REPO_FALLBACK:-fdm-d2e-reproduction}"
MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_luma2_actual_prefix320k_epoch3}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-scripts/run_g005_idm_temporal_luma2_actual_prefix320k_epoch3.sh}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
LOG="${LOG:-$REPO_LOCAL/artifacts/cluster/g005_4xh200_auto_launch_20260529.log}"
SUMMARY="${SUMMARY:-$REPO_LOCAL/artifacts/cluster/g005_4xh200_auto_launch_20260529.json}"

mkdir -p "$(dirname "$LOG")" "$(dirname "$SUMMARY")"
cd "$REPO_LOCAL"
exec >>"$LOG" 2>&1
printf '%s watcher_start reservation=%s script=%s\n' "$(date -Iseconds)" "$RES_ID" "$LAUNCH_SCRIPT"

get_token() {
  python3 - <<'PY'
from pathlib import Path
import os
import re
import sys
for name in ("RESERVATION_API_TOKEN", "MLXP_API_TOKEN"):
    value = os.environ.get(name)
    if value:
        print(value)
        sys.exit(0)
for candidate in (Path("mlxp.md"), Path(".env")):
    if not candidate.exists():
        continue
    text = candidate.read_text(errors="replace")
    for pat in [
        r"Access\s+Token[^:]*:\s*([^\s`]+)",
        r"RESERVATION_API_TOKEN\s*=\s*[\"']?([^\"'\s`]+)",
        r"MLXP_API_TOKEN\s*=\s*[\"']?([^\"'\s`]+)",
        r"Authorization:\s*Bearer\s+([^\s`]+)",
    ]:
        m = re.search(pat, text, re.I)
        if m:
            token = m.group(1).strip().strip("\"'")
            if token and "<redacted>" not in token.lower():
                print(token)
                sys.exit(0)
raise SystemExit("token_not_found")
PY
}

TOKEN="$(get_token)"
DETAIL="/tmp/g005_4xh200_detail_${RES_ID}.json"
POD=""
STATUS=""
END_AT=""
for i in $(seq 1 "$MAX_POLLS"); do
  curl -sS "$BASE_URL/api/projects/$PROJECT_ID/reservations/${RES_ID}" \
    -H "Authorization: Bearer ${TOKEN}" > "$DETAIL" || true
  STATUS=$(python3 - <<'PY' "$DETAIL"
import json, sys
try: print((json.load(open(sys.argv[1])).get('status') or ''))
except Exception: print('')
PY
)
  POD=$(python3 - <<'PY' "$DETAIL"
import json, sys
try: print((json.load(open(sys.argv[1])).get('pod_name') or ''))
except Exception: print('')
PY
)
  END_AT=$(python3 - <<'PY' "$DETAIL"
import json, sys
try: print((json.load(open(sys.argv[1])).get('end_at') or ''))
except Exception: print('')
PY
)
  printf '%s poll=%s status=%s pod=%s end_at=%s\n' "$(date -Iseconds)" "$i" "$STATUS" "$POD" "$END_AT"
  if [[ "$STATUS" == "running" && -n "$POD" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

if [[ "$STATUS" == "running" && -n "$END_AT" ]]; then
  python3 - <<'PY' "$END_AT" "$MIN_REMAINING_SECONDS"
import datetime, sys
end = datetime.datetime.fromisoformat(sys.argv[1])
minimum = float(sys.argv[2])
now = datetime.datetime.now(end.tzinfo)
remaining = (end - now).total_seconds()
print(f"reservation_remaining_seconds={remaining:.0f}")
raise SystemExit(0 if remaining > minimum else 4)
PY
fi

if [[ "$STATUS" != "running" || -z "$POD" ]]; then
  python3 - <<'PY' "$SUMMARY" "$RES_ID" "$STATUS" "$POD" "$END_AT"
import datetime, json, sys
out, res, status, pod, end_at = sys.argv[1:]
json.dump(
    {
        "status": "blocked_pod_not_running",
        "reservation_id": res,
        "reservation_status": status,
        "pod_name": pod,
        "reservation_end_at": end_at,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    },
    open(out, "w"),
    indent=2,
)
PY
  exit 2
fi

# Wait for any old single-GPU actual-luma run in a reused pod/path before reset.
for j in $(seq 1 40); do
  if KUBECONFIG="$KUBECONFIG_PATH" kubectl --request-timeout=60s -n "$NAMESPACE" exec -i "$POD" -- bash -lc "pgrep -af 'g005_idm_temporal_masked_diffusion_luma2_actual_fast80k|run_g005_idm_temporal_luma2_actual_fast80k|train_idm_temporal_masked_diffusion.py --config configs/model/idm_temporal_masked_diffusion_d2e_luma2_actual_fast80k' >/dev/null"; then
    printf '%s old_fast80k_still_running wait=%s\n' "$(date -Iseconds)" "$j"
    sleep 60
  else
    break
  fi
done

REMOTE_OUTPUT="/tmp/g005_4xh200_launch_remote_${RES_ID}.txt"
set +e
KUBECONFIG="$KUBECONFIG_PATH" kubectl --request-timeout=300s -n "$NAMESPACE" exec -i "$POD" -- bash -s > "$REMOTE_OUTPUT" 2>&1 <<REMOTE
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
BASE="$REMOTE_BASE"
cd "\$BASE"
if [[ -d "$REMOTE_REPO_PRIMARY/.git" ]]; then
  REPO="\$BASE/$REMOTE_REPO_PRIMARY"
elif [[ -d "$REMOTE_REPO_FALLBACK/.git" ]]; then
  REPO="\$BASE/$REMOTE_REPO_FALLBACK"
else
  git clone https://github.com/PJH6029/fdm-d2e-reproduction.git "$REMOTE_REPO_FALLBACK"
  REPO="\$BASE/$REMOTE_REPO_FALLBACK"
fi
cd "\$REPO"
if pgrep -af '$MODEL_SLUG|run_g005_idm_temporal_luma2_actual_prefix320k' >/dev/null; then
  echo "prefix320k_already_running"
  exit 0
fi
git fetch origin main
git reset --hard origin/main
mkdir -p outputs/cluster artifacts/idm
if [[ ! -f .env && -f "\$BASE/$REMOTE_REPO_PRIMARY/.env" ]]; then
  cp "\$BASE/$REMOTE_REPO_PRIMARY/.env" .env
fi
NPROC_PER_NODE="$NPROC_PER_NODE" nohup bash "$LAUNCH_SCRIPT" \
  > "artifacts/idm/${MODEL_SLUG}_h200.log" 2>&1 &
PID=\$!
echo "\$PID" > "outputs/cluster/${MODEL_SLUG}.pid"
echo "launched_prefix320k pid=\$PID repo=\$REPO head=\$(git rev-parse --short HEAD) date=\$(date -Iseconds)"
REMOTE
rc=$?
set -e
cat "$REMOTE_OUTPUT"
python3 - <<'PY' "$SUMMARY" "$RES_ID" "$STATUS" "$POD" "$END_AT" "$rc" "$REMOTE_OUTPUT"
import datetime, json, pathlib, sys
out, res, status, pod, end_at, rc, remote = sys.argv[1:]
text = pathlib.Path(remote).read_text(errors="replace")
json.dump(
    {
        "status": "launched" if rc == "0" else "launch_failed",
        "reservation_id": res,
        "reservation_status": status,
        "pod_name": pod,
        "reservation_end_at": end_at,
        "kubectl_exit_code": int(rc),
        "remote_output_tail": text[-4000:],
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    },
    open(out, "w"),
    indent=2,
)
PY
exit "$rc"
