#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${PID_FILE:-outputs/cluster/g003_full_compact_accel64.pid}"
LOG_PATH="${LOG_PATH:-artifacts/idm/g003_d2e_full_idm_run_accel64_isolated.log}"
GPU_MONITOR="${GPU_MONITOR:-artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64.csv}"
GPU_MONITOR_META="${GPU_MONITOR_META:-artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor_accel64_attached.json}"
GPU_MONITOR_PID="${GPU_MONITOR_PID:-outputs/cluster/g003_accel64_attached_gpu_monitor.pid}"
WATCHER_PID="${WATCHER_PID:-outputs/cluster/g003_accel64_postrun_watcher.pid}"
WATCHER_LOG="${WATCHER_LOG:-artifacts/idm/g003_accel64_postrun_watcher.log}"
WATCHER_SUMMARY="${WATCHER_SUMMARY:-artifacts/idm/g003_accel64_postrun_watcher_summary.json}"
FINALIZATION_SUMMARY="${FINALIZATION_SUMMARY:-artifacts/idm/g003_accel64_integrated_finalization_summary.json}"

mkdir -p outputs/cluster artifacts/idm artifacts/sources/g003_accel64
if [[ -s "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${existing_pid}" ]] && ps -p "${existing_pid}" >/dev/null 2>&1; then
    echo "g003 accel64 already running pid=${existing_pid} pid_file=${PID_FILE}"
    exit 0
  fi
fi

nohup bash scripts/run_g003_d2e_full_idm_accel64_isolated.sh > "${LOG_PATH}" 2>&1 &
run_pid="$!"
echo "${run_pid}" > "${PID_FILE}"
echo "launched g003 accel64 pid=${run_pid} pid_file=${PID_FILE} log=${LOG_PATH}"

nohup uv run python scripts/attach_g003_gpu_monitor.py \
  --pid-file "${PID_FILE}" \
  --output "${GPU_MONITOR}" \
  --metadata-out "${GPU_MONITOR_META}" \
  --monitor-pid-file "${GPU_MONITOR_PID}" \
  --interval-seconds 30 \
  > artifacts/idm/g003_accel64_attached_gpu_monitor.log 2>&1 &
echo "launched/attached accel64 gpu monitor pid=$! output=${GPU_MONITOR}"

nohup uv run python scripts/watch_g003_then_finalize.py \
  --output "${WATCHER_SUMMARY}" \
  --watcher-pid-file "${WATCHER_PID}" \
  --g003-finalization-summary "${FINALIZATION_SUMMARY}" \
  --split-stats-config configs/eval/g003_split_statistics_accel64.yaml \
  --split-stats-summary artifacts/eval/g003_split_statistical_comparisons_accel64_summary.json \
  --g003-completion-config configs/eval/g003_full_idm_completion_accel64.yaml \
  --g003-audit-output artifacts/idm/g003_full_idm_completion_accel64_audit.json \
  --integrated-run-evidence artifacts/idm/g003_d2e_full_idm_run_accel64.json \
  --idm-summary artifacts/idm/idm_streaming_d2e_full_compact_accel64_summary.json \
  --checkpoint-metadata outputs/idm_streaming_d2e_full_compact_accel64/checkpoint_metadata.json \
  --metrics outputs/idm_streaming_d2e_full_compact_accel64/metrics.json \
  --gpu-monitor "${GPU_MONITOR}" \
  --attached-monitor-metadata "${GPU_MONITOR_META}" \
  --train-run-summary artifacts/idm/g003_d2e_full_idm_4xh200_train_run_accel64.json \
  --shard-root outputs/data/d2e_full_corpus_shards_accel64 \
  --log-dir artifacts/sources/g003_accel64 \
  --data-output-dir outputs/data/d2e_full_corpus_accel64 \
  --idm-output-dir outputs/idm_streaming_d2e_full_compact_accel64 \
  --pid-file "${PID_FILE}" \
  --repair-pid-glob outputs/cluster/g003_accel64_shard_'*'_repair.pid \
  --num-shards 64 \
  > "${WATCHER_LOG}" 2>&1 &
echo "launched accel64 postrun watcher pid=$! summary=${WATCHER_SUMMARY}"
