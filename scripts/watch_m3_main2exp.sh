#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/watch_m3_main2exp.sh [run_name] [config_path]
#
# This watchdog ensures the M3 expansion pipeline stays alive.
# If the pipeline exits, it relaunches in resume mode and records events.

RUN_NAME="${1:-m3_main2exp_v2}"
CONFIG_PATH="${2:-configs/experiment_config.mac_m3_main2_expansion.yaml}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
STATE_DIR="$ROOT_DIR/run_state"
PIPELINE_LOG="$LOG_DIR/${RUN_NAME}_live.log"
WATCHDOG_LOG="$LOG_DIR/${RUN_NAME}_watchdog.log"
PID_FILE="$STATE_DIR/${RUN_NAME}.pid"
STATUS_FILE="$STATE_DIR/${RUN_NAME}.status"
LAST_CHECK_FILE="$STATE_DIR/${RUN_NAME}.last_check"

CONDA_BIN="${CONDA_BIN:-/opt/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-dino311}"
DATASET_DIR="${DATASET_DIR:-$HOME/Downloads}"
DINO_WM_DEVICE="${DINO_WM_DEVICE:-mps}"
CHECK_EVERY_SECONDS="${CHECK_EVERY_SECONDS:-300}"

mkdir -p "$LOG_DIR" "$STATE_DIR"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

is_alive() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  if ! ps -p "$pid" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

find_worker_pid() {
  # Prefer active python workers for this run name.
  local pid=""
  pid="$(pgrep -f "python -m experiments.make_paired_targets .*--run-name ${RUN_NAME}" | head -n 1 || true)"
  if [[ -n "$pid" ]]; then
    echo "$pid"
    return 0
  fi
  pid="$(pgrep -f "python -m experiments.run_wall_grid .*--run-name ${RUN_NAME}" | head -n 1 || true)"
  if [[ -n "$pid" ]]; then
    echo "$pid"
    return 0
  fi
  pid="$(pgrep -f "python -m experiments.dino_runner .*--run-name ${RUN_NAME}" | head -n 1 || true)"
  if [[ -n "$pid" ]]; then
    echo "$pid"
    return 0
  fi
  # Fallback to any launcher process with this run name.
  pid="$(pgrep -f "run_main2_expansion_pipeline_m3.sh .* ${RUN_NAME}" | head -n 1 || true)"
  if [[ -n "$pid" ]]; then
    echo "$pid"
    return 0
  fi
  return 1
}

start_pipeline() {
  echo "[$(timestamp)] starting pipeline: $RUN_NAME" | tee -a "$WATCHDOG_LOG"
  (
    cd "$ROOT_DIR"
    export DATASET_DIR DINO_WM_DEVICE PYTHONUNBUFFERED=1
    nohup "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
      bash scripts/run_main2_expansion_pipeline_m3.sh "$CONFIG_PATH" "$RUN_NAME" \
      >> "$PIPELINE_LOG" 2>&1 &
    echo $! > "$PID_FILE"
  )
  local pid
  pid="$(cat "$PID_FILE")"
  echo "[$(timestamp)] pid=$pid" | tee -a "$WATCHDOG_LOG"
  echo "RUNNING pid=$pid started_at=$(timestamp)" > "$STATUS_FILE"
}

while true; do
  echo "$(timestamp)" > "$LAST_CHECK_FILE"
  pid=""
  worker_pid=""
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" || true)"
  fi

  worker_pid="$(find_worker_pid || true)"
  if [[ -n "$worker_pid" ]]; then
    echo "$worker_pid" > "$PID_FILE"
    echo "RUNNING pid=$worker_pid checked_at=$(timestamp)" > "$STATUS_FILE"
  elif is_alive "$pid"; then
    echo "RUNNING pid=$pid checked_at=$(timestamp)" > "$STATUS_FILE"
  else
    echo "[$(timestamp)] detected dead/missing process pid=${pid:-none}" | tee -a "$WATCHDOG_LOG"
    start_pipeline
  fi

  sleep "$CHECK_EVERY_SECONDS"
done
