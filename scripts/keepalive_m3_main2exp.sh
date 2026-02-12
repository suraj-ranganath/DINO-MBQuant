#!/usr/bin/env bash
set -u

RUN_NAME="${1:-m3_main2exp_v2}"
CONFIG_PATH="${2:-configs/experiment_config.mac_m3_main2_expansion.yaml}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
STATE_DIR="$ROOT_DIR/run_state"
PIPELINE_LOG="$LOG_DIR/${RUN_NAME}_live.log"
KEEPALIVE_LOG="$LOG_DIR/${RUN_NAME}_keepalive.log"
STATUS_FILE="$STATE_DIR/${RUN_NAME}.status"
LAST_CHECK_FILE="$STATE_DIR/${RUN_NAME}.last_check"
PID_FILE="$STATE_DIR/${RUN_NAME}.pid"

CONDA_BIN="${CONDA_BIN:-/opt/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-dino311}"
DATASET_DIR="${DATASET_DIR:-$HOME/Downloads}"
DINO_WM_DEVICE="${DINO_WM_DEVICE:-mps}"
CHECK_EVERY_SECONDS="${CHECK_EVERY_SECONDS:-15}"

mkdir -p "$LOG_DIR" "$STATE_DIR"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

find_worker() {
  pgrep -f "python -m experiments.make_paired_targets .*--run-name ${RUN_NAME}" | head -n 1 && return 0
  pgrep -f "python -m experiments.run_wall_grid .*--run-name ${RUN_NAME}" | head -n 1 && return 0
  pgrep -f "python -m experiments.dino_runner .*--run-name ${RUN_NAME}" | head -n 1 && return 0
  pgrep -f "run_main2_expansion_pipeline_m3.sh .* ${RUN_NAME}" | head -n 1 && return 0
  return 1
}

start_pipeline() {
  echo "[$(ts)] keepalive: starting pipeline ${RUN_NAME}" >> "$KEEPALIVE_LOG"
  (
    cd "$ROOT_DIR" || exit 1
    export DATASET_DIR DINO_WM_DEVICE PYTHONUNBUFFERED=1
    nohup "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
      bash scripts/run_main2_expansion_pipeline_m3.sh "$CONFIG_PATH" "$RUN_NAME" \
      >> "$PIPELINE_LOG" 2>&1 &
    echo $! > "$PID_FILE"
  )
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  echo "[$(ts)] keepalive: started pid=${pid:-unknown}" >> "$KEEPALIVE_LOG"
}

while true; do
  echo "$(ts)" > "$LAST_CHECK_FILE"
  worker_pid="$(find_worker 2>/dev/null || true)"
  if [[ -n "${worker_pid:-}" ]]; then
    echo "$worker_pid" > "$PID_FILE"
    echo "RUNNING pid=$worker_pid checked_at=$(ts)" > "$STATUS_FILE"
  else
    echo "STOPPED checked_at=$(ts) - relaunching" > "$STATUS_FILE"
    start_pipeline
  fi
  sleep "$CHECK_EVERY_SECONDS"
done

