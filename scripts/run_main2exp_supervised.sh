#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_NAME="${1:-m3_main2exp_v2}"
CONFIG_PATH="${2:-$ROOT_DIR/configs/experiment_config.mac_m3_main2_expansion.yaml}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
SLEEP_SECONDS="${SLEEP_SECONDS:-15}"

LOG_DIR="$ROOT_DIR/logs"
RUN_LOG="$LOG_DIR/${RUN_NAME}_supervised.log"
PIPELINE_LOG="$LOG_DIR/${RUN_NAME}_live.log"
STATE_DIR="$ROOT_DIR/run_state"
STATUS_FILE="$STATE_DIR/${RUN_NAME}.status"
LAST_CHECK_FILE="$STATE_DIR/${RUN_NAME}.last_check"

mkdir -p "$LOG_DIR" "$STATE_DIR"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

attempt=0
while true; do
  now="$(ts)"
  echo "$now" > "$LAST_CHECK_FILE"
  echo "SUPERVISOR attempt=$((attempt + 1)) started_at=$now" > "$STATUS_FILE"
  echo "[$now] [supervisor] starting attempt $((attempt + 1))" | tee -a "$RUN_LOG"

  set +e
  (
    cd "$ROOT_DIR"
    export PATH="/opt/anaconda3/envs/dino311/bin:$PATH"
    export DATASET_DIR="${DATASET_DIR:-$HOME/Downloads}"
    export DINO_WM_DEVICE="${DINO_WM_DEVICE:-mps}"
    export PYTHONUNBUFFERED=1
    bash scripts/run_main2_expansion_pipeline_m3.sh "$CONFIG_PATH" "$RUN_NAME"
  ) >> "$PIPELINE_LOG" 2>&1
  ec=$?
  set -e

  now="$(ts)"
  echo "$now" > "$LAST_CHECK_FILE"
  if [[ $ec -eq 0 ]]; then
    echo "COMPLETED exit_code=0 completed_at=$now" > "$STATUS_FILE"
    echo "[$now] [supervisor] completed successfully" | tee -a "$RUN_LOG"
    exit 0
  fi

  attempt=$((attempt + 1))
  echo "FAILED exit_code=$ec failed_at=$now attempt=$attempt" > "$STATUS_FILE"
  echo "[$now] [supervisor] attempt $attempt failed with exit code $ec" | tee -a "$RUN_LOG"

  if [[ $attempt -ge $MAX_RESTARTS ]]; then
    echo "[$now] [supervisor] reached MAX_RESTARTS=$MAX_RESTARTS; exiting" | tee -a "$RUN_LOG"
    exit "$ec"
  fi

  echo "[$now] [supervisor] sleeping $SLEEP_SECONDS seconds before restart" | tee -a "$RUN_LOG"
  sleep "$SLEEP_SECONDS"
done
