#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/experiment_config.mac_transition_study.yaml}"
RUN_NAME="${2:-}"
RESUME_TAG="${3:-resume_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "$RUN_NAME" ]]; then
  echo "Usage: bash scripts/resume_transition_run.sh <config_path> <run_name> [resume_tag]"
  echo "Example: bash scripts/resume_transition_run.sh configs/experiment_config.mac_transition_study.yaml transition_run_v2 resume_try1"
  exit 1
fi

cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs/$RUN_NAME"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RESUME_TAG}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[info] Config: $CONFIG_PATH"
echo "[info] Run name: $RUN_NAME"
echo "[info] Resume tag: $RESUME_TAG"
echo "[info] Log file: $LOG_FILE"

bash scripts/mac_preflight.sh "$CONFIG_PATH"
export DINO_WM_DEVICE="${DINO_WM_DEVICE:-mps}"
echo "[info] DINO_WM_DEVICE=$DINO_WM_DEVICE"

echo "[stage] Resume Budget B into isolated subfolder"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --allow-existing \
  --skip-existing \
  --stage-subdir "resume/${RESUME_TAG}/budget_bB" \
  --budget-ids bB \
  --variants fp16,uniform_int4,mixed_int4,uniform_int3

echo "[stage] Resume layerwise INT4 into isolated subfolder"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --allow-existing \
  --skip-existing \
  --stage-subdir "resume/${RESUME_TAG}/layerwise_int4" \
  --budget-ids bA \
  --seeds 1,2 \
  --variants encfp16_0,encfp16_25,encfp16_50,encfp16_75,encfp16_100

echo "[stage] Rebuild summaries + analyses + figures for full run"
python -m experiments.aggregate \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --allow-existing \
  --summary-out results/summary.csv \
  --grouped-out results/summary_grouped.csv

python -m experiments.extract_episode_outcomes \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --out results/episode_outcomes.csv

python -m experiments.analyze_paired_deltas \
  --run-name "$RUN_NAME" \
  --outcomes results/episode_outcomes.csv \
  --budget-id bA \
  --variant-a mixed_int4 \
  --variant-b uniform_int4 \
  --out results/paired_delta_bA_mixed_vs_uniform_int4.json

python -m experiments.analyze_paired_deltas \
  --run-name "$RUN_NAME" \
  --outcomes results/episode_outcomes.csv \
  --budget-id bB \
  --variant-a mixed_int4 \
  --variant-b uniform_int4 \
  --out results/paired_delta_bB_mixed_vs_uniform_int4.json

python -m experiments.analyze_mechanistic_metrics \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --summary results/summary.csv \
  --out-csv results/mechanistic_metrics.csv \
  --out-json results/mechanistic_correlations.json

python -m experiments.make_transition_figures \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --grouped results/summary_grouped.csv

echo "[done] Resume completed."
echo "[next] New stage folders:"
echo "  results/wall_transition/$RUN_NAME/resume/$RESUME_TAG/budget_bB"
echo "  results/wall_transition/$RUN_NAME/resume/$RESUME_TAG/layerwise_int4"
