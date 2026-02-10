#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/experiment_config.gpu_transition_smoke.yaml}"
RUN_NAME="${2:-transition_smoke_gpu_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT_DIR"

echo "[info] Config: $CONFIG_PATH"
echo "[info] Run name: $RUN_NAME"

bash scripts/gpu_preflight.sh "$CONFIG_PATH"
export DINO_WM_DEVICE="${DINO_WM_DEVICE:-cuda}"
echo "[info] DINO_WM_DEVICE=$DINO_WM_DEVICE"

echo "[stage] Generate paired targets"
python -m experiments.make_paired_targets --config "$CONFIG_PATH" --run-name "$RUN_NAME"

echo "[stage] Optional sanity build for variants"
python -m experiments.build_variants --config "$CONFIG_PATH" --run-name "$RUN_NAME"

echo "[stage] Smoke grid (Budget A, fp16/int8/int4)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --budget-ids bA \
  --variants fp16,uniform_int8,mixed_int4

echo "[stage] Aggregate + analyses"
python -m experiments.aggregate --config "$CONFIG_PATH" --run-name "$RUN_NAME" --summary-out results/summary.csv --grouped-out results/summary_grouped.csv
python -m experiments.extract_episode_outcomes --config "$CONFIG_PATH" --run-name "$RUN_NAME" --out results/episode_outcomes.csv
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bA --variant-a mixed_int4 --variant-b uniform_int8 --out results/paired_delta_bA_mixed_int4_vs_uniform_int8.json

echo "[done] Smoke pipeline completed for run '$RUN_NAME'"
echo "[next] Review:"
echo "  - results/$RUN_NAME/summary.csv"
echo "  - results/$RUN_NAME/summary_grouped.csv"
echo "  - results/$RUN_NAME/paired_delta_bA_mixed_int4_vs_uniform_int8.json"
