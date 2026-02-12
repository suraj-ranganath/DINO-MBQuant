#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/experiment_config.mac_m3_main2_expansion_smoke.yaml}"
RUN_NAME="${2:-m3_main2exp_smoke_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT_DIR"

echo "[info] Config: $CONFIG_PATH"
echo "[info] Run name: $RUN_NAME"

bash scripts/mac_preflight.sh "$CONFIG_PATH"
export DINO_WM_DEVICE="${DINO_WM_DEVICE:-mps}"
echo "[info] DINO_WM_DEVICE=$DINO_WM_DEVICE"

echo "[stage] Generate paired targets"
python -m experiments.make_paired_targets --config "$CONFIG_PATH" --run-name "$RUN_NAME" --budget-ids bA

echo "[stage] Smoke grid"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --budget-ids bA \
  --variants fp16,uniform_int8,uniform_int4,mixed_int4

echo "[stage] Aggregate + paired check"
python -m experiments.aggregate --config "$CONFIG_PATH" --run-name "$RUN_NAME" --summary-out results/summary.csv --grouped-out results/summary_grouped.csv
python -m experiments.extract_episode_outcomes --config "$CONFIG_PATH" --run-name "$RUN_NAME" --out results/episode_outcomes.csv
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bA --variant-a mixed_int4 --variant-b uniform_int4 --out results/paired_delta_bA_mixed_vs_uniform_int4.json

echo "[done] Smoke completed for run '$RUN_NAME'"
echo "  - results/$RUN_NAME/summary.csv"
echo "  - results/$RUN_NAME/paired_delta_bA_mixed_vs_uniform_int4.json"
