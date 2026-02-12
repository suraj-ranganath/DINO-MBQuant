#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/experiment_config.gpu_transition_study.yaml}"
RUN_NAME="${2:-transition_gpu_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT_DIR"

echo "[info] Config: $CONFIG_PATH"
echo "[info] Run name: $RUN_NAME"

bash scripts/gpu_preflight.sh "$CONFIG_PATH"
export DINO_WM_DEVICE="${DINO_WM_DEVICE:-cuda}"
export DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/dino/data}"
echo "[info] DATASET_DIR=$DATASET_DIR"
echo "[info] DINO_WM_DEVICE=$DINO_WM_DEVICE"

echo "[stage] Generate paired targets"
python -m experiments.make_paired_targets --config "$CONFIG_PATH" --run-name "$RUN_NAME"

echo "[stage] Optional sanity build for variants"
python -m experiments.build_variants --config "$CONFIG_PATH" --run-name "$RUN_NAME"

echo "[stage] Main paired frontier (Budget A)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --budget-ids bA \
  --variants fp16,uniform_int8,mixed_int8,uniform_int4,mixed_int4,uniform_int3,mixed_int3

echo "[stage] Budget sensitivity subset (Budget B)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --budget-ids bB \
  --variants fp16,uniform_int4,mixed_int4,uniform_int3

echo "[stage] Layerwise encoder-retention curve (INT4, Budget A, seeds 1-2)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --budget-ids bA \
  --seeds 1,2 \
  --variants encfp16_0,encfp16_25,encfp16_50,encfp16_75,encfp16_100

echo "[stage] Aggregate + analyses"
python -m experiments.aggregate --config "$CONFIG_PATH" --run-name "$RUN_NAME" --summary-out results/summary.csv --grouped-out results/summary_grouped.csv
python -m experiments.extract_episode_outcomes --config "$CONFIG_PATH" --run-name "$RUN_NAME" --out results/episode_outcomes.csv
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bA --variant-a mixed_int4 --variant-b uniform_int4 --out results/paired_delta_bA_mixed_vs_uniform_int4.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bB --variant-a mixed_int4 --variant-b uniform_int4 --out results/paired_delta_bB_mixed_vs_uniform_int4.json
python -m experiments.analyze_mechanistic_metrics --config "$CONFIG_PATH" --run-name "$RUN_NAME" --summary results/summary.csv --out-csv results/mechanistic_metrics.csv --out-json results/mechanistic_correlations.json
python -m experiments.make_transition_figures --config "$CONFIG_PATH" --run-name "$RUN_NAME" --grouped results/summary_grouped.csv

echo "[done] Transition pipeline completed for run '$RUN_NAME'"
echo "[next] Review:"
echo "  - results/$RUN_NAME/summary.csv"
echo "  - results/$RUN_NAME/summary_grouped.csv"
echo "  - results/$RUN_NAME/paired_delta_bA_mixed_vs_uniform_int4.json"
echo "  - figures_transition/$RUN_NAME/transition_frontier.pdf"
