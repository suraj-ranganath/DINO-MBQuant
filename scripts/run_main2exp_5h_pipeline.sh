#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/experiment_config.mac_m3_main2_5h.yaml}"
RUN_NAME="${2:-m3_main2exp_5h_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT_DIR"

echo "[info] Config: $CONFIG_PATH"
echo "[info] Run name: $RUN_NAME"

bash scripts/mac_preflight.sh "$CONFIG_PATH"
export DINO_WM_DEVICE="${DINO_WM_DEVICE:-mps}"
echo "[info] DINO_WM_DEVICE=$DINO_WM_DEVICE"

echo "[stage] Generate paired targets (bA,bB)"
python -m experiments.make_paired_targets \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --budget-ids bA,bB \
  --allow-existing

echo "[stage] Core claim: mixed_int4 vs uniform_int4 (+fp16/int8 anchors)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --allow-existing \
  --skip-existing \
  --budget-ids bA,bB \
  --seeds 1,2 \
  --variants fp16,uniform_int8,mixed_int8,uniform_int4,mixed_int4

echo "[stage] Where-bits check: tail vs head retention at INT4 (bA, seed1)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --allow-existing \
  --skip-existing \
  --budget-ids bA \
  --seeds 1 \
  --variants encfp16_50,encheadfp16_50

echo "[stage] Asymmetric allocation check: E8/P4 vs E4/P8 (bA, seed1)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --allow-existing \
  --skip-existing \
  --budget-ids bA \
  --seeds 1 \
  --variants mixed_e8_p4,mixed_e4_p8

echo "[stage] Random-goal robustness mini-check (bA_rand, seed1)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --allow-existing \
  --skip-existing \
  --budget-ids bA_rand \
  --seeds 1 \
  --variants fp16,uniform_int4,mixed_int4

echo "[stage] Aggregate + paired-delta sanity outputs"
python -m experiments.aggregate \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
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

python -m experiments.analyze_paired_deltas \
  --run-name "$RUN_NAME" \
  --outcomes results/episode_outcomes.csv \
  --budget-id bA \
  --variant-a encfp16_50 \
  --variant-b encheadfp16_50 \
  --out results/paired_delta_bA_tail50_vs_head50_int4.json

python -m experiments.analyze_paired_deltas \
  --run-name "$RUN_NAME" \
  --outcomes results/episode_outcomes.csv \
  --budget-id bA \
  --variant-a mixed_e8_p4 \
  --variant-b mixed_e4_p8 \
  --out results/paired_delta_bA_e8p4_vs_e4p8.json

echo "[done] 5h-constrained pipeline completed for run '$RUN_NAME'"
echo "[next] Review:"
echo "  - results/$RUN_NAME/summary.csv"
echo "  - results/$RUN_NAME/summary_grouped.csv"
echo "  - results/$RUN_NAME/paired_delta_bA_mixed_vs_uniform_int4.json"
echo "  - results/$RUN_NAME/paired_delta_bB_mixed_vs_uniform_int4.json"
