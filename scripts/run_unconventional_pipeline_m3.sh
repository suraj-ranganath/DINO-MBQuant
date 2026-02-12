#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/experiment_config.mac_m3_unconventional.yaml}"
RUN_NAME="${2:-m3_unconventional_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT_DIR"

echo "[info] Config: $CONFIG_PATH"
echo "[info] Run name: $RUN_NAME"

bash scripts/mac_preflight.sh "$CONFIG_PATH"
export DINO_WM_DEVICE="${DINO_WM_DEVICE:-mps}"
echo "[info] DINO_WM_DEVICE=$DINO_WM_DEVICE"

echo "[stage] Generate paired targets"
python -m experiments.make_paired_targets --config "$CONFIG_PATH" --run-name "$RUN_NAME"

echo "[stage] Build variants (sanity)"
python -m experiments.build_variants --config "$CONFIG_PATH" --run-name "$RUN_NAME"

echo "[stage] A. Frontier + 6-bit midpoint sweep (bA,bB,bC)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --budget-ids bA,bB,bC \
  --variants fp16,uniform_int8,mixed_int8,uniform_int6,mixed_int6,uniform_int4,mixed_int4,uniform_int3,mixed_int3

echo "[stage] B. Head-vs-tail encoder retention at INT4 (bA,bB,bC)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --budget-ids bA,bB,bC \
  --variants encfp16_0,encfp16_25,encfp16_50,encfp16_75,encfp16_100,encheadfp16_25,encheadfp16_50

echo "[stage] C. Extra seeds for core claim only (mixed_int4 vs uniform_int4)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --budget-ids bA,bB,bC \
  --seeds 1,2,3,4,5 \
  --variants uniform_int4,mixed_int4

echo "[stage] D. Asymmetric cross-allocation families (bA,bB,bC)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --budget-ids bA,bB,bC \
  --variants mixed_e8_p4,mixed_e4_p8,mixed_e6_p4,mixed_e4_p6

echo "[stage] E. Goal-source robustness (random_state budgets)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --budget-ids bA_rand,bB_rand \
  --variants fp16,uniform_int4,mixed_int4,encfp16_50,encheadfp16_50

echo "[stage] Aggregate + paired analyses"
python -m experiments.aggregate --config "$CONFIG_PATH" --run-name "$RUN_NAME" --summary-out results/summary.csv --grouped-out results/summary_grouped.csv
python -m experiments.extract_episode_outcomes --config "$CONFIG_PATH" --run-name "$RUN_NAME" --out results/episode_outcomes.csv

python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bA --variant-a mixed_int4 --variant-b uniform_int4 --out results/paired_delta_bA_mixed_vs_uniform_int4.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bB --variant-a mixed_int4 --variant-b uniform_int4 --out results/paired_delta_bB_mixed_vs_uniform_int4.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bC --variant-a mixed_int4 --variant-b uniform_int4 --out results/paired_delta_bC_mixed_vs_uniform_int4.json

python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bA --variant-a mixed_int6 --variant-b uniform_int6 --out results/paired_delta_bA_mixed_vs_uniform_int6.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bB --variant-a mixed_int6 --variant-b uniform_int6 --out results/paired_delta_bB_mixed_vs_uniform_int6.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bC --variant-a mixed_int6 --variant-b uniform_int6 --out results/paired_delta_bC_mixed_vs_uniform_int6.json

python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bA --variant-a encfp16_50 --variant-b encheadfp16_50 --out results/paired_delta_bA_tail50_vs_head50_int4.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bB --variant-a encfp16_50 --variant-b encheadfp16_50 --out results/paired_delta_bB_tail50_vs_head50_int4.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bC --variant-a encfp16_50 --variant-b encheadfp16_50 --out results/paired_delta_bC_tail50_vs_head50_int4.json

python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bA --variant-a mixed_e8_p4 --variant-b mixed_e4_p8 --out results/paired_delta_bA_e8p4_vs_e4p8.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bB --variant-a mixed_e8_p4 --variant-b mixed_e4_p8 --out results/paired_delta_bB_e8p4_vs_e4p8.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bC --variant-a mixed_e8_p4 --variant-b mixed_e4_p8 --out results/paired_delta_bC_e8p4_vs_e4p8.json

python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bA_rand --variant-a mixed_int4 --variant-b uniform_int4 --out results/paired_delta_bA_rand_mixed_vs_uniform_int4.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bB_rand --variant-a mixed_int4 --variant-b uniform_int4 --out results/paired_delta_bB_rand_mixed_vs_uniform_int4.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bA_rand --variant-a encfp16_50 --variant-b encheadfp16_50 --out results/paired_delta_bA_rand_tail50_vs_head50_int4.json
python -m experiments.analyze_paired_deltas --run-name "$RUN_NAME" --outcomes results/episode_outcomes.csv --budget-id bB_rand --variant-a encfp16_50 --variant-b encheadfp16_50 --out results/paired_delta_bB_rand_tail50_vs_head50_int4.json

python -m experiments.analyze_mechanistic_metrics --config "$CONFIG_PATH" --run-name "$RUN_NAME" --summary results/summary.csv --out-csv results/mechanistic_metrics.csv --out-json results/mechanistic_correlations.json
python -m experiments.make_transition_figures --config "$CONFIG_PATH" --run-name "$RUN_NAME" --grouped results/summary_grouped.csv

echo "[done] Unconventional M3 pipeline completed for run '$RUN_NAME'"
echo "[next] Review:"
echo "  - results/$RUN_NAME/summary.csv"
echo "  - results/$RUN_NAME/summary_grouped.csv"
echo "  - results/$RUN_NAME/paired_delta_bA_mixed_vs_uniform_int4.json"
echo "  - results/$RUN_NAME/paired_delta_bC_mixed_vs_uniform_int4.json"
echo "  - results/$RUN_NAME/paired_delta_bA_tail50_vs_head50_int4.json"
echo "  - results/$RUN_NAME/paired_delta_bA_e8p4_vs_e4p8.json"
echo "  - results/$RUN_NAME/paired_delta_bA_rand_mixed_vs_uniform_int4.json"
echo "  - figures_m3_unconventional/$RUN_NAME/transition_frontier.pdf"
