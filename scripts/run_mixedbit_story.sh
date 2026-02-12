#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/experiment_config.mac_mixedbit_story.yaml}"
RUN_NAME="${2:-mixedbit_$(date +%Y%m%d_%H%M%S)}"
RESUME_TAG="${3:-}"

cd "$ROOT_DIR"

RESUME_MODE=0
if [[ -n "$RESUME_TAG" ]]; then
  RESUME_MODE=1
fi

STAGE_PREFIX="stage"
GRID_ARGS=()
AGG_ARGS=()
TARGET_ARGS=()
if [[ "$RESUME_MODE" == "1" ]]; then
  STAGE_PREFIX="resume/${RESUME_TAG}"
  GRID_ARGS+=(--allow-existing --skip-existing)
  AGG_ARGS+=(--allow-existing)
  TARGET_ARGS+=(--allow-existing)
fi

echo "[info] Config: $CONFIG_PATH"
echo "[info] Run name: $RUN_NAME"
if [[ "$RESUME_MODE" == "1" ]]; then
  echo "[info] Resume tag: $RESUME_TAG"
fi

bash scripts/mac_preflight.sh "$CONFIG_PATH"
export DINO_WM_DEVICE="${DINO_WM_DEVICE:-mps}"
echo "[info] DINO_WM_DEVICE=$DINO_WM_DEVICE"

PAIRED_MANIFEST="$ROOT_DIR/results/paired_targets/$RUN_NAME/paired_targets_manifest.json"
if [[ "$RESUME_MODE" == "1" && -f "$PAIRED_MANIFEST" ]]; then
  echo "[stage] Reusing existing paired targets: $PAIRED_MANIFEST"
else
  echo "[stage] Generate paired targets"
  python -m experiments.make_paired_targets \
    --config "$CONFIG_PATH" \
    --run-name "$RUN_NAME" \
    "${TARGET_ARGS[@]}"
fi

if [[ "$RESUME_MODE" == "0" ]]; then
  echo "[stage] Optional sanity build for variants"
  python -m experiments.build_variants \
    --config "$CONFIG_PATH" \
    --run-name "$RUN_NAME"
fi

echo "[stage] Main mixed-bit frontier (Budget A)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  "${GRID_ARGS[@]}" \
  --stage-subdir "${STAGE_PREFIX}/frontier_bA" \
  --budget-ids bA \
  --variants fp16,uniform_int8,mixed_int8,uniform_int6,mixed_int6,uniform_int4,mixed_int4,uniform_int3,mixed_int3,enc8_pred4,enc6_pred4,enc4_pred8,enc4_pred6

echo "[stage] Budget robustness subset (Budget B)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  "${GRID_ARGS[@]}" \
  --stage-subdir "${STAGE_PREFIX}/budget_bB" \
  --budget-ids bB \
  --variants fp16,uniform_int6,mixed_int6,uniform_int4,mixed_int4,uniform_int3,mixed_int3,enc8_pred4

echo "[stage] Encoder-retention curve (Budget A, seeds 1-2)"
python -m experiments.run_wall_grid \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  "${GRID_ARGS[@]}" \
  --stage-subdir "${STAGE_PREFIX}/layerwise_int4" \
  --budget-ids bA \
  --seeds 1,2 \
  --variants encfp16_0,encfp16_25,encfp16_50,encfp16_75,encfp16_100

echo "[stage] Aggregate + analyses + figures"
python -m experiments.aggregate \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  "${AGG_ARGS[@]}" \
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
  --variant-a mixed_int3 \
  --variant-b uniform_int3 \
  --out results/paired_delta_bA_mixed_vs_uniform_int3.json

python -m experiments.analyze_mechanistic_metrics \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --summary results/summary.csv \
  --out-csv results/mechanistic_metrics.csv \
  --out-json results/mechanistic_correlations.json

python -m experiments.analyze_mixedbit_stats \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --summary results/summary.csv \
  --outcomes results/episode_outcomes.csv \
  --out-csv results/mixedbit_pairwise_stats.csv \
  --out-json results/mixedbit_pairwise_stats.json \
  --note-out notes/mixedbit_story_notes.md

python -m experiments.make_transition_figures \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --grouped results/summary_grouped.csv

python -m experiments.make_mixedbit_appendix_figures \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  --grouped results/summary_grouped.csv \
  --stats results/mixedbit_pairwise_stats.csv

echo "[done] Mixed-bit story pipeline completed for run '$RUN_NAME'"
echo "[next] Review:"
echo "  - results/$RUN_NAME/summary.csv"
echo "  - results/$RUN_NAME/mixedbit_pairwise_stats.csv"
echo "  - notes/$RUN_NAME/mixedbit_story_notes.md"
echo "  - figures_mixedbit_story/$RUN_NAME/appendix_bit_ladder.pdf"
