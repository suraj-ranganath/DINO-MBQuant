#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/experiment_config.mac.yaml}"
RUN_NAME="${2:-run_$(date +%Y%m%d_%H%M%S)}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"

cd "$ROOT_DIR"

bash scripts/mac_preflight.sh "$CONFIG_PATH"
export DINO_WM_DEVICE="${DINO_WM_DEVICE:-mps}"
echo "[info] DINO_WM_DEVICE=$DINO_WM_DEVICE"
echo "[info] RUN_NAME=$RUN_NAME"

ALLOW_EXISTING_ARGS=()
if [[ "$ALLOW_EXISTING_RUN" == "1" ]]; then
  ALLOW_EXISTING_ARGS+=(--allow-existing)
fi

python -m experiments.run_baseline_wall --config "$CONFIG_PATH" --run-name "$RUN_NAME" "${ALLOW_EXISTING_ARGS[@]}" --n-evals 2
python -m experiments.parse_success --run-dir "results/baseline/$RUN_NAME/fp16" --n-evals 2 --out "results/baseline/$RUN_NAME/fp16/baseline_success.json"

python -m experiments.build_variants --config "$CONFIG_PATH" --run-name "$RUN_NAME" "${ALLOW_EXISTING_ARGS[@]}"
python -m experiments.run_wall_grid --config "$CONFIG_PATH" --run-name "$RUN_NAME" "${ALLOW_EXISTING_ARGS[@]}"
python -m experiments.aggregate --config "$CONFIG_PATH" --run-name "$RUN_NAME" "${ALLOW_EXISTING_ARGS[@]}" --summary-out results/summary.csv --grouped-out results/summary_grouped.csv
python -m experiments.make_figures --config "$CONFIG_PATH" --run-name "$RUN_NAME" "${ALLOW_EXISTING_ARGS[@]}" --summary results/summary.csv --grouped results/summary_grouped.csv
python -m experiments.export_demo_artifacts --config "$CONFIG_PATH" --run-name "$RUN_NAME" "${ALLOW_EXISTING_ARGS[@]}"
python -m experiments.summarize_for_paper --config "$CONFIG_PATH" --run-name "$RUN_NAME" "${ALLOW_EXISTING_ARGS[@]}" --summary results/summary.csv --grouped results/summary_grouped.csv --out notes/paper_numbers.md

if [[ ! -f paper/main.tex ]]; then
  bash scripts/setup_paper.sh
fi
bash scripts/compile_paper.sh

echo "[done] Mac pipeline completed"
echo "[next] Review notes/$RUN_NAME/paper_numbers.md and finalize paper/main.tex"
