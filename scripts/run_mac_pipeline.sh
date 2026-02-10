#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/experiment_config.mac.yaml}"

cd "$ROOT_DIR"

bash scripts/mac_preflight.sh "$CONFIG_PATH"
export DINO_WM_DEVICE="${DINO_WM_DEVICE:-mps}"
echo "[info] DINO_WM_DEVICE=$DINO_WM_DEVICE"

python -m experiments.run_baseline_wall --config "$CONFIG_PATH" --n-evals 2
python -m experiments.parse_success --run-dir results/baseline/fp16 --n-evals 2 --out results/baseline/fp16/baseline_success.json

python -m experiments.build_variants --config "$CONFIG_PATH"
python -m experiments.run_wall_grid --config "$CONFIG_PATH"
python -m experiments.aggregate --config "$CONFIG_PATH"
python -m experiments.make_figures --config "$CONFIG_PATH"
python -m experiments.export_demo_artifacts --config "$CONFIG_PATH"
python -m experiments.summarize_for_paper --config "$CONFIG_PATH" --summary results/summary.csv --out notes/paper_numbers.md

if [[ ! -f paper/main.tex ]]; then
  bash scripts/setup_paper.sh
fi
bash scripts/compile_paper.sh

echo "[done] Mac pipeline completed"
echo "[next] Review notes/paper_numbers.md and finalize paper/main.tex"
