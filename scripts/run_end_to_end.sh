#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/experiment_config.yaml}"

python -m experiments.build_variants --config "$CONFIG_PATH"
python -m experiments.run_wall_grid --config "$CONFIG_PATH"
python -m experiments.aggregate --config "$CONFIG_PATH"
python -m experiments.make_figures --config "$CONFIG_PATH"
python -m experiments.export_demo_artifacts --config "$CONFIG_PATH"

bash scripts/setup_paper.sh
bash scripts/compile_paper.sh

echo "Pipeline completed."
