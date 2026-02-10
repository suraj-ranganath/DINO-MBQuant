#!/usr/bin/env bash
set -euo pipefail

# Usage (example):
# bash scripts/setup_nautilus.sh /path/to/workspace /path/to/checkpoints /path/to/data

WORK_DIR="${1:-$PWD}"
CKPT_BASE_PATH="${2:-/ABS/PATH/TO/checkpoints}"
DATASET_DIR="${3:-/ABS/PATH/TO/data}"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d dino_wm ]; then
  git clone https://github.com/gaoyuezhou/dino_wm
fi

cd dino_wm

if command -v conda >/dev/null 2>&1; then
  conda env create -f environment.yaml || true
  conda activate dino_wm
fi

export DATASET_DIR="$DATASET_DIR"

python plan.py --config-name plan_wall.yaml \
  ckpt_base_path="$CKPT_BASE_PATH" \
  model_name=wall \
  n_evals=2 \
  planner.sub_planner.opt_steps=10

