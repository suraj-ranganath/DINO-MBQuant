#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_ROOT="${1:-/path/to/assets_root}"
MODE="${2:-symlink}" # symlink | copy

SRC_DATASET="$SRC_ROOT/wall_single"
SRC_HYDRA="$SRC_ROOT/outputs/wall_single/hydra.yaml"
SRC_MODEL="$SRC_ROOT/outputs/wall_single/checkpoints/model_latest.pth"

DST_DATA_ROOT="$ROOT_DIR/dino/data"
DST_CKPT_ROOT="$ROOT_DIR/dino/checkpoints/outputs/wall_single"
DST_CKPT_DIR="$DST_CKPT_ROOT/checkpoints"
DST_DATASET="$DST_DATA_ROOT/wall_single"
DST_HYDRA="$DST_CKPT_ROOT/hydra.yaml"
DST_MODEL="$DST_CKPT_DIR/model_latest.pth"

mkdir -p "$DST_DATA_ROOT" "$DST_CKPT_DIR"

if [[ ! -d "$SRC_DATASET" ]]; then
  echo "[error] Missing source dataset dir: $SRC_DATASET"
  exit 1
fi
if [[ ! -f "$SRC_HYDRA" ]]; then
  echo "[error] Missing source hydra file: $SRC_HYDRA"
  exit 1
fi
if [[ ! -f "$SRC_MODEL" ]]; then
  echo "[error] Missing source checkpoint file: $SRC_MODEL"
  exit 1
fi

if [[ "$MODE" == "symlink" ]]; then
  ln -snf "$SRC_DATASET" "$DST_DATASET"
  ln -snf "$SRC_HYDRA" "$DST_HYDRA"
  ln -snf "$SRC_MODEL" "$DST_MODEL"
elif [[ "$MODE" == "copy" ]]; then
  rm -rf "$DST_DATASET"
  cp -R "$SRC_DATASET" "$DST_DATASET"
  cp "$SRC_HYDRA" "$DST_HYDRA"
  cp "$SRC_MODEL" "$DST_MODEL"
else
  echo "[error] MODE must be 'symlink' or 'copy'. Got: $MODE"
  exit 1
fi

echo "[ok] Internal dino assets prepared."
echo "[info] dataset: $DST_DATASET"
echo "[info] hydra:   $DST_HYDRA"
echo "[info] model:   $DST_MODEL"
echo "[next] run smoke:"
echo "  bash scripts/run_transition_smoke_gpu.sh configs/experiment_config.gpu_transition_smoke.yaml smoke_gpu_v1"
