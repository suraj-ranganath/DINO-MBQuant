#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/experiment_config.gpu_transition_study.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[error] Config not found: $CONFIG_PATH"
  exit 1
fi

python - <<'PY'
import sys
import importlib
if sys.version_info < (3, 10) or sys.version_info >= (3, 12):
    raise SystemExit(
        f"[error] Python {sys.version.split()[0]} detected. Use Python 3.10 or 3.11 for DINO-WM compatibility."
    )
mods = ["torch", "torchvision", "gym", "einops", "submitit", "wandb", "hydra", "omegaconf", "pandas", "matplotlib", "yaml", "scipy", "psutil", "bitsandbytes"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"[error] Missing Python packages: {missing}. Install requirements first.")
PY

if [[ -z "${DATASET_DIR:-}" ]]; then
  echo "[error] DATASET_DIR is not set. Example: export DATASET_DIR=/ABS/PATH/TO/data"
  exit 1
fi

if [[ ! -d "$DATASET_DIR/wall_single" ]]; then
  echo "[error] Missing dataset folder: $DATASET_DIR/wall_single"
  exit 1
fi

CKPT_BASE_PATH="$(python - <<'PY' "$CONFIG_PATH"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg['dino']['ckpt_base_path'])
PY
)"
MODEL_NAME="$(python - <<'PY' "$CONFIG_PATH"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg['dino'].get('model_name', 'wall'))
PY
)"

if [[ "$CKPT_BASE_PATH" == "/ABS/PATH/TO/checkpoints" ]]; then
  echo "[error] ckpt_base_path is still placeholder in $CONFIG_PATH"
  exit 1
fi

if [[ ! -f "$CKPT_BASE_PATH/outputs/$MODEL_NAME/hydra.yaml" ]]; then
  echo "[error] Missing checkpoint hydra config: $CKPT_BASE_PATH/outputs/$MODEL_NAME/hydra.yaml"
  exit 1
fi

MODEL_EPOCH="$(python - <<'PY' "$CONFIG_PATH"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg['dino'].get('model_epoch', 'final'))
PY
)"

if [[ ! -f "$CKPT_BASE_PATH/outputs/$MODEL_NAME/checkpoints/model_${MODEL_EPOCH}.pth" ]]; then
  echo "[warn] model_${MODEL_EPOCH}.pth not found. Available files:"
  ls -lah "$CKPT_BASE_PATH/outputs/$MODEL_NAME/checkpoints" || true
  exit 1
fi

FREE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR==2{print $4}')"
FREE_GB="$((FREE_KB / 1024 / 1024))"
if [[ -n "$FREE_GB" ]] && [[ "$FREE_GB" -lt 30 ]]; then
  echo "[warn] Low free disk (<30GB). Runs may fail while writing outputs/videos."
fi

python - <<'PY'
import torch
print(f"[info] torch version={torch.__version__}")
print(f"[info] cuda available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[info] cuda device={torch.cuda.get_device_name(0)}")
    print(f"[info] cuda capability={torch.cuda.get_device_capability(0)}")
PY

echo "[ok] Preflight passed"
echo "[info] DATASET_DIR=$DATASET_DIR"
echo "[info] ckpt_base_path=$CKPT_BASE_PATH"
echo "[info] model_name=$MODEL_NAME"
echo "[info] model_epoch=$MODEL_EPOCH"
