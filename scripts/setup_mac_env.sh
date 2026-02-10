#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${1:-python3.11}"
VENV_DIR="${2:-$ROOT_DIR/.venv311}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] $PYTHON_BIN not found. Install Python 3.11 and retry."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 12):
    raise SystemExit(f"[error] Python {sys.version.split()[0]} unsupported. Need 3.10 or 3.11.")
print(f"[ok] Using Python {sys.version.split()[0]}")
PY

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements-mac.txt"

echo "[ok] Mac environment ready at $VENV_DIR"
echo "[next] source $VENV_DIR/bin/activate"
