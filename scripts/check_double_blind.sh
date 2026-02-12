#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Patterns that should never appear in a double-blind package.
PATTERN='suraj|anishpatnaik|ranganath|DINO-MBQuant|/Users/'

echo "[check] scanning tracked files for deanonymizing strings..."
if git grep -n -E "$PATTERN" -- . ':!paper/iclr2026/*' ':!paper/natbib.sty' ':!paper/iclr2026_conference.sty' ':!paper/iclr2026_conference.bst'; then
  echo "[fail] deanonymizing strings found in tracked files."
  exit 1
fi

echo "[ok] no deanonymizing strings found in tracked source files."
