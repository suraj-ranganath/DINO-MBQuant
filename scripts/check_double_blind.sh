#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Patterns that should never appear in a double-blind package.
PATTERN='suraj|anishpatnaik|ranganath|DINO-MBQuant|/Users/[A-Za-z0-9._-]+'

echo "[check] scanning tracked files for deanonymizing strings..."
MATCHED=0
while IFS= read -r f; do
  [[ -f "$f" ]] || continue
  if rg -n -e "$PATTERN" "$f"; then
    MATCHED=1
  fi
done < <(git ls-files \
  | rg -v '^scripts/check_double_blind\.sh$' \
  | rg -v '^paper/iclr2026/' \
  | rg -v '^paper/natbib\.sty$' \
  | rg -v '^paper/iclr2026_conference\.sty$' \
  | rg -v '^paper/iclr2026_conference\.bst$')

if [[ "$MATCHED" -eq 1 ]]; then
  echo "[fail] deanonymizing strings found in tracked files."
  exit 1
fi

echo "[ok] no deanonymizing strings found in tracked source files."
