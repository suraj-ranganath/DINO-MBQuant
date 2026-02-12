#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/paper"

# Build directly to a single canonical artifact: paper.pdf
latexmk -pdf -interaction=nonstopmode -jobname=paper main.tex

# Remove legacy duplicate if present.
rm -f main.pdf

echo "Compiled paper/paper.pdf"
