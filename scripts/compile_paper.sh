#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/paper"

latexmk -pdf -interaction=nonstopmode main.tex
cp -f main.pdf paper.pdf

echo "Compiled paper/main.pdf and paper/paper.pdf"
