#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p release

if [[ ! -f paper/paper.pdf ]]; then
  echo "[error] Missing paper/paper.pdf"
  exit 1
fi
cp -f paper/paper.pdf release/paper.pdf

SUPP_DIR="release/supplemental"
rm -rf "$SUPP_DIR"
mkdir -p "$SUPP_DIR"

cp -f results/summary.csv "$SUPP_DIR" 2>/dev/null || true
cp -f results/summary_grouped.csv "$SUPP_DIR" 2>/dev/null || true
cp -f figures/success_vs_opt_steps.pdf "$SUPP_DIR" 2>/dev/null || true
cp -f figures/efficiency_table.pdf "$SUPP_DIR" 2>/dev/null || true
cp -f figures/efficiency_tradeoff.pdf "$SUPP_DIR" 2>/dev/null || true
cp -f figures/mac_predictor_latency.pdf "$SUPP_DIR" 2>/dev/null || true
cp -f notes/paper_numbers.md "$SUPP_DIR" 2>/dev/null || true
cp -f notes/paper_numbers_harder.md "$SUPP_DIR" 2>/dev/null || true
cp -f notes/mac_quant_findings.md "$SUPP_DIR" 2>/dev/null || true
cp -f configs/experiment_config.mac.yaml "$SUPP_DIR" 2>/dev/null || true
cp -f configs/experiment_config.mac_harder.yaml "$SUPP_DIR" 2>/dev/null || true
cp -f demo/demo_artifacts/manifest.json "$SUPP_DIR" 2>/dev/null || true
cp -f results/mac_quant/*.json "$SUPP_DIR" 2>/dev/null || true

if [[ -d demo/demo_artifacts ]]; then
  mkdir -p "$SUPP_DIR/demo_artifacts"
  cp -R demo/demo_artifacts/* "$SUPP_DIR/demo_artifacts/" 2>/dev/null || true
fi

cp -f README.md "$SUPP_DIR" 2>/dev/null || true
cp -f release/README.txt "$SUPP_DIR" 2>/dev/null || true

(cd release && zip -qr supplemental.zip supplemental)

echo "[ok] Built release/paper.pdf and release/supplemental.zip"
