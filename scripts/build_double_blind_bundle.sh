#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/check_double_blind.sh

mkdir -p release
BUNDLE_DIR="release/review_bundle"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/repo"

# Copy source-only repository snapshot for reviewer artifact.
rsync -a \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.venv311' \
  --exclude '__pycache__' \
  --exclude 'results' \
  --exclude 'logs' \
  --exclude 'saved_runs' \
  --exclude 'run_state' \
  --exclude 'release' \
  --exclude 'demo/demo_artifacts*' \
  --exclude 'paper/*.pdf' \
  ./ "$BUNDLE_DIR/repo/"

# Include submission PDF if already compiled.
if [[ -f paper/paper.pdf ]]; then
  cp -f paper/paper.pdf "$BUNDLE_DIR/"
fi

cat > "$BUNDLE_DIR/README_REVIEW.md" <<'EOT'
Double-blind reviewer bundle

Contents:
- repo/: source snapshot without runtime artifacts
- paper.pdf: submission PDF (if present at bundle time)

Quick reproduction:
1) Setup environment from repo/README.md
2) Fill paths in configs/*.yaml
3) Run scripts/check_double_blind.sh before submission artifacts
EOT

(cd release && zip -qr review_bundle.zip review_bundle)

echo "[ok] Built release/review_bundle.zip"
