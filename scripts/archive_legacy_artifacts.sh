#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="${1:-$ROOT_DIR/saved_runs/archive/$STAMP}"

mkdir -p "$ARCHIVE_DIR"

move_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
    echo "[moved] $src -> $dst"
  fi
}

# Root-level legacy figure collections not used by the current mixed-bit paper.
move_if_exists "$ROOT_DIR/figures_bitsweep_h9_mixed" "$ARCHIVE_DIR/root/figures_bitsweep_h9_mixed"
move_if_exists "$ROOT_DIR/figures_claim_fast" "$ARCHIVE_DIR/root/figures_claim_fast"
move_if_exists "$ROOT_DIR/figures_claim_lowbudget_fixseed" "$ARCHIVE_DIR/root/figures_claim_lowbudget_fixseed"
move_if_exists "$ROOT_DIR/figures_harder" "$ARCHIVE_DIR/root/figures_harder"
move_if_exists "$ROOT_DIR/figures_transition" "$ARCHIVE_DIR/root/figures_transition"

# Legacy root logs.
move_if_exists "$ROOT_DIR/logs_transition_run_v1.txt" "$ARCHIVE_DIR/root/logs_transition_run_v1.txt"
move_if_exists "$ROOT_DIR/logs_transition_run_v2.txt" "$ARCHIVE_DIR/root/logs_transition_run_v2.txt"

# Legacy notes from prior run branches.
move_if_exists "$ROOT_DIR/notes/smoke_artifacts" "$ARCHIVE_DIR/notes/smoke_artifacts"
move_if_exists "$ROOT_DIR/notes/transition_run_v2" "$ARCHIVE_DIR/notes/transition_run_v2"

echo "[done] Archived legacy artifacts to: $ARCHIVE_DIR"
echo "[note] Current paper-critical figures under figures_mixedbit_story/ were not moved."

