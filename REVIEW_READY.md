# Review-Ready Checklist (ICLR Workshop Double-Blind)

This branch (`review-ready`) is prepared for anonymous reviewer access.

## What was cleaned

- Removed tracked runtime/state artifacts:
  - `logs/`
  - `saved_runs/`
  - `run_state/`
  - generated files under `release/`
  - generated paper PDFs (`paper/*.pdf`)
- Replaced machine-specific paths (`/Users/...`) in tracked configs/docs with placeholders.
- Added automated anonymity checks and bundle generation scripts.

## Pre-submission commands

```bash
# 1) Verify no deanonymizing strings remain in tracked sources
bash scripts/check_double_blind.sh

# 2) Compile paper (optional if already compiled)
bash scripts/compile_paper.sh

# 3) Build anonymized reviewer bundle
bash scripts/build_double_blind_bundle.sh
```

## Expected output

- `release/review_bundle.zip` containing:
  - clean source snapshot (`repo/`)
  - `paper.pdf` if available
  - reviewer README

## Notes

- You still need to set your own local paths in `configs/*.yaml` before running experiments.
- Keep author identities out of supplementary text files and logs before uploading.
