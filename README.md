# EAI_DINO: ES-Reasoning Tiny Paper MVP

This repository contains a fast, reproducible pipeline for:
- running DINO-WM Wall planning experiments across FP16 / uniform INT8 / mixed INT8 variants,
- aggregating metrics and producing paper-ready figures,
- generating replay artifacts and a Mac-friendly Streamlit demo,
- compiling an anonymized 4-page Tiny paper (ICLR 2026 workshop format).

## Quick Start

1. Install Python deps:
   ```bash
   # Use Python 3.10 or 3.11 for DINO-WM compatibility.
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Or use one-command Mac setup:
   ```bash
   bash scripts/setup_mac_env.sh python3.11 .venv311
   source .venv311/bin/activate
   ```
   For Mac local experiment runs:
   ```bash
   pip install -r requirements-mac.txt
   ```
   For GPU quantization runs on Nautilus:
   ```bash
   pip install -r requirements-gpu.txt
   ```
2. Get dataset + pretrained checkpoint (source of truth):
   - DINO-WM OSF bundle (dataset + checkpoints):  
     `https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28`
   - In this project, current local paths are:
     - checkpoint: `/Users/suraj/Downloads/outputs/wall_single/checkpoints/model_latest.pth`
     - dataset (Wall): `/Users/suraj/Downloads/wall_single`
   - Set environment/config:
   ```bash
   export DATASET_DIR=/Users/suraj/Downloads
   # Then set dino.ckpt_base_path in config to:
   # /Users/suraj/Downloads/outputs/wall_single/checkpoints/model_latest.pth
   ```
3. Configure paths in `configs/experiment_config.yaml`.
4. Build variants:
   ```bash
   python -m experiments.build_variants --config configs/experiment_config.yaml
   ```
5. Run FP16 baseline only (optional gate check):
   ```bash
   python -m experiments.run_baseline_wall --config configs/experiment_config.yaml
   ```
6. Run grid:
   ```bash
   python -m experiments.run_wall_grid --config configs/experiment_config.yaml
   ```
7. Aggregate + figures:
   ```bash
   python -m experiments.aggregate --config configs/experiment_config.yaml
   python -m experiments.make_figures --config configs/experiment_config.yaml
   ```
8. Export demo artifacts + launch demo:
   ```bash
   python -m experiments.export_demo_artifacts --config configs/experiment_config.yaml
   streamlit run demo/app.py
   ```
9. Prepare paper scaffold and compile:
   ```bash
   bash scripts/setup_paper.sh
   bash scripts/compile_paper.sh
   ```

## Mac-Only 6-7 Hour Path

Use this when running only on a local MacBook (no CUDA):

1. Configure Mac preset:
   - set `dino.ckpt_base_path` in `configs/experiment_config.mac.yaml`
   - set `export DATASET_DIR=/ABS/PATH/TO/data`
   - set `export DINO_WM_DEVICE=mps` (or `cpu` if MPS op support is problematic)
2. Run preflight + pipeline:
   ```bash
   bash scripts/run_mac_pipeline.sh configs/experiment_config.mac.yaml my_run_name
   ```
   Each run is isolated under run-specific folders (for example, `results/.../my_run_name`, `figures/.../my_run_name`, `notes/my_run_name/...`), so new runs do not overwrite old results.
3. Launch replay demo:
   ```bash
   streamlit run demo/app.py
   ```
4. Build release artifacts:
   ```bash
   bash scripts/build_release_bundle.sh
   ```
5. Use `notes/paper_numbers.md` to quickly fill result text in the tiny paper.

## Transition-Study Pipeline (Paired + Budget + Mechanistic)

Use this for the stronger workshop narrative (paired evaluation, budget sensitivity, encoder-retention at INT4):

```bash
bash scripts/run_transition_pipeline.sh configs/experiment_config.mac_transition_study.yaml my_transition_run
```

Key outputs:
- `results/my_transition_run/summary.csv`
- `results/my_transition_run/summary_grouped.csv`
- `results/my_transition_run/episode_outcomes.csv`
- `results/my_transition_run/paired_delta_bA_mixed_vs_uniform_int4.json`
- `results/my_transition_run/mechanistic_correlations.json`
- `figures_transition/my_transition_run/transition_frontier.pdf`
- `figures_transition/my_transition_run/budget_sensitivity.pdf`
- `figures_transition/my_transition_run/encoder_retention_curve.pdf`

Note: ES-Reasoning Tiny paper formatting can be targeted to 5 pages including references, with unlimited appendix for additional ablations.

## Notes
- Use GPU (A6000/A100) for experiment runs.
- Use Mac for replay demo and paper finalization.
- Nautilus bootstrap helper: `bash scripts/setup_nautilus.sh <work_dir> <ckpt_base_path> <dataset_dir>`.
- Mac runbook: `scripts/mac_6h_runbook.md`.
