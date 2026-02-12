# Where Bits Matter in World-Model Planning: A Paired Mixed-Bit Study for Efficient Spatial Reasoning

Mixed-bit quantization experiments for efficient world-model planning, centered on DINO-WM and the ES-Reasoning workshop workflow.

This repo provides:
- experiment runners for uniform, mixed, asymmetric, and layerwise quantization variants,
- reproducible run orchestration scripts (fresh run + resume),
- aggregation/statistics/figure pipelines for paper-ready outputs,
- LaTeX paper sources and release bundling.

## Project Goal

Investigate **where bits matter** in world-model planning under efficiency constraints:
- Is performance driven mostly by total bitwidth?
- Or by how precision is allocated across encoder vs predictor?

Current experiments focus on DINO-WM on Wall with paired-goal evaluation and mixed-bit ablations.

## Repository Layout

Core code:
- `configs/` experiment configurations
- `experiments/` Python modules for runs, aggregation, analysis, and plotting
- `scripts/` setup/run/resume/release shell entrypoints
- `paper/` LaTeX sources and compiled paper output
- `demo/` optional replay viewer
- `third_party_dino_wm/` upstream dependency

Generated artifacts (expected after runs):
- `results/`
- `figures*/`
- `logs/`
- `release/`
- `saved_runs/`

## Quick Start (Mac)

1. Setup environment:

```bash
bash scripts/setup_mac_env.sh python3.11 .venv311
source .venv311/bin/activate
```

2. Set environment:

```bash
export DATASET_DIR=/ABS/PATH/TO/DATA_ROOT
export DINO_WM_DEVICE=mps   # or cpu
```

3. Configure checkpoints in (recommended):
- `configs/experiment_config.mac_mixedbit_story.yaml`

Required fields:

```yaml
dino:
  ckpt_base_path: /ABS/PATH/TO/CHECKPOINT_ROOT
  model_name: wall_single
  model_epoch: latest
```

4. Validate setup:

```bash
bash scripts/mac_preflight.sh configs/experiment_config.mac_mixedbit_story.yaml
```

5. Run main mixed-bit pipeline:

```bash
RUN_NAME=mixedbit_$(date +%Y%m%d_%H%M%S)
bash scripts/run_mixedbit_story.sh configs/experiment_config.mac_mixedbit_story.yaml "$RUN_NAME" 2>&1 | tee "logs/${RUN_NAME}.log"
```

## Main Workflows

Mixed-bit story (recommended):

```bash
bash scripts/run_mixedbit_story.sh configs/experiment_config.mac_mixedbit_story.yaml <run_name>
```

Resume mixed-bit run:

```bash
bash scripts/run_mixedbit_story.sh configs/experiment_config.mac_mixedbit_story.yaml <run_name> <resume_tag>
```

Transition study:

```bash
bash scripts/run_transition_pipeline.sh configs/experiment_config.mac_transition_study.yaml <run_name>
```

Baseline/grid:

```bash
bash scripts/run_mac_pipeline.sh configs/experiment_config.mac.yaml <run_name>
```

## Paper + Release

Compile canonical submission PDF:

```bash
bash scripts/compile_paper.sh
```

Output:
- `paper/paper.pdf`

Build release bundle:

```bash
bash scripts/build_release_bundle.sh
```

Outputs:
- `release/paper.pdf`
- `release/supplemental.zip`

## Dataset / External Links

- DINO-WM assets (dataset + checkpoints):  
  `https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28`
- DINO-WM repo: `https://github.com/gaoyuezhou/dino_wm`
- DINO-WM paper: `https://arxiv.org/abs/2411.04983`
- ES-Reasoning workshop: `https://sites.google.com/ucsd.edu/efficient-spatial-reasoning/home?authuser=0`
- CFP: `https://sites.google.com/ucsd.edu/efficient-spatial-reasoning/call-for-papers`
- OpenReview venue: `https://openreview.net/group?id=ICLR.cc%2F2026%2FWorkshop%2FES-Reasoning`

## Housekeeping / Repo Hygiene

Archive legacy artifacts out of repo root (without deleting your history):

```bash
bash scripts/archive_legacy_artifacts.sh
```

This moves older figures/logs/notes to:
- `saved_runs/archive/<timestamp>/...`

## Troubleshooting

If you see `Config not found: configs/`, you passed a directory instead of a YAML file.

If MPS fails on an op/kernel:

```bash
export DINO_WM_DEVICE=cpu
```
