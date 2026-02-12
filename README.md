# EAI_DINO: Mixed-Bit Quantization Experiments (Mac Collaborator Guide)

This README is the **single setup + runbook** for collaborators using a local Mac (Apple Silicon).

It covers:
- exact setup commands,
- where to download data/checkpoints,
- how to set `ckpt_base_path` correctly,
- how to run experiments with logs,
- how to resume without overwriting,
- where outputs/figures/paper files are written.

---

## 1) What This Repo Produces

Main pipelines in this repo generate:
- run-scoped experiment metrics (`results/.../<run_name>/...`),
- run-scoped figures (`figures.../<run_name>/...`),
- notes for paper writing (`notes/<run_name>/...`),
- optional replay artifacts for demo (`demo/.../<run_name>/...`),
- paper PDF (`paper/main.pdf` and copied `paper/paper.pdf`).

The recommended research pipeline for current work is:
- `scripts/run_mixedbit_story.sh` (mixed-bit main study + appendix analysis).

---

## 2) Prerequisites (Mac)

- macOS on Apple Silicon (M-series)
- Python `3.10` or `3.11` (required; 3.12 is not supported by this project)
- Git
- Enough free disk space (recommend 15GB+ free)

Check Python:

```bash
python3.11 --version
```

If unavailable, install Python 3.11 first.

---

## 3) Clone + Environment Setup

From scratch:

```bash
git clone <your-fork-or-repo-url>
cd EAI_DINO
bash scripts/setup_mac_env.sh python3.11 .venv311
source .venv311/bin/activate
```

This installs `requirements-mac.txt` and prepares a compatible venv.

---

## 4) Download Dataset + Checkpoint (Required)

Primary source:
- DINO-WM assets (dataset + checkpoints):  
  `https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28`

Workshop/reference links:
- Workshop home: `https://sites.google.com/ucsd.edu/efficient-spatial-reasoning/home?authuser=0`
- CFP: `https://sites.google.com/ucsd.edu/efficient-spatial-reasoning/call-for-papers`
- OpenReview: `https://openreview.net/group?id=ICLR.cc%2F2026%2FWorkshop%2FES-Reasoning`
- DINO-WM repo: `https://github.com/gaoyuezhou/dino_wm`
- DINO-WM paper: `https://arxiv.org/abs/2411.04983`

### Expected Local Layout

After download/unzip, your local structure should match:

```text
/Users/<you>/Downloads/
  wall_single/                               # dataset folder
  outputs/
    wall_single/
      hydra.yaml
      checkpoints/
        model_latest.pth
```

Given the path you shared earlier, this is correct:
- checkpoint: `/Users/suraj/Downloads/outputs/wall_single/checkpoints/model_latest.pth`
- data folder: `/Users/suraj/Downloads/wall_single`

---

## 5) Configure Paths Correctly

Use one of the Mac configs (recommended for mixed-bit story):
- `configs/experiment_config.mac_mixedbit_story.yaml`

Ensure these fields are set:

```yaml
dino:
  ckpt_base_path: /Users/suraj/Downloads
  model_name: wall_single
  model_epoch: latest
```

Critical notes:
- `ckpt_base_path` is **not** the checkpoint file path.
- It must be the parent that contains `outputs/<model_name>/...`.
- With current assets, that parent is `/Users/suraj/Downloads`.

Set dataset + device env vars in shell:

```bash
export DATASET_DIR=/Users/suraj/Downloads
export DINO_WM_DEVICE=mps
```

If MPS op issues occur, switch to CPU:

```bash
export DINO_WM_DEVICE=cpu
```

---

## 6) Preflight Check (Do This First)

```bash
bash scripts/mac_preflight.sh configs/experiment_config.mac_mixedbit_story.yaml
```

It validates:
- Python version,
- package imports,
- `DATASET_DIR/wall_single` exists,
- checkpoint files exist from config,
- free disk warning threshold.

If preflight fails, fix that before running experiments.

---

## 7) Recommended Main Run (Mixed-Bit Story)

Run with explicit run name + log capture:

```bash
RUN_NAME=mixedbit_$(date +%Y%m%d_%H%M%S)
bash scripts/run_mixedbit_story.sh configs/experiment_config.mac_mixedbit_story.yaml "$RUN_NAME" 2>&1 | tee "logs/${RUN_NAME}.log"
```

What this pipeline runs:
- paired target generation,
- main mixed-bit frontier (Budget A),
- budget robustness subset (Budget B),
- encoder-retention curve,
- aggregations/statistics,
- transition + appendix figures.

---

## 8) Resume an Interrupted Run (No Overwrite)

If interrupted, resume safely by adding a resume tag:

```bash
bash scripts/run_mixedbit_story.sh configs/experiment_config.mac_mixedbit_story.yaml <existing_run_name> resume_try1 2>&1 | tee "logs/<existing_run_name>_resume_try1.log"
```

Resume writes new stage outputs under:
- `results/wall_mixedbit_story/<run>/resume/resume_try1/...`

It does **not** erase completed stage outputs.

---

## 9) Other Useful Pipelines

### A) Standard Mac baseline/grid pipeline

```bash
bash scripts/run_mac_pipeline.sh configs/experiment_config.mac.yaml my_mac_run 2>&1 | tee logs/my_mac_run.log
```

### B) Transition study pipeline

```bash
bash scripts/run_transition_pipeline.sh configs/experiment_config.mac_transition_study.yaml transition_run_01 2>&1 | tee logs/transition_run_01.log
```

Resume transition run:

```bash
bash scripts/resume_transition_run.sh configs/experiment_config.mac_transition_study.yaml transition_run_01 resume_try1 2>&1 | tee logs/transition_run_01_resume.log
```

---

## 10) Where Outputs Go

For run name `<run_name>` in mixed-bit story:

- Metrics/root outputs:
  - `results/wall_mixedbit_story/<run_name>/...`
  - `results/<run_name>/summary.csv`
  - `results/<run_name>/summary_grouped.csv`
  - `results/<run_name>/mixedbit_pairwise_stats.csv`
- Notes:
  - `notes/<run_name>/mixedbit_story_notes.md`
- Figures:
  - `figures_mixedbit_story/<run_name>/transition_frontier.pdf`
  - `figures_mixedbit_story/<run_name>/budget_sensitivity.pdf`
  - `figures_mixedbit_story/<run_name>/encoder_retention_curve.pdf`
  - `figures_mixedbit_story/<run_name>/appendix_*.pdf`

General no-overwrite behavior:
- use a **new run name** for a fresh run,
- use resume mode for continuation into run-specific resume subfolders.

---

## 11) Paper Build

Compile paper:

```bash
bash scripts/compile_paper.sh
```

Outputs:
- `paper/main.pdf`
- `paper/paper.pdf` (copy of `main.pdf`)

Submission recommendation:
- use `paper/paper.pdf` (or `release/paper.pdf` after bundling).

Create release bundle:

```bash
bash scripts/build_release_bundle.sh
```

Outputs:
- `release/paper.pdf`
- `release/supplemental.zip`

---

## 12) Demo (Optional)

If demo artifacts exist:

```bash
streamlit run demo/app.py
```

---

## 13) Common Errors + Fixes

### `Config not found: configs/`
You passed a directory instead of a YAML file. Use a concrete config path, e.g.:

```bash
bash scripts/run_mixedbit_story.sh configs/experiment_config.mac_mixedbit_story.yaml my_run
```

### `ckpt_base_path is still placeholder`
Edit config and set:

```yaml
ckpt_base_path: /Users/suraj/Downloads
```

### Missing checkpoint file for model epoch
If config has `model_epoch: latest`, ensure:
- `/Users/suraj/Downloads/outputs/wall_single/checkpoints/model_latest.pth`

### Gym warning about maintenance / NumPy 2.0
This warning is expected in this stack and does not by itself mean failure. Continue if runs proceed normally.

### MPS errors
Fallback to CPU:

```bash
export DINO_WM_DEVICE=cpu
```

### Existing run folder errors
Use a new run name for fresh runs, or resume mode for continuation.

---

## 14) Minimal Collaborator Quickstart (Copy/Paste)

```bash
cd /path/to/EAI_DINO
bash scripts/setup_mac_env.sh python3.11 .venv311
source .venv311/bin/activate
export DATASET_DIR=/Users/suraj/Downloads
export DINO_WM_DEVICE=mps

# verify config has:
# dino.ckpt_base_path: /Users/suraj/Downloads
# dino.model_name: wall_single
# dino.model_epoch: latest

bash scripts/mac_preflight.sh configs/experiment_config.mac_mixedbit_story.yaml
RUN_NAME=mixedbit_$(date +%Y%m%d_%H%M%S)
bash scripts/run_mixedbit_story.sh configs/experiment_config.mac_mixedbit_story.yaml "$RUN_NAME" 2>&1 | tee "logs/${RUN_NAME}.log"
```

This is the fastest path from zero setup to paper-ready run outputs.
