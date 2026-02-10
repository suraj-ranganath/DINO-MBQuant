# Mac 6-Hour Runbook (Wall MVP)

## 0) Set paths

```bash
cd /Users/suraj/Desktop/EAI_DINO
bash scripts/setup_mac_env.sh python3.11 .venv311
source .venv311/bin/activate
export DATASET_DIR=/ABS/PATH/TO/data
export DINO_WM_DEVICE=mps   # set cpu if mps causes operator issues
# Edit ckpt_base_path in configs/experiment_config.mac.yaml
```

## 1) Run full pipeline

```bash
bash scripts/run_mac_pipeline.sh configs/experiment_config.mac.yaml
```

## 2) Launch demo

```bash
streamlit run demo/app.py
```

## 3) Finalize paper

- Open `notes/paper_numbers.md`
- Paste numbers into `paper/main.tex` (or `paper/paper.tex`)
- Compile:

```bash
bash scripts/compile_paper.sh
```

## 4) Build release bundle

```bash
bash scripts/build_release_bundle.sh
```

Outputs:
- `release/paper.pdf`
- `release/supplemental.zip`
