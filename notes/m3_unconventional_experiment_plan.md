# M3 Unconventional Experiment Plan

Goal: strengthen the claim that quantization outcome depends on both bitwidth (`how many bits`) and allocation (`where bits are allocated`), using experiments that are meaningfully different from `main2.pdf`.

## What is new vs `main2.pdf`

- Adds INT6 midpoint variants (`uniform_int6`, `mixed_int6`) to test whether allocation effects already appear above the 4-bit frontier.
- Adds asymmetric cross-allocation families (`E8/P4`, `E4/P8`, `E6/P4`, `E4/P6`) to test directional sensitivity, not just "mixed vs uniform".
- Adds positional encoder-retention comparison:
  - tail-preserved FP16 (`encfp16_*`) vs
  - head-preserved FP16 (`encheadfp16_*`).
- Adds a third harder budget (`bC`: `opt_steps=4`, `goal_H=12`, `planner_max_iter=4`) to test whether the mixed-vs-uniform INT4 gap widens under stress.
- Adds boosted-seed rerun focused only on `mixed_int4` vs `uniform_int4` for tighter paired confidence intervals.
- Adds goal-source robustness budgets (`bA_rand`, `bB_rand`) with `goal_source=random_state` to check that paired-file setup is not creating the observed effect.

## Preconditions

1. Python 3.10/3.11 environment.
2. `third_party_dino_wm` exists and imports successfully.
3. Dataset exists at `$DATASET_DIR/wall_single`.
4. Checkpoint exists at:
   - `/Users/anishpatnaik/Downloads/outputs/wall_single/hydra.yaml`
   - `/Users/anishpatnaik/Downloads/outputs/wall_single/checkpoints/model_latest.pth`

Config used: `configs/experiment_config.mac_m3_unconventional.yaml`.

## Main pipeline command

```bash
source .venv311/bin/activate
export DATASET_DIR=/Users/anishpatnaik/Downloads
export DINO_WM_DEVICE=mps
bash scripts/run_unconventional_pipeline_m3.sh configs/experiment_config.mac_m3_unconventional.yaml m3_unconventional_v1 2>&1 | tee logs_m3_unconventional_v1.txt
```

## Experiment blocks

### A) Frontier + midpoint sweep

- Budgets: `bA,bB,bC`
- Variants:
  - `fp16`
  - `uniform_int8`, `mixed_int8`
  - `uniform_int6`, `mixed_int6`
  - `uniform_int4`, `mixed_int4`
  - `uniform_int3`, `mixed_int3`
- Question answered:
  - Is module-aware allocation still beneficial at 4-bit under multiple budgets?
  - Do allocation effects also appear at a less extreme bitwidth (6-bit)?

### B) Positional retention ablation (new)

- Budgets: `bA,bB,bC`
- Variants:
  - tail-preserved: `encfp16_0,25,50,75,100`
  - head-preserved: `encheadfp16_25,50`
- Question answered:
  - If the same percent of encoder layers are FP16, does retaining tail layers outperform retaining head layers?
  - This tests whether *where inside encoder* precision is allocated matters.

### C) Focused statistical tightening for core claim

- Budgets: `bA,bB,bC`
- Variants: `uniform_int4`, `mixed_int4`
- Seeds override: `1,2,3,4,5`
- Question answered:
  - Is `mixed_int4 > uniform_int4` robust with larger paired sample size?

### D) Asymmetric cross-allocation families (new)

- Budgets: `bA,bB,bC`
- Variants:
  - `mixed_e8_p4`, `mixed_e4_p8`, `mixed_e6_p4`, `mixed_e4_p6`
- Question answered:
  - If average precision is similar, does putting higher precision in encoder vs predictor change success asymmetrically?
  - This targets directionality of precision placement.

### E) Goal-source robustness (new)

- Budgets: `bA_rand`, `bB_rand`
- Variants: `fp16`, `uniform_int4`, `mixed_int4`, `encfp16_50`, `encheadfp16_50`
- Question answered:
  - Does the core INT4 allocation effect persist when goals are sampled from `random_state` instead of fixed paired target files?

## Primary success criteria

1. Core claim:
   - Paired delta `mixed_int4 - uniform_int4` is positive in `bA` and `bB`, preferably also `bC`.
2. Allocation-within-encoder claim:
   - `encfp16_50 - encheadfp16_50` paired delta positive in at least two budgets.
3. Threshold claim:
   - INT3 shows clear collapse relative to INT4.
4. Directionality claim:
   - `mixed_e8_p4` outperforms `mixed_e4_p8` in at least two budgets.

## Key outputs

- Summary tables:
  - `results/<run_name>/summary.csv`
  - `results/<run_name>/summary_grouped.csv`
- Episode-level paired outcomes:
  - `results/<run_name>/episode_outcomes.csv`
- Paired deltas (claim-focused):
  - `results/<run_name>/paired_delta_bA_mixed_vs_uniform_int4.json`
  - `results/<run_name>/paired_delta_bB_mixed_vs_uniform_int4.json`
  - `results/<run_name>/paired_delta_bC_mixed_vs_uniform_int4.json`
  - `results/<run_name>/paired_delta_bA_tail50_vs_head50_int4.json`
  - `results/<run_name>/paired_delta_bA_e8p4_vs_e4p8.json`
  - `results/<run_name>/paired_delta_bA_rand_mixed_vs_uniform_int4.json`
- Mechanistic:
  - `results/<run_name>/mechanistic_metrics.csv`
  - `results/<run_name>/mechanistic_correlations.json`
- Figures:
  - `figures_m3_unconventional/<run_name>/transition_frontier.pdf`
  - `figures_m3_unconventional/<run_name>/budget_sensitivity.pdf`
  - `figures_m3_unconventional/<run_name>/encoder_retention_curve.pdf`
## Runtime control (if needed)

- Fast smoke for this config:
  - `python -m experiments.run_wall_grid --config configs/experiment_config.mac_m3_unconventional.yaml --run-name <run_name> --budget-ids bA --variants fp16,uniform_int4,mixed_int4 --seeds 1 --max-runs 3`
- Resume interrupted run:
  - add `--allow-existing --skip-existing` to `experiments.run_wall_grid` calls.
