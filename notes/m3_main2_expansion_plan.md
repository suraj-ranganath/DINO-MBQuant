# M3 Main2-Expansion Plan (RQ-Driven)

This plan expands `main2.pdf` rather than repeating it.  
`main2.pdf` establishes three preliminary claims: (1) INT8 stable, (2) INT3 collapse, (3) INT4 transition where mixed (`enc FP16 + pred INT4`) beats uniform INT4.

The expanded plan deepens each claim with broader budgets, stronger paired statistics, and more granular allocation tests.

## RQ1 (Primary): Is mixed INT4 consistently better than uniform INT4?

- Motivation from `main2.pdf`: mixed INT4 > uniform INT4 at one budget.
- Expansion:
  - Evaluate on three budgets (`bA`,`bB`,`bC`), not one.
  - Increase seeds to `1..6` for key variants.
  - Use paired-goal deltas with bootstrap CI.
- Evidence files:
  - `paired_delta_bA_mixed_vs_uniform_int4.json`
  - `paired_delta_bB_mixed_vs_uniform_int4.json`
  - `paired_delta_bC_mixed_vs_uniform_int4.json`

## RQ2: How close can mixed INT4 get to INT8 and FP16?

- Motivation from `main2.pdf`: INT8 ~= FP16, mixed INT4 lower but promising.
- Expansion:
  - Include `uniform_int6/mixed_int6` to map the slope from 8->6->4 bits.
  - Re-estimate gaps from `mixed_int4` to `mixed_int8`, `uniform_int8`, and `fp16` across budgets.
- Desired result framing:
  - “Mixed INT4 is within X absolute success of INT8/FP16 under budget Y.”

## RQ3: Does location of preserved encoder precision matter (head vs tail)?

- Motivation from `main2.pdf`: preserving encoder helps, but location untested.
- Expansion:
  - Compare `encfp16_25/50/75` (tail-preserved) against `encheadfp16_25/50` (head-preserved).
  - Analyze paired delta `encfp16_50 - encheadfp16_50`.
- Evidence files:
  - `paired_delta_b*_tail50_vs_head50_int4.json`

## RQ4: Is allocation directionality important at similar average precision?

- Motivation from `main2.pdf`: mixed can help, but asymmetric direction not isolated.
- Expansion:
  - Compare `mixed_e8_p4` vs `mixed_e4_p8`.
  - Compare `mixed_e6_p4` vs `mixed_e4_p6`.
- Evidence files:
  - `paired_delta_b*_e8p4_vs_e4p8.json`

## RQ5: Is the mixed-INT4 gain robust to goal source?

- Motivation from `main2.pdf`: results from fixed setup; test generalization.
- Expansion:
  - Add random-goal budgets (`bA_rand`,`bB_rand`) with `goal_source=random_state`.
  - Re-test key variants: `fp16,uniform_int8,mixed_int8,uniform_int4,mixed_int4`.
- Evidence files:
  - `paired_delta_bA_rand_mixed_vs_uniform_int4.json`
  - `paired_delta_bB_rand_mixed_vs_uniform_int4.json`

## RQ6 (Replacement): Does mixed INT4 win on the same episodes, not just in mean rate?

- Motivation from `main2.pdf`: reported mean success deltas, but no episode-level dominance analysis.
- Expansion:
  - Compute paired contingency over identical `(budget,seed,episode)` tuples:
    - `mixed wins` (`mixed=1, uniform=0`)
    - `uniform wins` (`mixed=0, uniform=1`)
    - `both win`, `both fail`
  - Run this for `mixed_int4 vs uniform_int4` and `encfp16_50 vs encheadfp16_50`.
- Evidence files:
  - `episode_matchup_int4_mixed_vs_uniform.json`
  - `episode_matchup_tail50_vs_head50.json`

## Pipeline scripts

- Full expansion:
  - `scripts/run_main2_expansion_pipeline_m3.sh`
- Smoke:
  - `scripts/run_main2_expansion_smoke_m3.sh`

Configs:
- Full: `configs/experiment_config.mac_m3_main2_expansion.yaml`
- Smoke: `configs/experiment_config.mac_m3_main2_expansion_smoke.yaml`

## Run order

1. Smoke:
```bash
source .venv311/bin/activate
export DATASET_DIR=/Users/anishpatnaik/Downloads
export DINO_WM_DEVICE=mps
bash scripts/run_main2_expansion_smoke_m3.sh configs/experiment_config.mac_m3_main2_expansion_smoke.yaml m3_main2exp_smoke_v1
```

2. Full expansion:
```bash
bash scripts/run_main2_expansion_pipeline_m3.sh configs/experiment_config.mac_m3_main2_expansion.yaml m3_main2exp_v1 2>&1 | tee logs_m3_main2exp_v1.txt
```

## Paper production plan (ICLR 2026, 4-8 pages)

Use `paper/main.tex` as base and rewrite sections around RQ1-RQ6.

- Section 1 (Intro): restate `main2.pdf` preliminary observation and define expansion hypotheses.
- Section 2 (Protocol): paired vs random-goal budgets, variant families, paired CI method.
- Section 3 (Results):
  - Fig A: multi-budget frontier (`fp16/int8/int6/int4`).
  - Fig B: paired deltas for `mixed_int4 - uniform_int4` across budgets.
  - Fig C: head-vs-tail retention at INT4.
  - Table A: closeness gaps of `mixed_int4` vs `int8/fp16`.
  - Table B: asymmetric allocation comparisons (`e8/p4` vs `e4/p8`).
  - Table C: episode-level matchup contingency (`mixed wins` vs `uniform wins`).
- Section 4 (Discussion): what supports/weakens the main claim.
- Section 5 (Limitations): M3-only, Wall-only, weight-only quantization.
- Appendix: all paired delta JSON summaries and additional grouped tables.
