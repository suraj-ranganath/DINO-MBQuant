# Paper Numbers (Auto-generated)

Summary source: `/Users/suraj/Desktop/EAI_DINO/results/transition_run_v2/summary.csv`

Grouped source: `/Users/suraj/Desktop/EAI_DINO/results/transition_run_v2/summary_grouped.csv`

## Headline (highest opt_steps)

- opt_steps used for headline: `2`
- FP16 success rate: `0.533`
- Uniform INT8 success rate: `0.533`
- Mixed INT8 success rate: `0.533`
- Mixed - Uniform gap: `+0.000`

## Mixed vs Uniform by opt_steps

| opt_steps | Uniform mean | Mixed mean | Gap (Mixed-Uniform) | Uniform CI95 | Mixed CI95 |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.533 | 0.533 | +0.000 | [0.468, 0.599] | [0.468, 0.599] |
## Per-setting results

| Variant | opt_steps | Success mean | Success std | Avg plan time (s) | Model size (MB) | Peak mem (MB) |
|---|---:|---:|---:|---:|---:|---:|
| INT4 (0% enc FP16) | 2 | 0.100 | 0.000 | 41.610 | 68.12 | 0.00 |
| INT4 (100% enc FP16) | 2 | 0.250 | 0.071 | 42.473 | 138.84 | 0.00 |
| INT4 (25% enc FP16) | 2 | 0.050 | 0.071 | 41.994 | 85.80 | 0.00 |
| INT4 (50% enc FP16) | 2 | 0.150 | 0.071 | 42.181 | 103.48 | 0.00 |
| INT4 (75% enc FP16) | 2 | 0.050 | 0.071 | 42.247 | 121.16 | 0.00 |
| FP16 | 2 | 0.533 | 0.058 | 49.664 | 204.99 | 0.00 |
| Mixed INT3 (enc FP16) | 2 | 0.000 | 0.000 | 41.531 | 136.47 | 0.00 |
| Mixed INT4 (enc FP16) | 2 | 0.267 | 0.058 | 48.360 | 138.84 | 0.00 |
| Mixed INT8 (enc FP16) | 2 | 0.533 | 0.058 | 48.294 | 148.31 | 0.00 |
| Uniform INT3 | 2 | 0.000 | 0.000 | 41.105 | 63.23 | 0.00 |
| Uniform INT4 | 2 | 0.067 | 0.058 | 46.540 | 68.12 | 0.00 |
| Uniform INT8 | 2 | 0.533 | 0.058 | 50.441 | 87.72 | 0.00 |

## Suggested abstract sentence

- On the Wall task at opt_steps=2 (Mac-only preliminary setting), mixed INT8 improved success over uniform INT8 by +0.000 absolute while retaining similar local efficiency footprint.
