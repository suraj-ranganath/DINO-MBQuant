# Paper Numbers (Auto-generated)

Summary source: `results/summary_bitsweep_h9_mixed.csv`

Grouped source: `results/summary_bitsweep_h9_mixed_grouped.csv`

## Headline (highest opt_steps)

- opt_steps used for headline: `2`
- FP16 success rate: `0.500`
- Uniform INT8 success rate: `0.500`
- Mixed INT8 success rate: `0.500`
- Mixed - Uniform gap: `+0.000`

## Mixed vs Uniform by opt_steps

| opt_steps | Uniform mean | Mixed mean | Gap (Mixed-Uniform) | Uniform CI95 | Mixed CI95 |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.500 | 0.500 | +0.000 | [0.173, 0.827] | [0.173, 0.827] |
## Per-setting results

| Variant | opt_steps | Success mean | Success std | Avg plan time (s) | Model size (MB) | Peak mem (MB) |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 2 | 0.500 | 0.236 | 54.150 | 204.99 | 0.00 |
| Mixed E4/P3 | 2 | 0.000 | 0.000 | 56.352 | 65.76 | 0.00 |
| Mixed E8/P3 | 2 | 0.000 | 0.000 | 54.323 | 75.88 | 0.00 |
| Mixed E8/P4 | 2 | 0.083 | 0.118 | 53.456 | 78.25 | 0.00 |
| Mixed INT3 (enc FP16) | 2 | 0.000 | 0.000 | 53.214 | 136.47 | 0.00 |
| Mixed INT4 (enc FP16) | 2 | 0.167 | 0.000 | 52.735 | 138.84 | 0.00 |
| Mixed INT8 (enc FP16) | 2 | 0.500 | 0.236 | 54.462 | 148.31 | 0.00 |
| Uniform INT3 | 2 | 0.000 | 0.000 | 52.515 | 63.23 | 0.00 |
| Uniform INT4 | 2 | 0.083 | 0.118 | 54.986 | 68.12 | 0.00 |
| Uniform INT8 | 2 | 0.500 | 0.236 | 55.486 | 87.72 | 0.00 |

## Suggested abstract sentence

- On the Wall task at opt_steps=2 (Mac-only preliminary setting), mixed INT8 improved success over uniform INT8 by +0.000 absolute while retaining similar local efficiency footprint.
