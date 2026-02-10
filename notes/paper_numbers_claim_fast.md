# Paper Numbers (Auto-generated)

Summary source: `results/summary_claim_fast.csv`

Grouped source: `results/summary_claim_fast_grouped.csv`

## Headline (highest opt_steps)

- opt_steps used for headline: `20`
- FP16 success rate: `1.000`
- Uniform INT8 success rate: `1.000`
- Mixed INT8 success rate: `1.000`
- Mixed - Uniform gap: `+0.000`

## Mixed vs Uniform by opt_steps

| opt_steps | Uniform mean | Mixed mean | Gap (Mixed-Uniform) | Uniform CI95 | Mixed CI95 |
|---:|---:|---:|---:|---:|---:|
| 5 | 1.000 | 1.000 | +0.000 | [1.000, 1.000] | [1.000, 1.000] |
| 10 | 1.000 | 1.000 | +0.000 | [1.000, 1.000] | [1.000, 1.000] |
| 20 | 1.000 | 1.000 | +0.000 | [1.000, 1.000] | [1.000, 1.000] |
## Per-setting results

| Variant | opt_steps | Success mean | Success std | Avg plan time (s) | Model size (MB) | Peak mem (MB) |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 5 | 1.000 | 0.000 | 7.820 | 204.99 | 0.00 |
| Mixed INT8 (encoder FP16) | 5 | 1.000 | 0.000 | 8.065 | 148.31 | 0.00 |
| Uniform INT8 | 5 | 1.000 | 0.000 | 7.945 | 87.72 | 0.00 |
| FP16 | 10 | 1.000 | 0.000 | 9.073 | 204.99 | 0.00 |
| Mixed INT8 (encoder FP16) | 10 | 1.000 | 0.000 | 9.195 | 148.31 | 0.00 |
| Uniform INT8 | 10 | 1.000 | 0.000 | 9.144 | 87.72 | 0.00 |
| FP16 | 20 | 1.000 | 0.000 | 9.279 | 204.99 | 0.00 |
| Mixed INT8 (encoder FP16) | 20 | 1.000 | 0.000 | 9.272 | 148.31 | 0.00 |
| Uniform INT8 | 20 | 1.000 | 0.000 | 9.347 | 87.72 | 0.00 |

## Suggested abstract sentence

- On the Wall task at opt_steps=20 (Mac-only preliminary setting), mixed INT8 improved success over uniform INT8 by +0.000 absolute while retaining similar local efficiency footprint.
